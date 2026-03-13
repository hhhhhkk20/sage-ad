"""
SAGE-AD: LLM Inference Strategies for AD Prediction
=====================================================
Implements multi-strategy LLM inference for Alzheimer's disease risk prediction
from longitudinal survey profiles (zero-shot, few-shot, chain-of-thought).

Supports OpenAI-compatible APIs (GPT, Qwen, DeepSeek, SiliconFlow),
Google Gemini, and Anthropic Claude.
"""

import os
import re
import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed


class InferenceStrategy(Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "cot"


@dataclass
class TemporalSymptomProfile:
    """Longitudinal health profile for a single survey participant."""
    participant_id: str
    age_at_assessments: List[int]
    cognitive_features: List[Dict[str, any]]
    functional_features: List[Dict[str, any]]
    physiological_features: List[Dict[str, any]]
    neuropsychiatric_features: List[Dict[str, any]]
    demographics: Dict[str, any]

    def to_narrative_text(self) -> str:
        """Convert structured profile to natural language narrative."""
        narrative = f"Patient ID: {self.participant_id}\n"
        narrative += f"Demographics: {self._format_demographics()}\n\n"
        narrative += "Longitudinal Health Profile:\n"
        for idx, age in enumerate(self.age_at_assessments):
            narrative += f"\n--- Assessment at age {age} ---\n"
            narrative += f"Cognitive: {self._format_features(self.cognitive_features[idx])}\n"
            narrative += f"Functional: {self._format_features(self.functional_features[idx])}\n"
            narrative += f"Physiological: {self._format_features(self.physiological_features[idx])}\n"
            narrative += f"Neuropsychiatric: {self._format_features(self.neuropsychiatric_features[idx])}\n"
        return narrative

    def _format_demographics(self) -> str:
        return ", ".join([f"{k}: {v}" for k, v in self.demographics.items()])

    def _format_features(self, features: Dict[str, any]) -> str:
        return ", ".join([f"{k}={v}" for k, v in features.items()])


class PromptTemplate:
    """
    Prompt templates for different inference strategies.
    Based on Extended Data Fig. 2 in the SAGE-AD paper.
    """

    SYSTEM_ROLE = (
        "You are a highly qualified specialist in Alzheimer's disease, "
        "with extensive clinical and research experience. "
        "Your task is to analyze longitudinal health profiles from community-based "
        "surveys and predict the likelihood of AD onset."
    )

    @staticmethod
    def get_task_specification(prediction_horizon: int) -> str:
        return (
            f"Based on the provided longitudinal health profile, predict whether "
            f"this participant will develop Alzheimer's disease within "
            f"{prediction_horizon} year(s). "
            "Consider cognitive function, functional status, physiological health, "
            "and neuropsychiatric symptoms across all available time points."
        )

    @staticmethod
    def get_output_constraints() -> str:
        return (
            "Provide your prediction in the following format:\n"
            "Prediction: [Yes/No]\n"
            "Confidence: [Low/Medium/High]\n"
            "Brief Justification: [1-2 sentences]"
        )

    @staticmethod
    def get_zero_shot_output_constraints() -> str:
        """Strict zero-shot output — Yes/No only."""
        return (
            "Additional Instructions:\n"
            "• Respond ONLY with either 'Yes' or 'No'.\n"
            "• Do not include any explanations or additional text."
        )

    @staticmethod
    def get_cot_instructions() -> str:
        return (
            "Before making your final prediction, reason step by step:\n"
            "1. Identify concerning patterns within each clinical dimension "
            "(cognitive, functional, physiological, neuropsychiatric).\n"
            "2. Integrate findings across dimensions.\n"
            "3. Consider temporal trajectories and progressive decline patterns.\n"
            "4. Synthesize observations into a diagnostic conclusion.\n\n"
            "Show your step-by-step reasoning, then provide the final prediction "
            "in the format:\nPrediction: [Yes/No]"
        )

    # Built-in few-shot example pairs (drawn from ELSA balanced profiles)
    FEW_SHOT_EXAMPLES = [
        {
            "profile": (
                "Demographics: Age: 68, Sex: Female, Education: 12 years\n\n"
                "--- Assessment at age 64 ---\n"
                "Cognitive: word_recall_immediate=9, word_recall_delayed=7, orientation=5\n"
                "Functional: ADL_impairment=0, IADL_impairment=0\n"
                "Physiological: grip_strength=24, gait_speed=1.1\n"
                "Neuropsychiatric: depression_score=2, anxiety=low\n\n"
                "--- Assessment at age 66 ---\n"
                "Cognitive: word_recall_immediate=6, word_recall_delayed=4, orientation=4\n"
                "Functional: ADL_impairment=0, IADL_impairment=1\n"
                "Physiological: grip_strength=22, gait_speed=0.95\n"
                "Neuropsychiatric: depression_score=4, anxiety=moderate\n\n"
                "--- Assessment at age 68 ---\n"
                "Cognitive: word_recall_immediate=4, word_recall_delayed=2, orientation=3\n"
                "Functional: ADL_impairment=1, IADL_impairment=3\n"
                "Physiological: grip_strength=19, gait_speed=0.8\n"
                "Neuropsychiatric: depression_score=6, anxiety=moderate"
            ),
            "label": "Yes",
            "justification": (
                "Progressive decline in memory recall, increasing functional impairment, "
                "and worsening neuropsychiatric symptoms indicate high AD risk."
            ),
        },
        {
            "profile": (
                "Demographics: Age: 72, Sex: Male, Education: 16 years\n\n"
                "--- Assessment at age 68 ---\n"
                "Cognitive: word_recall_immediate=8, word_recall_delayed=6, orientation=5\n"
                "Functional: ADL_impairment=0, IADL_impairment=0\n"
                "Physiological: grip_strength=34, gait_speed=1.3\n"
                "Neuropsychiatric: depression_score=1, anxiety=low\n\n"
                "--- Assessment at age 70 ---\n"
                "Cognitive: word_recall_immediate=8, word_recall_delayed=7, orientation=5\n"
                "Functional: ADL_impairment=0, IADL_impairment=0\n"
                "Physiological: grip_strength=32, gait_speed=1.25\n"
                "Neuropsychiatric: depression_score=1, anxiety=low\n\n"
                "--- Assessment at age 72 ---\n"
                "Cognitive: word_recall_immediate=9, word_recall_delayed=7, orientation=5\n"
                "Functional: ADL_impairment=0, IADL_impairment=0\n"
                "Physiological: grip_strength=31, gait_speed=1.2\n"
                "Neuropsychiatric: depression_score=2, anxiety=low"
            ),
            "label": "No",
            "justification": (
                "Stable or slightly improving cognitive performance with no functional "
                "decline suggests low AD risk."
            ),
        },
    ]


class LLMInference:
    """
    LLM inference engine supporting multiple model backends and strategies.

    Supported backends (auto-detected from model_name):
      - OpenAI  (gpt-*, o1-*, o3-*)
      - Anthropic Claude  (claude-*)
      - Google Gemini  (gemini-*)
      - Qwen / DashScope  (qwen-*, qwen3-*)
      - DeepSeek  (deepseek-*)
      - SiliconFlow / any OpenAI-compatible  (pass base_url explicitly)
    """

    _API_KEY_ENV = {
        "openai":     "OPENAI_API_KEY",
        "anthropic":  "ANTHROPIC_API_KEY",
        "google":     "GOOGLE_API_KEY",
        "dashscope":  "DASHSCOPE_API_KEY",
        "siliconflow": "SILICONFLOW_API_KEY",
    }

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        max_retries: int = 3,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        profile: TemporalSymptomProfile,
        strategy: InferenceStrategy,
        prediction_horizon: int = 1,
        few_shot_examples: Optional[List[Tuple[TemporalSymptomProfile, bool]]] = None,
    ) -> Dict[str, any]:
        """Make AD prediction from a TemporalSymptomProfile."""
        prompt = self._build_prompt(profile, strategy, prediction_horizon, few_shot_examples)
        raw = self._call_llm_api(prompt)
        return self._parse_response(raw, strategy)

    def predict_from_text(
        self,
        profile_text: str,
        strategy: InferenceStrategy = InferenceStrategy.ZERO_SHOT,
        prediction_horizon: int = 1,
    ) -> Dict[str, any]:
        """
        Predict directly from free-text profile (.txt files as used in ELSA/HRS/SHARE).
        """
        system = PromptTemplate.SYSTEM_ROLE
        task = PromptTemplate.get_task_specification(prediction_horizon)

        if strategy == InferenceStrategy.ZERO_SHOT:
            constraints = PromptTemplate.get_zero_shot_output_constraints()
            prompt = f"{system}\n\n{task}\n\n{constraints}\n\nPatient profile:\n{profile_text}"

        elif strategy == InferenceStrategy.FEW_SHOT:
            examples_text = self._format_builtin_examples()
            constraints = PromptTemplate.get_zero_shot_output_constraints()
            prompt = (
                f"{system}\n\n{task}\n\n"
                f"=== Demonstrative Examples ===\n{examples_text}\n\n"
                f"{constraints}\n\nPatient profile:\n{profile_text}"
            )

        else:  # CoT
            cot = PromptTemplate.get_cot_instructions()
            prompt = f"{system}\n\n{task}\n\n{cot}\n\nPatient profile:\n{profile_text}"

        raw = self._call_llm_api(prompt)
        return self._parse_response(raw, strategy)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, profile, strategy, prediction_horizon, few_shot_examples):
        parts = [
            PromptTemplate.SYSTEM_ROLE,
            PromptTemplate.get_task_specification(prediction_horizon),
        ]

        if strategy == InferenceStrategy.FEW_SHOT:
            parts.append("=== Demonstrative Examples ===")
            if few_shot_examples:
                for i, (ex, label) in enumerate(few_shot_examples, 1):
                    parts.append(f"\nExample {i}:\n{ex.to_narrative_text()}")
                    parts.append(f"Ground Truth: {'Yes (AD positive)' if label else 'No (Healthy)'}")
            else:
                parts.append(self._format_builtin_examples())

        if strategy == InferenceStrategy.CHAIN_OF_THOUGHT:
            parts.append(PromptTemplate.get_cot_instructions())

        parts.append("=== Target Patient Profile ===")
        parts.append(profile.to_narrative_text())
        parts.append(PromptTemplate.get_output_constraints())

        return "\n\n".join(parts)

    def _format_builtin_examples(self) -> str:
        lines = []
        for i, ex in enumerate(PromptTemplate.FEW_SHOT_EXAMPLES, 1):
            lines.append(f"Example {i}:\n{ex['profile']}")
            lines.append(f"Prediction: {ex['label']}")
            lines.append(f"Brief Justification: {ex['justification']}\n")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # API dispatch
    # ------------------------------------------------------------------

    def _call_llm_api(self, prompt: str) -> str:
        name = self.model_name.lower()
        if self.base_url:
            return self._call_openai_compat(prompt, self.base_url, self._get_key("openai"))
        if "claude" in name:
            return self._call_claude(prompt)
        if "gemini" in name:
            return self._call_gemini(prompt)
        if any(k in name for k in ("qwen", "deepseek", "baichuan", "llama", "mistral")):
            return self._call_openai_compat(prompt, self._infer_base_url(), self._get_key("dashscope"))
        return self._call_openai_compat(prompt, "https://api.openai.com/v1", self._get_key("openai"))

    def _infer_base_url(self) -> str:
        name = self.model_name.lower()
        if "qwen" in name:
            return "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if "deepseek" in name:
            return "https://api.deepseek.com/v1"
        return "https://api.siliconflow.cn/v1"

    def _get_key(self, backend: str) -> str:
        if self.api_key:
            return self.api_key
        env = self._API_KEY_ENV.get(backend, "OPENAI_API_KEY")
        key = os.environ.get(env, "")
        if not key:
            raise EnvironmentError(
                f"API key not found. Set {env} or pass api_key= to LLMInference."
            )
        return key

    def _call_openai_compat(self, prompt: str, base_url: str, api_key: str) -> str:
        """Call any OpenAI-compatible endpoint (GPT, Qwen, DeepSeek, SiliconFlow …)."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai>=1.0.0")

        client = OpenAI(api_key=api_key, base_url=base_url)
        for attempt in range(self.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"  [attempt {attempt+1}] error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        return "Error"

    def _call_claude(self, prompt: str) -> str:
        """Call Anthropic Claude API."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic>=0.18.0")

        client = anthropic.Anthropic(api_key=self._get_key("anthropic"))
        for attempt in range(self.max_retries):
            try:
                msg = client.messages.create(
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text.strip()
            except Exception as e:
                print(f"  [attempt {attempt+1}] Claude error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        return "Error"

    def _call_gemini(self, prompt: str) -> str:
        """Call Google Gemini API."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("pip install google-generativeai>=0.3.0")

        genai.configure(api_key=self._get_key("google"))
        model = genai.GenerativeModel(self.model_name)
        for attempt in range(self.max_retries):
            try:
                resp = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                    ),
                )
                return resp.text.strip()
            except Exception as e:
                print(f"  [attempt {attempt+1}] Gemini error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        return "Error"

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str, strategy: InferenceStrategy) -> Dict[str, any]:
        result = {
            "prediction": None,
            "confidence": "Unknown",
            "reasoning": "",
            "raw_response": raw,
        }
        if not raw or raw == "Error":
            return result

        lower = raw.lower().strip()

        # Prediction: look for explicit "Prediction: Yes/No" line
        m = re.search(r"prediction\s*:\s*(yes|no)", lower)
        if m:
            result["prediction"] = m.group(1) == "yes"
        else:
            if re.search(r"\byes\b", lower):
                result["prediction"] = True
            elif re.search(r"\bno\b", lower):
                result["prediction"] = False

        # Confidence
        m = re.search(r"confidence\s*:\s*(low|medium|high)", lower)
        if m:
            result["confidence"] = m.group(1).capitalize()

        # Reasoning
        for marker in ("brief justification:", "justification:", "reasoning:"):
            idx = lower.find(marker)
            if idx != -1:
                result["reasoning"] = raw[idx + len(marker):].strip().split("\n")[0].strip()
                break
        if not result["reasoning"] and strategy == InferenceStrategy.CHAIN_OF_THOUGHT:
            result["reasoning"] = raw

        return result


class BatchInference:
    """Parallel batch evaluation of an LLM over a cohort of profiles."""

    def __init__(self, model_name: str, max_workers: int = 4, **engine_kwargs):
        self.model_name = model_name
        self.max_workers = max_workers
        self._engine_kwargs = engine_kwargs

    def evaluate_cohort(
        self,
        profiles: List[TemporalSymptomProfile],
        true_labels: List[bool],
        strategy: InferenceStrategy,
        prediction_horizon: int = 1,
        few_shot_examples=None,
    ) -> Dict[str, any]:
        """Evaluate all profiles in parallel."""

        def _infer(idx_profile):
            idx, profile = idx_profile
            engine = LLMInference(self.model_name, **self._engine_kwargs)
            r = engine.predict(profile, strategy, prediction_horizon, few_shot_examples)
            return idx, r

        preds = [None] * len(profiles)
        confs = [None] * len(profiles)
        reasons = [None] * len(profiles)

        print(f"Evaluating {len(profiles)} profiles [{self.model_name}] "
              f"strategy={strategy.value} horizon={prediction_horizon}y")

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(_infer, (i, p)): i for i, p in enumerate(profiles)}
            done = 0
            for fut in as_completed(futs):
                idx, r = fut.result()
                preds[idx] = r["prediction"]
                confs[idx] = r["confidence"]
                reasons[idx] = r["reasoning"]
                done += 1
                if done % 50 == 0 or done == len(profiles):
                    print(f"  {done}/{len(profiles)} done")

        return {
            "predictions": preds,
            "true_labels": true_labels,
            "confidences": confs,
            "reasonings": reasons,
            "strategy": strategy.value,
            "model": self.model_name,
            "prediction_horizon": prediction_horizon,
        }

    def evaluate_from_files(
        self,
        profile_dir: str,
        gt_dict: Dict[int, int],
        strategy: InferenceStrategy = InferenceStrategy.ZERO_SHOT,
        prediction_horizon: int = 1,
        max_samples: Optional[int] = None,
    ) -> Dict[str, any]:
        """
        Evaluate profiles stored as individual .txt files (ELSA/HRS/SHARE format).
        Filename stem must be the integer patient ID.
        """
        from pathlib import Path

        files = sorted(Path(profile_dir).glob("*.txt"))
        if max_samples:
            files = files[:max_samples]

        def _process(f):
            pid = int(f.stem)
            if pid not in gt_dict:
                return None
            engine = LLMInference(self.model_name, **self._engine_kwargs)
            text = f.read_text(encoding="utf-8")
            r = engine.predict_from_text(text, strategy, prediction_horizon)
            return pid, r["prediction"], gt_dict[pid], r["raw_response"]

        predictions, actuals, ids, responses = [], [], [], []

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = [ex.submit(_process, f) for f in files]
            done = 0
            for fut in as_completed(futs):
                r = fut.result()
                if r is None:
                    continue
                pid, pred, label, raw = r
                if pred is None:
                    continue
                predictions.append(int(pred))
                actuals.append(int(label))
                ids.append(pid)
                responses.append(raw)
                done += 1
                if done % 50 == 0:
                    print(f"  {done} profiles processed")

        print(f"Done. Valid samples: {len(predictions)}")
        return {
            "predictions": predictions,
            "true_labels": actuals,
            "patient_ids": ids,
            "raw_responses": responses,
            "strategy": strategy.value,
            "model": self.model_name,
            "prediction_horizon": prediction_horizon,
        }
