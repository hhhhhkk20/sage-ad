"""
SAGE-AD: Multi-Specialist Agent System for AD Risk Assessment
=============================================================
Implements a panel of five virtual specialist agents that collaboratively
assess Alzheimer's disease risk from longitudinal health profiles.

Architecture (mirrors the CARE-AD pipeline):
  1. PrimaryCareDoctor  – overall health trajectory
  2. Neurologist        – memory, executive function, neurological status
  3. Psychiatrist       – mood, behaviour, neuropsychiatric symptoms
  4. Geriatrician       – frailty, functional decline, comorbidities
  5. ClinicalPsychologist – cognitive domain profiling
  6. ADSpecialist       – integrative final decision

Each specialist is stateless; the ADSpecialist receives all five analyses
and produces the binary (Yes / No) AD risk prediction.
"""

import re
import time
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from .llm_inference import LLMInference
except ImportError:
    from llm_inference import LLMInference  # flat import when src/ is on sys.path


# ---------------------------------------------------------------------------
# Specialist definitions
# ---------------------------------------------------------------------------

SPECIALIST_CONFIGS: Dict[str, Dict[str, str]] = {
    "primary_care": {
        "role": "a primary care physician with comprehensive knowledge of adult health and preventive care",
        "instruction": (
            "Please assess the patient's overall physical health, medical history, "
            "comorbid conditions, and daily functioning. Identify any concerns that "
            "may be related to Alzheimer's disease, and briefly explain why these "
            "findings are concerning."
        ),
        "additional": (
            "• Highlight red flags that warrant further neurological or cognitive evaluation.\n"
            "• Summarise key clinical observations that could guide next steps."
        ),
    },
    "neurologist": {
        "role": "a neurologist specialising in disorders of the brain and nervous system",
        "instruction": (
            "Please evaluate the patient's neurological status, including memory, "
            "executive function, language, and any motor or sensory findings. "
            "Identify concerns that may be related to Alzheimer's disease and explain "
            "the reasoning behind each concern."
        ),
        "additional": (
            "• Focus on memory loss, language difficulties, and other cognitive deficits.\n"
            "• Provide a concise rationale for suspected AD-related signs."
        ),
    },
    "psychiatrist": {
        "role": "a psychiatrist specialising in mental and emotional health",
        "instruction": (
            "Please evaluate the patient's psychiatric presentation, including mood, "
            "affect, and behavioural changes. Identify any concerns potentially linked "
            "to Alzheimer's disease and articulate the basis for each concern."
        ),
        "additional": (
            "• Assess depression, anxiety, or other psychiatric comorbidities that may "
            "mask or exacerbate cognitive decline.\n"
            "• Note behavioural disturbances, personality changes, or psychotic features "
            "indicative of AD."
        ),
    },
    "geriatrician": {
        "role": "a geriatrician with expertise in the care of older adults",
        "instruction": (
            "Please assess the patient's overall functional status, comorbidities, and "
            "geriatric syndromes (e.g., fall risk, frailty, polypharmacy). Identify any "
            "concerns related to Alzheimer's disease and provide a brief explanation."
        ),
        "additional": (
            "• Consider the patient's aging trajectory, functional independence, and "
            "support system.\n"
            "• Highlight interactions between comorbidities and cognitive decline."
        ),
    },
    "clinical_psychologist": {
        "role": "a clinical psychologist focusing on cognitive assessment",
        "instruction": (
            "Please evaluate the patient's cognitive domains (memory, attention, language, "
            "executive function, and visuospatial skills). Identify concerns that may "
            "indicate Alzheimer's disease and explain the underlying reasons."
        ),
        "additional": (
            "• Highlight patterns of cognitive performance characteristic of AD.\n"
            "• Reference relevant neuropsychological models where appropriate."
        ),
    },
}

AD_SPECIALIST_SYSTEM = (
    "You are a highly qualified specialist in Alzheimer's disease, with extensive "
    "clinical and research experience. You will receive a comprehensive longitudinal "
    "profile of a patient's AD-related signs and symptoms, together with assessments "
    "from five specialty clinicians. Integrate this information in light of recognised "
    "diagnostic criteria and determine whether the patient is likely to develop "
    "Alzheimer's disease in the future.\n\n"
    "Instructions:\n"
    "• Respond ONLY with either 'Yes' or 'No'.\n"
    "• Do not include any explanations or additional text."
)


def _build_specialist_prompt(role: str, instruction: str, additional: str, profile_text: str) -> str:
    return (
        f"You are {role}.\n"
        "A patient has been followed longitudinally in an ageing cohort study. "
        f"{instruction}\n\n"
        "Additional guidelines:\n"
        f"{additional}\n\n"
        "Respond in the format:\n"
        "Assessment: [Yes/No]\n"
        "Explanation: [brief explanation]\n\n"
        f"Patient profile:\n{profile_text}"
    )


def _build_integrator_prompt(
    profile_text: str, specialist_analyses: Dict[str, str]
) -> str:
    analyses_block = "\n".join(
        f"- {name.replace('_', ' ').title()}: {analysis}"
        for name, analysis in specialist_analyses.items()
    )
    return (
        f"{AD_SPECIALIST_SYSTEM}\n\n"
        f"Patient profile:\n{profile_text}\n\n"
        f"Specialist analyses:\n{analyses_block}"
    )


# ---------------------------------------------------------------------------
# Main multi-agent class
# ---------------------------------------------------------------------------

class MultiSpecialistPanel:
    """
    Convenes a panel of virtual specialists to assess AD risk.

    Each specialist calls the LLM independently (optionally in parallel).
    The AD Specialist then integrates all analyses into a final binary verdict.

    Args:
        model_name: LLM to use for all agents (can be overridden per specialist)
        base_url: Optional OpenAI-compatible base URL
        api_key: API key (falls back to environment variables)
        parallel: Whether to query specialists in parallel (default True)
        max_workers: Thread count for parallel inference
        temperature: Sampling temperature (low recommended for reproducibility)
        max_retries: Retries on transient API errors
    """

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        parallel: bool = True,
        max_workers: int = 5,
        temperature: float = 0.1,
        max_retries: int = 3,
    ):
        self.model_name = model_name
        self.parallel = parallel
        self.max_workers = max_workers
        self._engine_kwargs = dict(
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=256,
            max_retries=max_retries,
        )

    # ------------------------------------------------------------------

    def assess(self, profile_text: str) -> Dict[str, any]:
        """
        Run the full multi-specialist assessment pipeline.

        Args:
            profile_text: Free-text longitudinal patient profile

        Returns:
            dict with keys:
                - prediction (bool | None)  — final AD risk verdict
                - raw_prediction (str)      — raw LLM output from AD Specialist
                - specialist_analyses (dict)— per-specialist text outputs
        """
        # Step 1: gather specialist analyses
        if self.parallel:
            specialist_analyses = self._parallel_specialist_queries(profile_text)
        else:
            specialist_analyses = self._serial_specialist_queries(profile_text)

        # Step 2: AD Specialist integrates and decides
        integrator_prompt = _build_integrator_prompt(profile_text, specialist_analyses)
        engine = LLMInference(self.model_name, **self._engine_kwargs)
        raw = engine._call_llm_api(integrator_prompt)

        prediction = self._parse_binary(raw)

        return {
            "prediction": prediction,
            "raw_prediction": raw,
            "specialist_analyses": specialist_analyses,
        }

    def batch_assess(
        self,
        profile_dir: str,
        gt_dict: Dict[int, int],
        max_samples: Optional[int] = None,
    ) -> Dict[str, any]:
        """
        Batch evaluate profiles stored as .txt files.

        Args:
            profile_dir: Directory of per-patient .txt files (stem = patient ID)
            gt_dict: {patient_id: label} mapping
            max_samples: Limit to first N files

        Returns:
            Result dict with predictions, actuals, patient_ids
        """
        from pathlib import Path

        files = sorted(Path(profile_dir).glob("*.txt"))
        if max_samples:
            files = files[:max_samples]

        predictions, actuals, ids = [], [], []

        def _process(f):
            pid = int(f.stem)
            if pid not in gt_dict:
                return None
            text = f.read_text(encoding="utf-8")
            result = self.assess(text)
            return pid, result["prediction"], gt_dict[pid]

        print(f"Multi-agent assessment on {len(files)} profiles ...")

        with ThreadPoolExecutor(max_workers=max(1, self.max_workers // 5)) as ex:
            futs = [ex.submit(_process, f) for f in files]
            done = 0
            for fut in as_completed(futs):
                r = fut.result()
                if r is None:
                    continue
                pid, pred, label = r
                if pred is None:
                    continue
                predictions.append(int(pred))
                actuals.append(int(label))
                ids.append(pid)
                done += 1
                if done % 20 == 0:
                    print(f"  {done}/{len(files)} completed")

        print(f"Done. Valid samples: {len(predictions)}")
        return {"predictions": predictions, "true_labels": actuals, "patient_ids": ids}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parallel_specialist_queries(self, profile_text: str) -> Dict[str, str]:
        """Query all five specialists concurrently."""
        results = {}

        def _query(name: str, cfg: Dict[str, str]) -> Tuple[str, str]:
            prompt = _build_specialist_prompt(
                cfg["role"], cfg["instruction"], cfg["additional"], profile_text
            )
            engine = LLMInference(self.model_name, **self._engine_kwargs)
            return name, engine._call_llm_api(prompt)

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(_query, name, cfg): name for name, cfg in SPECIALIST_CONFIGS.items()}
            for fut in as_completed(futs):
                name, response = fut.result()
                results[name] = response

        return results

    def _serial_specialist_queries(self, profile_text: str) -> Dict[str, str]:
        """Query specialists sequentially (useful for rate-limited APIs)."""
        results = {}
        for name, cfg in SPECIALIST_CONFIGS.items():
            prompt = _build_specialist_prompt(
                cfg["role"], cfg["instruction"], cfg["additional"], profile_text
            )
            engine = LLMInference(self.model_name, **self._engine_kwargs)
            results[name] = engine._call_llm_api(prompt)
        return results

    @staticmethod
    def _parse_binary(raw: str) -> Optional[bool]:
        """Parse Yes/No from AD Specialist output."""
        if not raw or raw == "Error":
            return None
        lower = raw.lower().strip()
        if re.search(r"\byes\b", lower):
            return True
        if re.search(r"\bno\b", lower):
            return False
        return None
