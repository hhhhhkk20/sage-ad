"""
SAGE-AD: Interpretability Analysis Module
==========================================
This module implements interpretability analysis methods:
1. Feature domain ablation analysis
2. Dominance ratio computation
3. Feature interaction analysis
4. Reasoning process analysis (word cloud generation)
5. Temporal contribution analysis

Based on Figure 7 and interpretability methods in the SAGE-AD paper.
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from collections import Counter
from itertools import combinations
import warnings


@dataclass
class AblationResult:
    """
    Container for ablation experiment results.

    Attributes:
        configuration: Which dimensions were retained
        f1_score: F1 score for this configuration
        accuracy: Accuracy for this configuration
        n_dimensions: Number of dimensions retained
    """
    configuration: Tuple[str, ...]
    f1_score: float
    accuracy: float
    n_dimensions: int


class FeatureDomainAblation:
    """
    Systematic feature ablation to quantify domain contributions.
    """

    DOMAIN_NAMES = ["cognitive", "functional", "neuropsychiatric", "physiological", "demographic"]

    def __init__(self, inference_engine):
        """
        Initialize ablation analyzer.

        Args:
            inference_engine: LLM inference engine (from llm_inference module)
        """
        self.inference_engine = inference_engine

    def generate_all_configurations(self) -> List[Tuple[str, ...]]:
        """
        Generate all possible domain combinations (2^5 = 32 configurations).

        Returns:
            List of domain combinations
        """
        all_configs = []

        # Generate all subsets from size 0 to 5
        for size in range(len(self.DOMAIN_NAMES) + 1):
            for combo in combinations(self.DOMAIN_NAMES, size):
                all_configs.append(combo)

        return all_configs

    def ablate_profile(
        self,
        profile,  # TemporalSymptomProfile
        retained_domains: Tuple[str, ...]
    ):
        """
        Create ablated profile with only specified domains.

        Args:
            profile: Original temporal symptom profile
            retained_domains: Domains to retain

        Returns:
            Ablated profile
        """
        # Create a copy of the profile
        ablated_profile = profile.__class__(
            participant_id=profile.participant_id,
            age_at_assessments=profile.age_at_assessments,
            cognitive_features=profile.cognitive_features if "cognitive" in retained_domains else [{}] * len(profile.age_at_assessments),
            functional_features=profile.functional_features if "functional" in retained_domains else [{}] * len(profile.age_at_assessments),
            physiological_features=profile.physiological_features if "physiological" in retained_domains else [{}] * len(profile.age_at_assessments),
            neuropsychiatric_features=profile.neuropsychiatric_features if "neuropsychiatric" in retained_domains else [{}] * len(profile.age_at_assessments),
            demographics=profile.demographics if "demographic" in retained_domains else {}
        )

        return ablated_profile

    def run_ablation_experiment(
        self,
        profiles: List,  # List[TemporalSymptomProfile]
        true_labels: List[bool],
        prediction_horizon: int = 1,
        strategy=None  # InferenceStrategy
    ) -> Dict[Tuple[str, ...], AblationResult]:
        """
        Run comprehensive ablation experiment across all configurations.

        Args:
            profiles: List of participant profiles
            true_labels: Ground truth labels
            prediction_horizon: Years before diagnosis
            strategy: Inference strategy to use

        Returns:
            Dictionary mapping configurations to results
        """
        from evaluation_metrics import PerformanceEvaluator

        all_configs = self.generate_all_configurations()
        results = {}

        print(f"Running ablation experiment with {len(all_configs)} configurations...")

        for idx, config in enumerate(all_configs):
            # Ablate profiles
            ablated_profiles = [
                self.ablate_profile(p, config) for p in profiles
            ]

            # Make predictions
            predictions = []
            for profile in ablated_profiles:
                result = self.inference_engine.predict(
                    profile=profile,
                    strategy=strategy,
                    prediction_horizon=prediction_horizon
                )
                predictions.append(result["prediction"])

            # Compute metrics
            evaluator = PerformanceEvaluator()
            metrics = evaluator.compute_metrics(true_labels, predictions)

            results[config] = AblationResult(
                configuration=config,
                f1_score=metrics.f1_score,
                accuracy=metrics.accuracy,
                n_dimensions=len(config)
            )

            if (idx + 1) % 10 == 0:
                print(f"  Completed {idx + 1}/{len(all_configs)} configurations")

        return results


class DominanceAnalysis:
    """
    Compute dominance ratios and feature importance.
    """

    @staticmethod
    def compute_dominance_ratios(
        ablation_results: Dict[Tuple[str, ...], AblationResult]
    ) -> Dict[str, float]:
        """
        Compute dominance ratio for each domain.

        Dominance ratio = Average contribution of a domain across all configurations
        where it appears.

        Args:
            ablation_results: Results from ablation experiment

        Returns:
            Dictionary mapping domain names to dominance ratios
        """
        domain_names = ["cognitive", "functional", "neuropsychiatric", "physiological", "demographic"]
        dominance_ratios = {}

        # Get baseline (all domains included)
        all_domains_config = tuple(domain_names)
        if all_domains_config not in ablation_results:
            warnings.warn("All-domains configuration not found in ablation results")
            return {}

        baseline_f1 = ablation_results[all_domains_config].f1_score

        for domain in domain_names:
            # Find all configurations that EXCLUDE this domain
            configs_without_domain = [
                config for config in ablation_results.keys()
                if domain not in config and len(config) == len(domain_names) - 1
            ]

            if not configs_without_domain:
                dominance_ratios[domain] = 0.0
                continue

            # Average performance drop when domain is removed
            avg_f1_without = np.mean([
                ablation_results[config].f1_score
                for config in configs_without_domain
            ])

            # Contribution = baseline - performance without domain
            contribution = baseline_f1 - avg_f1_without

            dominance_ratios[domain] = contribution

        # Normalize to percentages
        total_contribution = sum(dominance_ratios.values())
        if total_contribution > 0:
            dominance_ratios = {
                domain: (contrib / total_contribution) * 100
                for domain, contrib in dominance_ratios.items()
            }

        return dominance_ratios

    @staticmethod
    def compute_marginal_contribution(
        ablation_results: Dict[Tuple[str, ...], AblationResult],
        domain: str
    ) -> float:
        """
        Compute marginal contribution of a domain.

        Args:
            ablation_results: Results from ablation experiment
            domain: Domain name to analyze

        Returns:
            Marginal contribution (F1 improvement)
        """
        # Find pairs of configurations that differ only by the target domain
        contributions = []

        for config_with in ablation_results.keys():
            if domain not in config_with:
                continue

            # Create corresponding configuration without this domain
            config_without = tuple(d for d in config_with if d != domain)

            if config_without in ablation_results:
                f1_with = ablation_results[config_with].f1_score
                f1_without = ablation_results[config_without].f1_score
                contribution = f1_with - f1_without
                contributions.append(contribution)

        return np.mean(contributions) if contributions else 0.0


class InteractionAnalysis:
    """
    Analyze feature interactions between domains.
    """

    @staticmethod
    def compute_interaction_matrix(
        ablation_results: Dict[Tuple[str, ...], AblationResult]
    ) -> np.ndarray:
        """
        Compute pairwise interaction matrix between domains.

        Interaction strength = Performance with both domains - sum of individual contributions

        Args:
            ablation_results: Results from ablation experiment

        Returns:
            5x5 interaction matrix
        """
        domain_names = ["cognitive", "functional", "neuropsychiatric", "physiological", "demographic"]
        n_domains = len(domain_names)
        interaction_matrix = np.zeros((n_domains, n_domains))

        # Get baseline (no features)
        baseline_f1 = ablation_results[()].f1_score if () in ablation_results else 0.0

        for i, domain_i in enumerate(domain_names):
            for j, domain_j in enumerate(domain_names):
                if i >= j:
                    continue

                # Individual contributions
                config_i = (domain_i,)
                config_j = (domain_j,)
                config_ij = tuple(sorted([domain_i, domain_j]))

                if config_i not in ablation_results or config_j not in ablation_results or config_ij not in ablation_results:
                    continue

                f1_i = ablation_results[config_i].f1_score - baseline_f1
                f1_j = ablation_results[config_j].f1_score - baseline_f1
                f1_ij = ablation_results[config_ij].f1_score - baseline_f1

                # Interaction = joint contribution - sum of individual contributions
                interaction = f1_ij - (f1_i + f1_j)

                interaction_matrix[i, j] = interaction
                interaction_matrix[j, i] = interaction

        # Normalize to [0, 1] range
        max_val = np.abs(interaction_matrix).max()
        if max_val > 0:
            interaction_matrix = (interaction_matrix + max_val) / (2 * max_val)

        return interaction_matrix


class ReasoningAnalysis:
    """
    Analyze LLM reasoning processes through text analysis.
    """

    # Clinical keywords organized by semantic domain
    SEMANTIC_DOMAINS = {
        "Cognitive Assessment": [
            "cognitive impairment", "memory decline", "word recall", "delayed recall",
            "immediate recall", "orientation", "executive function", "attention",
            "confusion", "forgetfulness"
        ],
        "Functional Status": [
            "functional impairment", "mobility", "IADL", "ADL", "activities of daily living",
            "instrumental activities", "disability", "independence"
        ],
        "Neuropsychiatric Symptoms": [
            "depression", "apathy", "anxiety", "behavioral changes", "agitation",
            "irritability", "mood", "psychiatric"
        ],
        "Risk Factors": [
            "age", "diabetes", "hypertension", "family history", "cardiovascular",
            "vascular", "stroke", "comorbidities"
        ],
        "Progressive Patterns": [
            "progressive decline", "change over time", "gradual worsening",
            "trajectory", "deterioration", "accelerating", "temporal"
        ],
        "Clinical Diagnosis": [
            "Alzheimer", "dementia", "AD", "neurodegenerative", "diagnosis",
            "pathology", "biomarker"
        ]
    }

    @staticmethod
    def extract_keywords(
        reasoning_texts: List[str],
        top_n: int = 50
    ) -> Counter:
        """
        Extract and count keywords from reasoning texts.

        Args:
            reasoning_texts: List of reasoning text from LLM outputs
            top_n: Number of top keywords to return

        Returns:
            Counter object with keyword frequencies
        """
        # Tokenize and clean
        all_words = []

        for text in reasoning_texts:
            # Convert to lowercase
            text = text.lower()

            # Remove punctuation and split
            words = re.findall(r'\b[a-z]+\b', text)

            # Remove common stop words
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                'would', 'should', 'could', 'may', 'might', 'must', 'this', 'that',
                'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
            }

            words = [w for w in words if w not in stop_words and len(w) > 2]
            all_words.extend(words)

        # Count frequencies
        keyword_counter = Counter(all_words)

        return keyword_counter.most_common(top_n)

    @staticmethod
    def categorize_keywords_by_domain(
        keyword_frequencies: Counter
    ) -> Dict[str, List[Tuple[str, int]]]:
        """
        Categorize keywords into semantic domains.

        Args:
            keyword_frequencies: Counter of keyword frequencies

        Returns:
            Dictionary mapping domain names to keyword lists
        """
        categorized = {domain: [] for domain in ReasoningAnalysis.SEMANTIC_DOMAINS.keys()}
        uncategorized = []

        for keyword, freq in keyword_frequencies.items():
            matched = False

            for domain, domain_keywords in ReasoningAnalysis.SEMANTIC_DOMAINS.items():
                # Check if keyword appears in domain keywords
                if any(keyword in dk.lower() for dk in domain_keywords):
                    categorized[domain].append((keyword, freq))
                    matched = True
                    break

            if not matched:
                uncategorized.append((keyword, freq))

        # Add uncategorized keywords
        categorized["Other"] = uncategorized

        return categorized

    @staticmethod
    def generate_wordcloud_data(
        reasoning_texts: List[str],
        top_n: int = 50
    ) -> Dict[str, int]:
        """
        Generate word cloud data from reasoning texts.

        Args:
            reasoning_texts: List of reasoning texts
            top_n: Number of top words

        Returns:
            Dictionary mapping words to frequencies
        """
        keyword_frequencies = ReasoningAnalysis.extract_keywords(reasoning_texts, top_n)
        return dict(keyword_frequencies)


class TemporalContributionAnalysis:
    """
    Analyze how domain contributions evolve across prediction horizons.
    """

    @staticmethod
    def compute_temporal_contributions(
        ablation_results_by_horizon: Dict[int, Dict[Tuple[str, ...], AblationResult]]
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute domain contributions across prediction horizons.

        Args:
            ablation_results_by_horizon: Ablation results for each horizon

        Returns:
            Dictionary mapping horizons to dominance ratios
        """
        temporal_contributions = {}

        for horizon, ablation_results in ablation_results_by_horizon.items():
            dominance_ratios = DominanceAnalysis.compute_dominance_ratios(ablation_results)
            temporal_contributions[horizon] = dominance_ratios

        return temporal_contributions

    @staticmethod
    def analyze_contribution_evolution(
        temporal_contributions: Dict[int, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze how each domain's contribution evolves over time.

        Args:
            temporal_contributions: Contributions by horizon

        Returns:
            Dictionary with evolution analysis for each domain
        """
        domain_names = ["cognitive", "functional", "neuropsychiatric", "physiological", "demographic"]
        evolution_analysis = {}

        for domain in domain_names:
            contributions = [
                temporal_contributions[h][domain]
                for h in sorted(temporal_contributions.keys())
            ]

            evolution_analysis[domain] = {
                "initial": contributions[0],
                "final": contributions[-1],
                "change": contributions[-1] - contributions[0],
                "relative_change": ((contributions[-1] - contributions[0]) / contributions[0] * 100)
                                   if contributions[0] > 0 else 0,
                "trajectory": contributions
            }

        return evolution_analysis


class CognitiveFeatureAnalysis:
    """
    Specific analysis comparing configurations with vs without cognitive features.
    """

    @staticmethod
    def compare_cognitive_inclusion(
        ablation_results: Dict[Tuple[str, ...], AblationResult]
    ) -> Dict[str, any]:
        """
        Compare performance of configurations with vs without cognitive features.

        Args:
            ablation_results: Ablation experiment results

        Returns:
            Comparison statistics
        """
        # Split configurations
        with_cognitive = [
            result for config, result in ablation_results.items()
            if "cognitive" in config
        ]

        without_cognitive = [
            result for config, result in ablation_results.items()
            if "cognitive" not in config and len(config) > 0  # Exclude empty config
        ]

        # Compute statistics
        f1_with = [r.f1_score for r in with_cognitive]
        f1_without = [r.f1_score for r in without_cognitive]

        mean_with = np.mean(f1_with)
        mean_without = np.mean(f1_without)

        # Contribution as percentage
        total_contribution = mean_with + mean_without
        contrib_with_pct = (mean_with / total_contribution) * 100 if total_contribution > 0 else 0
        contrib_without_pct = (mean_without / total_contribution) * 100 if total_contribution > 0 else 0

        # Statistical test
        from scipy import stats
        statistic, p_value = stats.mannwhitneyu(f1_with, f1_without, alternative='greater')

        return {
            "mean_f1_with_cognitive": mean_with,
            "mean_f1_without_cognitive": mean_without,
            "contribution_with_cognitive_pct": contrib_with_pct,
            "contribution_without_cognitive_pct": contrib_without_pct,
            "performance_gap": mean_with - mean_without,
            "p_value": p_value,
            "n_configs_with": len(with_cognitive),
            "n_configs_without": len(without_cognitive)
        }


# Example usage
if __name__ == "__main__":
    # Example: Dominance analysis with mock ablation results
    mock_results = {
        ("cognitive", "functional", "neuropsychiatric", "physiological", "demographic"):
            AblationResult(("cognitive", "functional", "neuropsychiatric", "physiological", "demographic"), 0.58, 0.75, 5),
        ("functional", "neuropsychiatric", "physiological", "demographic"):
            AblationResult(("functional", "neuropsychiatric", "physiological", "demographic"), 0.32, 0.55, 4),
        ("cognitive", "neuropsychiatric", "physiological", "demographic"):
            AblationResult(("cognitive", "neuropsychiatric", "physiological", "demographic"), 0.52, 0.70, 4),
        ("cognitive", "functional", "physiological", "demographic"):
            AblationResult(("cognitive", "functional", "physiological", "demographic"), 0.55, 0.72, 4),
        ("cognitive", "functional", "neuropsychiatric", "demographic"):
            AblationResult(("cognitive", "functional", "neuropsychiatric", "demographic"), 0.54, 0.71, 4),
        ("cognitive", "functional", "neuropsychiatric", "physiological"):
            AblationResult(("cognitive", "functional", "neuropsychiatric", "physiological"), 0.56, 0.73, 4),
    }

    # Compute dominance ratios
    dominance_ratios = DominanceAnalysis.compute_dominance_ratios(mock_results)
    print("Dominance Ratios:")
    for domain, ratio in dominance_ratios.items():
        print(f"  {domain}: {ratio:.1f}%")

    # Example: Reasoning analysis
    mock_reasoning = [
        "The patient shows progressive memory decline with poor word recall performance.",
        "Cognitive impairment is evident with functional decline in ADL and IADL.",
        "Depression and anxiety symptoms combined with vascular risk factors suggest increased AD risk."
    ]

    wordcloud_data = ReasoningAnalysis.generate_wordcloud_data(mock_reasoning, top_n=20)
    print("\nTop Keywords from Reasoning:")
    for word, freq in sorted(wordcloud_data.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {word}: {freq}")
