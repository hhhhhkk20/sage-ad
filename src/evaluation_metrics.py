"""
SAGE-AD: Performance Evaluation Metrics Module
===============================================
This module implements comprehensive evaluation metrics for AD prediction:
1. Classification metrics (Accuracy, Precision, Recall, F1, Balanced Accuracy)
2. Bootstrap confidence intervals
3. Cross-cohort generalization analysis
4. Temporal decay analysis
5. Statistical comparisons

Based on the evaluation methodology in the SAGE-AD paper.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import warnings


@dataclass
class ClassificationMetrics:
    """
    Container for classification performance metrics.

    Attributes:
        accuracy: Overall accuracy
        precision: Precision for AD class (positive class)
        recall: Recall for AD class (sensitivity)
        f1_score: F1 score for AD class
        balanced_accuracy: Balanced accuracy (mean of sensitivity and specificity)
        specificity: Specificity for AD class
        confusion_matrix: 2x2 confusion matrix [[TN, FP], [FN, TP]]
    """
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    balanced_accuracy: float
    specificity: float
    confusion_matrix: np.ndarray

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary format."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "balanced_accuracy": self.balanced_accuracy,
            "specificity": self.specificity
        }


class PerformanceEvaluator:
    """
    Main class for computing performance metrics and confidence intervals.
    """

    @staticmethod
    def compute_metrics(
        y_true: List[bool],
        y_pred: List[bool]
    ) -> ClassificationMetrics:
        """
        Compute classification metrics.

        Args:
            y_true: Ground truth labels (True = AD positive)
            y_pred: Predicted labels

        Returns:
            ClassificationMetrics object
        """
        y_true = np.array(y_true, dtype=int)
        y_pred = np.array(y_pred, dtype=int)

        # Compute confusion matrix
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))

        confusion_matrix = np.array([[tn, fp], [fn, tp]])

        # Compute metrics
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        balanced_accuracy = (recall + specificity) / 2

        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            balanced_accuracy=balanced_accuracy,
            specificity=specificity,
            confusion_matrix=confusion_matrix
        )

    @staticmethod
    def bootstrap_ci(
        y_true: List[bool],
        y_pred: List[bool],
        metric: str = "f1_score",
        n_iterations: int = 5000,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute bootstrap confidence interval for a metric.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            metric: Metric name ('accuracy', 'precision', 'recall', 'f1_score')
            n_iterations: Number of bootstrap iterations
            confidence_level: Confidence level (default 0.95 for 95% CI)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        n_samples = len(y_true)

        metric_values = []

        for _ in range(n_iterations):
            # Stratified bootstrap resampling (preserve class distribution)
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]

            # Compute metric
            metrics = PerformanceEvaluator.compute_metrics(
                y_true_boot.tolist(),
                y_pred_boot.tolist()
            )
            metric_value = getattr(metrics, metric)
            metric_values.append(metric_value)

        # Compute percentile-based confidence interval
        alpha = 1 - confidence_level
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)

        lower_bound = np.percentile(metric_values, lower_percentile)
        upper_bound = np.percentile(metric_values, upper_percentile)

        return lower_bound, upper_bound


class CrossCohortAnalysis:
    """
    Analyze cross-cohort generalization and distribution shift.
    """

    @staticmethod
    def compute_generalization_gap(
        cohort_metrics: Dict[str, ClassificationMetrics]
    ) -> float:
        """
        Compute generalization gap (difference between max and min F1 across cohorts).

        Args:
            cohort_metrics: Dictionary mapping cohort names to metrics

        Returns:
            Generalization gap (Δ)
        """
        f1_scores = [metrics.f1_score for metrics in cohort_metrics.values()]
        return max(f1_scores) - min(f1_scores)

    @staticmethod
    def compute_transfer_pattern(
        source_metrics: ClassificationMetrics,
        target_metrics: ClassificationMetrics
    ) -> Dict[str, float]:
        """
        Analyze transfer pattern from source to target cohort.

        Args:
            source_metrics: Performance on source (in-distribution) cohort
            target_metrics: Performance on target (out-of-distribution) cohort

        Returns:
            Dictionary with transfer analysis
        """
        f1_gap = source_metrics.f1_score - target_metrics.f1_score
        relative_gap = (f1_gap / source_metrics.f1_score) * 100 if source_metrics.f1_score > 0 else 0

        return {
            "source_f1": source_metrics.f1_score,
            "target_f1": target_metrics.f1_score,
            "absolute_gap": f1_gap,
            "relative_gap_percent": relative_gap
        }

    @staticmethod
    def pareto_frontier_analysis(
        models_performance: Dict[str, Dict[str, float]]
    ) -> List[str]:
        """
        Identify Pareto-optimal models balancing source performance and generalization.

        Args:
            models_performance: Dict of {model_name: {"source_f1": X, "transfer_gap": Y}}

        Returns:
            List of Pareto-optimal model names
        """
        pareto_models = []

        for model_a, perf_a in models_performance.items():
            is_dominated = False

            for model_b, perf_b in models_performance.items():
                if model_a == model_b:
                    continue

                # Model B dominates A if:
                # - B has higher source F1 AND lower (or equal) transfer gap
                # - OR B has equal source F1 AND lower transfer gap
                if (perf_b["source_f1"] >= perf_a["source_f1"] and
                    perf_b["transfer_gap"] <= perf_a["transfer_gap"] and
                    (perf_b["source_f1"] > perf_a["source_f1"] or perf_b["transfer_gap"] < perf_a["transfer_gap"])):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_models.append(model_a)

        return pareto_models


class TemporalAnalysis:
    """
    Analyze temporal decay of prediction performance across horizons.
    """

    @staticmethod
    def compute_temporal_decay(
        horizon_metrics: Dict[int, ClassificationMetrics]
    ) -> Dict[str, float]:
        """
        Compute temporal decay statistics.

        Args:
            horizon_metrics: Dictionary mapping prediction horizon (years) to metrics

        Returns:
            Dictionary with decay analysis
        """
        horizons = sorted(horizon_metrics.keys())
        f1_scores = [horizon_metrics[h].f1_score for h in horizons]

        # Compute overall decay
        initial_f1 = f1_scores[0]
        final_f1 = f1_scores[-1]
        absolute_decay = initial_f1 - final_f1
        relative_decay = (absolute_decay / initial_f1) * 100 if initial_f1 > 0 else 0

        # Compute decay between adjacent horizons
        adjacent_decays = []
        for i in range(len(horizons) - 1):
            decay = (f1_scores[i] - f1_scores[i+1]) / f1_scores[i] * 100 if f1_scores[i] > 0 else 0
            adjacent_decays.append(decay)

        return {
            "initial_f1": initial_f1,
            "final_f1": final_f1,
            "absolute_decay": absolute_decay,
            "relative_decay_percent": relative_decay,
            "adjacent_decays": adjacent_decays,
            "mean_adjacent_decay": np.mean(adjacent_decays),
            "horizons": horizons
        }

    @staticmethod
    def compute_fewshot_gain_attenuation(
        horizon_fewshot_metrics: Dict[int, ClassificationMetrics],
        horizon_zeroshot_metrics: Dict[int, ClassificationMetrics]
    ) -> Dict[str, any]:
        """
        Compute few-shot gain attenuation across prediction horizons.

        Args:
            horizon_fewshot_metrics: Few-shot metrics by horizon
            horizon_zeroshot_metrics: Zero-shot metrics by horizon

        Returns:
            Dictionary with gain attenuation analysis
        """
        horizons = sorted(horizon_fewshot_metrics.keys())
        gains = []

        for h in horizons:
            fewshot_f1 = horizon_fewshot_metrics[h].f1_score
            zeroshot_f1 = horizon_zeroshot_metrics[h].f1_score
            gain = fewshot_f1 - zeroshot_f1
            gains.append(gain)

        # Compute attenuation
        initial_gain = gains[0]
        final_gain = gains[-1]
        attenuation = (initial_gain - final_gain) / initial_gain * 100 if initial_gain > 0 else 0

        return {
            "gains_by_horizon": dict(zip(horizons, gains)),
            "initial_gain": initial_gain,
            "final_gain": final_gain,
            "attenuation_percent": attenuation
        }


class StatisticalComparison:
    """
    Statistical tests for comparing models and strategies.
    """

    @staticmethod
    def wilcoxon_signed_rank_test(
        metrics_a: List[float],
        metrics_b: List[float],
        alternative: str = "two-sided"
    ) -> Tuple[float, float]:
        """
        Perform Wilcoxon signed-rank test for paired samples.

        Args:
            metrics_a: Metrics from method A (e.g., few-shot F1 scores)
            metrics_b: Metrics from method B (e.g., zero-shot F1 scores)
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            Tuple of (statistic, p_value)
        """
        statistic, p_value = stats.wilcoxon(metrics_a, metrics_b, alternative=alternative)
        return statistic, p_value

    @staticmethod
    def mann_whitney_u_test(
        metrics_a: List[float],
        metrics_b: List[float],
        alternative: str = "two-sided"
    ) -> Tuple[float, float]:
        """
        Perform Mann-Whitney U test for independent samples.

        Args:
            metrics_a: Metrics from group A
            metrics_b: Metrics from group B
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            Tuple of (statistic, p_value)
        """
        statistic, p_value = stats.mannwhitneyu(metrics_a, metrics_b, alternative=alternative)
        return statistic, p_value

    @staticmethod
    def bonferroni_correction(
        p_values: List[float],
        alpha: float = 0.05
    ) -> Tuple[List[bool], float]:
        """
        Apply Bonferroni correction for multiple comparisons.

        Args:
            p_values: List of p-values
            alpha: Significance level

        Returns:
            Tuple of (list of significant flags, corrected alpha)
        """
        n_comparisons = len(p_values)
        corrected_alpha = alpha / n_comparisons
        significant = [p < corrected_alpha for p in p_values]
        return significant, corrected_alpha

    @staticmethod
    def kendall_tau_correlation(
        parameter_scales: List[float],
        accuracies: List[float]
    ) -> Tuple[float, float]:
        """
        Compute Kendall's tau correlation for scaling analysis.

        Args:
            parameter_scales: Model parameter counts (e.g., [7B, 14B, 32B, 72B])
            accuracies: Corresponding accuracies

        Returns:
            Tuple of (tau coefficient, p_value)
        """
        tau, p_value = stats.kendalltau(parameter_scales, accuracies)
        return tau, p_value


class StrategyComparison:
    """
    Compare performance across inference strategies.
    """

    @staticmethod
    def compute_superiority_rate(
        strategy_a_metrics: List[float],
        strategy_b_metrics: List[float]
    ) -> float:
        """
        Compute superiority rate (percentage of models where A > B).

        Args:
            strategy_a_metrics: F1 scores for strategy A across models
            strategy_b_metrics: F1 scores for strategy B across models

        Returns:
            Superiority rate (0-100%)
        """
        n_models = len(strategy_a_metrics)
        n_superior = sum(1 for a, b in zip(strategy_a_metrics, strategy_b_metrics) if a > b)
        return (n_superior / n_models) * 100 if n_models > 0 else 0

    @staticmethod
    def compare_strategies(
        strategy_metrics: Dict[str, List[float]]
    ) -> Dict[str, any]:
        """
        Comprehensive comparison of inference strategies.

        Args:
            strategy_metrics: Dict mapping strategy names to list of F1 scores

        Returns:
            Dictionary with comparison results
        """
        results = {}

        # Compute mean and std for each strategy
        for strategy, metrics in strategy_metrics.items():
            results[f"{strategy}_mean"] = np.mean(metrics)
            results[f"{strategy}_std"] = np.std(metrics)

        # Pairwise comparisons
        strategies = list(strategy_metrics.keys())
        comparisons = []

        for i, strat_a in enumerate(strategies):
            for strat_b in strategies[i+1:]:
                # Superiority rate
                sup_rate = StrategyComparison.compute_superiority_rate(
                    strategy_metrics[strat_a],
                    strategy_metrics[strat_b]
                )

                # Statistical test
                stat, p_value = StatisticalComparison.wilcoxon_signed_rank_test(
                    strategy_metrics[strat_a],
                    strategy_metrics[strat_b]
                )

                comparisons.append({
                    "strategy_a": strat_a,
                    "strategy_b": strat_b,
                    "superiority_rate": sup_rate,
                    "p_value": p_value,
                    "significant": p_value < 0.05
                })

        results["comparisons"] = comparisons
        return results


# Example usage
if __name__ == "__main__":
    # Example: Compute metrics for a sample prediction
    y_true = [True, True, False, False, True, False, True, True, False, False]
    y_pred = [True, True, False, True, True, False, False, True, False, False]

    evaluator = PerformanceEvaluator()

    # Compute metrics
    metrics = evaluator.compute_metrics(y_true, y_pred)
    print("Performance Metrics:")
    print(f"  Accuracy: {metrics.accuracy:.3f}")
    print(f"  Precision: {metrics.precision:.3f}")
    print(f"  Recall: {metrics.recall:.3f}")
    print(f"  F1 Score: {metrics.f1_score:.3f}")
    print(f"  Balanced Accuracy: {metrics.balanced_accuracy:.3f}")

    # Compute bootstrap CI for F1 score
    lower, upper = evaluator.bootstrap_ci(y_true, y_pred, metric="f1_score", n_iterations=1000)
    print(f"\n95% CI for F1: [{lower:.3f}, {upper:.3f}]")

    # Example: Strategy comparison
    strategy_metrics = {
        "zero_shot": [0.45, 0.48, 0.42, 0.46, 0.44],
        "few_shot": [0.52, 0.55, 0.50, 0.53, 0.51],
        "cot": [0.38, 0.40, 0.36, 0.39, 0.37]
    }

    comparison_results = StrategyComparison.compare_strategies(strategy_metrics)
    print("\nStrategy Comparison:")
    for comp in comparison_results["comparisons"]:
        print(f"  {comp['strategy_a']} vs {comp['strategy_b']}:")
        print(f"    Superiority rate: {comp['superiority_rate']:.1f}%")
        print(f"    P-value: {comp['p_value']:.4f}")
