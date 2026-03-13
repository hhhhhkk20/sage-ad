"""
SAGE-AD: Main Execution Script
===============================
This script orchestrates the complete SAGE-AD benchmark evaluation pipeline:
1. Load and preprocess cohort data
2. Evaluate LLMs across multiple inference strategies
3. Compute performance metrics and confidence intervals
4. Perform cross-cohort generalization analysis
5. Conduct interpretability analysis

Usage:
    python main.py --config configs/experiment_config.json
"""

import argparse
import os
import sys
from typing import List, Dict

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from llm_inference import LLMInference, BatchInference, InferenceStrategy, TemporalSymptomProfile
from evaluation_metrics import (
    PerformanceEvaluator,
    CrossCohortAnalysis,
    TemporalAnalysis,
    StrategyComparison
)
from interpretability import (
    FeatureDomainAblation,
    DominanceAnalysis,
    InteractionAnalysis,
    ReasoningAnalysis,
    TemporalContributionAnalysis
)
from utils import (
    DataLoader,
    CohortProcessor,
    ResultsManager,
    Logger,
    ConfigManager
)


class SAGEADBenchmark:
    """
    Main benchmark orchestrator for SAGE-AD framework.
    """

    def __init__(self, config_path: str):
        self.config = ConfigManager.load_config(config_path)

        self.logger = Logger(
            log_dir=self.config.get("log_dir", "./logs"),
            experiment_name=self.config.get("experiment_name", "sage_ad_benchmark")
        )

        self.logger.log("Initializing SAGE-AD Benchmark Framework")
        self.logger.log(f"Configuration loaded from {config_path}")

        self.results_manager = ResultsManager()

        self.all_results = {
            "models": {},
            "strategies": {},
            "cohorts": {},
            "temporal": {},
            "interpretability": {}
        }

    def load_cohort_data(self, cohort_name: str) -> tuple:
        self.logger.log(f"Loading {cohort_name} cohort data...")

        data_loader = DataLoader()

        if cohort_name == "ELSA":
            df = data_loader.load_elsa_data(self.config["cohort_data_paths"]["ELSA"])
        elif cohort_name == "HRS":
            df = data_loader.load_hrs_data(self.config["cohort_data_paths"]["HRS"])
        elif cohort_name == "SHARE":
            df = data_loader.load_share_data(self.config["cohort_data_paths"]["SHARE"])
        else:
            raise ValueError(f"Unknown cohort: {cohort_name}")

        processor = CohortProcessor()
        cases, controls = processor.identify_cases_and_controls(df)

        combined_df = pd.concat([cases, controls])
        profiles = processor.create_temporal_profiles(combined_df)
        labels = combined_df["ad_diagnosis"].tolist()

        self.logger.log(f"Loaded {len(profiles)} profiles from {cohort_name}")
        self.logger.log(f"  AD cases: {sum(labels)}")
        self.logger.log(f"  Controls: {len(labels) - sum(labels)}")

        return profiles, labels

    def evaluate_model_strategy(
        self,
        model_name: str,
        profiles: List[TemporalSymptomProfile],
        labels: List[bool],
        strategy: InferenceStrategy,
        prediction_horizon: int
    ) -> Dict:
        self.logger.log(
            f"Evaluating {model_name} with {strategy.value} strategy "
            f"(horizon: {prediction_horizon} years)"
        )

        batch_inference = BatchInference(model_name)

        results = batch_inference.evaluate_cohort(
            profiles=profiles,
            true_labels=labels,
            strategy=strategy,
            prediction_horizon=prediction_horizon
        )

        evaluator = PerformanceEvaluator()
        metrics = evaluator.compute_metrics(
            y_true=results["true_labels"],
            y_pred=results["predictions"]
        )

        f1_ci = evaluator.bootstrap_ci(
            y_true=results["true_labels"],
            y_pred=results["predictions"],
            metric="f1_score",
            n_iterations=self.config.get("bootstrap_iterations", 5000)
        )

        result_dict = {
            "model": model_name,
            "strategy": strategy.value,
            "horizon": prediction_horizon,
            "metrics": metrics.to_dict(),
            "f1_ci": {"lower": f1_ci[0], "upper": f1_ci[1]},
            "predictions": results["predictions"],
            "confidences": results["confidences"]
        }

        self.logger.log_metrics(metrics.to_dict(), prefix=f"{model_name} ")

        return result_dict

    def run_strategy_comparison(
        self,
        cohort_name: str,
        profiles: List[TemporalSymptomProfile],
        labels: List[bool]
    ):
        self.logger.log(f"\n{'='*80}")
        self.logger.log(f"Strategy Comparison Analysis - {cohort_name}")
        self.logger.log(f"{'='*80}\n")

        strategy_results = {s.value: [] for s in InferenceStrategy}

        for model_name in self.config["models"]:
            for strategy in InferenceStrategy:
                result = self.evaluate_model_strategy(
                    model_name=model_name,
                    profiles=profiles,
                    labels=labels,
                    strategy=strategy,
                    prediction_horizon=1
                )
                strategy_results[strategy.value].append(result["metrics"]["f1_score"])

        strategy_comparison = StrategyComparison.compare_strategies(strategy_results)
        self.all_results["strategies"][cohort_name] = strategy_comparison
        self.logger.log(f"Strategy comparison completed for {cohort_name}")

    def run_temporal_analysis(
        self,
        model_name: str,
        profiles: List[TemporalSymptomProfile],
        labels: List[bool]
    ):
        self.logger.log(f"\n{'='*80}")
        self.logger.log(f"Temporal Decay Analysis - {model_name}")
        self.logger.log(f"{'='*80}\n")

        horizons = self.config.get("prediction_horizons", [1, 2, 3, 4])
        horizon_metrics = {}

        for horizon in horizons:
            result = self.evaluate_model_strategy(
                model_name=model_name,
                profiles=profiles,
                labels=labels,
                strategy=InferenceStrategy.FEW_SHOT,
                prediction_horizon=horizon
            )

            from evaluation_metrics import ClassificationMetrics
            metrics_obj = ClassificationMetrics(**result["metrics"])
            horizon_metrics[horizon] = metrics_obj

        temporal_decay = TemporalAnalysis.compute_temporal_decay(horizon_metrics)
        self.all_results["temporal"][model_name] = temporal_decay
        self.logger.log(f"Temporal decay: {temporal_decay['relative_decay_percent']:.1f}%")

    def run_cross_cohort_analysis(self):
        self.logger.log(f"\n{'='*80}")
        self.logger.log("Cross-Cohort Generalization Analysis")
        self.logger.log(f"{'='*80}\n")

        cohorts = self.config.get("cohorts", ["ELSA", "HRS", "SHARE"])
        cohort_data = {}

        for cohort_name in cohorts:
            profiles, labels = self.load_cohort_data(cohort_name)
            cohort_data[cohort_name] = (profiles, labels)

        for model_name in self.config["models"]:
            cohort_metrics = {}

            for cohort_name, (profiles, labels) in cohort_data.items():
                result = self.evaluate_model_strategy(
                    model_name=model_name,
                    profiles=profiles,
                    labels=labels,
                    strategy=InferenceStrategy.FEW_SHOT,
                    prediction_horizon=1
                )

                from evaluation_metrics import ClassificationMetrics
                metrics_obj = ClassificationMetrics(**result["metrics"])
                cohort_metrics[cohort_name] = metrics_obj

            gen_gap = CrossCohortAnalysis.compute_generalization_gap(cohort_metrics)

            self.all_results["cohorts"][model_name] = {
                "metrics_by_cohort": {
                    cohort: metrics.to_dict()
                    for cohort, metrics in cohort_metrics.items()
                },
                "generalization_gap": gen_gap
            }

            self.logger.log(f"{model_name} generalization gap: {gen_gap:.3f}")

    def run_interpretability_analysis(
        self,
        model_name: str,
        profiles: List[TemporalSymptomProfile],
        labels: List[bool]
    ):
        self.logger.log(f"\n{'='*80}")
        self.logger.log(f"Interpretability Analysis - {model_name}")
        self.logger.log(f"{'='*80}\n")

        inference_engine = LLMInference(model_name)
        ablation_analyzer = FeatureDomainAblation(inference_engine)

        sample_size = min(100, len(profiles))
        sample_indices = np.random.choice(len(profiles), sample_size, replace=False)
        sample_profiles = [profiles[i] for i in sample_indices]
        sample_labels = [labels[i] for i in sample_indices]

        self.logger.log(f"Running ablation on {sample_size} profiles...")

        ablation_results = ablation_analyzer.run_ablation_experiment(
            profiles=sample_profiles,
            true_labels=sample_labels,
            prediction_horizon=1,
            strategy=InferenceStrategy.FEW_SHOT
        )

        dominance_ratios = DominanceAnalysis.compute_dominance_ratios(ablation_results)
        interaction_matrix = InteractionAnalysis.compute_interaction_matrix(ablation_results)

        self.all_results["interpretability"][model_name] = {
            "dominance_ratios": dominance_ratios,
            "interaction_matrix": interaction_matrix.tolist()
        }

        self.logger.log("Dominance Ratios:")
        for domain, ratio in dominance_ratios.items():
            self.logger.log(f"  {domain}: {ratio:.1f}%")

    def run_complete_benchmark(self):
        self.logger.log("\n" + "="*80)
        self.logger.log("STARTING SAGE-AD BENCHMARK EVALUATION")
        self.logger.log("="*80 + "\n")

        try:
            primary_cohort = self.config.get("primary_cohort", "HRS")
            profiles, labels = self.load_cohort_data(primary_cohort)

            if self.config.get("run_strategy_comparison", True):
                self.run_strategy_comparison(primary_cohort, profiles, labels)

            if self.config.get("run_temporal_analysis", True):
                for model in self.config["models"][:3]:
                    self.run_temporal_analysis(model, profiles, labels)

            if self.config.get("run_cross_cohort", True):
                self.run_cross_cohort_analysis()

            if self.config.get("run_interpretability", True):
                self.run_interpretability_analysis(self.config["models"][0], profiles, labels)

            self.logger.log("\n" + "="*80)
            self.logger.log("BENCHMARK COMPLETED SUCCESSFULLY")
            self.logger.log("="*80 + "\n")

        except Exception as e:
            self.logger.log(f"ERROR: {str(e)}", level="ERROR")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="SAGE-AD: Benchmark framework for LLM-based AD prediction"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.json",
        help="Path to configuration JSON file"
    )

    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create default configuration file and exit"
    )

    args = parser.parse_args()

    if args.create_config:
        config_manager = ConfigManager()
        default_config = config_manager.create_default_config()
        config_manager.save_config(default_config, "configs/default_config.json")
        print("Default configuration created at configs/default_config.json")
        return

    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}")
        print("Run with --create-config to generate a default configuration")
        return

    benchmark = SAGEADBenchmark(config_path=args.config)
    benchmark.run_complete_benchmark()


if __name__ == "__main__":
    main()
