"""
SAGE-AD Framework: Example Usage
=================================

This script demonstrates how to use the SAGE-AD framework for evaluating
LLMs on AD prediction tasks.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from llm_inference import LLMInference, InferenceStrategy, TemporalSymptomProfile
from evaluation_metrics import PerformanceEvaluator, StrategyComparison
from interpretability import DominanceAnalysis
import numpy as np


def example_1_single_prediction():
    """
    Example 1: Make a single AD prediction using few-shot learning
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Prediction with Few-Shot Learning")
    print("="*80 + "\n")

    # Create a sample temporal symptom profile
    profile = TemporalSymptomProfile(
        participant_id="DEMO_001",
        age_at_assessments=[65, 67, 69, 71],
        cognitive_features=[
            {"word_recall_immediate": 8, "word_recall_delayed": 6, "orientation": 5},
            {"word_recall_immediate": 7, "word_recall_delayed": 5, "orientation": 5},
            {"word_recall_immediate": 5, "word_recall_delayed": 3, "orientation": 4},
            {"word_recall_immediate": 4, "word_recall_delayed": 2, "orientation": 3}
        ],
        functional_features=[
            {"ADL_impairment": 0, "IADL_impairment": 0},
            {"ADL_impairment": 0, "IADL_impairment": 1},
            {"ADL_impairment": 1, "IADL_impairment": 2},
            {"ADL_impairment": 2, "IADL_impairment": 3}
        ],
        physiological_features=[
            {"BMI": 25.3, "grip_strength": 32, "gait_speed": 1.2},
            {"BMI": 24.8, "grip_strength": 30, "gait_speed": 1.1},
            {"BMI": 24.2, "grip_strength": 28, "gait_speed": 1.0},
            {"BMI": 23.5, "grip_strength": 25, "gait_speed": 0.9}
        ],
        neuropsychiatric_features=[
            {"depression_score": 2, "anxiety": "low"},
            {"depression_score": 3, "anxiety": "low"},
            {"depression_score": 5, "anxiety": "moderate"},
            {"depression_score": 6, "anxiety": "moderate"}
        ],
        demographics={
            "age": 65,
            "sex": "Female",
            "education": "12 years"
        }
    )

    # Initialize LLM inference engine
    print("Initializing GPT-4o model...")
    model = LLMInference(model_name="gpt-4o")

    # Make prediction with few-shot strategy
    print("Making prediction with few-shot strategy (2-year horizon)...")
    result = model.predict(
        profile=profile,
        strategy=InferenceStrategy.FEW_SHOT,
        prediction_horizon=2
    )

    print(f"\n📊 Prediction Results:")
    print(f"  Prediction: {'AD Positive' if result['prediction'] else 'Healthy'}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Reasoning: {result['reasoning'][:200]}...")


def example_2_performance_evaluation():
    """
    Example 2: Evaluate model performance and compute metrics
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Performance Evaluation with Bootstrap CI")
    print("="*80 + "\n")

    # Simulate predictions
    np.random.seed(42)
    n_samples = 100

    # Ground truth (30% AD positive)
    y_true = np.random.choice([True, False], size=n_samples, p=[0.3, 0.7])

    # Model predictions (with some errors)
    y_pred = y_true.copy()
    # Introduce 15% error rate
    error_indices = np.random.choice(n_samples, size=int(n_samples * 0.15), replace=False)
    y_pred[error_indices] = ~y_pred[error_indices]

    # Compute metrics
    evaluator = PerformanceEvaluator()
    metrics = evaluator.compute_metrics(y_true.tolist(), y_pred.tolist())

    print("📊 Classification Metrics:")
    print(f"  Accuracy: {metrics.accuracy:.3f}")
    print(f"  Precision: {metrics.precision:.3f}")
    print(f"  Recall: {metrics.recall:.3f}")
    print(f"  F1 Score: {metrics.f1_score:.3f}")
    print(f"  Balanced Accuracy: {metrics.balanced_accuracy:.3f}")

    # Bootstrap confidence intervals
    print("\nComputing 95% confidence intervals (1000 iterations)...")
    f1_ci = evaluator.bootstrap_ci(
        y_true.tolist(),
        y_pred.tolist(),
        metric="f1_score",
        n_iterations=1000
    )

    precision_ci = evaluator.bootstrap_ci(
        y_true.tolist(),
        y_pred.tolist(),
        metric="precision",
        n_iterations=1000
    )

    recall_ci = evaluator.bootstrap_ci(
        y_true.tolist(),
        y_pred.tolist(),
        metric="recall",
        n_iterations=1000
    )

    print("\n📊 95% Confidence Intervals:")
    print(f"  F1 Score: [{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]")
    print(f"  Precision: [{precision_ci[0]:.3f}, {precision_ci[1]:.3f}]")
    print(f"  Recall: [{recall_ci[0]:.3f}, {recall_ci[1]:.3f}]")


def example_3_strategy_comparison():
    """
    Example 3: Compare inference strategies
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Strategy Comparison Across Models")
    print("="*80 + "\n")

    # Simulate performance of different strategies across models
    np.random.seed(42)

    strategy_metrics = {
        "zero_shot": np.random.normal(0.45, 0.05, 10).tolist(),
        "few_shot": np.random.normal(0.55, 0.04, 10).tolist(),
        "cot": np.random.normal(0.40, 0.06, 10).tolist()
    }

    # Perform comparison
    comparison = StrategyComparison.compare_strategies(strategy_metrics)

    print("📊 Strategy Performance:")
    for strategy in ["zero_shot", "few_shot", "cot"]:
        mean = comparison[f"{strategy}_mean"]
        std = comparison[f"{strategy}_std"]
        print(f"  {strategy.replace('_', '-').title()}: {mean:.3f} ± {std:.3f}")

    print("\n📊 Pairwise Comparisons:")
    for comp in comparison["comparisons"]:
        print(f"\n  {comp['strategy_a']} vs {comp['strategy_b']}:")
        print(f"    Superiority rate: {comp['superiority_rate']:.1f}%")
        print(f"    P-value: {comp['p_value']:.4f}")
        print(f"    Significant: {'Yes' if comp['significant'] else 'No'}")


def example_4_interpretability():
    """
    Example 4: Interpretability analysis with mock ablation results
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Interpretability Analysis - Domain Contributions")
    print("="*80 + "\n")

    # Mock ablation results
    from interpretability import AblationResult

    mock_ablation_results = {
        ("cognitive", "functional", "neuropsychiatric", "physiological", "demographic"):
            AblationResult(
                configuration=("cognitive", "functional", "neuropsychiatric", "physiological", "demographic"),
                f1_score=0.58,
                accuracy=0.75,
                n_dimensions=5
            ),
        ("functional", "neuropsychiatric", "physiological", "demographic"):
            AblationResult(
                configuration=("functional", "neuropsychiatric", "physiological", "demographic"),
                f1_score=0.32,
                accuracy=0.55,
                n_dimensions=4
            ),
        ("cognitive", "neuropsychiatric", "physiological", "demographic"):
            AblationResult(
                configuration=("cognitive", "neuropsychiatric", "physiological", "demographic"),
                f1_score=0.52,
                accuracy=0.70,
                n_dimensions=4
            ),
        ("cognitive", "functional", "physiological", "demographic"):
            AblationResult(
                configuration=("cognitive", "functional", "physiological", "demographic"),
                f1_score=0.55,
                accuracy=0.72,
                n_dimensions=4
            ),
        ("cognitive", "functional", "neuropsychiatric", "demographic"):
            AblationResult(
                configuration=("cognitive", "functional", "neuropsychiatric", "demographic"),
                f1_score=0.54,
                accuracy=0.71,
                n_dimensions=4
            ),
        ("cognitive", "functional", "neuropsychiatric", "physiological"):
            AblationResult(
                configuration=("cognitive", "functional", "neuropsychiatric", "physiological"),
                f1_score=0.56,
                accuracy=0.73,
                n_dimensions=4
            ),
    }

    # Compute dominance ratios
    dominance_ratios = DominanceAnalysis.compute_dominance_ratios(mock_ablation_results)

    print("📊 Domain Contribution Analysis:")
    print("\nDominance Ratios (Percentage Contribution):")

    # Sort by contribution
    sorted_domains = sorted(dominance_ratios.items(), key=lambda x: x[1], reverse=True)

    for domain, ratio in sorted_domains:
        bar_length = int(ratio / 2)  # Scale for visualization
        bar = "█" * bar_length
        print(f"  {domain:20s}: {bar} {ratio:5.1f}%")

    print("\n💡 Interpretation:")
    top_domain = sorted_domains[0][0]
    print(f"  • {top_domain.title()} features are the most important predictors")
    print(f"  • Contribute {sorted_domains[0][1]:.1f}% to overall performance")
    print(f"  • Consistent with known AD pathophysiology")


def main():
    """
    Run all examples
    """
    print("\n" + "="*80)
    print("SAGE-AD FRAMEWORK: USAGE EXAMPLES")
    print("="*80)

    try:
        # Example 1: Single prediction
        # example_1_single_prediction()  # Uncomment when API keys are configured

        # Example 2: Performance evaluation
        example_2_performance_evaluation()

        # Example 3: Strategy comparison
        example_3_strategy_comparison()

        # Example 4: Interpretability
        example_4_interpretability()

        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*80 + "\n")

        print("📚 Next Steps:")
        print("  1. Configure API keys for LLM models")
        print("  2. Prepare your cohort data")
        print("  3. Customize configs/default_config.json")
        print("  4. Run: python main.py --config configs/default_config.json")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nNote: Example 1 requires valid API keys and may be commented out")


if __name__ == "__main__":
    main()
