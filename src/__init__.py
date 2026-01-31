"""
SAGE-AD Framework
=================

A comprehensive benchmark framework for evaluating Large Language Models
in community-based early prediction of Alzheimer's disease.

Modules:
    - llm_inference: LLM inference strategies (zero-shot, few-shot, CoT)
    - evaluation_metrics: Performance metrics and statistical analysis
    - interpretability: Feature ablation and reasoning analysis
    - utils: Data loading, preprocessing, and result management
"""

__version__ = "1.0.0"
__author__ = "SAGE-AD Research Team"

from .llm_inference import (
    LLMInference,
    BatchInference,
    InferenceStrategy,
    TemporalSymptomProfile,
    PromptTemplate
)

from .evaluation_metrics import (
    PerformanceEvaluator,
    ClassificationMetrics,
    CrossCohortAnalysis,
    TemporalAnalysis,
    StrategyComparison,
    StatisticalComparison
)

from .interpretability import (
    FeatureDomainAblation,
    DominanceAnalysis,
    InteractionAnalysis,
    ReasoningAnalysis,
    TemporalContributionAnalysis,
    CognitiveFeatureAnalysis
)

from .utils import (
    DataLoader,
    CohortProcessor,
    ResultsManager,
    Logger,
    ConfigManager
)

__all__ = [
    # LLM Inference
    "LLMInference",
    "BatchInference",
    "InferenceStrategy",
    "TemporalSymptomProfile",
    "PromptTemplate",

    # Evaluation
    "PerformanceEvaluator",
    "ClassificationMetrics",
    "CrossCohortAnalysis",
    "TemporalAnalysis",
    "StrategyComparison",
    "StatisticalComparison",

    # Interpretability
    "FeatureDomainAblation",
    "DominanceAnalysis",
    "InteractionAnalysis",
    "ReasoningAnalysis",
    "TemporalContributionAnalysis",
    "CognitiveFeatureAnalysis",

    # Utils
    "DataLoader",
    "CohortProcessor",
    "ResultsManager",
    "Logger",
    "ConfigManager"
]
