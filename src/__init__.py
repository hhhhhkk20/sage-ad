"""SAGE-AD Framework"""

__version__ = "1.0.0"

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
