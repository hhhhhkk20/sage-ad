# SAGE-AD: Survey Assessment for Generalizable Early AD Detection

A comprehensive benchmark framework for evaluating Large Language Models (LLMs) in community-based early prediction of Alzheimer's disease using longitudinal survey data.

##  Overview

This repository contains the implementation of the SAGE-AD framework described in the paper:

**"SAGE-AD: A benchmark framework for large language models in community-based early prediction of Alzheimer's disease"**

The framework evaluates 49 state-of-the-art LLMs across three international population-based cohorts (ELSA, HRS, SHARE) using longitudinal health profiles derived from community surveys.

### Key Features

- **Multiple Inference Strategies**: Zero-shot, few-shot, and chain-of-thought prompting
- **Comprehensive Evaluation Metrics**: Accuracy, precision, recall, F1, balanced accuracy with bootstrap confidence intervals
- **Cross-Cohort Generalization Analysis**: Distribution shift evaluation across populations
- **Temporal Decay Analysis**: Performance across 1-4 year prediction horizons
- **Interpretability Analysis**: Feature ablation, dominance ratios, and interaction analysis
- **Support for 49+ LLMs**: Both proprietary (GPT, Gemini, Claude) and open-source models

## 🏗️ Project Structure

```
sage_ad_framework/
├── main.py                      # Main execution script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── configs/                     # Configuration files
│   └── default_config.json
├── src/                         # Source code modules
│   ├── llm_inference.py        # LLM inference strategies
│   ├── evaluation_metrics.py   # Performance evaluation
│   ├── interpretability.py     # Interpretability analysis
│   └── utils.py                # Utility functions
├── data/                        # Data directory (cohort data)
└── results/                     # Output results
```

##  Quick Start

### 1. Installation

```bash
git clone https://github.com/YOUR_USERNAME/sage_ad_framework.git
cd sage_ad_framework

python -m venv venv
source venv/bin/activate 

pip install -r requirements.txt
```

### 2. Configuration

Create a configuration file or use the default:

```bash
python main.py --create-config
```

Edit `configs/default_config.json` to customize:
- Models to evaluate
- Cohort data paths
- Inference strategies
- Output directories

### 3. Prepare Data

Place your cohort data files in the `data/` directory:

```
data/
├── ELSA/
│   └── elsa_data.csv
├── HRS/
│   └── hrs_data.csv
└── SHARE/
    └── share_data.csv
```

### 4. Run Benchmark

```bash
python main.py --config configs/default_config.json
```

##  Usage Examples

### Example 1: Evaluate Single Model with Few-Shot Strategy

```python
from src.llm_inference import LLMInference, InferenceStrategy, TemporalSymptomProfile

model = LLMInference(model_name="gpt-4o")

profile = TemporalSymptomProfile(
    participant_id="SUBJ_001",
    age_at_assessments=[65, 67, 69, 71],
    cognitive_features=[...],
    functional_features=[...],
    physiological_features=[...],
    neuropsychiatric_features=[...],
    demographics={"age": 65, "sex": "Female", "education": "12 years"}
)

result = model.predict(
    profile=profile,
    strategy=InferenceStrategy.FEW_SHOT,
    prediction_horizon=2
)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
```

### Example 2: Compute Performance Metrics

```python
from src.evaluation_metrics import PerformanceEvaluator

evaluator = PerformanceEvaluator()

y_true = [True, True, False, False, True]
y_pred = [True, True, False, True, True]

metrics = evaluator.compute_metrics(y_true, y_pred)
print(f"F1 Score: {metrics.f1_score:.3f}")
print(f"Precision: {metrics.precision:.3f}")
print(f"Recall: {metrics.recall:.3f}")

# Bootstrap confidence intervals
lower, upper = evaluator.bootstrap_ci(y_true, y_pred, metric="f1_score")
print(f"95% CI: [{lower:.3f}, {upper:.3f}]")
```

### Example 3: Feature Ablation Analysis

```python
from src.interpretability import FeatureDomainAblation, DominanceAnalysis

# Initialize ablation analyzer
ablation = FeatureDomainAblation(inference_engine=model)

ablation_results = ablation.run_ablation_experiment(
    profiles=profiles,
    true_labels=labels,
    prediction_horizon=1,
    strategy=InferenceStrategy.FEW_SHOT
)

dominance_ratios = DominanceAnalysis.compute_dominance_ratios(ablation_results)

for domain, ratio in dominance_ratios.items():
    print(f"{domain}: {ratio:.1f}%")
```

##  Key Components

### 1. LLM Inference Strategies (`llm_inference.py`)

Three inference paradigms are supported:

- **Zero-Shot**: Models rely entirely on pre-trained knowledge
- **Few-Shot**: Models receive 3-5 demonstrative examples before prediction
- **Chain-of-Thought (CoT)**: Models generate explicit step-by-step reasoning

### 2. Performance Evaluation (`evaluation_metrics.py`)

Comprehensive metrics including:

- Classification metrics (accuracy, precision, recall, F1, balanced accuracy)
- Bootstrap confidence intervals (5000 iterations by default)
- Cross-cohort generalization gaps
- Temporal decay analysis
- Statistical comparisons (Wilcoxon, Mann-Whitney U)

### 3. Interpretability Analysis (`interpretability.py`)

Multiple interpretability methods:

- **Feature Domain Ablation**: Systematic removal of feature domains
- **Dominance Analysis**: Quantify contribution of each domain
- **Interaction Analysis**: Pairwise feature interactions
- **Reasoning Analysis**: Extract clinical keywords from LLM outputs

## Results

The framework outputs comprehensive results including:

### Model Performance
- F1 scores across all models and strategies
- Precision-recall trade-offs
- Bootstrap confidence intervals
- Performance stratified by model category

### Cross-Cohort Analysis
- Generalization gaps across populations
- Transfer patterns (source → target)
- Pareto-optimal models

### Temporal Dynamics
- Performance decay across prediction horizons (1-4 years)
- Few-shot gain attenuation
- Temporal stability rankings

### Interpretability
- Domain contribution percentages
- Interaction matrices
- Word clouds from reasoning processes

## 🔧 Configuration

Example configuration file (`configs/default_config.json`):

```json
{
  "experiment_name": "sage_ad_benchmark",
  "models": [
    "gpt-4o",
    "gemini-2.5-flash",
    "claude-4.5-sonnet"
  ],
  "cohorts": ["ELSA", "HRS", "SHARE"],
  "cohort_data_paths": {
    "ELSA": "data/ELSA/elsa_data.csv",
    "HRS": "data/HRS/hrs_data.csv",
    "SHARE": "data/SHARE/share_data.csv"
  },
  "inference_strategies": ["zero_shot", "few_shot", "cot"],
  "prediction_horizons": [1, 2, 3, 4],
  "primary_cohort": "HRS",
  "bootstrap_iterations": 5000,
  "confidence_level": 0.95,
  "run_strategy_comparison": true,
  "run_temporal_analysis": true,
  "run_cross_cohort": true,
  "run_interpretability": true,
  "output_dir": "./results",
  "cache_dir": "./cache",
  "log_dir": "./logs",
}
```

## 🔑 API Keys

To use proprietary LLMs, set your API keys as environment variables:

```bash
# OpenAI (GPT models)
export OPENAI_API_KEY="your-api-key"

# Google (Gemini models)
export GOOGLE_API_KEY="your-api-key"

# Anthropic (Claude models)
export ANTHROPIC_API_KEY="your-api-key"
```

Or create a `.env` file:

```
OPENAI_API_KEY=your-api-key
GOOGLE_API_KEY=your-api-key
ANTHROPIC_API_KEY=your-api-key
```

##  Testing

Run tests to verify installation:

```bash
pytest tests/
```





### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/

# Run linting
flake8 src/
```

---

**Version**: 1.0.0
**Last Updated**: 2025-01-31
