# SAGE-AD: Survey Assessment for Generalizable Early AD Detection

A benchmark framework for evaluating Large Language Models in community-based early prediction of Alzheimer's disease using longitudinal survey data.

## Overview

SAGE-AD evaluates 49 state-of-the-art LLMs across three international population-based cohorts (ELSA, HRS, SHARE) using longitudinal health profiles derived from community surveys.

### Key Features

- Multiple inference strategies (zero-shot, few-shot, chain-of-thought)
- Comprehensive evaluation metrics with bootstrap confidence intervals
- Cross-cohort generalization analysis
- Temporal decay analysis (1-4 year prediction horizons)
- Interpretability analysis through feature ablation

## Quick Start

```bash
# Installation
git clone https://github.com/hhhhhkk20/sage-ad.git
cd sage-ad
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configuration
python main.py --create-config

# Run benchmark
python main.py --config configs/default_config.json
```

## Usage

```python
from src import LLMInference, PerformanceEvaluator, InferenceStrategy

# Make prediction
model = LLMInference("gpt-4o")
result = model.predict(profile, strategy=InferenceStrategy.FEW_SHOT, horizon=2)

# Evaluate performance
evaluator = PerformanceEvaluator()
metrics = evaluator.compute_metrics(y_true, y_pred)
lower, upper = evaluator.bootstrap_ci(y_true, y_pred, "f1_score")
```

## API Keys

```bash
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

## Project Structure

```
sage-ad/
├── src/
│   ├── llm_inference.py        # LLM inference strategies
│   ├── evaluation_metrics.py   # Performance evaluation
│   ├── interpretability.py     # Interpretability analysis
│   └── utils.py                # Utility functions
├── configs/                     # Configuration files
├── data/                        # Cohort data
└── results/                     # Output results
```

## License

MIT License


