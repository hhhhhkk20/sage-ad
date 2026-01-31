# SAGE-AD Framework: Quick Start Guide

This guide will help you get started with the SAGE-AD framework in 10 minutes.

## 🚀 Installation (2 minutes)

### Step 1: Clone and Setup

```bash
cd /path/to/your/projects
git clone https://github.com/YOUR_USERNAME/sage_ad_framework.git
cd sage_ad_framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🔑 API Configuration (3 minutes)

### Step 2: Set up API Keys

Create a `.env` file in the project root:

```bash
# .env file
OPENAI_API_KEY=sk-your-openai-key-here
GOOGLE_API_KEY=your-google-api-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
```

Or set environment variables:

```bash
export OPENAI_API_KEY="sk-your-key"
export GOOGLE_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

**Getting API Keys:**
- OpenAI (GPT): https://platform.openai.com/api-keys
- Google (Gemini): https://makersuite.google.com/app/apikey
- Anthropic (Claude): https://console.anthropic.com/

## 📊 Run Examples (2 minutes)

### Step 3: Test Installation

```bash
# Run example usage script
python example_usage.py
```

This will demonstrate:
- ✅ Performance evaluation
- ✅ Strategy comparison
- ✅ Interpretability analysis

Expected output:
```
📊 Classification Metrics:
  Accuracy: 0.850
  Precision: 0.765
  Recall: 0.765
  F1 Score: 0.765
  Balanced Accuracy: 0.838
```

## 🎯 Your First Benchmark (3 minutes)

### Step 4: Create Configuration

```bash
# Generate default config
python main.py --create-config
```

### Step 5: Edit Configuration

Edit `configs/default_config.json`:

```json
{
  "experiment_name": "my_first_benchmark",
  "models": [
    "gpt-4o"  // Start with one model
  ],
  "cohorts": ["HRS"],  // Start with one cohort
  "prediction_horizons": [1, 2],  // Start with shorter horizons
  "run_strategy_comparison": true,
  "run_temporal_analysis": false,  // Skip for first run
  "run_cross_cohort": false,  // Skip for first run
  "run_interpretability": false  // Skip for first run
}
```

### Step 6: Prepare Sample Data (Optional)

For testing, create a minimal CSV file `data/HRS/hrs_sample.csv`:

```csv
participant_id,wave,age,ad_diagnosis,word_recall_immediate,word_recall_delayed,orientation,ADL_impairment,IADL_impairment,BMI,grip_strength,gait_speed,depression_score,anxiety,sex,education_years
SUBJ001,1,65,0,8,6,5,0,0,25.3,32,1.2,2,low,Female,12
SUBJ001,2,67,0,7,5,5,0,1,24.8,30,1.1,3,low,Female,12
SUBJ001,3,69,1,5,3,4,1,2,24.2,28,1.0,5,moderate,Female,12
SUBJ002,1,70,0,10,8,5,0,0,27.1,35,1.3,1,none,Male,16
SUBJ002,2,72,0,9,8,5,0,0,26.8,34,1.2,1,none,Male,16
SUBJ002,3,74,0,9,7,5,0,0,26.5,33,1.2,2,low,Male,16
```

### Step 7: Run Benchmark

```bash
python main.py --config configs/default_config.json
```

## 📈 View Results

Results will be saved in:
```
results/
  └── my_first_benchmark_YYYYMMDD_HHMMSS.json

logs/
  └── my_first_benchmark_YYYYMMDD_HHMMSS.log
```

Open the JSON file to see:
- Model performance metrics
- F1 scores with confidence intervals
- Strategy comparisons
- Detailed predictions

## 🎓 Next Steps

### Learn More

1. **Read the Full README**: [README.md](README.md)
2. **Explore Examples**: Run `python example_usage.py`
3. **Customize Analysis**: Edit configuration file
4. **Add More Models**: Update `models` list in config

### Common Tasks

#### Add a New Model

```json
{
  "models": [
    "gpt-4o",
    "gemini-2.5-flash",  // Add this
    "claude-4.5-sonnet"   // And this
  ]
}
```

#### Enable All Analysis Types

```json
{
  "run_strategy_comparison": true,
  "run_temporal_analysis": true,
  "run_cross_cohort": true,
  "run_interpretability": true
}
```

#### Compare Multiple Cohorts

```json
{
  "cohorts": ["ELSA", "HRS", "SHARE"],
  "run_cross_cohort": true
}
```

## 🐛 Troubleshooting

### Problem: Import Error

```bash
# Solution: Ensure you're in virtual environment
source venv/bin/activate
pip install -r requirements.txt
```

### Problem: API Key Error

```bash
# Solution: Check environment variables
echo $OPENAI_API_KEY

# Or create .env file
echo "OPENAI_API_KEY=your-key" > .env
```

### Problem: Out of Memory

```json
// Solution: Reduce sample size
{
  "interpretability": {
    "ablation_sample_size": 50  // Reduce from 100
  }
}
```

### Problem: Rate Limits

```json
// Solution: Add delays between requests
{
  "api_settings": {
    "retry_attempts": 5,
    "timeout_seconds": 120
  }
}
```

## 📚 Additional Resources

- **Paper**: [SAGE-AD Benchmark Paper]
- **Documentation**: Full API documentation in README.md
- **Issues**: Report bugs on GitHub
- **Contact**: 3139490837@qq.com

## ✅ Verification Checklist

Before running full benchmark:

- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] All dependencies installed
- [ ] API keys configured
- [ ] Configuration file created
- [ ] Data files prepared
- [ ] Example script runs successfully

## 💡 Pro Tips

1. **Start Small**: Test with 1 model, 1 cohort, 1 horizon first
2. **Use Caching**: Enable cache to avoid re-running expensive predictions
3. **Monitor Costs**: Track API usage to manage costs
4. **Parallel Processing**: Enable for faster processing (requires more memory)
5. **Save Reasoning**: Keep reasoning traces for debugging

```json
{
  "cache_dir": "./cache",
  "save_predictions": true,
  "save_reasoning": true,
  "parallel_processing": {
    "enabled": true,
    "max_workers": 4
  }
}
```

## 🎉 Success!

You're now ready to run comprehensive benchmarks with SAGE-AD!

For questions or issues:
- 📧 Email: 3139490837@qq.com
- 🐛 GitHub Issues: [Report a bug]
- 📖 Documentation: [Full README](README.md)

---

**Time to Complete**: ~10 minutes
**Difficulty**: Beginner-Friendly
**Prerequisites**: Python 3.8+, API keys
