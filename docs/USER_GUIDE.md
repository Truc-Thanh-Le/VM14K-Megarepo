# VM14K-Megarepo User Guide

## Overview

This user guide walks you through using the VM14K-Megarepo system to process medical data and benchmark large language models (LLMs) on Vietnamese medical questions. The system provides a complete pipeline from data acquisition to performance evaluation.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Data Pipeline](#data-pipeline)
3. [LLM Benchmarking](#llm-benchmarking)
4. [Evaluation and Analysis](#evaluation-and-analysis)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites
- Complete the [Setup Guide](SETUP.md)
- Have your medical data sources ready
- API keys for LLM providers (if using API services)

### Quick Start Workflow
1. **Data Acquisition**: Use DataPlatformEtlService to extract raw medical data
2. **Data Cleaning**: Process data with LLM-based cleaning tools
3. **Deduplication**: Remove duplicate and irrelevant questions
4. **Inference**: Run LLM benchmarking on processed data
5. **Evaluation**: Analyze model performance with metrics and visualizations

## Data Pipeline

### 1. Data Extraction with ETL Service

#### Starting the ETL Service
```bash
cd DataPlatformEtlService
python server.py
```

#### Access the API Documentation
Open your browser to `http://localhost:8000/docs` to see available endpoints.

#### Basic Data Extraction
```python
import requests

# Example: Extract data from a database
extraction_config = {
    "source_type": "postgresql",
    "connection": {
        "host": "localhost",
        "port": 5432,
        "database": "medical_data",
        "username": "your_user",
        "password": "your_password"
    },
    "query": "SELECT * FROM medical_questions WHERE language = 'vietnamese'",
    "output_format": "csv"
}

response = requests.post(
    "http://localhost:8000/extract",
    json=extraction_config
)
```

#### Supported Data Sources
- **Databases**: PostgreSQL, MongoDB, SQL Server
- **Cloud Storage**: AWS S3, Google Cloud Storage
- **Web APIs**: RESTful endpoints, GraphQL
- **Files**: CSV, JSON, Parquet

#### Configuration Examples

**MongoDB Extraction**:
```json
{
    "source_type": "mongodb",
    "connection": {
        "uri": "mongodb://localhost:27017/medical_db"
    },
    "collection": "vietnamese_medical_qa",
    "filter": {"category": "clinical"},
    "output_format": "jsonl"
}
```

**Web Scraping**:
```json
{
    "source_type": "web_scraper",
    "config": {
        "urls": ["https://medical-site.com/questions"],
        "selectors": {
            "question": ".question-text",
            "options": ".answer-option",
            "correct_answer": ".correct-answer"
        }
    },
    "output_format": "csv"
}
```

### 2. Data Cleaning

#### Preparing Your Data
Ensure your raw data contains:
- Medical questions in Vietnamese
- Multiple choice options (A-G)
- Additional metadata (subject, difficulty, etc.)

#### Running the Cleaning Process
```bash
cd DataCleaning

# Configure API keys
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"

# Run cleaning with default settings
python classification.py \
    --input raw_medical_data.csv \
    --output cleaned_medical_data.csv \
    --model gpt-3.5-turbo \
    --batch_size 50
```

#### Advanced Cleaning Options
```bash
# Use different LLM provider
python classification.py \
    --input raw_data.csv \
    --output cleaned_data.csv \
    --model claude-3-sonnet \
    --provider anthropic \
    --temperature 0.1 \
    --max_tokens 1000

# Custom prompt templates
python classification.py \
    --input raw_data.csv \
    --output cleaned_data.csv \
    --prompt_file custom_prompts.py \
    --cleaning_tasks "html_removal,format_standardization,option_normalization"
```

#### Cleaning Tasks Performed
- **HTML Tag Removal**: Strips HTML markup from questions and answers
- **Format Standardization**: Ensures consistent question formatting
- **Option Normalization**: Standardizes multiple choice options (A, B, C, D, E, F, G)
- **Language Verification**: Confirms content is in Vietnamese
- **Medical Content Validation**: Ensures questions are medically relevant

#### Custom Prompt Configuration
Edit `prompts.py` to customize cleaning behavior:
```python
CLEANING_PROMPTS = {
    "html_removal": """
Remove all HTML tags and formatting from the following medical question
while preserving the essential content and structure.
    """,
    "format_standardization": """
Reformat this Vietnamese medical question to follow standard multiple choice format:
- Clear question statement
- Options labeled A through G
- Proper punctuation and spacing
    """
}
```

### 3. Deduplication

#### Basic Deduplication
```bash
cd Deduplication

# Run three-tier deduplication process
python deduplication.py --input_path cleaned_medical_data.csv
```

This creates three output files:
- `dedup_v1.csv`: Exact duplicates removed
- `dedup_v2.csv`: Near-duplicates removed (>0.9 similarity)
- `dedup_v3.csv`: Off-topic questions filtered

#### Advanced Deduplication
```bash
# Custom similarity threshold
python deduplication.py \
    --input_path cleaned_data.csv \
    --similarity_threshold 0.85 \
    --output_dir ./dedup_results \
    --batch_size 500

# Specific deduplication levels
python deduplication.py \
    --input_path cleaned_data.csv \
    --exact_match_only \
    --output_file exact_dedup.csv
```

#### Understanding Deduplication Levels

**Level 1 - Exact Match Removal**:
- Normalizes text (lowercase, remove punctuation)
- Removes identical questions

**Level 2 - Similarity-based Removal**:
- Uses edit distance algorithm
- Removes questions with >90% similarity
- Preserves question variations

**Level 3 - Topic Filtering**:
- Identifies off-topic questions
- Removes non-medical content
- Maintains medical domain focus

## LLM Benchmarking

### 1. API Services Inference

#### Setup Environment
```bash
cd Inference/APIServices
source .venv/bin/activate

# Configure your API keys in .env file
cat > .env << EOF
OPENAI_API_KEY=your_openai_key
AZURE_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_API_KEY=your_azure_key
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
DEEPSEEK_ENDPOINT=your_deepseek_endpoint
DEEPSEEK_API_KEY=your_deepseek_key
EOF
```

#### Prepare Input Data
Place your deduplicated data in the `raw_data/` directory:
```bash
cp ../../../Deduplication/dedup_v3.csv raw_data/medical_questions.csv
```

#### Run Batch Inference
```bash
# Run inference across all configured providers
uv run batch_run.py

# Run specific model
uv run batch_run.py --model gpt-4 --provider openai

# Custom configuration
uv run batch_run.py \
    --input raw_data/medical_questions.csv \
    --output_dir batch_out \
    --batch_size 100 \
    --temperature 0.1 \
    --max_tokens 500
```

#### Supported LLM Providers

**OpenAI**:
- GPT-4, GPT-4-turbo
- GPT-3.5-turbo
- Custom fine-tuned models

**Azure OpenAI**:
- Same models as OpenAI
- Enterprise-grade deployment
- Regional compliance options

**AWS Bedrock**:
- Claude 3 (Sonnet, Haiku, Opus)
- Llama 2/3 variants
- Titan models

**Google Gemini**:
- Gemini Pro
- Gemini Pro Vision
- PaLM 2 models

**Groq**:
- Llama 3 (8B, 70B)
- Mixtral 8x7B
- High-speed inference

**DeepSeek**:
- DeepSeek R1
- Code and reasoning models

#### Monitoring Inference Progress
```bash
# Check batch job status
tail -f batch_out/*/status.log

# Monitor costs
cat batch_out/cost_summary.json

# View partial results
head -n 10 batch_out/openai/results.jsonl
```

### 2. Self-Hosted Inference

#### Setup vLLM Server
```bash
cd Inference/SelfHost

# Install vLLM with GPU support
pip install vllm[cuda]

# Configure models in run.py
nano run.py
```

#### Configure Models
Edit `run.py` to specify models:
```python
MODELS = [
    "microsoft/DialoGPT-medium",
    "vinai/phobert-base",
    "VietAI/viet-llama2-7b-chat",
    "meta-llama/Llama-2-7b-chat-hf",
    "huatuogpt/HuatuoGPT-o1",
    "R1-Distill-Llama-8B"
]
```

#### Run Inference
```bash
# Start inference for all models
python run.py

# Run specific model types
python run_reasoning.py  # For reasoning models
python run_nonreasoning.py  # For standard models
```

#### GPU Memory Management
```python
# In run_nonreasoning.py, configure vLLM
vllm_config = {
    "model": model_name,
    "gpu_memory_utilization": 0.8,
    "max_model_len": 2048,
    "tensor_parallel_size": 2,  # For multi-GPU
    "dtype": "float16"
}
```

## Evaluation and Analysis

### 1. Running Benchmark Analysis

#### Start Jupyter Environment
```bash
cd Evaluation
jupyter notebook benchmark_analysis.ipynb
```

#### Key Analysis Components

**Load Results**:
```python
import pandas as pd
import numpy as np

# Load inference results
results_df = pd.read_csv('path/to/inference_results.csv')

# Load ground truth
ground_truth = pd.read_csv('path/to/ground_truth.csv')
```

**Calculate Pass@k Metrics**:
```python
def calculate_pass_at_k(predictions, ground_truth, k_values=[1, 3, 5]):
    """Calculate pass@k accuracy for different k values"""
    results = {}
    for k in k_values:
        # Implementation in notebook
        pass_at_k = compute_pass_at_k(predictions, ground_truth, k)
        results[f'pass@{k}'] = pass_at_k
    return results
```

**Ensemble Analysis**:
```python
def ensemble_evaluation(model_predictions):
    """Evaluate ensemble of multiple models"""
    # Majority voting
    ensemble_pred = majority_vote(model_predictions)
    
    # Weighted ensemble
    weighted_pred = weighted_ensemble(model_predictions, weights)
    
    return ensemble_pred, weighted_pred
```

### 2. Visualization and Reporting

#### Performance Comparison Charts
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Model performance comparison
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x='model', y='accuracy')
plt.title('Model Performance Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_comparison.png')
```

#### Statistical Analysis
```python
from scipy import stats

# Statistical significance testing
def compare_models(model1_scores, model2_scores):
    t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
    return {
        'statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```

#### Generate Reports
```python
def generate_report(results_dict):
    """Generate comprehensive evaluation report"""
    report = {
        'summary': calculate_summary_stats(results_dict),
        'model_rankings': rank_models(results_dict),
        'statistical_tests': run_significance_tests(results_dict),
        'recommendations': generate_recommendations(results_dict)
    }
    return report
```

### 3. Custom Metrics

#### Define Domain-Specific Metrics
```python
def medical_accuracy_metric(predictions, ground_truth, medical_categories):
    """Calculate accuracy by medical specialty"""
    category_scores = {}
    for category in medical_categories:
        mask = ground_truth['category'] == category
        cat_pred = predictions[mask]
        cat_truth = ground_truth[mask]
        category_scores[category] = accuracy_score(cat_truth, cat_pred)
    return category_scores
```

#### Confidence Interval Calculation
```python
def bootstrap_confidence_interval(scores, n_bootstrap=1000, confidence=0.95):
    """Calculate bootstrap confidence intervals"""
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrap_scores.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_scores, 100 * alpha/2)
    upper = np.percentile(bootstrap_scores, 100 * (1 - alpha/2))
    return lower, upper
```

## Best Practices

### Data Quality
1. **Validate Input Data**: Ensure proper encoding and format
2. **Review Cleaning Results**: Manually inspect sample cleaned data
3. **Monitor Deduplication**: Check for over-aggressive removal
4. **Version Control**: Track data versions and transformations

### Inference Optimization
1. **Batch Processing**: Use appropriate batch sizes for efficiency
2. **Rate Limiting**: Respect API provider limits
3. **Error Handling**: Implement robust retry mechanisms
4. **Cost Monitoring**: Track API usage and costs

### Evaluation Rigor
1. **Multiple Metrics**: Use diverse evaluation measures
2. **Statistical Testing**: Verify significance of results
3. **Cross-Validation**: Use multiple evaluation sets
4. **Documentation**: Record experimental parameters

### Performance Tips
1. **Parallel Processing**: Utilize multiple cores/GPUs
2. **Memory Management**: Monitor and optimize memory usage
3. **Caching**: Cache intermediate results when appropriate
4. **Profiling**: Identify and optimize bottlenecks

## Troubleshooting

### Common Issues

#### Data Processing Errors
```bash
# Check data format
head -n 5 your_data.csv
file your_data.csv

# Validate encoding
python -c "open('your_data.csv', 'r', encoding='utf-8').read()"
```

#### API Connection Issues
```bash
# Test API connectivity
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models

# Check rate limits
python -c "import openai; print(openai.api_key)"
```

#### Memory Problems
```bash
# Monitor memory usage
htop
nvidia-smi  # For GPU memory

# Reduce batch sizes in config
```

#### Evaluation Errors
```bash
# Check result format compatibility
python -c "
import pandas as pd
df = pd.read_csv('results.csv')
print(df.columns)
print(df.dtypes)
"
```

### Performance Optimization

#### Data Pipeline
- Use streaming for large datasets
- Implement parallel processing
- Optimize database queries
- Cache frequently accessed data

#### LLM Inference
- Batch requests when possible
- Use connection pooling
- Implement exponential backoff
- Monitor and optimize token usage

#### Analysis and Visualization
- Use efficient data structures
- Leverage vectorized operations
- Implement lazy loading for large datasets
- Cache computed metrics

## Next Steps

After completing your first benchmark:

1. **Analyze Results**: Identify top-performing models for your use case
2. **Fine-tune Models**: Consider domain-specific fine-tuning
3. **Expand Dataset**: Add more diverse medical questions
4. **Custom Metrics**: Develop domain-specific evaluation measures
5. **Deploy Models**: Set up production inference pipelines
6. **Continuous Monitoring**: Track model performance over time

## Support and Resources

- **Documentation**: Check component-specific READMEs
- **Issue Tracking**: Use GitHub issues for bug reports
- **Community**: Join discussions and share insights
- **Updates**: Monitor for new model providers and features

For additional help, consult the [Architecture Documentation](ARCHITECTURE.md) and [Setup Guide](SETUP.md).