# VM14K-Megarepo: Medical Data Processing and LLM Benchmarking Pipeline
![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
This repository contains the complete pipeline for the paper "VM14K: First Vietnamese Medical Benchmark". It implements a comprehensive workflow from medical data acquisition to large language model (LLM) evaluation, including data scraping, cleaning, deduplication, inference benchmarking, and performance assessment. 

<a href="https://venera-ai.github.io/VM14K/" target="_blank" style="display: inline-block; padding: 6px 10px; background-color: #0d6efd; color: white; text-decoration: none; border-radius: 8px; font-weight: bold; text-shadow: 0 0 5px rgba(255,255,255,0.5); box-shadow: 0 0 15px rgba(13, 110, 253, 0.7);">🌟 Website</a>
&nbsp;&nbsp;
<a href="https://huggingface.co/datasets/venera-ai/VietnameseMedBench/" target="_blank" style="display: inline-block; padding: 6px 10px; background-color: #0d6efd; color: white; text-decoration: none; border-radius: 8px; font-weight: bold; text-shadow: 0 0 5px rgba(255,255,255,0.5); box-shadow: 0 0 15px rgba(13, 110, 253, 0.7);">💎 Dataset</a>
## Table of Contents
- [📚 Complete Documentation](#-complete-documentation)
- [License](#license)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Data Pipeline](#data-pipeline)
- [LLM Benchmarking](#llm-benchmarking)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [Contributing](#contributing)
- [Contact](#contact)

## 📚 Complete Documentation

For comprehensive guides, detailed setup instructions, and API references, visit our **[documentation suite](docs/README.md)**:

| Document | Description | When to Use |
|----------|-------------|-------------|
| **[Documentation Overview](docs/README.md)** | Navigation hub for all documentation | Start here for complete guidance |
| **[Architecture Guide](docs/ARCHITECTURE.md)** | System design and technical details | Understanding system internals |
| **[Setup Guide](docs/SETUP.md)** | Complete installation and configuration | Setting up the system |
| **[User Guide](docs/USER_GUIDE.md)** | Step-by-step usage instructions | Learning how to use the system |
| **[API Reference](docs/API_REFERENCE.md)** | REST API documentation | Building integrations |

💡 **New users**: Start with the [Setup Guide](docs/SETUP.md)  
🔧 **Developers**: Check the [Architecture Guide](docs/ARCHITECTURE.md)  
📊 **Researchers**: Follow the [User Guide](docs/USER_GUIDE.md)
## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE.md) file for details.
## Repository Structure
```
VM14K-Megarepo/
├── DataPlatformEtlService/    # Medical data scraping and ETL pipeline
├── DataCleaning/              # LLM-powered data cleaning and standardization
├── Deduplication/             # Three-tier deduplication process
├── Inference/                 # Multi-provider LLM inference framework
│   ├── APIServices/           # API-based inference (OpenAI, Azure, AWS, etc.)
│   └── SelfHost/              # Self-hosted inference with vLLM
├── Evaluation/                # Comprehensive model evaluation and analysis
├── docs/                      # Complete documentation suite
│   ├── README.md              # Documentation overview and navigation
│   ├── ARCHITECTURE.md        # System design and component details
│   ├── SETUP.md               # Installation and configuration guide
│   ├── USER_GUIDE.md          # Step-by-step usage instructions
│   └── API_REFERENCE.md       # Complete API documentation
├── README.md                  # Project overview (this file)
└── LICENSE.md                 # Apache 2.0 license
```
## Quick Start

### 1. Basic Setup (5 minutes)
```bash
# Clone repository
git clone https://github.com/your-org/VM14K-Megarepo.git
cd VM14K-Megarepo

# Setup ETL service
cd DataPlatformEtlService
pip install -r requirements.txt
python server.py
```

### 2. Complete Setup
For full installation including all components, databases, and inference providers, follow our [Setup Guide](docs/SETUP.md).

### 3. Usage Examples
Quick examples for common tasks:

```bash
# Extract medical data
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"source_type": "postgresql", "query": "SELECT * FROM medical_questions"}'

# Clean data with LLM
cd DataCleaning
python classification.py --input raw_data.csv --output cleaned_data.csv

# Remove duplicates
cd ../Deduplication
python deduplication.py --input_path cleaned_data.csv

# Run LLM inference
cd ../Inference/APIServices
uv run batch_run.py

# Analyze results
cd ../../Evaluation
jupyter notebook benchmark_analysis.ipynb
```

📖 **For detailed instructions**: See the [User Guide](docs/USER_GUIDE.md)

## Data Pipeline

### DataPlatformEtlService
The complete data processing workflow includes:
- **Multi-source extraction**: PostgreSQL, MongoDB, S3, web scraping
- **Real-time processing**: Streaming and batch processing capabilities
- **Scalable transformations**: DuckDB for analytics, Spark for big data
- **Multiple output formats**: CSV, JSON, Parquet, Delta Lake

### DataCleaning 
The cleaning process utilizes LLM to remove extra data like HTML tags, indices, etc. from the benchmark data. It also reformats all questions into the same standardized format:
- **HTML tag removal**: Clean web-scraped content
- **Format standardization**: Consistent question structure
- **Language validation**: Ensure Vietnamese content quality
- **Medical relevance**: Verify medical domain specificity

### Deduplication
Three-tier deduplication process to remove duplicate questions:
- **Level 1**: Exact match removal after text normalization
- **Level 2**: Near-duplicate detection (>90% similarity threshold)
- **Level 3**: Off-topic question filtering

## LLM Benchmarking

### APIServices
The benchmarking framework supports multiple LLM providers:
- **OpenAI**: GPT-4, GPT-3.5-turbo, fine-tuned models
- **Azure OpenAI**: Enterprise deployment with regional compliance
- **AWS Bedrock**: Claude 3, Llama variants, Titan models
- **Google Gemini**: Gemini Pro, PaLM 2 models
- **Groq**: High-speed inference with Llama 3, Mixtral
- **DeepSeek**: R1 and reasoning models

**Features**:
- Parallel inference processing
- Comprehensive cost tracking
- Error handling and retry mechanisms
- Batch processing optimization

### SelfHost
Local model deployment with vLLM:
- **GPU optimization**: Efficient memory utilization
- **Model variety**: Support for open-source medical models
- **Custom endpoints**: Flexible deployment options
- **Resource management**: Configurable memory and compute allocation

## Evaluation
The evaluation suite provides comprehensive model assessment:
- **Pass@k metrics**: Success rates at different k values (1, 3, 5)
- **F1 accuracy assessment**: Precision and recall balance
- **Ensemble evaluation**: Combined model performance analysis
- **Statistical testing**: Confidence intervals and significance tests
- **Performance visualization**: Charts and comparative analysis
- **Cost efficiency analysis**: Performance per dollar metrics

📊 **For detailed evaluation guide**: See [User Guide - Evaluation](docs/USER_GUIDE.md#evaluation-and-analysis)

📊 **For detailed evaluation guide**: See [User Guide - Evaluation](docs/USER_GUIDE.md#evaluation-and-analysis)

## Replicating VLLM Medical QA Evaluation Results

If you want to replicate the results of the **VLLM Medical QA Evaluation with Cross-Language Analysis** evaluation framework, the Python and Bash scripts to run the evaluation are located in the **`Eval-vLLM/`** folder. Follow the comprehensive instructions below:

---

# VLLM Medical QA Evaluation with Cross-Language Analysis

> A comprehensive evaluation framework for testing large language models on medical question-answering benchmarks using vLLM for efficient batch inference with advanced cross-language performance normalization.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![vLLM](https://img.shields.io/badge/vLLM-0.6.0+-green.svg)](https://github.com/vllm-project/vllm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

### Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Cross-Language Analysis Workflow](#cross-language-analysis-workflow)
- [Usage](#usage)
- [Configuration](#configuration)
- [Dataset Format](#dataset-format)
- [Output Files](#output-files)
- [Stratified Language Adjustment](#stratified-language-adjustment)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)


---

### Overview

This evaluation system assesses language model performance on medical QA datasets with support for:

- ✅ Multiple-choice medical questions
- 📊 Topic-based accuracy analysis
- 📈 Difficulty-level performance metrics
- 🔢 Token usage statistics
- 🌍 Bilingual support (English and Vietnamese)
- ⚖️ **Cross-language performance normalization with stratified adjustment factors**
- 📉 Comprehensive visualizations (10+ charts)
- 🎯 Response deviation/hallucination detection

---

### Features

#### Core Capabilities

- **Stratified Language Adjustment** - Sophisticated cross-language performance normalization using topic-specific and difficulty-specific adjustment factors for fair comparison between English and Vietnamese datasets
- **Comprehensive Evaluation** - Measures overall accuracy with detailed breakdowns by medical topic and difficulty level
- **Response Quality Analysis** - Identifies and quantifies response deviations (hallucinations) where the model generates invalid answers
- **Performance Metrics** - Calculates token generation speed (tokens per second) with batch analysis by topic and difficulty
- **Batch Processing** - Efficient batch inference using vLLM with configurable batch sizes and GPU utilization
- **Detailed Logging** - Generates comprehensive output files with individual question analysis, ranked topic performance, and token statistics
- **Option Shuffling** - Prevents position bias by randomly shuffling answer choices
- **Cross-Dataset Integration** - Seamlessly loads and analyzes multiple datasets for adjustment factor calculation

#### Advanced Features

- **StratifiedLanguageAdjuster Class** - ~200 lines of sophisticated adjustment logic
- **Topic-Difficulty Stratification** - Calculates adjustment factors at three levels: topic-only, difficulty-only, and combined
- **Confidence Scoring** - Each adjustment factor includes confidence scores based on sample sizes
- **Fallback Mechanisms** - Gracefully handles insufficient data with intelligent fallbacks
- **Language-Adjusted Metrics** - Both raw and adjusted metrics preserved in all outputs
- **Enhanced Visualizations** - All charts include language adjustment context when applicable

---

### Requirements

#### Hardware Requirements

- CUDA-compatible GPU (tested with single GPU setup)
- Minimum 16GB GPU memory (recommended: 24GB+)
- 32GB+ system RAM (48GB+ recommended for cross-language analysis)

#### Software Requirements

- Python 3.8+
- CUDA 11.8 or higher
- PyTorch 2.3.0+
- vLLM 0.6.0+

---

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/vllm-medical-qa-eval.git
cd vllm-medical-qa-eval
```

2. **Install dependencies**
```bash
pip install "vllm>=0.6.0" "torch>=2.3.0" transformers pandas numpy matplotlib tqdm
```

3. **Configure environment variables** (optional)
```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRITON_DISABLE_LINE_INFO=1
export CUDA_VISIBLE_DEVICES=0
```

---

### Quick Start

#### Using the Shell Script (Recommended)

```bash
# Make script executable
chmod +x run_evaluation_enhanced.sh

# Run with default settings (no cross-language analysis)
./run_evaluation_enhanced.sh -m Qwen/Qwen3-4B-Instruct-2507 -d medqa -l english

# Run with cross-language analysis (RECOMMENDED for fair comparison)
./run_evaluation_enhanced.sh -m Qwen/Qwen3-4B-Instruct-2507 -d vm14k -l vietnamese --auto-cross
```

#### Using Python Directly

```bash
# Basic evaluation
python vllm_qa_evaluation_enhanced.py \
  --model_name "Qwen/Qwen3-4B-Instruct-2507" \
  --dataset_path "/path/to/dataset.jsonl" \
  --dataset_language "english"

# With cross-language analysis
python vllm_qa_evaluation_enhanced.py \
  --model_name "Qwen/Qwen3-4B-Instruct-2507" \
  --dataset_path "/path/to/VM14K.jsonl" \
  --cross_dataset_path "/path/to/MedQA.jsonl" \
  --dataset_language "vietnamese" \
  --enable_topic_batch_analysis \
  --enable_difficulty_batch_analysis
```

---

### Cross-Language Analysis Workflow

#### ⚠️ IMPORTANT: Dataset Evaluation Order

**For proper cross-language analysis with language adjustment factors and comprehensive token analysis, you MUST evaluate datasets in this specific order:**

#### Step 1: Evaluate English (MedQA) Dataset First

```bash
./run_evaluation_enhanced.sh \
  -m Qwen/Qwen3-4B-Instruct-2507 \
  -d medqa \
  -l english \
  --enable-all-batch
```

**Why English first?**
- Establishes the English baseline (adjustment factor = 1.0)
- Generates reference token metrics for comparison
- Creates baseline performance data
- English is used as the reference language for all adjustments

**Output:** 
- Standard evaluation results
- Topic and difficulty performance metrics
- Token generation statistics
- Visualization charts

#### Step 2: Evaluate Vietnamese (VM14K) Dataset with Cross-Dataset

```bash
./run_evaluation_enhanced.sh \
  -m Qwen/Qwen3-4B-Instruct-2507 \
  -d vm14k \
  -l vietnamese \
  -c medqa \
  --enable-all-batch
```

**Why second with cross-dataset?**
- Uses English dataset as reference for adjustment calculations
- Calculates topic-specific and difficulty-specific adjustment factors
- Normalizes Vietnamese performance to English-equivalent metrics
- Enables fair cross-language comparison

**Output:** 
- Standard evaluation results
- **`language_adjustment_factors.txt`** - Complete adjustment factor analysis
- **Language-adjusted metrics** in all text outputs (infer_result.txt, topics_ranked_by_accuracy.txt)
- Enhanced visualizations with adjustment context
- Both raw and adjusted performance metrics

#### What Happens Without This Order?

❌ **Running Vietnamese First or Without Cross-Dataset:**
- No language adjustment factors generated
- Cannot fairly compare performance with English
- Missing normalized token efficiency metrics
- Vietnamese appears artificially faster (due to fewer tokens per question)

✅ **Running in Correct Order:**
- Fair performance comparison enabled
- Adjustment factors account for linguistic differences
- Token efficiency properly normalized
- Accurate cross-language performance metrics

#### Alternative: Auto-Cross Mode (Simplified)

```bash
# Step 1: English (same as above)
./run_evaluation_enhanced.sh -m MODEL -d medqa -l english --enable-all-batch

# Step 2: Vietnamese with auto-cross (automatically uses medqa as cross-dataset)
./run_evaluation_enhanced.sh -m MODEL -d vm14k -l vietnamese --auto-cross --enable-all-batch
```

#### Manual Paths (Non-Preset Datasets)

```bash
# Step 1: English evaluation
python vllm_qa_evaluation_enhanced.py \
  --model_name MODEL \
  --dataset_path /path/to/MedQA.jsonl \
  --dataset_language english \
  --enable_topic_batch_analysis \
  --enable_difficulty_batch_analysis

# Step 2: Vietnamese with cross-dataset
python vllm_qa_evaluation_enhanced.py \
  --model_name MODEL \
  --dataset_path /path/to/VM14K.jsonl \
  --cross_dataset_path /path/to/MedQA.jsonl \
  --dataset_language vietnamese \
  --enable_topic_batch_analysis \
  --enable_difficulty_batch_analysis
```

---

### Usage

#### Basic Example (Single Language)

```bash
# Evaluate English only
./run_evaluation_enhanced.sh \
  -m Qwen/Qwen3-4B-Instruct-2507 \
  -d /path/to/english_dataset.jsonl \
  -l english \
  -s 10000
```

#### Cross-Language Example (Recommended)

```bash
# Step 1: English
./run_evaluation_enhanced.sh \
  -m Qwen/Qwen3-4B-Instruct-2507 \
  -d /path/to/MedQA.jsonl \
  -l english \
  --enable-all-batch

# Step 2: Vietnamese with cross-analysis
./run_evaluation_enhanced.sh \
  -m Qwen/Qwen3-4B-Instruct-2507 \
  -d /path/to/VM14K.jsonl \
  -c /path/to/MedQA.jsonl \
  -l vietnamese \
  --enable-all-batch
```

#### Multi-GPU Example

```bash
./run_evaluation_enhanced.sh \
  -m meta-llama/Llama-2-70b-chat-hf \
  -d medqa \
  -l english \
  --tensor-parallel 4 \
  --gpu-memory 0.95
```

#### Configuration

**Shell Script Options (`run_evaluation_enhanced.sh`):**

| Flag | Description | Example |
|------|-------------|---------|
| `-m, --model` | Model name or path (required) | `-m Qwen/Qwen3-4B-Instruct-2507` |
| `-d, --dataset` | Dataset name or path (required) | `-d medqa` or `-d /path/to/data.jsonl` |
| `-c, --cross` | Cross-dataset path for language adjustment | `-c medqa` or `-c /path/to/MedQA.jsonl` |
| `-l, --language` | Dataset language (english/vietnamese) | `-l english` |
| `-s, --subset` | Subset size (default: 10000) | `-s 5000` |
| `--tensor-parallel` | Number of GPUs for tensor parallelism | `--tensor-parallel 4` |
| `--gpu-memory` | GPU memory utilization (0.0-1.0) | `--gpu-memory 0.9` |
| `--max-len` | Maximum model length in tokens | `--max-len 2048` |
| `--enable-topic-batch` | Enable topic-based batch analysis | `--enable-topic-batch` |
| `--enable-difficulty-batch` | Enable difficulty-based batch analysis | `--enable-difficulty-batch` |
| `--enable-all-batch` | Enable all batch analyses | `--enable-all-batch` |
| `--auto-cross` | Automatically use preset cross-dataset | `--auto-cross` |

**Preset Datasets:**

| Name | Language | Description |
|------|----------|-------------|
| `medqa` | English | US Medical Licensing Exam questions |
| `vm14k` | Vietnamese | Vietnamese Medical Benchmark |

**Python Script Arguments:**

```python
--model_name                      # Model identifier
--dataset_path                    # Path to evaluation dataset
--cross_dataset_path              # Path to cross-dataset (optional)
--dataset_language                # "english" or "vietnamese"
--subset_size                     # Number of questions to evaluate
--enable_topic_batch_analysis     # Enable topic batching
--enable_difficulty_batch_analysis # Enable difficulty batching
--tensor_parallel_size            # Number of GPUs
--gpu_memory_utilization          # GPU memory fraction (0.0-1.0)
--max_model_len                   # Maximum context length
```

---

### Dataset Format

Your dataset should be in JSONL format with the following structure:

```json
{
  "question": "What is the primary function of insulin?",
  "options": {
    "A": "Regulate blood glucose",
    "B": "Increase heart rate",
    "C": "Stimulate appetite",
    "D": "Lower blood pressure"
  },
  "answer_index": 0,
  "medical_topic": "Endocrinology",
  "difficulty_level": "Moderate"
}
```

**Required Fields:**
- `question` (string): The medical question text
- `options` (dict): Answer choices with keys "A", "B", "C", etc.
- `answer_index` (int): Correct answer index (0-based, so 0 = A, 1 = B, etc.)

**Optional Fields (Required for Cross-Language Analysis):**
- `medical_topic` (string): Medical specialty (e.g., "Cardiology", "Neurology")
- `difficulty_level` (string): Question difficulty ("Easy", "Moderate", "Hard")

**Supported Option Counts:**
- Minimum: 2 options (A-B)
- Maximum: 8 options (A-H)
- Most common: 4 options (A-D)

---

### Output Files

#### Standard Output Files (All Evaluations)

1. **`model_output.txt`** - Raw model responses for all questions
   ```
   Question 1:
   What is the primary function of insulin?
   
   Model Response:
   The primary function of insulin is to regulate blood glucose...
   Final Answer: A
   ```

2. **`infer_result.txt`** - Comprehensive evaluation summary
   ```
   Total Questions: 10000
   Correct: 8520
   Overall Accuracy: 85.20%
   Average Tokens per Question: 156.3
   Tokens per Second: 245.7
   ```

3. **`topics_ranked_by_accuracy.txt`** - Performance by medical topic
   ```
   Rank | Topic           | Accuracy | Questions | Tokens/Q | Tokens/Sec
   -----|-----------------|----------|-----------|----------|------------
   1    | Cardiology      | 92.3%    | 850       | 168.2    | 235.6
   2    | Neurology       | 88.7%    | 720       | 172.5    | 228.3
   ```

4. **Visualization Files** (10+ PNG charts):
   - `accuracy_by_topic.png`
   - `accuracy_by_difficulty.png`
   - `tokens_per_second_by_topic.png`
   - `tokens_per_second_by_difficulty.png`
   - `deviation_rate_by_topic.png`
   - `deviation_rate_by_difficulty.png`
   - Plus additional topic-difficulty heatmaps

#### Cross-Language Specific Output

5. **`language_adjustment_factors.txt`** ⭐ (Only with `--cross_dataset_path`)
   ```
   Cross-Language Performance Normalization Factors
   ================================================
   
   Topic-Specific Factors:
   -----------------------
   Cardiology:
     Token Ratio (VI/ENG): 0.611
     Length Ratio (VI/ENG): 0.650  
     Combined Factor: 0.631
     Sample Sizes - ENG: 850, VI: 1200
     Confidence: 1.0 (85 samples per strata)
   
   Difficulty-Specific Factors:
   ----------------------------
   Moderate:
     Token Ratio (VI/ENG): 0.623
     Combined Factor: 0.645
     Sample Sizes - ENG: 2500, VI: 3200
   ```

**Enhanced Text Files with Language Adjustment:**

When cross-dataset is provided, `infer_result.txt` and `topics_ranked_by_accuracy.txt` include both raw and adjusted metrics:

```
Performance Metrics:
  Raw Tokens/Sec: 245.7
  Adjusted Tokens/Sec (Language Normalized): 389.4
  
  Raw Questions/Sec: 1.57
  Adjusted Questions/Sec (Language Normalized): 0.99
```

---

### Stratified Language Adjustment

The stratified language adjustment system enables fair cross-language comparison by normalizing token efficiency differences between English and Vietnamese datasets.

#### Why Language Adjustment is Necessary

**Problem:** Vietnamese text requires significantly fewer tokens than English for the same content:
- English medical question: ~180 tokens
- Vietnamese medical question: ~110 tokens

**Impact:** Without adjustment:
- Vietnamese appears 60% faster (more questions/second)
- Unfair comparison between language performances
- Token efficiency metrics are misleading

**Solution:** Calculate adjustment factors that normalize performance based on linguistic token efficiency.

#### How It Works

##### 1. Token Ratio Calculation

For each medical topic and difficulty level:

```
Token Ratio = Average VI Tokens / Average ENG Tokens
Length Ratio = Average VI Character Length / Average ENG Character Length
Combined Factor = (Token Ratio + Length Ratio) / 2
```

##### 2. Stratification Levels

Three levels of adjustment factors are calculated:

- **Topic-Only**: Factors specific to each medical specialty (e.g., Cardiology, Neurology)
- **Difficulty-Only**: Factors specific to question difficulty (Easy, Moderate, Hard)
- **Topic-Difficulty Combined**: Most granular, combining both dimensions

##### 3. Adjustment Factor Application

The factors are applied to normalize Vietnamese performance metrics:

```
Adjusted Performance Metrics:
  adjusted_tokens_per_second = raw_tokens_per_second × adjustment_factor
  adjusted_questions_per_second = raw_questions_per_second ÷ adjustment_factor
```

##### 4. Fallback Hierarchy

When insufficient data exists for a specific combination:

1. **Best**: Use topic-difficulty specific factor (requires 3+ samples)
2. **Good**: Use topic-only factor (requires 5+ samples)
3. **Acceptable**: Use difficulty-only factor (requires 5+ samples)
4. **Default**: No adjustment (factor = 1.0)

#### Example Adjustment

**Real-world scenario:**

```
Cardiology Topic, Moderate Difficulty:
  English samples: 850 questions, avg 180 tokens per question
  Vietnamese samples: 1200 questions, avg 110 tokens per question
  
  Token ratio: 110/180 = 0.611
  Length ratio: (vietnamese avg chars) / (english avg chars) = 0.650
  Combined factor: (0.611 + 0.650) / 2 = 0.631
  
  Raw Vietnamese performance: 140 questions/second
  Adjusted performance: 140 ÷ 0.631 = 221.9 questions/second
  
  Interpretation: When accounting for token efficiency, Vietnamese 
  requires ~58% more computational effort than raw metrics suggest
```

#### Generated Factor File

The `language_adjustment_factors.txt` contains:

```
Topic: Cardiology
  Token Ratio (VI/ENG): 0.611
  Combined Factor: 0.631
  Sample Sizes - ENG: 850, VI: 1200
  Confidence: 1.0 (85 samples)

Difficulty: Moderate
  Token Ratio (VI/ENG): 0.623
  Combined Factor: 0.645
  Sample Sizes - ENG: 2500, VI: 3200
  Confidence: 1.0 (250 samples)

Topic-Difficulty: Cardiology + Moderate
  Token Ratio (VI/ENG): 0.608
  Combined Factor: 0.628
  Sample Sizes - ENG: 420, VI: 580
  Confidence: 1.0 (42 samples)
```

---

### Performance

#### Typical Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Processing speed | ~200 Q/s | With vLLM optimization |
| GPU memory | 8-12GB | For 4B parameter models |
| Evaluation time | 5-10 min | For 10K questions |
| Cross-dataset loading | +30 sec | One-time calculation |
| Batch analysis overhead | +2-3 min | Per batch type enabled |

#### Performance Impact of Cross-Language Analysis

| Component | Memory | Time | Notes |
|-----------|--------|------|-------|
| Base evaluation | 8-12GB | 5-10 min | Standard |
| Cross-dataset loading | +500MB | +5 sec | One-time |
| Factor calculation | +10MB | +30 sec | One-time |
| Adjusted metrics | +50MB | +2 sec | During analysis |
| **Total overhead** | **+560MB** | **+37 sec** | **~10% increase** |

#### Answer Extraction Methods

The system uses multiple fallback methods to extract answers:

1. **Exact pattern matching** - "Final Answer: A"
2. **Contextual keywords** - "The answer is A", "Correct answer: A"
3. **Parenthetical** - "(A)" or "[A]"
4. **Direct letter** - First valid letter (A-H) found
5. **None detection** - Identifies response deviations

#### Batch Analysis Benefits

When enabled via `--enable-all-batch`:

- **Topic-based analysis** - Reveals which medical specialties generate tokens faster
- **Difficulty-based analysis** - Shows how question complexity affects speed
- **Combined insights** - Identifies optimization opportunities

**Trade-off:** Adds 2-3 minutes per analysis type, but provides valuable performance insights.

---

### Troubleshooting

#### Out of Memory Errors

```bash
# Reduce GPU memory utilization
--gpu_memory_utilization 0.7

# Decrease maximum model length
--max_model_len 1024

# Reduce subset size
--subset_size 5000

# Use single batch analysis instead of both
--enable-topic-batch  # Instead of --enable-all-batch
```

#### Cross-Dataset Not Found

```bash
# Check file paths
ls -l /path/to/MedQA.jsonl
ls -l /path/to/VM14K.jsonl

# Use absolute paths
./run_evaluation_enhanced.sh -m MODEL -d /full/path/to/vm14k.jsonl -c /full/path/to/medqa.jsonl
```

#### Missing Language Adjustment Factors

**Problem:** `language_adjustment_factors.txt` not generated

**Solutions:**
1. Ensure you provided `--cross_dataset_path`
2. Check cross-dataset has sufficient samples (100+ recommended)
3. Verify cross-dataset has matching fields (medical_topic, difficulty_level)

#### Slow Performance

```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Monitor GPU
watch -n 1 nvidia-smi

# Reduce context if not needed
--max_model_len 1024

# Disable batch analysis for speed
# Don't use --enable-all-batch
```

#### Import Errors

```bash
# Reinstall vLLM
pip install --upgrade --force-reinstall vllm

# Check PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"

# Verify all packages
pip install --upgrade transformers pandas numpy matplotlib tqdm
```

#### Incorrect Language Labels in Output

**Problem:** Vietnamese output shows English metrics or vice versa

**Solution:** Always specify `--dataset_language` correctly:

```bash
# Correct
-l english  # When evaluating English dataset
-l vietnamese  # When evaluating Vietnamese dataset

# Incorrect (will cause wrong adjustments)
-l vietnamese  # But evaluating English dataset
```

---

### Best Practices

#### 1. Always Follow Evaluation Order

✅ **Correct:**
```bash
# First: English
./run_evaluation_enhanced.sh -m MODEL -d medqa -l english

# Second: Vietnamese with cross-dataset
./run_evaluation_enhanced.sh -m MODEL -d vm14k -l vietnamese -c medqa
```

❌ **Incorrect:**
```bash
# Vietnamese without cross-dataset or before English
./run_evaluation_enhanced.sh -m MODEL -d vm14k -l vietnamese
```

#### 2. Use Batch Analysis for Comprehensive Results

```bash
# Recommended for production evaluation
--enable-all-batch

# Use for quick testing only
# (no batch analysis flags)
```

#### 3. Verify Dataset Quality

```bash
# Check sample count
wc -l dataset.jsonl

# Verify required fields
head -1 dataset.jsonl | jq 'keys'

# Should show: ["question", "options", "answer_index", "medical_topic", "difficulty_level"]
```

#### 4. Monitor Resource Usage

```bash
# Before starting
nvidia-smi
free -h

# During evaluation (separate terminal)
watch -n 2 nvidia-smi
```

---

### Quick Reference Card

#### Most Common Commands

```bash
# 1. English evaluation (run first)
./run_evaluation_enhanced.sh -m Qwen/Qwen3-4B-Instruct-2507 -d medqa -l english --enable-all-batch

# 2. Vietnamese with cross-language analysis (run second)
./run_evaluation_enhanced.sh -m Qwen/Qwen3-4B-Instruct-2507 -d vm14k -l vietnamese -c medqa --enable-all-batch

# 3. Quick test (small subset)
./run_evaluation_enhanced.sh -m Qwen/Qwen3-4B-Instruct-2507 -d medqa -l english -s 1000

# 4. Multi-GPU
./run_evaluation_enhanced.sh -m MODEL -d medqa --tensor-parallel 4
```

#### Expected Output Files

**English Only:**
- model_output.txt
- infer_result.txt
- topics_ranked_by_accuracy.txt
- 10+ PNG visualization files

**Vietnamese with Cross-Dataset (adds):**
- ✨ language_adjustment_factors.txt
- Enhanced metrics in all text files
- Adjustment context in visualizations

---

**Remember:** For proper cross-language token analysis, always run MedQA (English) first, then VM14K (Vietnamese) with cross-dataset enabled!

---


## Citation
If you use this repository in your research, please cite our paper:
```bibtex
@misc{nguyen2025vm14kvietnamesemedicalbenchmark,
      title={VM14K: First Vietnamese Medical Benchmark}, 
      author={Thong Nguyen and Duc Nguyen and Minh Dang and Thai Dao and Long Nguyen and Quan H. Nguyen and Dat Nguyen and Kien Tran and Minh Tran},
      year={2025},
      eprint={2506.01305},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.01305}, 
}
``` 
<!-- ## Contributing
Contributions are welcome. Please open an issue first to discuss proposed changes. -->
<!-- ## Contact
For questions about this repository, please contact:
- [Your Name] (your.email@institution.edu)
- [Co-author Name] (coauthor.email@institution.edu) -->
