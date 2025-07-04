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
