# VM14K-Megarepo Setup and Installation Guide

## Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 18.04+), macOS (10.15+), or Windows 10+
- **Python**: 3.8 or higher
- **Memory**: Minimum 8GB RAM (16GB+ recommended for large datasets)
- **Storage**: At least 50GB free space for data processing
- **GPU**: Optional but recommended for self-hosted inference (CUDA 11.8+ compatible)

### Required Software
- [Git](https://git-scm.com/)
- [Docker](https://docs.docker.com/get-docker/) (optional but recommended)
- [Python 3.8+](https://www.python.org/downloads/)
- [Node.js](https://nodejs.org/) (for some development tools)

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/VM14K-Megarepo.git
cd VM14K-Megarepo
```

### 2. Choose Your Setup Path

#### Option A: Full Local Setup
Best for development and customization
```bash
# Follow detailed instructions below
```

#### Option B: Docker Setup
Best for quick deployment and testing
```bash
# See Docker Setup section
```

#### Option C: Individual Component Setup
Best for using specific components only
```bash
# See Component-Specific Setup section
```

## Full Local Setup

### 1. DataPlatformEtlService Setup

#### Install Dependencies
```bash
cd DataPlatformEtlService
pip install -r requirements.txt
```

#### Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your configurations
nano .env
```

Required environment variables:
```env
# Database connections
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=medical_data
POSTGRES_USER=your_username
POSTGRES_PASSWORD=your_password

# MongoDB (if used)
MONGODB_URI=mongodb://localhost:27017/medical_data

# ClickHouse (if used)
CLICKHOUSE_HOST=localhost
CLICKHOUSE_PORT=9000

# AWS credentials (if using AWS services)
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=us-east-1

# Spark configuration
SPARK_MASTER=local[*]
SPARK_DRIVER_MEMORY=4g
```

#### Start the Service
```bash
# Development mode with hot reload
uvicorn server:app --reload --host 0.0.0.0 --port 8000

# Or production mode
python server.py
```

#### Verify Installation
```bash
# Test API endpoints
curl http://localhost:8000/docs
```

### 2. DataCleaning Setup

#### Install Dependencies
```bash
cd DataCleaning
pip install openai anthropic google-generativeai
# Add other LLM provider SDKs as needed
```

#### Configure API Keys
```bash
# Create environment file
echo "OPENAI_API_KEY=your_openai_key" > .env
echo "ANTHROPIC_API_KEY=your_anthropic_key" >> .env
echo "GOOGLE_API_KEY=your_google_key" >> .env
```

#### Test the Cleaning Process
```bash
# Prepare sample data
# Run cleaning script
python classification.py --input sample_data.csv --output cleaned_data.csv
```

### 3. Deduplication Setup

#### Install Dependencies
```bash
cd Deduplication
pip install -r requirements.txt
```

#### Run Deduplication
```bash
# Basic usage
python deduplication.py --input_path your_input_file.csv

# Advanced options
python deduplication.py \
    --input_path your_input_file.csv \
    --output_dir ./output \
    --similarity_threshold 0.9 \
    --batch_size 1000
```

### 4. Inference Setup

#### APIServices Setup
```bash
cd Inference/APIServices

# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
# OR
pip install uv

# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate     # Windows

# Install dependencies
uv sync
```

#### Configure API Keys
```bash
# Create .env file with API credentials
cat > .env << EOF
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
AZURE_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_API_KEY=your_azure_key
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_ROLE_ARN=your_aws_role
GEMINI_API_KEY=your_gemini_key
DEEPSEEK_ENDPOINT=your_deepseek_endpoint
DEEPSEEK_API_KEY=your_deepseek_key
EOF
```

#### SelfHost Setup
```bash
cd Inference/SelfHost

# Install vLLM
pip install vllm

# For GPU support
pip install vllm[cuda]

# Configure models in run.py
```

### 5. Evaluation Setup

#### Install Dependencies
```bash
cd Evaluation
pip install jupyter pandas numpy matplotlib seaborn scikit-learn
```

#### Start Jupyter Notebook
```bash
jupyter notebook benchmark_analysis.ipynb
```

## Docker Setup

### 1. Build All Services
```bash
# Build ETL service
cd DataPlatformEtlService
docker build -t vm14k-etl .

# Build other services as needed
```

### 2. Use Docker Compose (if available)
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 3. Individual Container Setup
```bash
# Run ETL service
docker run -p 8000:8000 --env-file .env vm14k-etl

# Run with volume mounts for data
docker run -p 8000:8000 \
    --env-file .env \
    -v $(pwd)/data:/app/data \
    vm14k-etl
```

## Component-Specific Setup

### DataPlatformEtlService Only
```bash
cd DataPlatformEtlService
pip install -r requirements.txt
cp .env.example .env
# Configure .env file
python server.py
```

### DataCleaning Only
```bash
cd DataCleaning
pip install openai pandas numpy
# Configure API keys
python classification.py --help
```

### Deduplication Only
```bash
cd Deduplication
pip install -r requirements.txt
python deduplication.py --input_path your_data.csv
```

### Inference Only
```bash
# For API services
cd Inference/APIServices
uv venv && source .venv/bin/activate
uv sync
# Configure .env
uv run batch_run.py

# For self-hosted
cd Inference/SelfHost
pip install vllm
python run.py
```

### Evaluation Only
```bash
cd Evaluation
pip install jupyter pandas numpy matplotlib seaborn
jupyter notebook benchmark_analysis.ipynb
```

## Database Setup

### PostgreSQL Setup
```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql
CREATE DATABASE medical_data;
CREATE USER vm14k_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE medical_data TO vm14k_user;
\q
```

### MongoDB Setup (Optional)
```bash
# Install MongoDB
sudo apt-get install mongodb

# Start MongoDB service
sudo systemctl start mongodb
sudo systemctl enable mongodb
```

### ClickHouse Setup (Optional)
```bash
# Install ClickHouse
curl https://clickhouse.com/ | sh
sudo ./clickhouse install

# Start ClickHouse
sudo systemctl start clickhouse-server
```

## Development Environment Setup

### 1. Install Development Tools
```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Install testing frameworks
pip install pytest pytest-cov

# Install linting tools
pip install black flake8 mypy
```

### 2. IDE Configuration

#### VS Code
Recommended extensions:
- Python
- Jupyter
- Docker
- GitLens
- Python Docstring Generator

#### PyCharm
Configure Python interpreter and enable:
- Code inspection
- PEP 8 formatting
- Type checking

### 3. Environment Variables
Create a comprehensive `.env` file:
```bash
# Copy template
cp .env.example .env

# Add all required variables
nano .env
```

## Testing Setup

### Unit Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test suite
pytest tests/test_etl.py
```

### Integration Tests
```bash
# Test database connections
python -m pytest tests/integration/test_db.py

# Test API endpoints
python -m pytest tests/integration/test_api.py
```

### End-to-End Tests
```bash
# Full pipeline test
python scripts/test_pipeline.py
```

## Troubleshooting

### Common Issues

#### 1. Python Version Conflicts
```bash
# Use pyenv to manage Python versions
pyenv install 3.9.16
pyenv local 3.9.16
```

#### 2. Dependency Conflicts
```bash
# Use virtual environments
python -m venv vm14k_env
source vm14k_env/bin/activate
pip install -r requirements.txt
```

#### 3. Memory Issues
```bash
# Increase swap space
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 4. GPU Issues (for self-hosted inference)
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Install appropriate PyTorch version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 5. API Rate Limits
- Configure retry mechanisms
- Use batch processing
- Implement exponential backoff
- Monitor usage quotas

#### 6. Database Connection Issues
```bash
# Test database connectivity
python -c "import psycopg2; print('PostgreSQL connection available')"
python -c "import pymongo; print('MongoDB connection available')"
```

### Performance Optimization

#### 1. Data Processing
- Use appropriate batch sizes
- Enable parallel processing
- Configure memory limits
- Use streaming for large files

#### 2. Inference Optimization
- Batch requests when possible
- Use connection pooling
- Implement caching
- Monitor response times

#### 3. Database Optimization
- Index frequently queried columns
- Use connection pooling
- Optimize query patterns
- Regular maintenance tasks

## Next Steps

After successful setup:

1. **Data Preparation**: Gather your medical data sources
2. **Pipeline Configuration**: Customize ETL processes for your data
3. **Model Selection**: Choose appropriate LLMs for your use case
4. **Evaluation Setup**: Configure metrics and benchmarks
5. **Production Deployment**: Set up monitoring and scaling

## Support

For additional help:
- Check the [Architecture Documentation](ARCHITECTURE.md)
- Review component-specific READMEs
- Open an issue on GitHub
- Consult the troubleshooting section

## Validation Checklist

After setup, verify:
- [ ] ETL service responds to health checks
- [ ] Database connections work
- [ ] API keys are configured correctly
- [ ] Sample data processing completes
- [ ] Inference endpoints respond
- [ ] Evaluation notebooks run successfully
- [ ] All tests pass