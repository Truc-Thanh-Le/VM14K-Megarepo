# VM14K-Megarepo Architecture Documentation

## Overview

The VM14K-Megarepo is a comprehensive medical data processing and LLM benchmarking pipeline designed to create, clean, deduplicate, and evaluate Vietnamese medical datasets. This document outlines the system architecture, data flow, and component interactions.

## System Architecture

```mermaid
graph TB
    subgraph "Data Acquisition Layer"
        A[DataPlatformEtlService]
        A1[Web Scrapers]
        A2[API Connectors]
        A3[Database Extractors]
    end
    
    subgraph "Data Processing Layer"
        B[DataCleaning]
        B1[LLM-based Cleaning]
        B2[Format Standardization]
        B3[HTML Tag Removal]
        
        C[Deduplication]
        C1[Exact Match Removal]
        C2[Similarity Detection]
        C3[Off-topic Filtering]
    end
    
    subgraph "Inference Layer"
        D[APIServices]
        D1[OpenAI/GPT Models]
        D2[Azure OpenAI]
        D3[AWS Bedrock]
        D4[Google Gemini]
        D5[Groq]
        D6[DeepSeek]
        
        E[SelfHost]
        E1[vLLM Server]
        E2[Local Models]
        E3[Custom Endpoints]
    end
    
    subgraph "Evaluation Layer"
        F[Evaluation]
        F1[Benchmark Analysis]
        F2[Pass@k Metrics]
        F3[Ensemble Evaluation]
        F4[Performance Visualization]
    end
    
    subgraph "Data Storage"
        G[Raw Data]
        H[Cleaned Data]
        I[Deduplicated Data]
        J[Inference Results]
        K[Evaluation Metrics]
    end
    
    A --> A1
    A --> A2
    A --> A3
    A1 --> G
    A2 --> G
    A3 --> G
    
    G --> B
    B --> B1
    B --> B2
    B --> B3
    B --> H
    
    H --> C
    C --> C1
    C --> C2
    C --> C3
    C --> I
    
    I --> D
    I --> E
    D --> D1
    D --> D2
    D --> D3
    D --> D4
    D --> D5
    D --> D6
    E --> E1
    E --> E2
    E --> E3
    
    D --> J
    E --> J
    
    J --> F
    F --> F1
    F --> F2
    F --> F3
    F --> F4
    F --> K
```

## Component Details

### 1. DataPlatformEtlService
**Purpose**: Medical data acquisition and initial processing

**Architecture**:
- **FastAPI Backend**: RESTful API for pipeline management
- **ETL Engine**: Multi-source extraction with DuckDB transformations
- **Spark Integration**: Scalable data processing for large datasets
- **Multi-destination Loading**: Support for various output formats

**Key Features**:
- MongoDB, PostgreSQL, SQL Server connectors
- AWS and GCP cloud integration
- Dockerized deployment
- Configuration-driven ETL processes

### 2. DataCleaning
**Purpose**: LLM-powered data cleaning and standardization

**Components**:
- `classification.py`: Main cleaning script using LLM APIs
- `prompts.py`: Prompt templates for different cleaning tasks
- `final_transform.ipynb`: Data shuffling and preparation

**Cleaning Operations**:
- HTML tag removal
- Index marker elimination
- Question format standardization
- Answer option normalization

### 3. Deduplication
**Purpose**: Remove duplicate and irrelevant questions

**Three-tier Approach**:
- **Level 1**: Exact match removal after text normalization
- **Level 2**: Near-duplicate detection (>0.9 edit distance similarity)
- **Level 3**: Off-topic question filtering

**Input Requirements**:
- CSV format with standardized columns (question, optionA-G)
- UTF-8 encoding support for Vietnamese text

### 4. Inference Framework
**Purpose**: Multi-provider LLM inference for benchmarking

#### APIServices
**Supported Providers**:
- OpenAI (GPT family)
- Azure OpenAI
- AWS Bedrock
- Google Gemini
- Groq
- DeepSeek R1

**Features**:
- Batch processing capabilities
- Cost tracking and optimization
- Parallel inference execution
- Error handling and retry mechanisms

#### SelfHost
**Components**:
- vLLM server integration
- Local model deployment
- Custom endpoint configuration
- Resource management

### 5. Evaluation Suite
**Purpose**: Comprehensive model performance assessment

**Metrics**:
- **Pass@k**: Success rate at different k values
- **F1 Score**: Precision and recall balance
- **Ensemble Metrics**: Combined model performance
- **Statistical Analysis**: Confidence intervals and significance tests

## Data Flow Pipeline

### 1. Data Acquisition
```
Raw Medical Sources → ETL Service → Structured Data
```

### 2. Data Processing
```
Structured Data → LLM Cleaning → Format Standardization → Clean Dataset
```

### 3. Deduplication
```
Clean Dataset → Exact Dedup → Similarity Dedup → Topic Filter → Final Dataset
```

### 4. Inference
```
Final Dataset → Model APIs/Local → Inference Results → Aggregation
```

### 5. Evaluation
```
Inference Results → Metric Calculation → Analysis → Visualization → Reports
```

## Technology Stack

### Backend Services
- **FastAPI**: REST API framework
- **DuckDB**: In-memory analytics database
- **Apache Spark**: Distributed data processing
- **PostgreSQL**: Persistent data storage

### Data Processing
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning utilities
- **NLTK/spaCy**: Text processing

### LLM Integration
- **OpenAI SDK**: GPT model access
- **Google AI**: Gemini integration
- **AWS Boto3**: Bedrock services
- **vLLM**: Self-hosted inference

### Infrastructure
- **Docker**: Containerization
- **UV**: Python package management
- **Git**: Version control
- **GitHub Actions**: CI/CD (planned)

## Scalability Considerations

### Horizontal Scaling
- Microservice architecture enables independent scaling
- API-based inference supports load balancing
- Batch processing for large datasets

### Performance Optimization
- DuckDB for fast analytics queries
- Spark for distributed processing
- Async processing in FastAPI
- Connection pooling for databases

### Resource Management
- Memory-efficient data processing
- Streaming for large files
- Configurable batch sizes
- GPU optimization for inference

## Security and Compliance

### API Security
- API key management for external services
- Rate limiting and quotas
- Request validation and sanitization

### Data Privacy
- Medical data anonymization
- Secure credential storage
- Audit logging capabilities

### Compliance
- GDPR considerations for data processing
- Medical data handling protocols
- Open source licensing (Apache 2.0)

## Monitoring and Observability

### Logging
- Structured logging across all components
- Error tracking and alerting
- Performance metrics collection

### Metrics
- ETL pipeline performance
- Model inference latency
- Cost tracking per provider
- Data quality metrics

### Debugging
- Detailed error messages
- Debug mode configurations
- Step-by-step pipeline tracing

## Extension Points

### Adding New Data Sources
1. Implement extractor interface
2. Configure connection parameters
3. Define data transformation rules
4. Update pipeline configuration

### Integrating New LLM Providers
1. Implement provider interface
2. Add authentication handling
3. Configure rate limiting
4. Update batch processing logic

### Custom Evaluation Metrics
1. Extend evaluation framework
2. Implement metric calculation
3. Add visualization support
4. Update reporting pipeline

## Future Enhancements

### Planned Features
- Real-time data streaming
- Advanced deduplication algorithms
- Multi-language support
- Automated model selection
- Enhanced visualization dashboard

### Research Directions
- Federated learning integration
- Privacy-preserving techniques
- Automated prompt optimization
- Domain-specific model fine-tuning