# DataPlatformEtlService API Reference

## Overview

The DataPlatformEtlService provides a RESTful API for extracting, transforming, and loading medical data from various sources. This service supports multiple data formats, databases, and cloud storage systems, making it the central hub for data acquisition in the VM14K pipeline.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API uses API key authentication. Include your API key in the request headers:

```bash
Authorization: Bearer YOUR_API_KEY
```

## Content Type

All requests should include:
```
Content-Type: application/json
```

## API Documentation

Interactive API documentation is available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Core Endpoints

### Health Check

#### GET /health
Check service health and status.

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2024-01-15T10:30:00Z",
    "version": "1.0.0",
    "uptime": "2h 15m 30s"
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

### Data Extraction

#### POST /extract
Extract data from various sources.

**Request Body:**
```json
{
    "source_type": "string",
    "connection": {
        "host": "string",
        "port": "integer", 
        "database": "string",
        "username": "string",
        "password": "string"
    },
    "query": "string",
    "output_format": "string",
    "batch_size": "integer",
    "filters": {}
}
```

**Response:**
```json
{
    "job_id": "uuid",
    "status": "started",
    "estimated_rows": 1000,
    "download_url": "/download/uuid",
    "created_at": "2024-01-15T10:30:00Z"
}
```

**Example - PostgreSQL Extraction:**
```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "postgresql",
    "connection": {
      "host": "localhost",
      "port": 5432,
      "database": "medical_data",
      "username": "user",
      "password": "password"
    },
    "query": "SELECT * FROM vietnamese_medical_questions WHERE category = '\''cardiology'\''",
    "output_format": "csv",
    "batch_size": 1000
  }'
```

**Example - MongoDB Extraction:**
```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "mongodb",
    "connection": {
      "uri": "mongodb://localhost:27017/medical_db"
    },
    "collection": "medical_questions",
    "filters": {"language": "vietnamese", "verified": true},
    "output_format": "jsonl",
    "batch_size": 500
  }'
```

### Data Transformation

#### POST /transform
Transform extracted data using DuckDB or Spark.

**Request Body:**
```json
{
    "input_source": "string",
    "transformation_type": "duckdb|spark",
    "sql_query": "string",
    "output_format": "string",
    "spark_config": {}
}
```

**Response:**
```json
{
    "job_id": "uuid",
    "status": "processing",
    "input_rows": 5000,
    "estimated_completion": "2024-01-15T10:45:00Z",
    "progress": 0.25
}
```

**Example - DuckDB Transformation:**
```bash
curl -X POST http://localhost:8000/transform \
  -H "Content-Type: application/json" \
  -d '{
    "input_source": "s3://bucket/raw-data.csv",
    "transformation_type": "duckdb",
    "sql_query": "SELECT question, optionA, optionB, optionC, optionD, correct_answer FROM input WHERE question IS NOT NULL",
    "output_format": "parquet"
  }'
```

**Example - Spark Transformation:**
```bash
curl -X POST http://localhost:8000/transform \
  -H "Content-Type: application/json" \
  -d '{
    "input_source": "s3://bucket/large-dataset/",
    "transformation_type": "spark",
    "sql_query": "SELECT *, CASE WHEN difficulty > 5 THEN '\''hard'\'' ELSE '\''easy'\'' END as difficulty_level FROM input",
    "spark_config": {
      "driver_memory": "4g",
      "executor_memory": "2g",
      "num_executors": 4
    },
    "output_format": "delta"
  }'
```

### Data Loading

#### POST /load
Load transformed data to destination systems.

**Request Body:**
```json
{
    "source_data": "string",
    "destination_type": "string",
    "connection": {},
    "table_name": "string",
    "write_mode": "append|overwrite|upsert",
    "partition_columns": ["string"]
}
```

**Response:**
```json
{
    "job_id": "uuid",
    "status": "loading",
    "rows_loaded": 0,
    "total_rows": 1000,
    "destination": "postgresql://localhost/medical_data/processed_questions"
}
```

**Example - Load to ClickHouse:**
```bash
curl -X POST http://localhost:8000/load \
  -H "Content-Type: application/json" \
  -d '{
    "source_data": "/tmp/processed_data.parquet",
    "destination_type": "clickhouse",
    "connection": {
      "host": "localhost",
      "port": 9000,
      "database": "medical_analytics"
    },
    "table_name": "vietnamese_medical_qa",
    "write_mode": "append",
    "partition_columns": ["category", "difficulty"]
  }'
```

**Example - Load to HuggingFace:**
```bash
curl -X POST http://localhost:8000/load \
  -H "Content-Type: application/json" \
  -d '{
    "source_data": "/tmp/final_dataset.jsonl",
    "destination_type": "huggingface",
    "connection": {
      "token": "hf_xxx",
      "organization": "venera-ai"
    },
    "dataset_name": "vietnamese-medical-benchmark-v2",
    "write_mode": "overwrite"
  }'
```

### Pipeline Management

#### POST /pipeline
Create and execute complete ETL pipelines.

**Request Body:**
```json
{
    "name": "string",
    "description": "string",
    "steps": [
        {
            "type": "extract|transform|load",
            "config": {},
            "depends_on": ["step_id"]
        }
    ],
    "schedule": "cron_expression",
    "notifications": {}
}
```

**Response:**
```json
{
    "pipeline_id": "uuid",
    "name": "Medical Data Pipeline",
    "status": "created",
    "steps": 3,
    "next_run": "2024-01-15T12:00:00Z"
}
```

**Example - Complete Pipeline:**
```bash
curl -X POST http://localhost:8000/pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Daily Medical Data Processing",
    "description": "Extract, clean, and load Vietnamese medical questions",
    "steps": [
      {
        "id": "extract_step",
        "type": "extract",
        "config": {
          "source_type": "postgresql",
          "connection": {
            "host": "source-db.com",
            "port": 5432,
            "database": "medical_raw"
          },
          "query": "SELECT * FROM new_questions WHERE created_date = CURRENT_DATE"
        }
      },
      {
        "id": "transform_step",
        "type": "transform",
        "config": {
          "transformation_type": "duckdb",
          "sql_query": "SELECT question, options, correct_answer FROM input WHERE question IS NOT NULL AND LENGTH(question) > 10"
        },
        "depends_on": ["extract_step"]
      },
      {
        "id": "load_step",
        "type": "load",
        "config": {
          "destination_type": "clickhouse",
          "table_name": "processed_questions",
          "write_mode": "append"
        },
        "depends_on": ["transform_step"]
      }
    ],
    "schedule": "0 2 * * *",
    "notifications": {
      "email": ["admin@example.com"],
      "webhook": "https://api.slack.com/webhooks/xxx"
    }
  }'
```

#### GET /pipeline/{pipeline_id}
Get pipeline details and status.

**Response:**
```json
{
    "pipeline_id": "uuid",
    "name": "Daily Medical Data Processing",
    "status": "running",
    "created_at": "2024-01-15T10:00:00Z",
    "last_run": "2024-01-15T02:00:00Z",
    "next_run": "2024-01-16T02:00:00Z",
    "steps": [
        {
            "id": "extract_step",
            "status": "completed",
            "started_at": "2024-01-15T02:00:00Z",
            "completed_at": "2024-01-15T02:05:00Z",
            "rows_processed": 1500
        }
    ],
    "metrics": {
        "total_runs": 45,
        "success_rate": 0.98,
        "avg_duration": "15m 30s"
    }
}
```

#### DELETE /pipeline/{pipeline_id}
Delete a pipeline.

**Response:**
```json
{
    "message": "Pipeline deleted successfully",
    "pipeline_id": "uuid"
}
```

### Job Management

#### GET /jobs
List all jobs with optional filtering.

**Query Parameters:**
- `status`: Filter by job status (pending, running, completed, failed)
- `type`: Filter by job type (extract, transform, load)
- `limit`: Number of results (default: 50)
- `offset`: Pagination offset (default: 0)

**Response:**
```json
{
    "jobs": [
        {
            "job_id": "uuid",
            "type": "extract",
            "status": "completed",
            "created_at": "2024-01-15T10:00:00Z",
            "completed_at": "2024-01-15T10:05:00Z",
            "source": "postgresql://medical_db",
            "rows_processed": 1000
        }
    ],
    "total": 150,
    "limit": 50,
    "offset": 0
}
```

**Example:**
```bash
curl "http://localhost:8000/jobs?status=running&type=extract&limit=10"
```

#### GET /jobs/{job_id}
Get detailed job status and logs.

**Response:**
```json
{
    "job_id": "uuid",
    "type": "extract",
    "status": "running",
    "progress": 0.75,
    "created_at": "2024-01-15T10:00:00Z",
    "started_at": "2024-01-15T10:01:00Z",
    "estimated_completion": "2024-01-15T10:15:00Z",
    "config": {
        "source_type": "postgresql",
        "query": "SELECT * FROM medical_questions"
    },
    "metrics": {
        "rows_processed": 7500,
        "estimated_total_rows": 10000,
        "processing_rate": "500 rows/sec"
    },
    "logs": [
        {
            "timestamp": "2024-01-15T10:01:00Z",
            "level": "INFO",
            "message": "Started database connection"
        },
        {
            "timestamp": "2024-01-15T10:01:05Z",
            "level": "INFO", 
            "message": "Query execution started"
        }
    ]
}
```

#### POST /jobs/{job_id}/cancel
Cancel a running job.

**Response:**
```json
{
    "message": "Job cancellation requested",
    "job_id": "uuid",
    "status": "cancelling"
}
```

### Data Sources Configuration

#### GET /sources
List configured data sources.

**Response:**
```json
{
    "sources": [
        {
            "id": "medical_db_prod",
            "type": "postgresql",
            "name": "Production Medical Database",
            "connection": {
                "host": "prod-db.example.com",
                "port": 5432,
                "database": "medical_data"
            },
            "status": "active",
            "last_tested": "2024-01-15T09:00:00Z"
        }
    ]
}
```

#### POST /sources
Add a new data source configuration.

**Request Body:**
```json
{
    "name": "string",
    "type": "postgresql|mongodb|s3|clickhouse",
    "connection": {},
    "description": "string",
    "tags": ["string"]
}
```

**Response:**
```json
{
    "source_id": "uuid",
    "name": "Test Medical Database",
    "status": "created",
    "test_connection": "successful"
}
```

#### POST /sources/{source_id}/test
Test connection to a data source.

**Response:**
```json
{
    "source_id": "uuid",
    "status": "success",
    "message": "Connection successful",
    "latency_ms": 45,
    "tested_at": "2024-01-15T10:30:00Z"
}
```

### File Operations

#### POST /upload
Upload files for processing.

**Form Data:**
- `file`: File to upload
- `format`: File format (csv, json, parquet)
- `description`: Optional description

**Response:**
```json
{
    "file_id": "uuid",
    "filename": "medical_questions.csv",
    "size_bytes": 1048576,
    "format": "csv",
    "upload_url": "/files/uuid",
    "status": "uploaded"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@medical_data.csv" \
  -F "format=csv" \
  -F "description=Vietnamese medical questions batch 1"
```

#### GET /download/{job_id}
Download processed data from a completed job.

**Response:**
Binary data with appropriate headers:
```
Content-Type: application/octet-stream
Content-Disposition: attachment; filename="processed_data.csv"
```

**Example:**
```bash
curl -O "http://localhost:8000/download/uuid"
```

### Monitoring and Metrics

#### GET /metrics
Get system metrics and performance data.

**Response:**
```json
{
    "system": {
        "cpu_usage": 0.45,
        "memory_usage": 0.67,
        "disk_usage": 0.23,
        "uptime": "2h 15m 30s"
    },
    "jobs": {
        "total_jobs": 1250,
        "completed_jobs": 1200,
        "failed_jobs": 25,
        "running_jobs": 3,
        "success_rate": 0.98
    },
    "data": {
        "total_rows_processed": 5000000,
        "avg_processing_rate": "1000 rows/sec",
        "data_sources_active": 8
    }
}
```

#### GET /logs
Get system logs with filtering options.

**Query Parameters:**
- `level`: Log level (DEBUG, INFO, WARNING, ERROR)
- `start_time`: Start timestamp (ISO 8601)
- `end_time`: End timestamp (ISO 8601)
- `limit`: Number of results
- `component`: Filter by component name

**Response:**
```json
{
    "logs": [
        {
            "timestamp": "2024-01-15T10:30:00Z",
            "level": "INFO",
            "component": "extractor",
            "message": "PostgreSQL extraction completed successfully",
            "job_id": "uuid",
            "metadata": {
                "rows_extracted": 1000,
                "duration_ms": 5000
            }
        }
    ],
    "total": 500,
    "filtered": 50
}
```

## Data Source Types

### PostgreSQL
```json
{
    "source_type": "postgresql",
    "connection": {
        "host": "localhost",
        "port": 5432,
        "database": "medical_data",
        "username": "user",
        "password": "password",
        "ssl_mode": "require"
    }
}
```

### MongoDB
```json
{
    "source_type": "mongodb",
    "connection": {
        "uri": "mongodb://username:password@host:port/database",
        "ssl": true,
        "auth_source": "admin"
    }
}
```

### AWS S3
```json
{
    "source_type": "s3",
    "connection": {
        "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
        "aws_secret_access_key": "secret",
        "region": "us-east-1",
        "bucket": "medical-data-bucket"
    }
}
```

### ClickHouse
```json
{
    "source_type": "clickhouse",
    "connection": {
        "host": "localhost",
        "port": 9000,
        "database": "medical_analytics",
        "username": "user",
        "password": "password",
        "secure": true
    }
}
```

### SQL Server
```json
{
    "source_type": "sqlserver",
    "connection": {
        "server": "localhost",
        "port": 1433,
        "database": "MedicalData",
        "username": "user",
        "password": "password",
        "driver": "ODBC Driver 17 for SQL Server"
    }
}
```

## Response Formats

### Output Formats
- `csv`: Comma-separated values
- `json`: JSON format
- `jsonl`: JSON Lines (newline-delimited JSON)
- `parquet`: Apache Parquet format
- `excel`: Microsoft Excel format
- `delta`: Delta Lake format

### Error Responses

#### 400 Bad Request
```json
{
    "error": "Bad Request",
    "message": "Invalid source type specified",
    "details": {
        "field": "source_type",
        "allowed_values": ["postgresql", "mongodb", "s3"]
    }
}
```

#### 401 Unauthorized
```json
{
    "error": "Unauthorized",
    "message": "Invalid or missing API key"
}
```

#### 404 Not Found
```json
{
    "error": "Not Found",
    "message": "Job not found",
    "job_id": "uuid"
}
```

#### 500 Internal Server Error
```json
{
    "error": "Internal Server Error",
    "message": "Database connection failed",
    "job_id": "uuid",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

## Rate Limits

- **Standard endpoints**: 100 requests per minute
- **Upload endpoints**: 10 requests per minute
- **Pipeline operations**: 50 requests per minute

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248600
```

## WebSocket Endpoints

### Real-time Job Updates
Connect to receive real-time job status updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/jobs/{job_id}');

ws.onmessage = function(event) {
    const update = JSON.parse(event.data);
    console.log('Job update:', update);
};
```

**Message Format:**
```json
{
    "job_id": "uuid",
    "status": "running",
    "progress": 0.45,
    "rows_processed": 4500,
    "timestamp": "2024-01-15T10:30:00Z"
}
```

## SDK Examples

### Python SDK Usage
```python
import requests
from datetime import datetime

class ETLClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def extract_data(self, config):
        response = requests.post(
            f'{self.base_url}/extract',
            json=config,
            headers=self.headers
        )
        return response.json()
    
    def get_job_status(self, job_id):
        response = requests.get(
            f'{self.base_url}/jobs/{job_id}',
            headers=self.headers
        )
        return response.json()

# Usage
client = ETLClient('http://localhost:8000', 'your-api-key')

# Extract medical data
config = {
    'source_type': 'postgresql',
    'connection': {
        'host': 'localhost',
        'database': 'medical_data'
    },
    'query': 'SELECT * FROM vietnamese_questions',
    'output_format': 'csv'
}

job = client.extract_data(config)
print(f"Job started: {job['job_id']}")

# Monitor progress
import time
while True:
    status = client.get_job_status(job['job_id'])
    if status['status'] in ['completed', 'failed']:
        break
    print(f"Progress: {status['progress'] * 100:.1f}%")
    time.sleep(5)
```

### JavaScript SDK Usage
```javascript
class ETLClient {
    constructor(baseUrl, apiKey) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json'
        };
    }
    
    async extractData(config) {
        const response = await fetch(`${this.baseUrl}/extract`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(config)
        });
        return await response.json();
    }
    
    async getJobStatus(jobId) {
        const response = await fetch(`${this.baseUrl}/jobs/${jobId}`, {
            headers: this.headers
        });
        return await response.json();
    }
}

// Usage
const client = new ETLClient('http://localhost:8000', 'your-api-key');

const config = {
    source_type: 'mongodb',
    connection: {
        uri: 'mongodb://localhost:27017/medical_db'
    },
    collection: 'questions',
    output_format: 'jsonl'
};

client.extractData(config)
    .then(job => {
        console.log(`Job started: ${job.job_id}`);
        return client.getJobStatus(job.job_id);
    })
    .then(status => {
        console.log(`Status: ${status.status}`);
    });
```

## Best Practices

### Performance Optimization
1. **Batch Size**: Use appropriate batch sizes (1000-5000 rows)
2. **Parallel Processing**: Run multiple extraction jobs in parallel
3. **Connection Pooling**: Reuse database connections when possible
4. **Compression**: Use compressed formats (parquet) for large datasets

### Error Handling
1. **Retry Logic**: Implement exponential backoff for failed requests
2. **Monitoring**: Set up alerts for job failures
3. **Validation**: Validate data sources before running large jobs
4. **Timeouts**: Set appropriate request timeouts

### Security
1. **API Keys**: Rotate API keys regularly
2. **Connections**: Use SSL/TLS for database connections
3. **Access Control**: Limit access to sensitive data sources
4. **Logging**: Monitor and audit API usage

### Cost Management
1. **Resource Limits**: Set appropriate limits for Spark jobs
2. **Scheduling**: Schedule heavy jobs during off-peak hours
3. **Monitoring**: Track data transfer and processing costs
4. **Cleanup**: Clean up temporary files and unused resources

## Support

For API support:
- **Documentation**: Check this reference and OpenAPI specs
- **Issues**: Report bugs on GitHub
- **Community**: Join discussions for best practices
- **Updates**: Monitor for API version updates and changes