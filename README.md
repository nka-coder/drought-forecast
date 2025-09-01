# Drought Forecast REST API Deployment Guide

## Overview
This API provides Drought forecasting capabilities for various sites in multiple countries. It uses historical Earth skin temperature (TS), rainfall and SPEI01 data, and machine learning models to predict drought risks based on current weather patterns.

## Prerequisites
- Docker installed on your system
- AWS CLI configured with appropriate permissions
- Access to the ECR repository containing the API image
- Store the dataset in an s3 bucket on your AWS account

## Deployment

### 1. Pull the Docker Image from AWS ECR
```bash
# Authenticate Docker with AWS ECR
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin 237710157910.dkr.ecr.eu-west-1.amazonaws.com

# Pull the image
docker pull 237710157910.dkr.ecr.eu-west-1.amazonaws.com/nestler/drought:latest
```

### 2. Environment Variables Configuration
Create a `.env` file with the following variables:

```ini
# S3 Configuration
S3_BUCKET=your-s3-bucket-name
S3_DATASET_PREFIX=dataset
AWS_REGION=your-aws-region

# Optional AWS credentials (only needed if not using IAM roles)
# AWS_ACCESS_KEY_ID=your-access-key
# AWS_SECRET_ACCESS_KEY=your-secret-key
```

### 3. Run the Container locally
```bash
docker run -d \
  --name drought \
  -p 8000:8000 \
  --env-file .env \
  237710157910.dkr.ecr.eu-west-1.amazonaws.com/nestler/drought:latest
```

### 4. Verify Deployment
Check the health endpoint:
```bash
curl http://localhost:8000/health
```

## API Endpoints

### 1. Root Endpoint
- **GET** `/`
- Returns a welcome message

### 2. Site List
- **GET** `/site-list`
- Returns a list of all available sites with their coordinates

### 4. Drought Forecast
- **POST** `/forecast/drought`
- Request Body:
  ```json
  {
    "country": "string",
    "sitename": "string",
    "spei_threshold": "float",
    "month_to_forecast": "int"
  }
  ```
- Returns drought forecast with model metrics

### 5. Health Check
- **GET** `/health`
- Returns service health status

## Usage Examples

### Get Site List
```bash
curl http://localhost:8000/site-list
```

### Get Drought Forecast
```bash
curl -X POST "http://localhost:8000/forecast/drought" \
  -H "Content-Type: application/json" \
  -d '{"country": "cameroon", "sitename": "douala", "spei_threshold": -1.0, "month_to_forcast": 10}'
```

## Deployment Options

### AWS ECS/Fargate
1. Create a task definition referencing your ECR image
2. Configure the environment variables in the task definition
3. Deploy as a service or run as a task

### AWS EC2 

### Your local Docker server

## Troubleshooting

1. **S3 Access Issues**:
   - Verify IAM permissions or AWS credentials
   - Check the S3 bucket name and region

2. **Dataset Not Loading**:
   - Check the S3_DATASET_PREFIX matches your S3 folder structure
   - Verify the health endpoint for dataset status

3. **CORS Issues**:
   - Ensure ALLOWED_ORIGINS includes your frontend domains
   - Separate multiple domains with commas (no spaces)

## Monitoring
The API includes a health endpoint (`/health`) that reports:
- Service status
- S3 connectivity
- Dataset availability

