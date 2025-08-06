from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from utils import create_dataset, create_model, get_monthly_weather
from datetime import datetime, timedelta

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import os
import boto3
from botocore.exceptions import ClientError
import logging
from tempfile import mkdtemp

app = FastAPI()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
#ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ROOT_DIR = os.getenv("ROOT_DIR", mkdtemp(dir="/tmp"))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset_drought/")
allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")

# S3 Configuration
S3_BUCKET = os.getenv("S3_BUCKET")
S3_DATASET_PREFIX = os.getenv("S3_DATASET_PREFIX")
AWS_REGION = os.getenv("AWS_REGION")

#  Initialize S3 client with IAM role fallback
try:
    # Try using IAM role first (recommended for Fargate)
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    
    # Verify we can access the bucket (will throw exception if not)
    #s3_client.head_bucket(Bucket=S3_BUCKET)
except ClientError as e:
    logger.warning(f"IAM role access failed: {e}. Falling back to explicit credentials.")
    try:
        s3_client = boto3.client(
            's3',
            region_name=AWS_REGION,
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
    except Exception as creds_error:
        logger.error("Failed to initialize S3 client with both IAM role and explicit credentials")
        raise RuntimeError("S3 client initialization failed") from creds_error


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    country: str
    sitename: str
    spei_threshold: float
    month_to_forcast: int

def download_dataset_from_s3():
    """Download dataset from S3 bucket to local directory"""
    try:
        # Create dataset directory if it doesn't exist
        os.makedirs(DATASET_DIR, exist_ok=True)
        
        # List objects in the S3 bucket
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_DATASET_PREFIX)
        
        if 'Contents' not in response:
            logger.error("No files found in S3 bucket")
            return False
            
        # Download each file
        for obj in response['Contents']:
            s3_key = obj['Key']
            local_path = os.path.join(DATASET_DIR, os.path.relpath(s3_key, S3_DATASET_PREFIX))
            
            # Create directory structure if needed
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Skip directories
            if s3_key.endswith('/'):
                continue
                
            logger.info(f"Downloading {s3_key} to {local_path}")
            s3_client.download_file(S3_BUCKET, s3_key, local_path)
            
        return True
    except ClientError as e:
        logger.error(f"Error downloading from S3: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

# Download dataset when starting up
def startup_event():
    """Initialize application state on startup"""
    logger.info("Application startup - initializing dataset")
    if not download_dataset_from_s3():
        logger.warning("Failed to download dataset from S3. Some functionality may not work.")
    print("Startup completed")

@app.get("/")
async def root():
    return {"greeting": "Drought forecast service"}

@app.get("/site-list", tags=["List of sites with their details"])
def get_country_city_coordinates():
    # Read the CSV file using pandas
    file = os.path.join(DATASET_DIR , "cities_coordinates.csv")
    df = pd.read_csv(file)  # Assuming the file is in a dataset folder
    
    # Convert the DataFrame to a list of dictionaries with the required structure
    result = []
    
    # Group by country and collect city information
    for country, group in df.groupby('country'):
        cities = []
        for _, row in group.iterrows():
            city_info = {
                'city': row['city'],
                'latitude': row['latitude'],
                'longitude': row['longitude']
            }
            cities.append(city_info)
        

        country_info = {
            'country': country,
            'cities': cities
        }
        result.append(country_info)
    
    return result


@app.post("/analyse/", tags=["Drought forecast of a site. To use the API, the following condition must be met:  -1.5 < spei_threshold <-0.5"])
def analyse(item: Item):
    if item.spei_threshold < -1.5 or item.spei_threshold > -0.5:
        input_data = {
            "sitename" : item.sitename,
            "spei_threshold" : item.spei_threshold,
            "month_to_forecast" : item.month_to_forcast
        }
        data = {
            "input_data": input_data,
            "warning": "spei_threshold is out of range. The following condition must be met:  -1.5 < spei_threshold <-0.5 "
        }
        return data

    m1 = (datetime.now().replace(day=1) - timedelta(days=1)).month
    m2 = item.month_to_forcast
    if m2 > m1:
        diff_month = m2 - m1
    if m2 < m1 :
        diff_month = m2 - (m1-12)
    if m2 == m1:
        diff_month = 1

    if diff_month <= 6 and diff_month > 1:
        x_train,x_test,y_train,y_test = create_dataset(item.country, item.sitename, item.spei_threshold, diff_month)

        m2_bin = str(bin(m2)[2:].zfill(4))
        pmonth_bit = []
        m1_bin = str(bin(m1)[2:].zfill(4))
        month_bit = []
        for i in range(4):
            pmonth_bit.append(m2_bin[i])
            month_bit.append(m1_bin[i])

        today = datetime.now()
        last_month = today.replace(day=1) - timedelta(days=1)
        year = last_month.year
        precipitation, temperature = get_monthly_weather(item.country, item.sitename, year, m1)
        data_in = { 
            "temperature": temperature,
            "precipitations" : precipitation, 
            "pmonth_bit_0" : pmonth_bit[0], 
            "pmonth_bit_1": pmonth_bit[1], 
            "pmonth_bit_2": pmonth_bit[2], 
            "pmonth_bit_3": pmonth_bit[3],
            "month_bit_0": month_bit[0], 
            "month_bit_1": month_bit[1], 
            "month_bit_2": month_bit[2], 
            "month_bit_3": month_bit[3]
        }
        df_in = pd.DataFrame([data_in])

        model, model_name, accuracy, precision, recall, specificity = create_model(x_train, y_train, x_test, y_test)
        y_pred = model.predict(df_in)

        #if int(y_pred[0])==2:
            #forcast = "extreme drought"

        if int(y_pred[0])==1:
            forcast = "drought"

        if int(y_pred[0])==0:
            forcast = "no drought"
        input_data = {
            "sitename" : item.sitename,
            "spei_threshold" : item.spei_threshold,
            "month_to_forecast" : item.month_to_forcast
        }
        data = {
            "forecast": forcast,
            "recall": recall,
            "specificity": specificity,
            "accuracy": accuracy,
            "precision": precision,
            "model_name": model_name,
            "input_data": input_data,
            "warning":""
        }
    else:
        input_data = {
            "sitename" : item.sitename,
            "spei_threshold" : item.spei_threshold,
            "month_to_forecast" : item.month_to_forcast
        }
        data = {
            "input_data": input_data,
            "warning": "The period interval to forecast is 1 to 6 months ahead"
        }

    return data

class HealthCheckResponse(BaseModel):
    status: str
    s3_accessible: bool
    dataset_ready: bool

@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """Health check: Dataset downlaoded."""
    s3_status = False
    dataset_status = False
    
    try:
        s3_client.head_bucket(Bucket=S3_BUCKET)
        s3_status = True
        dataset_status = os.path.exists(DATASET_DIR) and len(os.listdir(DATASET_DIR)) > 0
    except Exception as e:
        logger.warning(f"Health check failed: {e}")
    
    return {
        "status": "healthy" if s3_status and dataset_status else "degraded",
        "s3_accessible": s3_status,
        "dataset_ready": dataset_status
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)