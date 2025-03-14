import boto3
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Set up constants
BUCKET_NAME = 'sagemaker-eu-west-1-688567281415'

def list_s3_contents():
    """List the contents of the S3 bucket to find the PCA model"""
    s3_client = boto3.client('s3')
    
    # First try to find output directories of remote functions
    logging.info(f"Looking for remote function outputs in s3://{BUCKET_NAME}/")
    response = s3_client.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix='run-pca-remote-'
    )
    
    if 'Contents' in response:
        logging.info(f"Found {len(response['Contents'])} objects with prefix 'run-pca-remote-'")
        
        # Get the most recent remote function run
        remote_runs = {}
        for obj in response['Contents']:
            key = obj['Key']
            parts = key.split('/')
            if len(parts) > 0:
                remote_run = parts[0]
                if remote_run not in remote_runs:
                    remote_runs[remote_run] = {'last_modified': obj['LastModified'], 'keys': []}
                remote_runs[remote_run]['keys'].append(key)
        
        # Sort remote runs by last modified date
        sorted_runs = sorted(remote_runs.items(), key=lambda x: x[1]['last_modified'], reverse=True)
        
        if sorted_runs:
            latest_run, run_data = sorted_runs[0]
            logging.info(f"Latest remote function run: {latest_run}")
            
            # Look for model.tar.gz files in this run
            model_files = [k for k in run_data['keys'] if k.endswith('model.tar.gz')]
            if model_files:
                for model_file in model_files:
                    logging.info(f"Found model file: {model_file}")
                    # Check if the file exists
                    try:
                        s3_client.head_object(Bucket=BUCKET_NAME, Key=model_file)
                        logging.info(f"✅ CONFIRMED: This model file exists and is accessible")
                        logging.info(f"Full S3 URI: s3://{BUCKET_NAME}/{model_file}")
                        return model_file
                    except Exception as e:
                        logging.error(f"❌ ERROR: This model file cannot be accessed: {str(e)}")
    
    # If we didn't find anything with run-pca-remote, try looking in pca_output
    logging.info("\nLooking for model in pca_output directory...")
    response = s3_client.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix='pca_output'
    )
    
    if 'Contents' in response:
        logging.info(f"Found {len(response['Contents'])} objects with prefix 'pca_output'")
        model_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('model.tar.gz')]
        
        if model_files:
            for model_file in model_files:
                logging.info(f"Found model file: {model_file}")
                # Check if the file exists
                try:
                    s3_client.head_object(Bucket=BUCKET_NAME, Key=model_file)
                    logging.info(f"✅ CONFIRMED: This model file exists and is accessible")
                    logging.info(f"Full S3 URI: s3://{BUCKET_NAME}/{model_file}")
                    return model_file
                except Exception as e:
                    logging.error(f"❌ ERROR: This model file cannot be accessed: {str(e)}")
    
    # Last resort: list all model.tar.gz files in the bucket
    logging.info("\nSearching entire bucket for model.tar.gz files...")
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=BUCKET_NAME)
    
    found_models = []
    for page in page_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if key.endswith('model.tar.gz'):
                    found_models.append((key, obj['LastModified']))
    
    if found_models:
        # Sort by last modified
        found_models.sort(key=lambda x: x[1], reverse=True)
        logging.info(f"Found {len(found_models)} model.tar.gz files in the bucket")
        
        for model_file, last_modified in found_models[:5]:  # Show top 5
            logging.info(f"Model: {model_file}, Last Modified: {last_modified}")
            # Check if the file exists
            try:
                s3_client.head_object(Bucket=BUCKET_NAME, Key=model_file)
                logging.info(f"✅ CONFIRMED: This model file exists and is accessible")
                logging.info(f"Full S3 URI: s3://{BUCKET_NAME}/{model_file}")
                return model_file
            except Exception as e:
                logging.error(f"❌ ERROR: This model file cannot be accessed: {str(e)}")
    
    logging.error("No accessible model.tar.gz files found in the bucket")
    return None

if __name__ == "__main__":
    model_path = list_s3_contents()
    if model_path:
        print("\n" + "="*80)
        print(f"Found model at: s3://{BUCKET_NAME}/{model_path}")
        print("="*80)
        print("\nTo use this model in your interpret_pca_results.py script, update MODEL_KEY to:")
        print(f"MODEL_KEY = '{model_path}'")
    else:
        print("\n" + "="*80)
        print("No accessible model.tar.gz files found in the bucket.")
        print("="*80) 