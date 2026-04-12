"""
MLflow Configuration and Setup Guide for Diabetes Project

This guide helps you properly configure MLflow for your diabetes prediction project.
"""

import os
import json
from pathlib import Path


class MLflowConfig:
    """MLflow configuration for diabetes prediction project"""
    
    # Experiment names
    EXPERIMENT_NAME = "diabetes-prediction"
    TRACKING_URI = "file:./mlruns"  # Local tracking
    # TRACKING_URI = "http://localhost:5000"  # Remote server
    
    # Registry configuration
    REGISTRY_URI = "file:./mlruns"
    # REGISTRY_URI = "http://localhost:5000"
    
    # Artifact store
    ARTIFACT_STORE = "./mlruns"
    
    # S3 Configuration (optional)
    # Uncomment and fill in to use S3
    # S3_ENDPOINT_URL = "https://s3.us-east-1.amazonaws.com"
    # S3_BUCKET = "diabetes-ml-artifacts"
    # AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    # AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    @staticmethod
    def setup_local_mlflow():
        """Setup local MLflow configuration"""
        os.environ['MLFLOW_TRACKING_URI'] = MLflowConfig.TRACKING_URI
        os.environ['MLFLOW_REGISTRY_STORE'] = MLflowConfig.REGISTRY_URI
        
        # Create directories if needed
        Path(MLflowConfig.ARTIFACT_STORE).mkdir(parents=True, exist_ok=True)
        
        print("✓ MLflow local configuration set up")
    
    @staticmethod
    def setup_remote_mlflow(server_url="http://localhost:5000"):
        """Setup remote MLflow server configuration"""
        os.environ['MLFLOW_TRACKING_URI'] = server_url
        os.environ['MLFLOW_REGISTRY_STORE'] = server_url
        
        print(f"✓ MLflow remote configuration set to {server_url}")
    
    @staticmethod
    def setup_s3_backend(bucket_name, region='us-east-1'):
        """Setup S3 as artifact and registry backend"""
        os.environ['MLFLOW_TRACKING_URI'] = "file:./mlruns"
        os.environ['AWS_DEFAULT_REGION'] = region
        
        print(f"✓ MLflow S3 backend configured for bucket: {bucket_name}")
    
    @staticmethod
    def get_config_dict():
        """Get current configuration as dictionary"""
        return {
            'experiment_name': MLflowConfig.EXPERIMENT_NAME,
            'tracking_uri': os.getenv('MLFLOW_TRACKING_URI', MLflowConfig.TRACKING_URI),
            'registry_uri': os.getenv('MLFLOW_REGISTRY_STORE', MLflowConfig.REGISTRY_URI),
            'artifact_store': MLflowConfig.ARTIFACT_STORE
        }
    
    @staticmethod
    def save_config(output_path='config/mlflow_config.json'):
        """Save configuration to file"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        config = MLflowConfig.get_config_dict()
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ MLflow configuration saved to {output_path}")


# ============================================================================
# SETUP INSTRUCTIONS
# ============================================================================

"""
1. LOCAL SETUP (Development):
   
   In your Python script or notebook:
   ```python
   from config.mlflow_config import MLflowConfig
   MLflowConfig.setup_local_mlflow()
   
   import mlflow
   mlflow.set_experiment("diabetes-prediction")
   mlflow.start_run()
   # Your training code here
   ```
   
   To view results:
   ```bash
   mlflow ui
   ```
   Then open http://localhost:5000 in your browser.


2. REMOTE SERVER SETUP (Production):
   
   Start MLflow server:
   ```bash
   mlflow server --host 0.0.0.0 --port 5000 \
     --backend-store-uri postgresql://user:pass@localhost/mlflow \
     --default-artifact-root s3://bucket-name/mlflow
   ```
   
   In your Python script:
   ```python
   from config.mlflow_config import MLflowConfig
   MLflowConfig.setup_remote_mlflow("http://your-server:5000")
   ```


3. AWS S3 SETUP (Recommended for Production):
   
   a. Create S3 bucket:
   ```bash
   aws s3 mb s3://diabetes-ml-artifacts --region us-east-1
   ```
   
   b. Set AWS credentials:
   ```bash
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   ```
   
   c. In your Python script:
   ```python
   from config.mlflow_config import MLflowConfig
   MLflowConfig.setup_s3_backend("diabetes-ml-artifacts", "us-east-1")
   ```


4. DOCKER SETUP:
   
   Start MLflow with Docker:
   ```bash
   docker run -p 5000:5000 -v $(pwd)/mlruns:/mlflow \
     ghcr.io/mlflow/mlflow:latest mlflow server --host 0.0.0.0
   ```


5. COMMON MLFLOW COMMANDS:
   
   # View experiments and runs
   mlflow ui
   
   # Check registered models
   mlflow models list
   
   # Serve a model
   mlflow models serve -m models:/diabetes-random_forest/latest --port 5001
   
   # Export runs
   mlflow export-runs -e diabetes-prediction -o backup/
   
   # Clean up old runs
   mlflow gc --backend-store-uri mlruns/
"""


if __name__ == "__main__":
    # Setup and save configuration
    MLflowConfig.setup_local_mlflow()
    MLflowConfig.save_config()
    
    # Display current config
    config = MLflowConfig.get_config_dict()
    print("\nCurrent MLflow Configuration:")
    print(json.dumps(config, indent=2))
