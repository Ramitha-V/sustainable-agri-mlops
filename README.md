# MLOps Pipeline for Crop Yield Prediction

A DVC and Azure-based MLOps pipeline to train and deploy a crop yield prediction model.

## Tech Stack

| Category | Tools / Services |
|-----------|------------------|
| Cloud | Microsoft Azure (App Service, ACR, Blob Storage) |
| CI/CD | GitHub Actions |
| Versioning | DVC, Git |
| Containerization | Docker |
| Backend | FastAPI, Uvicorn |
| Machine Learning | Scikit-learn |

---

## 1. Local Setup

### Clone the Repository
```
git clone https://github.com/ramitha-v/sustainable-agri-mlops.git
cd sustainable-agri-mlops
```

### Create and Activate Virtual Environment
```
python -m venv venv
source venv/Scripts/activate  # On Windows
# OR
source venv/bin/activate      # On macOS/Linux
```

### Install Dependencies
```
pip install -r requirements.txt
```

### DVC Commands
 ```
# Configure Azure remote storage (set AZURE_STORAGE_CONNECTION_STRING first)
export AZURE_STORAGE_CONNECTION_STRING="your_connection_string"

# Pull data and model from DVC remote
dvc pull

# Reproduce the full ML pipeline if code/data changes
dvc repro
```

## 3. Local Testing

### A) Via Uvicorn Server

```
# Start the API server
uvicorn main:app --reload

```

### B) Via Docker Container

```
# Build the Docker image (set AZURE_STORAGE_CONNECTION_STRING first)
docker build . -t agri-predictor-local --build-arg AZURE_STORAGE_CONNECTION_STRING

# Run the container
docker run -p 8000:8000 agri-predictor-local

```

## 4. CI/CD Pipeline

The workflow defined in .github/workflows/ci-cd.yml automates build, train, and deploy stages.

Required GitHub Secrets

- AZURE_CREDENTIALS
- AZURE_STORAGE_CONNECTION_STRING
- ACR_LOGIN_SERVER
- ACR_USERNAME
- ACR_PASSWORD
- WEB_APP_NAME

## 5. Azure Infrastructure Setup (Azure CLI)

```
# Set environment variables
export RESOURCE_GROUP="SustainableAgriRG"
export LOCATION="southeastasia"
export ACR_NAME="agriacrdvc0405"
export APP_SERVICE_PLAN="agri-yield-plan"
export WEB_APP_NAME="agri-yield-app-0405"

# Create Azure Container Registry (ACR)
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic --admin-enabled true

# Create App Service Plan (B1 SKU is required for Docker)
az appservice plan create --name $APP_SERVICE_PLAN --resource-group $RESOURCE_GROUP --sku B1 --is-linux --location $LOCATION

# Create Web App
az webapp create --resource-group $RESOURCE_GROUP --plan $APP_SERVICE_PLAN --name $WEB_APP_NAME \
--deployment-container-image-name $ACR_NAME.azurecr.io/agri-yield-predictor:latest

# Configure App Service to pull from ACR
az webapp config container set --name $WEB_APP_NAME --resource-group $RESOURCE_GROUP \
--docker-custom-image-name $ACR_NAME.azurecr.io/agri-yield-predictor:latest \
--docker-registry-server-url https://$ACR_NAME.azurecr.io \
--docker-registry-server-user $ACR_USERNAME \
--docker-registry-server-password $ACR_PASSWORD


```

## 6 Deployment Summary

- Push changes to GitHub → triggers CI/CD workflow.
- Workflow builds Docker image and pushes to Azure Container Registry.
- App Service pulls latest image and redeploys automatically.
- Model artifacts are versioned with DVC and stored in Azure Blob Storage.

