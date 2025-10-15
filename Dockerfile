# Start with an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# --- DVC Authentication Fix ---
# Accept the connection string as a build argument from the CI/CD pipeline
ARG AZURE_STORAGE_CONNECTION_STRING
# Set it as an environment variable for DVC to use during the build
ENV AZURE_STORAGE_CONNECTION_STRING=$AZURE_STORAGE_CONNECTION_STRING

# Run DVC to pull the model and data needed for the API
RUN dvc pull

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run main.py when the container launches
# Use uvicorn to run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
