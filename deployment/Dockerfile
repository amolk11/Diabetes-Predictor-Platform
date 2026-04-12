# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY deployment/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY deployment/ deployment/
COPY models/ models/

# Expose port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]