# The Dockerfile sets up a lightweight Python environment, 
# installs dependencies, loads a pre-trained model, configures environment variables 
# for offline inference, ensures an output directory, and runs an inference script on startup.
FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
  build-essential \
  wget \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

COPY run_inference.py .
COPY model /model

# Environment variables for using the local model directory
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/model
ENV TRANSFORMERS_OFFLINE=1

# Ensure outputs directory exists and is writable
RUN mkdir -p /outputs && chmod 777 /outputs

ENTRYPOINT ["python", "-u", "/workspace/run_inference.py"]