FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Pin PyTorch CPU to avoid pulling GPU libs
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY wavetrader/ ./wavetrader/
COPY cli.py .

# Persistent data volume (checkpoints, logs)
VOLUME ["/data"]

# Default: run the streaming engine
CMD ["python", "-m", "wavetrader.streaming"]
