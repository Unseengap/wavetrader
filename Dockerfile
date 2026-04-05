FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && \
    rm -rf /var/lib/apt/lists/*

# Pin PyTorch CPU to avoid pulling GPU libs
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY wavetrader/ ./wavetrader/
COPY dashboard/ ./dashboard/
COPY cli.py .
COPY scripts/ ./scripts/

# Persistent data volume (checkpoints, logs)
VOLUME ["/data"]

# Health check for dashboard container
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Default: run the streaming engine
CMD ["python", "-m", "wavetrader.streaming"]
