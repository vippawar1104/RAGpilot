FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/data/huggingface \
    RAG_DEVICE=cpu

# Copy python packaging configuration and documentation
COPY pyproject.toml README.md ./

# Copy application source code
COPY src/ src/
COPY app.py .

# Install dependencies and local package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Force headless OpenCV to avoid libGL/OpenGL dependency issues
RUN pip uninstall -y opencv-python opencv-python-headless && \
    pip install --no-cache-dir opencv-python-headless

# Expose Streamlit default port
EXPOSE 8501

# Create the data directory for persistent mounting
RUN mkdir -p /app/data

# Run Streamlit (Using shell form to evaluate dynamic PORT variable)
CMD sh -c "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"
