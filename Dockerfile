FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install CPU-only PyTorch.
# Our semantic search engine does not need CUDA/GPU.
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# Install the remaining application dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application into the image.
COPY . .

EXPOSE 8000

# Render provides the PORT environment variable.
# Locally, if PORT is not set, we use 8000.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]