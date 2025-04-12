FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# Using default PyPI index for others
RUN pip install --no-cache-dir psutil pandas pyarrow scikit-learn


# Copy worker implementation
COPY ../distributed_train.py .

# Expose the TensorFlow server port
EXPOSE 5004

# Entry point for the worker
# CMD ["python", "distributed_train.py"]
CMD ["python", "/app/distributed_train.py"]