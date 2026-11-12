# Use a lightweight official Python 3.12 image
FROM python:3.12-slim

# Set environment variables for optimal Python execution in Docker
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install essential system dependencies (no-install-recommends minimizes image size)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code
COPY src/ ./src/

# Create a non-root user for enhanced security and switch to it
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose the port that the FastAPI application will run on
EXPOSE 8000

# Command to start the FastAPI server via Uvicorn
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]