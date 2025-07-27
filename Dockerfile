FROM python:3.12-slim

# Install curl for healthcheck and uv
RUN apt-get update && apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/* && \
    pip install uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY uv.lock ./

# Install dependencies
RUN uv pip install --system -r pyproject.toml

# Copy application code
COPY api.py ./

# Create directory for token storage
RUN mkdir -p /app/data

# Expose port
EXPOSE 8000

# Set environment variable for token location
ENV COPILOT_TOKEN_PATH=/app/data/.copilot_token

# Run the application
CMD ["python", "api.py"]