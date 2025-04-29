FROM python:3.13-slim AS builder

WORKDIR /app

# Install Poetry
RUN pip install poetry==2.1.2

# Copy poetry configuration files
COPY pyproject.toml poetry.lock* ./

# Configure poetry to not use a virtual environment
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry lock && poetry install --no-interaction --no-ansi

# Production stage
FROM python:3.13-slim

WORKDIR /app

# Copy dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose the application port
EXPOSE 8000

# Run the application
CMD ["python", "api.py"]