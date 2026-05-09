# Production Dockerfile for the FastAPI backend.
# Build context: this `backend/` folder. Run from `starter/backend/`:
#   docker build -t bocconi-backend .
#   railway up

FROM python:3.13-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

# Install dependencies first (cached layer if pyproject doesn't change)
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-install-project || uv sync --no-install-project || uv sync

# Copy backend source + bundled data/
COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uv run uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
