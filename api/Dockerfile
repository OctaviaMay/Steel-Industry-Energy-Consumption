# Build from the project root:
#   docker build -f api/Dockerfile -t steel-forecast-api .
#   docker run -p 8000:8000 steel-forecast-api
FROM python:3.11-slim

WORKDIR /app

COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# Only what the API needs: code, saved models, held-out test set.
COPY api/ api/
COPY models/ models/
COPY data/split/ data/split/

EXPOSE 8000
# PORT lets hosts like Render or Hugging Face Spaces set the port.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]