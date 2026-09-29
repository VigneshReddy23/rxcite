FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/opt/models

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

# Bake the embedding + reranking models into the image so startup needs no download.
RUN python -c "from rxcite.embeddings import FastEmbedder, CrossEncoderReranker; FastEmbedder(); CrossEncoderReranker()"

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "rxcite.api:app", "--host", "0.0.0.0", "--port", "8000"]
