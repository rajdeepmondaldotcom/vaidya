# ---- builder ----
FROM python:3.11-slim AS builder

ARG CACHE_BUST=2026-04-17-telephony-v7-welcome
WORKDIR /build

COPY pyproject.toml README.md ./
COPY src/ src/

RUN echo "cache_bust=$CACHE_BUST" && pip install --no-cache-dir --prefix=/install ".[telephony]"

# ---- runtime ----
FROM python:3.11-slim AS runtime

RUN groupadd --gid 1000 vaidya \
    && useradd --uid 1000 --gid vaidya --create-home vaidya

COPY --from=builder /install /usr/local

WORKDIR /app
RUN mkdir -p /app/data/audit /app/chroma_data \
    && chown -R vaidya:vaidya /app/data /app/chroma_data
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    HOME=/home/vaidya

USER vaidya

# Pre-bake ChromaDB's ONNX embedding model (~80MB) so container cold starts
# skip the ~50s first-boot download (the model otherwise downloads on first
# embed). Placed before the source COPY so this heavy layer stays cached
# across code-only deploys, and run as vaidya so it caches under the home
# dir the runtime process actually reads.
RUN python -c "import chromadb; c = chromadb.EphemeralClient(); col = c.create_collection('warmup'); col.add(ids=['1'], documents=['warm up the onnx embedding model'])"

COPY --chown=vaidya:vaidya src/ /app/src/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

ENTRYPOINT ["uvicorn", "vaidya.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
