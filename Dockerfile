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
COPY src/ /app/src/

WORKDIR /app
RUN mkdir -p /app/data/audit /app/chroma_data \
    && chown -R vaidya:vaidya /app/data /app/chroma_data
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

USER vaidya

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

ENTRYPOINT ["uvicorn", "vaidya.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
