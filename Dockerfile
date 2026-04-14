# ---- builder ----
FROM python:3.11-slim AS builder

WORKDIR /build

COPY pyproject.toml ./
COPY src/ src/

RUN pip install --no-cache-dir --prefix=/install .

# ---- runtime ----
FROM python:3.11-slim AS runtime

RUN groupadd --gid 1000 vaidya \
    && useradd --uid 1000 --gid vaidya --create-home vaidya

COPY --from=builder /install /usr/local
COPY src/ /app/src/
COPY data/ /app/data/

WORKDIR /app
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

USER vaidya

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

ENTRYPOINT ["uvicorn", "vaidya.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
