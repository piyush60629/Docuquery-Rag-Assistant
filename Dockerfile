# DocuQuery: one image that runs either the REST API (default) or the Streamlit UI.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

# Install dependencies first so this layer is cached when only the code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Run as a non-root user.
RUN useradd --create-home --uid 1000 docuquery && chown -R docuquery /app
USER docuquery

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
