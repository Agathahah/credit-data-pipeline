# Credit risk pipeline image. One requirements file is shared with local
# development so local, Docker and Kubernetes runs use the same library
# versions (the first version used different XGBoost/NumPy versions in Docker,
# which is why its scores differed in the 4th decimal).
FROM python:3.12-slim

LABEL description="Credit Risk ML Pipeline - ETL, PostgreSQL, XGBoost"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/ ./src/
COPY run_pipeline.py .

# Run as an unprivileged user.
RUN useradd --create-home pipeline && mkdir -p data/raw docs models && chown -R pipeline /app
USER pipeline

# Database credentials are NOT baked into the image. Pass them at runtime via
# docker compose (env_file: .env) or a Kubernetes Secret.
CMD ["python", "run_pipeline.py"]
