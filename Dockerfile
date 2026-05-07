# ============================================================
# Stage 1: Base image
# Pakai Python 3.10-slim (lebih ringan dari full Python image)
# ============================================================
FROM python:3.10-slim

# Metadata — bagus untuk dokumentasi image
LABEL maintainer="Agatha Ulina Silalahi <silalahiagatha15@gmail.com>"
LABEL description="Credit Risk ML Pipeline — ETL, PostgreSQL, XGBoost"
LABEL version="1.0"

# ============================================================
# System dependencies
# libpq-dev dibutuhkan oleh psycopg2 untuk koneksi PostgreSQL
# gcc dibutuhkan untuk compile beberapa Python packages
# ============================================================
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Set working directory di dalam container
# Semua perintah setelah ini akan dijalankan dari /app
# ============================================================
WORKDIR /app

# ============================================================
# Copy requirements dulu (SEBELUM copy source code)
# Teknik ini memanfaatkan Docker layer cache:
# kalau requirements.txt tidak berubah, layer ini di-cache
# dan tidak perlu install ulang setiap build
# ============================================================
COPY requirements_docker.txt .

# Install Python dependencies
# --no-cache-dir: hemat disk space di dalam container
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_docker.txt

# ============================================================
# Copy source code ke dalam container
# ============================================================
COPY src/ ./src/
COPY run_pipeline.py .

# ============================================================
# Buat direktori untuk data dan output
# ============================================================
RUN mkdir -p data/raw data/processed docs

# ============================================================
# Environment variables default
# Akan di-override oleh docker-compose atau --env-file
# ============================================================
ENV DB_HOST=postgres \
    DB_PORT=5432 \
    DB_NAME=credit_risk_db \
    DB_USER=dataengineer \
    DB_PASSWORD=de_password123 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ============================================================
# Health check: cek apakah Python environment beres
# ============================================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sqlalchemy; import xgboost; print('OK')" || exit 1

# ============================================================
# Default command: jalankan full pipeline
# Bisa di-override saat docker run
# ============================================================
CMD ["python", "run_pipeline.py"]
