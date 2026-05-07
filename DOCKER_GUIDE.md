# 🐳 Docker Guide — Credit Data Pipeline

Panduan lengkap untuk menjalankan credit-data-pipeline menggunakan Docker.

---

## Kenapa Docker?

Tanpa Docker:
- "Di komputer saya bisa jalan, di tempat lain tidak" → dependency hell
- Harus install PostgreSQL, Python, semua library secara manual
- Environment berbeda antara development dan production

Dengan Docker:
- Semua dependency terbungkus dalam satu image
- Jalan di mana saja: laptop, server, cloud
- Isolasi penuh: tidak ganggu environment lokal

---

## Prasyarat

```bash
# Install Docker Desktop (Mac/Windows) atau Docker Engine (Linux)
# Cek instalasi:
docker --version         # Docker version 24.x.x
docker-compose --version # Docker Compose version 2.x.x
```

---

## Struktur File Docker

```
credit-data-pipeline/
├── Dockerfile              ← Resep untuk build image Python pipeline
├── docker-compose.yml      ← Orkestrasi pipeline + PostgreSQL
├── requirements_docker.txt ← Dependencies versi ramping (tanpa Jupyter)
├── init.sql                ← SQL script inisialisasi database
├── .env.example            ← Template environment variables
├── .dockerignore           ← File yang tidak masuk image
├── Makefile                ← Shortcut perintah Docker
└── DOCKER_GUIDE.md         ← Panduan ini
```

---

## Cara Menjalankan

### Setup awal (sekali saja)

```bash
# 1. Clone repo
git clone https://github.com/Agathahah/credit-data-pipeline.git
cd credit-data-pipeline

# 2. Buat file .env dari template
cp .env.example .env

# 3. Taruh dataset di folder data/raw/
# Download dari: https://www.kaggle.com/c/GiveMeSomeCredit/data
# Simpan sebagai: data/raw/cs-training.csv

# 4. Build Docker image
make build
# atau: docker-compose build
```

### Jalankan full pipeline

```bash
# Jalankan PostgreSQL + pipeline sekaligus
make run
# atau: docker-compose up

# Jalankan di background
make run-detached
# atau: docker-compose up -d
```

### Jalankan bertahap (untuk development/debugging)

```bash
# Step 1: Jalankan PostgreSQL dulu
make db-only

# Step 2: Di terminal lain, jalankan pipeline
make pipeline-only
```

---

## Monitoring & Debugging

```bash
# Lihat status container
make ps

# Lihat logs real-time
make logs              # semua service
make logs-pipeline     # pipeline saja

# Masuk ke dalam container (untuk debug)
make shell             # bash di container pipeline
make psql              # psql di container PostgreSQL

# Test koneksi database
make test-db
```

---

## Perintah Docker Penting (yang sering ditanya interviewer)

```bash
# Build image
docker build -t credit-pipeline:v1 .

# Lihat image yang ada
docker images

# Jalankan container manual
docker run --env-file .env credit-pipeline:v1

# Lihat container yang berjalan
docker ps

# Lihat semua container (termasuk yang stop)
docker ps -a

# Lihat logs container
docker logs credit_pipeline

# Stop container
docker stop credit_pipeline

# Hapus container
docker rm credit_pipeline

# Masuk ke dalam container yang sedang berjalan
docker exec -it credit_pipeline /bin/bash
```

---

## Arsitektur Docker

```
┌─────────────────────────────────────────┐
│          Docker Network (bridge)        │
│                                         │
│  ┌──────────────┐   ┌────────────────┐  │
│  │   pipeline   │──▶│   postgres     │  │
│  │  container   │   │   container    │  │
│  │              │   │                │  │
│  │ Python 3.10  │   │ PostgreSQL 14  │  │
│  │ XGBoost      │   │ credit_risk_db │  │
│  │ SQLAlchemy   │   │                │  │
│  └──────────────┘   └───────┬────────┘  │
│                             │           │
└─────────────────────────────┼───────────┘
                              │ port 5433
                    ┌─────────▼─────────┐
                    │   Host Machine    │
                    │  (laptop kamu)    │
                    │                   │
                    │  data/raw/        │
                    │  (mounted volume) │
                    └───────────────────┘
```

**Penjelasan:**
- `pipeline` dan `postgres` berada dalam satu Docker network
- Di dalam network, `pipeline` bisa reach `postgres` via hostname `postgres`
- Dari laptop, PostgreSQL bisa diakses via `localhost:5433`
- Folder `data/` di laptop di-mount ke `/app/data` di dalam container

---

## Perbedaan `requirements.txt` vs `requirements_docker.txt`

| | requirements.txt | requirements_docker.txt |
|---|---|---|
| Tujuan | Development lokal | Docker container |
| Isi | Semua dependency + Jupyter | Hanya yang dibutuhkan pipeline |
| Ukuran | ~118 packages | ~14 packages |
| Jupyter | ✅ Ada | ❌ Tidak ada |

Memisahkan keduanya adalah best practice MLOps: container production harus sekecil dan sesederhana mungkin.

---

## Troubleshooting

**Error: `port 5432 already in use`**
```bash
# PostgreSQL lokal sedang berjalan, pakai port 5433 di host
# Sudah dikonfigurasi di docker-compose.yml: "5433:5432"
# Akses via: localhost:5433
```

**Error: `dataset not found`**
```bash
# Pastikan file ada di path yang benar:
ls data/raw/cs-training.csv
```

**Pipeline tidak bisa connect ke database**
```bash
# Cek apakah postgres container sehat:
docker-compose ps
# Tunggu hingga status postgres = "healthy" sebelum pipeline jalan
```

**Rebuild dari awal (kalau ada perubahan code)**
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up
```
