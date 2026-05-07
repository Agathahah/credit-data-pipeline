-- ============================================================
-- init.sql
-- Script ini dijalankan OTOMATIS oleh PostgreSQL container
-- saat pertama kali dijalankan (jika volume masih kosong)
-- ============================================================

-- Pastikan user dan database sudah ada
-- (biasanya sudah dibuat lewat env var, ini sebagai backup)

-- Beri privilege ke dataengineer
GRANT ALL PRIVILEGES ON DATABASE credit_risk_db TO dataengineer;

-- Set timezone
ALTER DATABASE credit_risk_db SET timezone TO 'Asia/Jakarta';

-- Log bahwa init berhasil
DO $$
BEGIN
    RAISE NOTICE 'Database credit_risk_db initialized successfully';
END $$;
