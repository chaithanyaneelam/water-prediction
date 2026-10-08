-- WATERNET hosted schema (Cloudflare D1, SQLite dialect)
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  email TEXT UNIQUE NOT NULL,
  display_name TEXT,
  password_hash TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS readings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  source TEXT NOT NULL DEFAULT 'manual',
  ph REAL, Hardness REAL, Solids REAL, Chloramines REAL, Sulfate REAL,
  Conductivity REAL, Organic_carbon REAL, Trihalomethanes REAL, Turbidity REAL
);

CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  reading_id INTEGER NOT NULL,
  potability INTEGER NOT NULL,
  probability REAL NOT NULL,
  ph_predicted REAL,
  ph_filled_by TEXT,
  is_anomaly INTEGER NOT NULL,
  anomaly_score REAL,
  anomaly_reason TEXT,
  FOREIGN KEY (reading_id) REFERENCES readings(id)
);
CREATE INDEX IF NOT EXISTS idx_predictions_reading ON predictions(reading_id);

CREATE TABLE IF NOT EXISTS sessions (
  token TEXT PRIMARY KEY,
  user_id INTEGER NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS irrigation_readings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  source TEXT NOT NULL DEFAULT 'manual',
  ph REAL, EC REAL, TDS REAL, CO3 REAL, HCO3 REAL, Cl REAL, F REAL, NO3 REAL,
  SO4 REAL, Na REAL, K REAL, Ca REAL, Mg REAL, TH REAL,
  SAR REAL, RSC REAL, ussl_class_rule TEXT, rsc_class_rule TEXT,
  suitable INTEGER, verdict_notes TEXT, ml_ussl_class TEXT, ml_probability REAL
);
