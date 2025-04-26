# Bottle Data Processing Scripts

This directory contains scripts for processing and ingesting bottle data into a PostgreSQL database with pgvector extension.

## Scripts Overview

### 1. `inventory_bottles.py`

This script manages bottle inventory data, with functionality to import, export, and enrich bottle information.

**Usage:**
```bash
# Enrich bottles with data from CSV file
cd scripts
python inventory_bottles.py enrich-csv "data/501 Bottle Dataset.csv"

# Enrich and save to a different file
python inventory_bottles.py enrich-csv "data/501 Bottle Dataset.csv" --output "data/bottles_enriched.json"

# Import bottles from JSON file
python inventory_bottles.py import-json "path/to/bottles.json"

# Export bottles to CSV format
python inventory_bottles.py export-csv "output.csv"

# Extract data from Baxus listings
python inventory_bottles.py extract-baxus "data/baxus-listings.json"
```

### 2. `ingest_bottles_to_pgvector.py`

This script loads bottle data, optionally enriches it with price information (using the `inventory_bottles.py` functionality), creates vector embeddings using OpenAI's models, and stores everything in a PostgreSQL database with the pgvector extension.

**Usage:**
```bash
# Basic usage - just ingest bottles.json
cd scripts
python ingest_bottles_to_pgvector.py

# Enrich data from CSV before ingestion
python ingest_bottles_to_pgvector.py --csv "data/501 Bottle Dataset.csv" 

# Enrich and save the enriched data
python ingest_bottles_to_pgvector.py --csv "data/501 Bottle Dataset.csv" --save-enriched

# Use a different bottles file
python ingest_bottles_to_pgvector.py --bottles "path/to/custom/bottles.json"
```

**Command-line options:**
- `--bottles`: Path to the bottles JSON file (default: data/bottles.json)
- `--csv`: Path to CSV dataset for enrichment (e.g., data/501 Bottle Dataset.csv)
- `--save-enriched`: Save the enriched bottles data to data/bottles_enriched.json

**Requirements:** PostgreSQL database with pgvector extension must be running and properly configured.

### 3. `run_enrichment_and_ingestion.py`

This script combines the enrichment and ingestion processes into a single pipeline. It's an alternative to using the enrichment capabilities directly.

**Usage:**
```bash
cd scripts
python run_enrichment_and_ingestion.py
```

## Database Schema

The ingestion script creates a PostgreSQL table with the following schema:

```sql
CREATE TABLE IF NOT EXISTS bottles (
    id SERIAL PRIMARY KEY,
    name TEXT,
    description TEXT,
    type TEXT,
    producer TEXT,
    series TEXT,
    abv FLOAT,
    age_statement TEXT,
    country TEXT,
    region TEXT,
    year_bottled TEXT,
    year_distilled TEXT,
    size TEXT,
    price FLOAT,
    fair_price FLOAT,
    shelf_price FLOAT,
    proof FLOAT,
    embedding vector(VECTOR_DIM)
);
```

## Data Enrichment Process

The enrichment process attempts to match bottles with entries in the `501 Bottle Dataset.csv` file using:

1. **Exact matching** - Using normalized bottle names and sizes
2. **Fuzzy matching** - For non-exact matches, using string similarity with a threshold of 80%

When a match is found, the following fields are added to the bottle data:
- `price` (from `avg_msrp` in CSV)
- `fair_price` (from CSV)
- `shelf_price` (from CSV)
- `proof` (from CSV)
- `abv` (if missing in bottles.json but present in CSV)

## Requirements

- Python 3.7+
- PostgreSQL database with pgvector extension
- Required Python packages: `psycopg`, `openai` (for embeddings)
- Data files: `bottles.json` and `501 Bottle Dataset.csv` in the `data/` directory
