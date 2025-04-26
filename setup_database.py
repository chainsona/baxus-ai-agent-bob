#!/usr/bin/env python3
"""
Database setup script for BAXUS whisky recommendation system.
This script:
1. Creates the necessary database schema
2. Loads bottle data from a CSV file
3. Generates vector embeddings for each bottle and stores them in the database

Usage:
    python setup_database.py --csv /path/to/bottles.csv

Requirements:
    - PostgreSQL with pgvector extension installed
    - OpenAI API key for generating embeddings
"""

import argparse
import asyncio
import csv
import os
import logging
import sys
from typing import List, Dict, Any, Optional

import pandas as pd
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import embedding utilities
def get_embedding(text: str) -> List[float]:
    """Get embedding for text using OpenAI's embedding model."""
    try:
        import openai
        
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set!")
            
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        
        return response.data[0].embedding
    except ImportError:
        logger.error("OpenAI package is not installed. Install with 'pip install openai'")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return []

def get_db_connection_string() -> str:
    """Get the database connection string from environment variables."""
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    dbname = os.getenv("DB_NAME", "baxus")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "postgres")
    
    return f"postgres://{user}:{password}@{host}:{port}/{dbname}"

async def setup_database():
    """Set up the database schema."""
    try:
        async with await psycopg.AsyncConnection.connect(get_db_connection_string()) as conn:
            async with conn.cursor() as cur:
                # Read schema.sql file and execute
                with open("schema.sql", "r") as f:
                    schema_sql = f.read()
                    
                await cur.execute(schema_sql)
                await conn.commit()
                
                logger.info("Database schema created successfully")
    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        sys.exit(1)

async def load_bottles_from_csv(csv_path: str):
    """Load bottle data from CSV file into the database."""
    try:
        # Read CSV data
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} bottles from CSV")
        
        # Clean and prepare data
        df = df.fillna({
            "description": "",
            "type": "Unknown",
            "producer": "",
            "series": "",
            "country": "",
            "region": ""
        })
        
        # Connect to database
        async with await psycopg.AsyncConnection.connect(get_db_connection_string()) as conn:
            conn.row_factory = dict_row
            async with conn.cursor() as cur:
                # Check if brands table needs to be populated
                await cur.execute("SELECT COUNT(*) FROM brands")
                brand_count = await cur.fetchone()
                
                # Create brands if needed
                if brand_count[0] == 0 and "brand_id" in df.columns and "brand" in df.columns:
                    brands = df[["brand_id", "brand"]].drop_duplicates()
                    for _, row in brands.iterrows():
                        if pd.notna(row["brand_id"]) and pd.notna(row["brand"]):
                            await cur.execute(
                                "INSERT INTO brands (id, name) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                                (int(row["brand_id"]), row["brand"])
                            )
                    
                    logger.info(f"Added {len(brands)} brands to database")
                
                # Process bottles in batches to avoid overwhelming the API
                batch_size = 50
                total_bottles = len(df)
                batches = (total_bottles + batch_size - 1) // batch_size
                
                for i in range(batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, total_bottles)
                    batch = df.iloc[start_idx:end_idx]
                    
                    logger.info(f"Processing batch {i+1}/{batches} ({start_idx+1}-{end_idx} of {total_bottles})")
                    
                    for _, row in batch.iterrows():
                        # Generate embedding for the bottle (combining name, type, and description)
                        embedding_text = f"{row['name']} {row.get('type', '')} {row.get('description', '')}"
                        embedding = get_embedding(embedding_text)
                        
                        # Convert numeric values
                        bottle_id = int(row["id"]) if pd.notna(row.get("id")) else None
                        brand_id = int(row["brand_id"]) if pd.notna(row.get("brand_id")) else None
                        abv = float(row["abv"]) if pd.notna(row.get("abv")) else None
                        proof = float(row["proof"]) if pd.notna(row.get("proof")) else (abv * 2 if abv else None)
                        shelf_price = float(row["shelf_price"]) if pd.notna(row.get("shelf_price")) else None
                        avg_msrp = float(row["avg_msrp"]) if pd.notna(row.get("avg_msrp")) else None
                        total_score = float(row["total_score"]) if pd.notna(row.get("total_score")) else None
                        vote_count = int(row["vote_count"]) if pd.notna(row.get("vote_count")) else 0
                        bar_count = int(row["bar_count"]) if pd.notna(row.get("bar_count")) else 0
                        
                        # Insert or update bottle
                        await cur.execute("""
                            INSERT INTO bottles (
                                id, name, description, type, producer, series,
                                abv, proof, age_statement, country, region,
                                shelf_price, avg_msrp, total_score, vote_count, bar_count,
                                brand_id, embedding
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (id) DO UPDATE SET
                                name = EXCLUDED.name,
                                description = EXCLUDED.description,
                                type = EXCLUDED.type,
                                producer = EXCLUDED.producer,
                                series = EXCLUDED.series,
                                abv = EXCLUDED.abv,
                                proof = EXCLUDED.proof,
                                age_statement = EXCLUDED.age_statement,
                                country = EXCLUDED.country,
                                region = EXCLUDED.region,
                                shelf_price = EXCLUDED.shelf_price,
                                avg_msrp = EXCLUDED.avg_msrp,
                                total_score = EXCLUDED.total_score,
                                vote_count = EXCLUDED.vote_count,
                                bar_count = EXCLUDED.bar_count,
                                brand_id = EXCLUDED.brand_id,
                                embedding = EXCLUDED.embedding,
                                updated_at = CURRENT_TIMESTAMP
                        """, (
                            bottle_id,
                            row["name"],
                            row.get("description", ""),
                            row.get("spirit_type", row.get("type", "Unknown")),
                            row.get("producer", ""),
                            row.get("series", ""),
                            abv,
                            proof,
                            row.get("age_statement", ""),
                            row.get("country", ""),
                            row.get("region", ""),
                            shelf_price,
                            avg_msrp,
                            total_score,
                            vote_count,
                            bar_count,
                            brand_id,
                            embedding
                        ))
                    
                    # Commit each batch
                    await conn.commit()
                    logger.info(f"Batch {i+1}/{batches} processed and committed")
                
                logger.info(f"Successfully loaded {total_bottles} bottles into database")
                
    except Exception as e:
        logger.error(f"Error loading bottles: {e}")
        sys.exit(1)

async def main():
    """Main function to set up database and load data."""
    parser = argparse.ArgumentParser(description="Set up BAXUS database and load bottle data")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file with bottle data")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        logger.error(f"CSV file not found: {args.csv}")
        sys.exit(1)
    
    # Create database schema
    logger.info("Setting up database schema...")
    await setup_database()
    
    # Load bottle data
    logger.info(f"Loading bottle data from {args.csv}...")
    await load_bottles_from_csv(args.csv)
    
    logger.info("Database setup complete!")

if __name__ == "__main__":
    asyncio.run(main()) 