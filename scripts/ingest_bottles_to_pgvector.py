#!/usr/bin/env python3
"""
Script to ingest bottles.json data into pgvector database.
This script can optionally enrich bottle data with 501 Bottle Dataset.csv before ingestion.
"""

from inventory_bottles import BottleInventory
from bottle_vector_utils import (
    VECTOR_DIM,
    generate_bottle_embedding,
    get_db_connection_string,
    logger
)
import asyncio
import json
import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import psycopg

# Add the current directory to path to allow importing bottle_vector_utils
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


# Import BottleInventory for enrichment functionality


async def setup_database():
    """Set up the database schema with pgvector extension."""
    try:
        # Print connection string for debugging (remove sensitive info)
        conn_string = get_db_connection_string()
        safe_conn_string = conn_string.replace(
            os.environ.get('PGVECTOR_PASSWORD', 'baxuspwd'), '***')
        logger.info(f"Attempting to connect with: {safe_conn_string}")

        async with await psycopg.AsyncConnection.connect(get_db_connection_string()) as conn:
            async with conn.cursor() as cur:
                # Create pgvector extension if it doesn't exist
                await cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # Create bottles table if it doesn't exist
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS bottles (
                        id SERIAL PRIMARY KEY,
                        name TEXT,
                        description TEXT,
                        type TEXT,
                        image_url TEXT,
                        animation_url TEXT,
                        producer TEXT,
                        series TEXT,
                        abv FLOAT,
                        age_statement TEXT,
                        country TEXT,
                        region TEXT,
                        year_bottled TEXT,
                        year_distilled TEXT,
                        size INT,
                        price FLOAT,
                        fair_price FLOAT,
                        shelf_price FLOAT,
                        proof FLOAT,
                        baxus_class_id TEXT,
                        baxus_class_name TEXT,
                        nft_address TEXT,
                        spirit_type TEXT,
                        brand_id FLOAT,
                        popularity FLOAT,
                        avg_msrp FLOAT,
                        total_score FLOAT,
                        wishlist_count FLOAT,
                        vote_count FLOAT,
                        bar_count FLOAT,
                        ranking FLOAT,
                        embedding vector({})
                    );
                """.format(VECTOR_DIM))

                # Create an index on the embedding vector for faster similarity searches
                await cur.execute("CREATE INDEX IF NOT EXISTS bottles_embedding_idx ON bottles USING ivfflat (embedding vector_l2_ops);")

                await conn.commit()
                logger.info("Database setup completed successfully")
    except psycopg.OperationalError as e:
        logger.error(f"Database connection error: {e}")
        logger.error(
            "Make sure your PostgreSQL container with pgvector is running")
        raise
    except Exception as e:
        logger.error(f"Database setup error: {e}")
        raise


def process_bottle(bottle):
    """Process a single bottle to generate its embedding."""
    try:
        embedding = generate_bottle_embedding(bottle)
        return bottle, embedding
    except Exception as e:
        logger.error(f"Error processing bottle {bottle.get('name')}: {e}")
        return bottle, None


async def ingest_bottle_batch(conn, batch_with_embeddings):
    """Ingest a batch of bottles with pre-generated embeddings."""
    async with conn.cursor() as cur:
        for bottle, embedding in batch_with_embeddings:
            if embedding is None:
                continue

            await cur.execute("""
                INSERT INTO bottles 
                (name, description, type, image_url, animation_url, producer, series, abv, age_statement,
                 country, region, year_bottled, year_distilled, size, price, 
                 fair_price, shelf_price, proof, baxus_class_id, baxus_class_name, nft_address,
                 spirit_type, brand_id, popularity, avg_msrp, total_score, wishlist_count,
                 vote_count, bar_count, ranking, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
            """, (
                bottle.get("name"),
                bottle.get("description"),
                bottle.get("type"),
                bottle.get("image_url"),
                bottle.get("animation_url"),
                bottle.get("producer"),
                bottle.get("series"),
                bottle.get("abv"),
                bottle.get("age_statement"),
                bottle.get("country"),
                bottle.get("region"),
                bottle.get("year_bottled"),
                bottle.get("year_distilled"),
                bottle.get("size"),
                bottle.get("price"),
                bottle.get("fair_price"),
                bottle.get("shelf_price"),
                bottle.get("proof"),
                bottle.get("baxus_class_id"),
                bottle.get("baxus_class_name"),
                bottle.get("nft_address"),
                bottle.get("spirit_type"),
                bottle.get("brand_id"),
                bottle.get("popularity"),
                bottle.get("avg_msrp"),
                bottle.get("total_score"),
                bottle.get("wishlist_count"),
                bottle.get("vote_count"),
                bottle.get("bar_count"),
                bottle.get("ranking"),
                embedding
            ))
        await conn.commit()


async def ingest_bottles(bottles: List[Dict[str, Any]]):
    """Ingest bottles into the pgvector database with parallel processing."""
    try:
        async with await psycopg.AsyncConnection.connect(get_db_connection_string()) as conn:
            logger.info(f"Processing {len(bottles)} bottles")

            # Truncate the bottles table to start fresh
            async with conn.cursor() as cur:
                await cur.execute("TRUNCATE TABLE bottles;")
                await conn.commit()

            # Process bottles in parallel batches
            batch_size = 100
            max_workers = min(
                32, (len(bottles) + batch_size - 1) // batch_size)

            # Process all batches of bottles
            for i in range(0, len(bottles), batch_size):
                batch = bottles[i:i+batch_size]
                logger.info(
                    f"Processing batch {i//batch_size + 1}/{(len(bottles) + batch_size - 1)//batch_size}")

                # Generate embeddings in parallel using ThreadPoolExecutor
                # (since OpenAI API calls are I/O bound but not async)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    batch_with_embeddings = list(
                        executor.map(process_bottle, batch))

                # Ingest the batch with pre-generated embeddings
                await ingest_bottle_batch(conn, batch_with_embeddings)
                logger.info(f"Committed batch {i//batch_size + 1}")

            logger.info("All bottles ingested successfully")
    except Exception as e:
        logger.error(f"Error ingesting bottles: {e}")
        raise


async def parallel_ingest_bottles(bottles: List[Dict[str, Any]]):
    """Ingest bottles with fully parallel processing across batches."""
    try:
        logger.info(
            f"Processing {len(bottles)} bottles with parallel ingestion")

        # Process bottles in parallel batches
        batch_size = 50  # Smaller batch size for more parallelism
        num_batches = (len(bottles) + batch_size - 1) // batch_size
        max_concurrent_batches = 5  # Limit concurrent DB connections

        # Truncate the bottles table to start fresh
        async with await psycopg.AsyncConnection.connect(get_db_connection_string()) as conn:
            async with conn.cursor() as cur:
                await cur.execute("TRUNCATE TABLE bottles;")
                await conn.commit()

        # Process batches in parallel with a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent_batches)

        # Store reference to process_bottle to avoid scoping issues
        bottle_processor = process_bottle

        async def process_batch(batch_idx):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(bottles))
            batch = bottles[start_idx:end_idx]

            logger.info(f"Starting batch {batch_idx + 1}/{num_batches}")

            # Generate embeddings in parallel
            with ThreadPoolExecutor(max_workers=min(20, len(batch))) as executor:
                batch_with_embeddings = list(
                    executor.map(bottle_processor, batch))

            # Acquire semaphore before connecting to DB
            async with semaphore:
                try:
                    async with await psycopg.AsyncConnection.connect(get_db_connection_string()) as conn:
                        await ingest_bottle_batch(conn, batch_with_embeddings)
                        logger.info(
                            f"Completed batch {batch_idx + 1}/{num_batches}")
                except Exception as e:
                    logger.error(f"Error in batch {batch_idx + 1}: {e}")

        # Create and gather all batch processing tasks
        tasks = [process_batch(i) for i in range(num_batches)]
        await asyncio.gather(*tasks)

        logger.info(
            "All bottles ingested successfully with parallel processing")
    except Exception as e:
        logger.error(f"Error in parallel ingestion: {e}")
        raise


async def main():
    """Main function to load bottles and ingest them into pgvector."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Ingest bottles data into pgvector database with optional enrichment')
    parser.add_argument('--bottles', type=str, default='data/bottles.json',
                        help='Path to bottles.json file (default: data/bottles.json)')
    parser.add_argument('--csv', type=str,
                        help='Path to CSV dataset for enrichment (e.g., data/501 Bottle Dataset.csv). If provided, enriches bottles data.')
    parser.add_argument('--save-enriched', action='store_true',
                        help='Save the enriched bottles data to data/bottles_enriched.json')
    args = parser.parse_args()

    try:
        # Path to the bottles.json file
        bottles_file = Path(args.bottles)

        # Check if the file exists
        if not bottles_file.exists():
            logger.error(f"File not found: {bottles_file}")
            return

        logger.info(f"Loading bottle data from {bottles_file}")

        # Load bottles data using BottleInventory
        inventory = BottleInventory(bottles_file)
        logger.info(f"Found {len(inventory)} bottles in data file")

        # Enrich data if CSV file is provided
        if args.csv:
            csv_file = Path(args.csv)
            logger.info(f"Enriching bottles data with {csv_file}")
            matched = inventory.enrich_from_csv(
                csv_file, save_enriched=args.save_enriched)
            logger.info(f"Enriched {matched} bottles from CSV data")

            # If save_enriched is True, the inventory is already saved in enrich_from_csv

            # Use the enriched bottles data for ingestion
            bottles_data = inventory.bottles
        else:
            # Use the original bottles data without enrichment
            bottles_data = inventory.bottles

        # Set up the database schema
        await setup_database()

        # Use parallel ingestion for better performance
        await parallel_ingest_bottles(bottles_data)

        logger.info("Bottle ingestion completed successfully")

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
