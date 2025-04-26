#!/usr/bin/env python3
"""
Script to run bottle enrichment followed by pgvector ingestion.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the scripts directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

async def main():
    try:
        # Import the modules we need
        try:
            from inventory_bottles import BottleInventory
            from ingest_bottles_to_pgvector import main as ingest_bottles
        except ImportError as e:
            logger.error(f"Could not import required modules: {e}")
            logger.error("Make sure inventory_bottles.py and ingest_bottles_to_pgvector.py are in the scripts directory.")
            return

        # First, run the enrichment process
        logger.info("Starting bottle enrichment process...")
        try:
            # Initialize inventory
            bottles_file = Path('data/bottles.json')
            csv_file = Path('data/501 Bottle Dataset.csv')
            enriched_file = Path('data/bottles_enriched.json')
            
            # Check if files exist
            if not bottles_file.exists():
                logger.error(f"Bottles file not found: {bottles_file}")
                return
                
            if not csv_file.exists():
                logger.error(f"CSV dataset file not found: {csv_file}")
                return
            
            # Load inventory and enrich
            inventory = BottleInventory(bottles_file)
            matched = inventory.enrich_from_csv(csv_file, save_enriched=True)
            
            if matched == 0:
                logger.warning("No bottles were enriched. Check that the CSV file format is correct.")
            else:
                logger.info(f"Enriched {matched} bottles from CSV data")
                
            # Create a copy with the enriched filename
            output_inventory = BottleInventory(enriched_file)
            output_inventory.bottles = inventory.bottles.copy()
            output_inventory.bottle_hashes = inventory.bottle_hashes.copy()
            output_inventory.save_inventory()
            logger.info(f"Saved enriched inventory to {enriched_file}")
            
        except Exception as e:
            logger.error(f"Error during enrichment process: {e}")
            return

        # Then, run the ingestion script
        logger.info("Starting bottle ingestion process...")
        try:
            # Use the enriched bottles file for ingestion
            await ingest_bottles()
            logger.info("Bottle ingestion completed")
        except Exception as e:
            logger.error(f"Error during ingestion process: {e}")
            return

        logger.info("Complete pipeline run successful!")

    except Exception as e:
        logger.error(f"Error in pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 