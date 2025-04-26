#!/usr/bin/env python3
"""
Script to search for bottles in pgvector database using vector similarity.
This script allows searching for bottles similar to a given query using
vector embeddings and pgvector's similarity search capabilities.
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import List, Dict, Any

import psycopg
from dotenv import load_dotenv

# Import shared utilities
from bottle_vector_utils import (
    get_embedding,
    get_db_connection_string,
    format_bottle_result,
    logger
)


async def search_bottles(query: str, limit: int = 5):
    """Search for bottles similar to the query using vector similarity."""
    try:
        # Generate embedding for the query
        query_embedding = get_embedding(query)
        
        async with await psycopg.AsyncConnection.connect(get_db_connection_string()) as conn:
            async with conn.cursor() as cur:
                # Perform similarity search using cosine distance
                await cur.execute("""
                    SELECT 
                        id, name, description, type, producer, series, 
                        abv, age_statement, country, region, 
                        1 - (embedding <=> %s::vector) AS similarity
                    FROM bottles
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, query_embedding, limit))
                
                results = await cur.fetchall()
                
                if not results:
                    print(f"No bottles found similar to '{query}'")
                    return
                
                print(f"\nTop {len(results)} bottles similar to '{query}':\n")
                
                for i, result in enumerate(results, 1):
                    formatted_result = format_bottle_result(result)
                    print(f"{i}. {formatted_result}\n")
    
    except Exception as e:
        logger.error(f"Error searching bottles: {e}")
        sys.exit(1)


async def main():
    """Main function to handle command line arguments and search for bottles."""
    parser = argparse.ArgumentParser(description="Search for bottles using vector similarity.")
    parser.add_argument("query", type=str, help="The search query for finding similar bottles")
    parser.add_argument("--limit", "-l", type=int, default=5, help="Maximum number of results to return (default: 5)")
    
    args = parser.parse_args()
    
    if not args.query:
        parser.print_help()
        sys.exit(1)
    
    await search_bottles(args.query, args.limit)


if __name__ == "__main__":
    asyncio.run(main()) 