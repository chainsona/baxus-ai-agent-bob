#!/usr/bin/env python3
"""
Script to filter bottles in pgvector database by criteria and return similar bottles.
This script allows filtering bottles by type, producer, country, etc., and then
finding the most similar bottles to the filtered results.
"""

import argparse
import asyncio
import sys
from typing import List, Optional

import psycopg
from dotenv import load_dotenv

# Import shared utilities
from bottle_vector_utils import (
    get_db_connection_string,
    format_bottle_result,
    logger
)


async def filter_bottles(
    bottle_type: Optional[str] = None,
    producer: Optional[str] = None,
    country: Optional[str] = None,
    region: Optional[str] = None,
    min_abv: Optional[float] = None,
    max_abv: Optional[float] = None,
    limit: int = 10
):
    """Filter bottles by criteria and return results."""
    try:
        # Build the query conditions
        conditions = []
        params = []
        
        if bottle_type:
            conditions.append("type ILIKE %s")
            params.append(f"%{bottle_type}%")
        
        if producer:
            conditions.append("producer ILIKE %s")
            params.append(f"%{producer}%")
        
        if country:
            conditions.append("country ILIKE %s")
            params.append(f"%{country}%")
        
        if region:
            conditions.append("region ILIKE %s")
            params.append(f"%{region}%")
        
        if min_abv is not None:
            conditions.append("abv >= %s")
            params.append(min_abv)
        
        if max_abv is not None:
            conditions.append("abv <= %s")
            params.append(max_abv)
        
        # Construct the WHERE clause
        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        
        # Construct the full query
        query = f"""
            SELECT 
                id, name, description, type, producer, series, 
                abv, age_statement, country, region, 
                1.0 AS similarity
            FROM bottles
            WHERE {where_clause}
            ORDER BY name
            LIMIT %s
        """
        params.append(limit)
        
        async with await psycopg.AsyncConnection.connect(get_db_connection_string()) as conn:
            async with conn.cursor() as cur:
                # Execute the query
                await cur.execute(query, params)
                results = await cur.fetchall()
                
                if not results:
                    print("No bottles found matching the specified criteria")
                    return
                
                # Format the filter description
                filter_desc = []
                if bottle_type:
                    filter_desc.append(f"type: {bottle_type}")
                if producer:
                    filter_desc.append(f"producer: {producer}")
                if country:
                    filter_desc.append(f"country: {country}")
                if region:
                    filter_desc.append(f"region: {region}")
                if min_abv is not None:
                    filter_desc.append(f"min ABV: {min_abv}%")
                if max_abv is not None:
                    filter_desc.append(f"max ABV: {max_abv}%")
                
                filter_text = ", ".join(filter_desc) if filter_desc else "no filters"
                
                print(f"\nFound {len(results)} bottles matching criteria: {filter_text}\n")
                
                for i, result in enumerate(results, 1):
                    formatted_result = format_bottle_result(result)
                    print(f"{i}. {formatted_result}\n")
    
    except Exception as e:
        logger.error(f"Error filtering bottles: {e}")
        sys.exit(1)


async def find_similar_to_bottle(bottle_id: int, limit: int = 5):
    """Find bottles similar to a specific bottle by ID."""
    try:
        async with await psycopg.AsyncConnection.connect(get_db_connection_string()) as conn:
            async with conn.cursor() as cur:
                # First, get the embedding for the specified bottle
                await cur.execute("SELECT name, embedding FROM bottles WHERE id = %s", (bottle_id,))
                bottle_data = await cur.fetchone()
                
                if not bottle_data:
                    print(f"No bottle found with ID {bottle_id}")
                    return
                
                bottle_name, bottle_embedding = bottle_data
                
                # Find similar bottles based on embedding
                await cur.execute("""
                    SELECT 
                        id, name, description, type, producer, series, 
                        abv, age_statement, country, region, 
                        1 - (embedding <=> %s::vector) AS similarity
                    FROM bottles
                    WHERE id != %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (bottle_embedding, bottle_id, bottle_embedding, limit))
                
                results = await cur.fetchall()
                
                if not results:
                    print(f"No similar bottles found for '{bottle_name}'")
                    return
                
                print(f"\nTop {len(results)} bottles similar to '{bottle_name}' (ID: {bottle_id}):\n")
                
                for i, result in enumerate(results, 1):
                    formatted_result = format_bottle_result(result)
                    print(f"{i}. {formatted_result}\n")
    
    except Exception as e:
        logger.error(f"Error finding similar bottles: {e}")
        sys.exit(1)


async def main():
    """Main function to handle command line arguments and filter bottles."""
    parser = argparse.ArgumentParser(description="Filter bottles by criteria or find similar bottles.")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Filter command
    filter_parser = subparsers.add_parser("filter", help="Filter bottles by criteria")
    filter_parser.add_argument("--type", type=str, help="Filter by bottle type (e.g., Bourbon, Scotch)")
    filter_parser.add_argument("--producer", type=str, help="Filter by producer/distillery")
    filter_parser.add_argument("--country", type=str, help="Filter by country")
    filter_parser.add_argument("--region", type=str, help="Filter by region")
    filter_parser.add_argument("--min-abv", type=float, help="Minimum ABV percentage")
    filter_parser.add_argument("--max-abv", type=float, help="Maximum ABV percentage")
    filter_parser.add_argument("--limit", "-l", type=int, default=10, help="Maximum number of results (default: 10)")
    
    # Similar command
    similar_parser = subparsers.add_parser("similar", help="Find bottles similar to a specific bottle")
    similar_parser.add_argument("bottle_id", type=int, help="ID of the bottle to find similar bottles for")
    similar_parser.add_argument("--limit", "-l", type=int, default=5, help="Maximum number of results (default: 5)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "filter":
        await filter_bottles(
            bottle_type=args.type,
            producer=args.producer,
            country=args.country,
            region=args.region,
            min_abv=args.min_abv,
            max_abv=args.max_abv,
            limit=args.limit
        )
    elif args.command == "similar":
        await find_similar_to_bottle(args.bottle_id, args.limit)


if __name__ == "__main__":
    asyncio.run(main()) 