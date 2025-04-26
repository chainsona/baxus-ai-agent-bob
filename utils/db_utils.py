"""
Database Utilities Module
------------------------
Centralized utilities for database operations.
"""

import os
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
import psycopg
from psycopg.rows import dict_row

# Import centralized utilities
from utils.logging_utils import get_logger

# Set up logging
logger = get_logger(__name__)

class DatabaseUtils:
    """Centralized utility functions for database operations."""

    @staticmethod
    def get_db_connection_string() -> str:
        """Get the database connection string from environment variables."""
        host = os.getenv("PGVECTOR_HOST", "localhost")
        port = os.getenv("PGVECTOR_PORT", "5432")
        dbname = os.getenv("PGVECTOR_DB", "baxus")
        user = os.getenv("PGVECTOR_USER", "postgres")
        password = os.getenv("PGVECTOR_PASSWORD", "postgres")

        connection_string = f"postgres://{user}:{password}@{host}:{port}/{dbname}"
        logger.debug(f"DB connection string created for host: {host}")
        return connection_string

    @staticmethod
    async def get_connection():
        """Get a database connection using the connection string."""
        db_conn_string = DatabaseUtils.get_db_connection_string()
        conn = await psycopg.AsyncConnection.connect(db_conn_string)
        # Set default row factory for easier result handling
        conn.row_factory = dict_row
        return conn

    @staticmethod
    async def find_bottle_by_id_or_name(
        conn, 
        bottle_id: Optional[Any] = None, 
        bottle_name: Optional[str] = None, 
        fields: str = "*", 
        limit: int = 1
    ) -> List[Dict[str, Any]]:
        """Find a bottle by ID or name, returning specified fields."""
        async with conn.cursor() as cur:
            if bottle_name:
                # Try to find the bottle by name first
                await cur.execute(
                    f"SELECT {fields} FROM bottles WHERE name ILIKE %s LIMIT {limit}",
                    (f"%{bottle_name}%",)
                )
                
                results = await cur.fetchall()
                if results:
                    return results
                    
            # If no results from name search or no name provided, try by ID
            if bottle_id:
                await cur.execute(
                    f"SELECT {fields} FROM bottles WHERE id = %s OR baxus_class_id = %s LIMIT {limit}",
                    (bottle_id, str(bottle_id))
                )
                results = await cur.fetchall()
                return results
                
            # No identifying information or no results
            return []

    @staticmethod
    async def search_by_similarity(
        conn,
        embedding: List[float],
        exclude_ids: Optional[List[int]] = None,
        limit: int = 5,
        extra_conditions: str = ""
    ) -> List[Dict[str, Any]]:
        """Search for bottles by similarity to an embedding vector."""
        async with conn.cursor() as cur:
            exclude_clause = ""
            params = []
            
            # Ensure we get the exact image_url and nft_address for each bottle
            base_query = """
                SELECT 
                    id, name, description, type, image_url, producer, region,
                    country, spirit_type, nft_address,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM bottles 
            """
            params.append(embedding)
            
            if exclude_ids and len(exclude_ids) > 0:
                # Convert all IDs to strings to match baxus_class_id type
                str_ids = [str(id) for id in exclude_ids]
                exclude_clause = "WHERE id::text != ALL(%s) AND baxus_class_id != ALL(%s)"
                params.extend([str_ids, str_ids])
            
            if extra_conditions:
                if exclude_clause:
                    exclude_clause += f" AND {extra_conditions}"
                else:
                    exclude_clause = f"WHERE {extra_conditions}"
            
            # Add order by clause
            order_by = " ORDER BY embedding <=> %s::vector LIMIT %s"
            params.extend([embedding, limit])
            
            query = base_query + exclude_clause + order_by
            
            await cur.execute(query, params)
            results = await cur.fetchall()
            
            # Verify we have complete data for each bottle
            enhanced_results = []
            for bottle in results:
                # Check if image_url is missing or null
                if not bottle.get("image_url") or not bottle.get("nft_address"):
                    # Fetch the exact bottle data by ID to ensure we have image_url
                    bottle_id = bottle.get("id")
                    if bottle_id:
                        await cur.execute("""
                            SELECT image_url, nft_address 
                            FROM bottles
                            WHERE id = %s OR baxus_class_id = %s
                            LIMIT 1
                        """, (bottle_id, str(bottle_id)))
                        exact_data = await cur.fetchone()
                        if exact_data:
                            if exact_data.get("image_url"):
                                bottle["image_url"] = exact_data["image_url"]
                            if exact_data.get("nft_address"):
                                bottle["nft_address"] = exact_data["nft_address"]
                
                enhanced_results.append(bottle)
            
            return enhanced_results

    @staticmethod
    def format_results_with_column_names(cur, results: List[Tuple]) -> List[Dict[str, Any]]:
        """Format query results with column names."""
        if not results:
            return []
            
        col_names = [desc[0] for desc in cur.description]
        formatted_results = []
        
        for row in results:
            formatted_results.append(dict(zip(col_names, row)))
            
        return formatted_results

    @staticmethod
    async def execute_query(query: str, params: List[Any] = None) -> List[Dict[str, Any]]:
        """Execute a query and return the results as dictionaries."""
        try:
            async with await DatabaseUtils.get_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, params or [])
                    results = await cur.fetchall()
                    return results
        except Exception as e:
            logger.error(f"Error executing query: {e}", exc_info=True)
            return []

    @staticmethod
    async def get_exact_bottle_data_by_name(
        conn,
        bottle_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get exact bottle data by name for accurate image and NFT information.
        
        This is a direct lookup, not a similarity search, to ensure we get the correct bottle data.
        """
        async with conn.cursor() as cur:
            # Try exact match first
            await cur.execute(
                "SELECT id, name, image_url, nft_address FROM bottles WHERE name = %s LIMIT 1",
                (bottle_name,)
            )
            result = await cur.fetchone()
            
            # If no exact match, try case insensitive match
            if not result:
                await cur.execute(
                    "SELECT id, name, image_url, nft_address FROM bottles WHERE LOWER(name) = LOWER(%s) LIMIT 1",
                    (bottle_name,)
                )
                result = await cur.fetchone()
                
            # If still no match, try partial match with ILIKE
            if not result:
                await cur.execute(
                    "SELECT id, name, image_url, nft_address FROM bottles WHERE name ILIKE %s LIMIT 1",
                    (f"%{bottle_name}%",)
                )
                result = await cur.fetchone()
            
            return result 