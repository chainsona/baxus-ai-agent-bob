"""
Bottle Recommender Module
-----------------------
Core recommendation engine for whisky bottles.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import psycopg
from psycopg.rows import dict_row

# Import local modules
from utils.bottle_utils import BottleUtils
from utils.db_utils import DatabaseUtils
from utils.error_utils import ErrorUtils

# Set up logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Core flavor categories
CORE_FLAVORS = ["sweet", "woody", "spicy", "smoky", "fruity", "smooth", "floral"]

# Flavor profile mappings for different spirits and whisky types
FLAVOR_PROFILES_BY_TYPE = {
    "Bourbon": {"sweet": 0.8, "woody": 0.7, "spicy": 0.5, "smoky": 0.2, "fruity": 0.3, "smooth": 0.6, "floral": 0.1},
    "Scotch": {"sweet": 0.3, "woody": 0.6, "spicy": 0.4, "smoky": 0.8, "fruity": 0.4, "smooth": 0.5, "floral": 0.2},
    "Rye": {"sweet": 0.3, "woody": 0.5, "spicy": 0.9, "smoky": 0.3, "fruity": 0.2, "smooth": 0.4, "floral": 0.2},
    "Irish": {"sweet": 0.6, "woody": 0.4, "spicy": 0.3, "smoky": 0.2, "fruity": 0.7, "smooth": 0.8, "floral": 0.4},
    "Japanese": {"sweet": 0.5, "woody": 0.7, "spicy": 0.4, "smoky": 0.3, "fruity": 0.6, "smooth": 0.8, "floral": 0.6},
    "Canadian": {"sweet": 0.7, "woody": 0.6, "spicy": 0.4, "smoky": 0.1, "fruity": 0.5, "smooth": 0.7, "floral": 0.2},
    "Canadian Whisky": {"sweet": 0.7, "woody": 0.6, "spicy": 0.4, "smoky": 0.1, "fruity": 0.5, "smooth": 0.7, "floral": 0.2},
    "Whiskey": {"sweet": 0.6, "woody": 0.6, "spicy": 0.4, "smoky": 0.3, "fruity": 0.4, "smooth": 0.5, "floral": 0.2},
    "Whisky": {"sweet": 0.6, "woody": 0.6, "spicy": 0.4, "smoky": 0.3, "fruity": 0.4, "smooth": 0.5, "floral": 0.2},
    "Blended": {"sweet": 0.6, "woody": 0.5, "spicy": 0.3, "smoky": 0.4, "fruity": 0.5, "smooth": 0.6, "floral": 0.3},
    "Single Malt": {"sweet": 0.4, "woody": 0.6, "spicy": 0.3, "smoky": 0.5, "fruity": 0.5, "smooth": 0.7, "floral": 0.4},
    "Single Malt Scotch Whisky": {"sweet": 0.4, "woody": 0.6, "spicy": 0.3, "smoky": 0.6, "fruity": 0.5, "smooth": 0.7, "floral": 0.4},
    "Tennessee": {"sweet": 0.7, "woody": 0.6, "spicy": 0.4, "smoky": 0.3, "fruity": 0.3, "smooth": 0.5, "floral": 0.1},
    "American": {"sweet": 0.6, "woody": 0.6, "spicy": 0.4, "smoky": 0.2, "fruity": 0.3, "smooth": 0.6, "floral": 0.2},
    "Rum": {"sweet": 0.9, "woody": 0.4, "spicy": 0.3, "smoky": 0.1, "fruity": 0.7, "smooth": 0.6, "floral": 0.2},
    "Tequila": {"sweet": 0.4, "woody": 0.3, "spicy": 0.6, "smoky": 0.2, "fruity": 0.5, "smooth": 0.4, "floral": 0.3},
    "Gin": {"sweet": 0.3, "woody": 0.1, "spicy": 0.5, "smoky": 0.1, "fruity": 0.5, "smooth": 0.4, "floral": 0.8},
    "Vodka": {"sweet": 0.2, "woody": 0.1, "spicy": 0.2, "smoky": 0.1, "fruity": 0.3, "smooth": 0.9, "floral": 0.3},
    "Cognac": {"sweet": 0.7, "woody": 0.5, "spicy": 0.3, "smoky": 0.1, "fruity": 0.8, "smooth": 0.7, "floral": 0.5}
}

class BottleRecommender:
    """Core recommendation engine for whisky bottles."""

    def __init__(self, csv_path: Optional[str] = None):
        """Initialize the recommender with optional dataset."""
        self.df = None
        if csv_path and os.path.exists(csv_path):
            logger.debug(f"Loading bottle data from CSV: {csv_path}")
            self.df = pd.read_csv(csv_path)
            # Clean and prepare the data
            self._prepare_data()
            logger.debug(f"Loaded {len(self.df)} bottles from CSV")
        else:
            logger.debug("No CSV data provided or file not found")

    def _prepare_data(self):
        """Clean and prepare the bottle dataset."""
        if self.df is not None:
            logger.debug("Preparing bottle dataset")
            # Fix missing values
            self.df["proof"] = self.df["proof"].fillna(self.df["abv"] * 2)
            self.df["abv"] = self.df["abv"].fillna(self.df["proof"] / 2)
            self.df["spirit_type"] = self.df["spirit_type"].fillna("Unknown")

            # Ensure numeric columns
            numeric_cols = ["avg_msrp", "price", "total_score",
                            "vote_count", "bar_count"]
            for col in numeric_cols:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(
                        self.df[col], errors="coerce").fillna(0)

            logger.debug(
                f"Data preparation complete. Missing values: {self.df.isna().sum().sum()}")

    @ErrorUtils.handle_async_exceptions(default_return={})
    async def get_bottle_flavor_profile(self, bottle_type: Optional[str] = None) -> Dict[str, float]:
        """
        Get the flavor profile for a specific bottle type.
        Returns a profile with core flavor attributes: sweet, woody, spicy, smoky, fruity, smooth, floral
        
        Args:
            bottle_type: The type of the bottle (e.g., 'Bourbon', 'Scotch')
            
        Returns:
            Dictionary containing flavor profile scores for the core attributes
        """
        logger.debug(f"Getting flavor profile for bottle type: {bottle_type}")
        
        # Use default profile if no type provided or type not found
        if not bottle_type or bottle_type not in FLAVOR_PROFILES_BY_TYPE:
            # Try to find a close match if exact match not found
            if bottle_type:
                bottle_type_lower = bottle_type.lower()
                for known_type in FLAVOR_PROFILES_BY_TYPE.keys():
                    if known_type.lower() in bottle_type_lower or bottle_type_lower in known_type.lower():
                        bottle_type = known_type
                        break
            
            # If still no match, use default profile (Whiskey)
            if not bottle_type or bottle_type not in FLAVOR_PROFILES_BY_TYPE:
                bottle_type = "Whiskey"
                logger.debug(f"No matching type found, using default profile: {bottle_type}")
        
        # Get the flavor profile for the bottle type
        profile = FLAVOR_PROFILES_BY_TYPE.get(bottle_type, FLAVOR_PROFILES_BY_TYPE["Whiskey"])
        
        # Ensure all core flavors are included
        flavor_profile = {flavor: profile.get(flavor, 0.0) for flavor in CORE_FLAVORS}
        
        logger.debug(f"Flavor profile for {bottle_type}: {flavor_profile}")
        return flavor_profile

    @ErrorUtils.handle_async_exceptions(default_return=[])
    async def search_similar_bottles(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for bottles similar to the query using pgvector similarity."""
        logger.debug(
            f"Searching for bottles similar to: '{query}' (limit: {limit})")
        
        # Generate embedding for the query
        query_embedding = BottleUtils.get_embedding(query)
        
        async with await DatabaseUtils.get_connection() as conn:
            return await DatabaseUtils.search_by_similarity(
                conn,
                embedding=query_embedding,
                limit=limit
            )

    @ErrorUtils.handle_async_exceptions(default_return=[])
    async def recommend_from_bottle_ids(
        self,
        bottle_ids: List[int],
        limit: int = 5,
        price_range: Optional[Tuple[float, float]] = None
    ) -> List[Dict[str, Any]]:
        """Recommend bottles based on a list of bottle IDs the user likes."""
        if not bottle_ids:
            logger.debug("No bottle IDs provided for recommendation")
            return []

        logger.debug(
            f"Recommending bottles based on {len(bottle_ids)} bottle IDs")

        async with await DatabaseUtils.get_connection() as conn:
            async with conn.cursor() as cur:
                # First get embeddings for the liked bottles
                logger.debug("Fetching embeddings for liked bottles")
                
                # Format for SQL IN clause
                bottle_ids_str = ','.join(str(id) for id in bottle_ids)
                
                await cur.execute(f"""
                    SELECT id, embedding
                    FROM bottles
                    WHERE id IN ({bottle_ids_str})
                """)

                liked_bottles = await cur.fetchall()
                if not liked_bottles:
                    logger.debug(
                        "No embeddings found for the provided bottle IDs")
                    return []

                # Average the embeddings of liked bottles to create a profile vector
                embeddings = [bottle["embedding"]
                              for bottle in liked_bottles]

                # Convert embeddings to numeric arrays if they're strings
                embeddings = self._convert_embeddings(embeddings)
                logger.debug(
                    f"Embeddings shape: {embeddings[0].shape if embeddings else 'no embeddings'}")

                avg_embedding = np.mean(embeddings, axis=0).tolist()
                logger.debug(
                    f"Created average embedding from {len(embeddings)} bottle embeddings")

                # Get price information for price filtering if needed
                extra_conditions = ""
                if price_range:
                    min_price, max_price = price_range
                    extra_conditions = f"price BETWEEN {min_price} AND {max_price}"
                    logger.debug(
                        f"Applying price filter: {min_price} to {max_price}")

                # Find similar bottles excluding the ones they already like
                logger.debug("Executing recommendation query")
                return await DatabaseUtils.search_by_similarity(
                    conn,
                    embedding=avg_embedding,
                    exclude_ids=bottle_ids,
                    limit=limit,
                    extra_conditions=extra_conditions
                )

    @ErrorUtils.handle_async_exceptions(default_return=[])
    async def recommend_diverse_bottles(
        self,
        bottle_ids: List[int],
        limit: int = 5,
        diversity_ratio: float = 0.4
    ) -> List[Dict[str, Any]]:
        """Recommend bottles that would diversify a collection."""
        if not bottle_ids:
            logger.debug(
                "No bottle IDs provided for diverse recommendations")
            return []

        bottle_ids_str = ','.join(str(id) for id in bottle_ids)
        logger.debug(
            f"Finding diverse recommendations for {len(bottle_ids)} bottle IDs with diversity ratio {diversity_ratio}")

        async with await DatabaseUtils.get_connection() as conn:
            async with conn.cursor() as cur:
                # First analyze what types the user already has
                logger.debug("Analyzing user's existing bottle types")
                await cur.execute(f"""
                    SELECT type, COUNT(*) as type_count
                    FROM bottles
                    WHERE id IN ({bottle_ids_str})
                    GROUP BY type
                """)

                user_types = await cur.fetchall()
                user_type_names = [t["type"].lower() for t in user_types]
                logger.debug(
                    f"User has {len(user_type_names)} different bottle types: {user_type_names}")

                # Get regions the user already has
                logger.debug("Analyzing user's existing regions")
                await cur.execute(f"""
                    SELECT region, COUNT(*) as region_count
                    FROM bottles
                    WHERE id IN ({bottle_ids_str}) AND region IS NOT NULL
                    GROUP BY region
                """)

                user_regions = await cur.fetchall()
                user_region_names = [r["region"]
                                     for r in user_regions if r["region"]]
                logger.debug(
                    f"User has {len(user_region_names)} different regions: {user_region_names}")

                # Adjust diversity based on the diversity_ratio
                # Higher ratio means we prioritize bottles that are more different from user's collection
                # Maps 0.0-1.0 to 0.2-0.8
                diversity_threshold = 0.2 + (0.6 * diversity_ratio)
                diversity_boost = 5 * diversity_ratio  # Boost for diversity sorting

                logger.debug(
                    f"Using diversity threshold {diversity_threshold} and boost {diversity_boost}")

                # Find types that are underrepresented or missing
                # With higher diversity_ratio, we put more emphasis on completely new types/regions
                logger.debug("Executing diverse recommendation query")

                # Base query
                query = f"""
                    WITH user_bottle_stats AS (
                        SELECT 
                            b.type,
                            COUNT(*) as type_count,
                            b.region,
                            COUNT(*) as region_count
                        FROM bottles b
                        WHERE b.id IN ({bottle_ids_str})
                        GROUP BY b.type, b.region
                    )
                    SELECT 
                        b.id, b.name, b.description, b.type, b.producer, b.series, 
                        b.abv, b.age_statement, b.country, b.region,
                        b.image_url, b.nft_address,
                        CASE 
                            WHEN b.type NOT IN (SELECT DISTINCT type FROM user_bottle_stats) THEN {diversity_boost}
                            ELSE 0
                        END +
                        CASE 
                            WHEN b.region IS NOT NULL AND b.region NOT IN (SELECT DISTINCT region FROM user_bottle_stats WHERE region IS NOT NULL) THEN {diversity_boost} 
                            ELSE 0
                        END +
                        CASE
                            WHEN b.country IS NOT NULL AND b.country NOT IN (SELECT DISTINCT country FROM bottles WHERE id IN ({bottle_ids_str}) AND country IS NOT NULL) THEN {diversity_boost * 0.5}
                            ELSE 0
                        END as diversity_score
                    FROM bottles b
                    WHERE b.id NOT IN ({bottle_ids_str})
                """

                # Apply diversity threshold filtering
                if diversity_ratio > 0.3:
                    query += f"""
                        AND (
                            b.type NOT IN (SELECT DISTINCT type FROM user_bottle_stats)
                            OR 
                            (b.region IS NOT NULL AND b.region NOT IN (SELECT DISTINCT region FROM user_bottle_stats WHERE region IS NOT NULL))
                            OR
                            (b.country IS NOT NULL AND b.country NOT IN (SELECT DISTINCT country FROM bottles WHERE id IN ({bottle_ids_str}) AND country IS NOT NULL))
                        )
                    """

                # If high diversity ratio, further prioritize completely new experiences
                if diversity_ratio > 0.7:
                    query += """
                        ORDER BY 
                            diversity_score DESC,
                            vote_count DESC
                    """
                else:
                    query += """
                        ORDER BY 
                            diversity_score DESC,
                            vote_count DESC
                    """

                query += f"LIMIT {limit}"

                await cur.execute(query)
                results = await cur.fetchall()

                logger.debug(
                    f"Found {len(results)} diverse bottle recommendations with diversity ratio {diversity_ratio}")

                # Add a reason for each recommendation
                for result in results:
                    if result["type"].lower() not in user_type_names:
                        result["reason"] = f"Adds {result['type']} to diversify your collection"
                    elif result["region"] and result["region"] not in user_region_names:
                        result["reason"] = f"Introduces a new region ({result['region']}) to your collection"
                    else:
                        result["reason"] = f"Provides a different experience from your current collection"

                return results

    # Helper function to convert string embeddings to numeric arrays
    def _convert_embeddings(self, embeddings: List[Any]) -> List[np.ndarray]:
        """Convert embeddings from string format to numeric arrays if needed."""
        if not embeddings:
            return []

        # Check if conversion is needed
        if isinstance(embeddings[0], str):
            logger.debug("Converting string embeddings to numeric arrays")
            # If embeddings are returned as strings, convert them to numeric arrays
            numeric_embeddings = []
            for emb in embeddings:
                # Parse string representation of vector to numeric array
                if isinstance(emb, str):
                    # Clean the string and split into values
                    clean_emb = emb.strip('[]()').replace(' ', '')
                    values = [float(val)
                              for val in clean_emb.split(',') if val]
                    numeric_embeddings.append(values)
                else:
                    numeric_embeddings.append(emb)
            embeddings = numeric_embeddings
            logger.debug(
                f"Converted {len(embeddings)} embeddings to numeric format")

        # Ensure all embeddings are properly formatted as numeric arrays
        return [np.array(emb, dtype=np.float32) for emb in embeddings]

    @ErrorUtils.handle_async_exceptions(default_return={"error": "Failed to analyze collection"})
    async def analyze_collection(self, bottle_ids: List[int]) -> Dict[str, Any]:
        """Analyze a user's collection to provide insights."""
        if not bottle_ids:
            logger.debug("No bottles in collection to analyze")
            return {"error": "No bottles in collection to analyze"}

        bottle_ids_str = ','.join(str(id) for id in bottle_ids)
        logger.debug(
            f"Analyzing collection with {len(bottle_ids)} bottles")

        async with await DatabaseUtils.get_connection() as conn:
            async with conn.cursor() as cur:
                # Get all bottle data
                logger.debug("Fetching bottle data for analysis")
                await cur.execute(f"""
                    SELECT *
                    FROM bottles
                    WHERE id IN ({bottle_ids_str})
                """)

                bottles = await cur.fetchall()
                logger.debug(
                    f"Retrieved {len(bottles)} bottles for analysis")

                # Calculate statistics
                types_count = {}
                regions_count = {}
                abvs = []

                for bottle in bottles:
                    # Count types
                    bottle_type = bottle.get("type", "Unknown")
                    types_count[bottle_type] = types_count.get(
                        bottle_type, 0) + 1

                    # Count regions
                    region = bottle.get("region", "Unknown")
                    regions_count[region] = regions_count.get(
                        region, 0) + 1

                    # Track prices and ABVs - remove price check as it doesn't exist
                    if bottle.get("abv"):
                        abvs.append(float(bottle["abv"]))

                # Calculate summary statistics
                avg_price = 0  # Set to 0 since price column doesn't exist
                avg_abv = round(np.mean(abvs), 1) if abvs else 0

                # Find favorite type and region
                favorite_type = max(types_count.items(), key=lambda x: x[1])[
                    0] if types_count else "Unknown"
                favorite_region = max(regions_count.items(), key=lambda x: x[1])[
                    0] if regions_count else "Unknown"

                logger.debug(
                    f"Analysis complete: {len(types_count)} types, {len(regions_count)} regions")
                return {
                    "bottle_count": len(bottles),
                    "types": types_count,
                    "regions": regions_count,
                    "avg_price": avg_price,
                    "avg_abv": avg_abv,
                    "favorite_type": favorite_type,
                    "favorite_region": favorite_region
                } 