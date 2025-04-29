"""
Bob - Whisky Expert AI Agent
-----------------------------------
An AI assistant that analyzes users' virtual bars within the BAXUS ecosystem
to provide personalized bottle recommendations and collection insights.

Bob uses pgvector for similarity searches and LangChain/LangGraph for the conversational interface.

Dependencies:
    pip install langchain langchain-openai langgraph psycopg python-dotenv pandas numpy aiohttp
"""

import os
import json
import logging
import asyncio
import aiohttp
import time
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
from dotenv import load_dotenv

# Add colored logging
import colorlog

# LangChain components
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Import local modules
from state.agent_state import AgentState
from utils.bottle_utils import BottleUtils
from utils.db_utils import DatabaseUtils
from utils.error_utils import ErrorUtils
from utils.logging_utils import get_logger
from utils.analysis_utils import generate_taste_profile, calculate_investment_stats, extract_collection_kpis
from recommender.bottle_recommender import BottleRecommender
from knowledge.retriever import BobKnowledgeRetriever
from agent.conversation_graph import ConversationGraph

# Import database utilities at the module level
try:
    import psycopg
    import numpy as np
    DB_IMPORTS_AVAILABLE = True
except ImportError:
    DB_IMPORTS_AVAILABLE = False

# Set up logging
logger = get_logger(__name__)

# Custom log levels for RAG bottle data
BOTTLE_RAG = 15  # Between DEBUG and INFO
logging.addLevelName(BOTTLE_RAG, "BOTTLE_RAG")

# Custom log levels for missing image/nft data
MISSING_DATA = 25  # Between INFO and WARNING
logging.addLevelName(MISSING_DATA, "MISSING_DATA")

# Add custom logging methods
def bottle_rag(self, message, *args, **kwargs):
    """Log RAG bottle data retrieval at custom level"""
    self.log(BOTTLE_RAG, message, *args, **kwargs)

def missing_data(self, message, *args, **kwargs):
    """Log missing bottle data with red color"""
    # Use ERROR level color (red) but with custom level name
    self._log(MISSING_DATA, "\033[31m" + message + "\033[0m", args, **kwargs)

# Add the custom methods to the logger class
logging.Logger.bottle_rag = bottle_rag
logging.Logger.missing_data = missing_data

# Load environment variables
load_dotenv()
logger.debug("Environment variables loaded")


class Bob:
    """
    Bob - The BAXUS whisky recommendation agent

    An AI assistant that analyzes users' virtual bars within the BAXUS ecosystem
    to provide personalized bottle recommendations and collection insights.
    """

    # System prompt for Bob's whisky expertise
    SYSTEM_PROMPT = """You are Bob, a whisky expert AI assistant for the BAXUS app.
    Answer the user's question about whisky accurately, helpfully, and conversationally.
    
    You are incredibly knowledgeable about whisky types (Bourbon, Scotch, Rye, etc.), 
    regions, production methods, flavor profiles, and tasting techniques.
    
    EXTREMELY IMPORTANT INSTRUCTIONS FOR BOTTLE IMAGES:
    1. ALWAYS use the EXACT image URL provided for each specific bottle
    2. NEVER mix up images between different bottles - each bottle must display its own correct image
    3. If recommending "Nikka Yoichi", show the Nikka Yoichi image, not any other bottle's image
    4. NEVER modify image URLs - use them EXACTLY as provided
    5. Always double-check that you're using the image that belongs to the bottle you're describing
    6. NEVER create or fabricate image URLs if they're missing - only use URLs from the database
    7. If a bottle has no image URL provided, DO NOT include an image in your recommendation
    
    EXTREMELY IMPORTANT INSTRUCTIONS FOR NFT/ASSET ADDRESSES:
    1. NEVER make up or invent NFT addresses - only use the exact addresses from the database
    2. If a bottle has no NFT address, DO NOT include a BAXUS link in your recommendation
    3. NEVER substitute a missing NFT address with another bottle's address
    4. Only show "View on BAXUS" links when you have a genuine NFT address from the database
    
    When recommending bottles, ALWAYS USE MARKDOWN FORMATTING:
    1. ONLY use exact image URLs from the database - NEVER create image URLs
    2. ONLY include a link to the BAXUS asset when a specific NFT address is available
    3. NEVER modify image URLs - copy the exact URL as-is, with no changes whatsoever
    4. If you receive an image URL like "https://assets.baxus.co/556/556.jpg", use EXACTLY that URL
    
    Format your recommendations using Markdown:
    ## [Bottle Name]
    [Description of the bottle]
    ![Bottle Image](exact image URL from database) - ONLY include if an image URL is provided
    [View on BAXUS](https://baxus.co/asset/NFT_ADDRESS) - ONLY if NFT address is available
    
    DO NOT include a link if no NFT address is available.
    DO NOT modify, change or create image URLs - use exactly what is provided from the database.
    DO NOT use placeholder image URLs or make up image URLs.
    DO NOT make up any bottle data - only use information you have from the database.

    ABSOLUTELY DO NOT answer questions unrelated to whisky.
    """

    def __init__(self):
        """Initialize Bob with the necessary components."""
        logger.debug("Initializing Bob")
        self.recommender = BottleRecommender()
        self.knowledge_retriever = BobKnowledgeRetriever()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.error("OPENAI_API_KEY environment variable is missing")
            raise ValueError("OPENAI_API_KEY environment variable is required")

        # Initialize LangChain components
        self.llm = ChatOpenAI(temperature=0.7)
        self.baxus_api_url = os.getenv(
            "BAXUS_API_URL", "https://services.baxus.co/api")

        # Initialize conversation graph
        self.conversation_graph = ConversationGraph(self.llm)

        # Initialize bottle cache to prevent redundant database queries
        self.bottle_cache = {}  # Key: bottle_id or name, Value: bottle data with image_url and nft_address

        logger.debug("Bob initialization complete")

    @ErrorUtils.handle_async_exceptions(default_return=[])
    async def fetch_user_collection(self, username: str) -> List[Dict[str, Any]]:
        """Fetch a user's collection from the BAXUS API."""
        logger.debug(f"Fetching collection for user: {username}")

        async with aiohttp.ClientSession() as session:
            url = f"{self.baxus_api_url}/bar/user/{username}"
            logger.debug(f"Making API request to: {url}")

            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"API request failed with status {response.status}: {await response.text()}")
                    return []

                data = await response.json()
                logger.debug(
                    f"Received data for {len(data)} bottles from API")

                # Enhance the collection data with complete bottle information from database
                enhanced_data = await self.enhance_collection_with_rag(data)
                return enhanced_data

    @ErrorUtils.handle_async_exceptions(default_return=[])
    async def enhance_collection_with_rag(self, collection_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance collection data with complete bottle information from the vector database."""
        logger.debug(
            f"Enhancing collection data with RAG for {len(collection_data)} bottles")

        enhanced_collection = []

        if not DB_IMPORTS_AVAILABLE:
            logger.error("Required database modules not available")
            return collection_data

        async with await DatabaseUtils.get_connection() as conn:
            for item in collection_data:
                product = item.get("product", {})
                product_id = product.get("id")
                product_name = product.get("name", "")

                # Log the bottle being processed
                logger.bottle_rag(f"Processing collection bottle: {product_name} (ID: {product_id})")
                
                # Check if bottle data is already in cache
                cache_key = str(product_name) if product_name else str(product_id)
                if cache_key and cache_key in self.bottle_cache:
                    # Use cached data
                    item["product"].update(self.bottle_cache[cache_key])
                    enhanced_collection.append(item)
                    logger.debug(f"Used cached data for bottle: {cache_key}")
                    continue

                # Prioritize name-based lookup
                bottle_data = None
                if product_name:
                    # First try to get bottle by name only
                    bottle_data = await BottleUtils.get_bottle_by_id_or_name(
                        bottle_id=None,
                        bottle_name=product_name
                    )
                    if bottle_data:
                        logger.debug(f"Found bottle data by name: {product_name}")
                
                # Fall back to ID if name lookup failed
                if not bottle_data and product_id:
                    bottle_data = await BottleUtils.get_bottle_by_id_or_name(
                        bottle_id=product_id,
                        bottle_name=None
                    )
                    if bottle_data:
                        logger.debug(f"Found bottle data by ID: {product_id}")

                if bottle_data:
                    # Update the product with detailed information
                    item["product"].update(bottle_data)
                    
                    # Check for missing image or NFT address
                    if not bottle_data.get("image_url"):
                        logger.missing_data(f"Missing image_url for bottle: {product_name}")
                    if not bottle_data.get("nft_address"):
                        logger.missing_data(f"Missing nft_address for bottle: {product_name}")
                    
                    # Cache the bottle data for future use
                    if cache_key:
                        self.bottle_cache[cache_key] = bottle_data
                else:
                    logger.missing_data(f"No bottle data found for: {product_name} (ID: {product_id})")

                enhanced_collection.append(item)

        logger.debug(
            f"Successfully enhanced {len(enhanced_collection)} bottles with RAG")
        return enhanced_collection

    @ErrorUtils.handle_async_exceptions(default_return={})
    async def analyze_user_bar(self, username: str) -> Dict[str, Any]:
        """Analyze a user's bar data to extract bottle IDs and preferences."""
        # Fetch user data from API
        user_data = await self.fetch_user_collection(username)

        logger.debug(f"Analyzing user bar with {len(user_data)} items")
        bottle_ids = []
        for item in user_data:
            product = item.get("product", {})
            if product_id := product.get("id"):
                bottle_ids.append(product_id)

        logger.debug(f"Extracted {len(bottle_ids)} bottle IDs from user data")
        # Get collection analysis
        analysis = await self.recommender.analyze_collection(bottle_ids)

        # Include the raw user data for investment calculations
        analysis["raw_data"] = user_data

        return {
            "bottle_ids": bottle_ids,
            "analysis": analysis
        }

    @ErrorUtils.handle_async_exceptions(default_return={})
    async def get_user_profile(self, username: str) -> Dict[str, Any]:
        """Get a user's profile including taste preferences and collection stats."""
        logger.debug(f"Getting profile for user: {username}")

        # Fetch and analyze the user's collection
        analysis_result = await self.analyze_user_bar(username)
        analysis = analysis_result.get("analysis", {})
        
        # Get the raw collection data 
        user_collection = analysis.get("raw_data", [])

        # Generate taste profile based on collection analysis
        taste_profile = generate_taste_profile(username, analysis)

        # Calculate investment statistics
        investment_stats = calculate_investment_stats(analysis)

        # Extract and format collection KPIs
        collection_kpis = extract_collection_kpis(analysis)

        # Ensure consistency between collection KPIs and investment stats
        if "bottle_count" in investment_stats and "bottle_count" in collection_kpis:
            # Use the more accurate count from investment_stats if available
            collection_kpis["bottle_count"] = investment_stats["bottle_count"]

        # Format the bottles data to include only the relevant information
        bottles = []
        for item in user_collection:
            product = item.get("product", {})
            bottle = {
                "id": product.get("id"),
                "name": product.get("name"),
                "image_url": product.get("image_url"),
                "type": product.get("type"),
                "spirit": product.get("spirit_type") or product.get("spirit"),
                "price": product.get("shelf_price") or product.get("price")
            }
            bottles.append(bottle)

        return {
            "username": username,
            "taste_profile": taste_profile,
            "collection": {
                "stats": collection_kpis,
                "investment": investment_stats,
                "bottles": bottles
            }
        }

    @ErrorUtils.handle_async_exceptions(default_return={"similar": [], "diverse": []})
    async def get_recommendations(
        self,
        username: str,
        num_similar: int = 3,
        num_diverse: int = 2,
        diversity_ratio: float = 0.4
    ) -> Dict[str, Any]:
        """Get both similar and diverse recommendations for a user's collection using RAG."""
        logger.debug(f"Getting recommendations for user: {username}")
        
        # Normalize diversity_ratio to be between 0 and 1
        diversity_ratio = max(0.0, min(1.0, diversity_ratio))
        logger.debug(f"Using normalized diversity_ratio: {diversity_ratio}")
        
        # Calculate total number of recommendations
        total_recommendations = num_similar + num_diverse
        
        # Adjust the number of similar and diverse recommendations based on diversity_ratio
        # diversity_ratio = 0 means all similar, diversity_ratio = 1 means all diverse
        adjusted_num_diverse = max(1, round(total_recommendations * diversity_ratio))
        adjusted_num_similar = total_recommendations - adjusted_num_diverse
        
        # Ensure at least one of each type when possible
        if total_recommendations > 1:
            if adjusted_num_similar == 0:
                adjusted_num_similar = 1
                adjusted_num_diverse = total_recommendations - 1
            elif adjusted_num_diverse == 0:
                adjusted_num_diverse = 1
                adjusted_num_similar = total_recommendations - 1
        
        logger.debug(f"Adjusted recommendation counts: {adjusted_num_similar} similar, {adjusted_num_diverse} diverse")

        # Fetch the user's collection from the API
        user_data = await self.fetch_user_collection(username)
        if not user_data:
            logger.error(f"No collection data found for user: {username}")
            return {"error": f"No collection data found for user: {username}"}

        logger.debug(f"Processing recommendations with {len(user_data)} items")

        if not DB_IMPORTS_AVAILABLE:
            raise ImportError("Required database modules not available")

        # Extract bottle IDs from user data
        bottle_ids = []
        bottle_types = set()
        bottle_regions = set()

        for item in user_data:
            product = item.get("product", {})
            if product_id := product.get("id"):
                bottle_ids.append(product_id)

            # Track bottle types and regions for diverse recommendations
            if bottle_type := product.get("type"):
                bottle_types.add(bottle_type)
            if region := product.get("region"):
                bottle_regions.add(region)

        if not bottle_ids:
            logger.warning(
                f"No valid bottle IDs found for user: {username}")
            return {"similar": [], "diverse": []}

        similar_recs = []
        diverse_recs = []

        async with await DatabaseUtils.get_connection() as conn:
            async with conn.cursor() as cur:
                # Get embeddings for user's bottles
                collection_embeddings = []
                for bottle_id in bottle_ids:
                    await cur.execute(
                        "SELECT embedding FROM bottles WHERE id = %s OR baxus_class_id = %s",
                        (bottle_id, str(bottle_id))
                    )
                    result = await cur.fetchone()
                    if result and result["embedding"]:
                        collection_embeddings.append(result["embedding"])

                if not collection_embeddings:
                    logger.warning(
                        f"No embeddings found for user's bottles: {username}")
                    return {"similar": [], "diverse": []}

                # Convert string embeddings to numeric arrays if needed
                numeric_embeddings = []
                for emb in collection_embeddings:
                    if isinstance(emb, str):
                        # Clean the string and split into values
                        clean_emb = emb.strip('[]()').replace(' ', '')
                        values = [float(val) for val in clean_emb.split(',') if val]
                        numeric_embeddings.append(np.array(values, dtype=np.float32))
                    else:
                        numeric_embeddings.append(np.array(emb, dtype=np.float32))

                # Calculate average embedding of user's collection
                avg_embedding = np.mean(numeric_embeddings, axis=0).tolist()

                # Get similar recommendations using the utility
                similar_results = await DatabaseUtils.search_by_similarity(
                    conn,
                    embedding=avg_embedding,
                    exclude_ids=bottle_ids,
                    limit=adjusted_num_similar
                )

                logger.bottle_rag(f"RAG fetched {len(similar_results)} similar bottles based on collection embedding")

                for bottle_data in similar_results:
                    # Create enhanced reason using description if available
                    region = bottle_data.get("region", "").strip()
                    bottle_type = bottle_data.get("type", "").strip()
                    producer = bottle_data.get("producer", "").strip()
                    
                    # Create more specific reason based on bottle attributes
                    if bottle_type and region:
                        base_reason = f"If you enjoy {bottle_type} from {region}, you'll appreciate this bottle"
                    elif bottle_type and producer:
                        base_reason = f"Based on your collection, this {producer} {bottle_type} matches your taste"
                    elif bottle_type:
                        base_reason = f"Complements your {bottle_type} collection"
                    elif producer:
                        base_reason = f"From {producer}, similar to bottles you already enjoy"
                    else:
                        base_reason = "Similar profile to bottles in your collection"
                    
                    if description := bottle_data.get("description"):
                        # Extract a meaningful snippet from the description (first 100 chars or so)
                        snippet = description.strip()[:120]
                        if len(snippet) == 120:
                            snippet += "..."
                        # Combine base reason with description snippet
                        bottle_data["reason"] = f"{base_reason}. {snippet}"
                    else:
                        bottle_data["reason"] = base_reason
                    
                    # Cache bottle info and ensure image and NFT details are included
                    bottle_id = bottle_data.get("id")
                    bottle_name = bottle_data.get("name", "")
                    
                    # Log the RAG fetch
                    logger.bottle_rag(f"RAG bottle: {bottle_name} (ID: {bottle_id})")
                    
                    # Check if we already have this bottle in cache
                    cache_key = str(bottle_name) if bottle_name else str(bottle_id)
                    if cache_key and cache_key in self.bottle_cache:
                        # Use cached image_url and nft_address if available
                        cached_bottle = self.bottle_cache[cache_key]
                        if "image_url" in cached_bottle and "image_url" not in bottle_data:
                            bottle_data["image_url"] = cached_bottle["image_url"]
                        if "nft_address" in cached_bottle and "nft_address" not in bottle_data:
                            bottle_data["nft_address"] = cached_bottle["nft_address"]
                    elif bottle_name:
                        # Get exact bottle data if not in cache
                        exact_bottle = await DatabaseUtils.get_exact_bottle_data_by_name(conn, bottle_name)
                        if exact_bottle:
                            # Update bottle with image_url and nft_address from exact match
                            if exact_bottle.get("image_url"):
                                bottle_data["image_url"] = exact_bottle["image_url"]
                                logger.debug(f"Using exact image for '{bottle_name}' via direct name lookup")
                            else:
                                logger.missing_data(f"Missing image_url for '{bottle_name}' in exact data lookup")
                                
                            if exact_bottle.get("nft_address"):
                                bottle_data["nft_address"] = exact_bottle["nft_address"]
                            else:
                                logger.missing_data(f"Missing nft_address for '{bottle_name}' in exact data lookup")
                            
                            # Cache the result for future use
                            self.bottle_cache[bottle_name] = {
                                "image_url": exact_bottle.get("image_url"),
                                "nft_address": exact_bottle.get("nft_address"),
                                "cached_at": time.time()
                            }
                            if bottle_id:
                                self.bottle_cache[str(bottle_id)] = self.bottle_cache[bottle_name]
                    
                    # Check if bottle is missing data after all lookups
                    if not bottle_data.get("image_url"):
                        logger.missing_data(f"Still missing image_url for bottle: {bottle_name}")
                    if not bottle_data.get("nft_address"):
                        logger.missing_data(f"Still missing nft_address for bottle: {bottle_name}")
                    
                    similar_recs.append(bottle_data)

                # Get diverse recommendations - find bottles different from user's collection
                # Different approach for diversity
                type_clause = ""
                if bottle_types:
                    type_clause = f"type NOT IN ({', '.join(['%s'] * len(bottle_types))})"

                extra_conditions = type_clause

                # Convert bottle_ids to strings for consistent comparison
                str_bottle_ids = [str(bid) for bid in bottle_ids]
                
                # Make sure we select all the necessary fields for a complete bottle record
                await cur.execute(f"""
                    SELECT id, name, description, type, image_url, producer, region,
                           country, spirit_type, nft_address 
                    FROM bottles 
                    WHERE id::text != ALL(%s) AND baxus_class_id != ALL(%s) {' AND ' if type_clause else ''}{type_clause}
                    ORDER BY RANDOM()
                    LIMIT %s
                """, (str_bottle_ids, str_bottle_ids, *bottle_types, adjusted_num_diverse))

                diverse_results = await cur.fetchall()
                logger.bottle_rag(f"RAG fetched {len(diverse_results)} diverse bottles by different types")
                
                # Get column names from cursor description
                col_names = [desc[0] for desc in cur.description]
                
                # Convert each row tuple to a dictionary using column names
                for row in diverse_results:
                    bottle_data = {}
                    for i, col_name in enumerate(col_names):
                        bottle_data[col_name] = row[col_name]
                    
                        # Create enhanced reason using description
                        bottle_type = bottle_data.get("type", "").strip()
                        if bottle_type:
                            base_reason = f"Adds {bottle_type} to your collection"
                        else:
                            base_reason = "Expands your whisky selection"
                            
                        if description := bottle_data.get("description"):
                            # Extract a meaningful snippet from the description
                            snippet = description.strip()[:120]
                            if len(snippet) == 120:
                                snippet += "..."
                            bottle_data["reason"] = f"{base_reason}. {snippet}"
                        else:
                            bottle_data["reason"] = base_reason
                    
                    # Log the RAG fetch
                    bottle_id = bottle_data.get("id")
                    bottle_name = bottle_data.get("name", "")
                    logger.bottle_rag(f"RAG diverse bottle (type): {bottle_name} (ID: {bottle_id})")
                    
                    # Check for missing data
                    if not bottle_data.get("image_url"):
                        logger.missing_data(f"Missing image_url for diverse bottle (type): {bottle_name}")
                    if not bottle_data.get("nft_address"):
                        logger.missing_data(f"Missing nft_address for diverse bottle (type): {bottle_name}")
                    
                    # Cache bottle data for future use
                    cache_key = str(bottle_name) if bottle_name else str(bottle_id)
                    cache_data = {
                        "image_url": bottle_data.get("image_url"),
                        "nft_address": bottle_data.get("nft_address"),
                        "cached_at": time.time()
                    }

                    if bottle_name:
                        self.bottle_cache[bottle_name] = cache_data
                    if bottle_id:
                        self.bottle_cache[str(bottle_id)] = cache_data
                    
                    diverse_recs.append(bottle_data)

                # If we don't have enough diverse recommendations, try bottles from different regions
                if len(diverse_recs) < adjusted_num_diverse and bottle_regions:
                    region_clause = f"region NOT IN ({', '.join(['%s'] * len(bottle_regions))})"
                    
                    # Convert IDs to strings for the diverse_recs also
                    diverse_rec_ids = [str(rec.get('id')) for rec in diverse_recs]
                    
                    await cur.execute(f"""
                        SELECT id, name, description, type, image_url, producer, region,
                               country, spirit_type, nft_address 
                        FROM bottles 
                        WHERE id::text != ALL(%s) AND baxus_class_id != ALL(%s) {' AND ' if region_clause else ''}{region_clause}
                        AND id::text != ALL(%s)  -- exclude already recommended bottles
                        ORDER BY RANDOM()
                        LIMIT %s
                    """, (
                        str_bottle_ids,
                        str_bottle_ids,
                        *bottle_regions,
                        diverse_rec_ids,
                        adjusted_num_diverse - len(diverse_recs)
                    ))

                    additional_diverse = await cur.fetchall()
                    logger.bottle_rag(f"RAG fetched {len(additional_diverse)} additional diverse bottles by region")
                    
                    # Convert each row tuple to a dictionary using column names
                    for row in additional_diverse:
                        bottle_data = {}
                        for i, col_name in enumerate(col_names):
                            bottle_data[col_name] = row[col_name]
                        
                        # Create enhanced reason using description
                        region = bottle_data.get("region", "").strip()
                        bottle_type = bottle_data.get("type", "").strip()
                        
                        if region:
                            base_reason = f"Introduces {bottle_type} from {region} to your collection"
                        elif bottle_type:
                            base_reason = f"Adds {bottle_type} to your collection"
                        else:
                            base_reason = "Expands your whisky selection"
                            
                        if description := bottle_data.get("description"):
                            # Extract a meaningful snippet from the description
                            snippet = description.strip()[:120]
                            if len(snippet) == 120:
                                snippet += "..."
                            bottle_data["reason"] = f"{base_reason}. {snippet}"
                        else:
                            bottle_data["reason"] = base_reason
                        
                        # Log the RAG fetch
                        bottle_id = bottle_data.get("id")
                        bottle_name = bottle_data.get("name", "")
                        logger.bottle_rag(f"RAG diverse bottle (region): {bottle_name} (ID: {bottle_id})")
                        
                        # Check for missing data
                        if not bottle_data.get("image_url"):
                            logger.missing_data(f"Missing image_url for diverse bottle (region): {bottle_name}")
                        if not bottle_data.get("nft_address"):
                            logger.missing_data(f"Missing nft_address for diverse bottle (region): {bottle_name}")
                        
                        # Cache bottle data for future use
                        cache_key = str(bottle_name) if bottle_name else str(bottle_id)
                        cache_data = {
                            "image_url": bottle_data.get("image_url"),
                            "nft_address": bottle_data.get("nft_address"),
                            "cached_at": time.time()
                        }
                        
                        if bottle_name:
                            self.bottle_cache[bottle_name] = cache_data
                        if bottle_id:
                            self.bottle_cache[str(bottle_id)] = cache_data
                        
                        diverse_recs.append(bottle_data)

                # If we still don't have enough diverse recommendations, add fallback recommendations
                # This is to ensure we always return the requested number of diverse recommendations
                if len(diverse_recs) < adjusted_num_diverse:
                    remaining_needed = adjusted_num_diverse - len(diverse_recs)
                    logger.warning(f"Not enough diverse recommendations found. Adding {remaining_needed} fallback recommendations.")
                    
                    await cur.execute("""
                        SELECT id, name, description, type, image_url, producer, region,
                               country, spirit_type, nft_address 
                        FROM bottles 
                        WHERE id::text != ALL(%s) AND baxus_class_id != ALL(%s)
                        ORDER BY RANDOM()
                        LIMIT %s
                    """, (str_bottle_ids, str_bottle_ids, remaining_needed))
                    
                    fallback_results = await cur.fetchall()
                    logger.bottle_rag(f"RAG fetched {len(fallback_results)} fallback diverse bottles")
                    
                    # Convert each row tuple to a dictionary using column names
                    for row in fallback_results:
                        bottle_data = {}
                        for i, col_name in enumerate(col_names):
                            bottle_data[col_name] = row[col_name]
                        
                        # Create enhanced reason using description
                        region = bottle_data.get("region", "").strip()
                        bottle_type = bottle_data.get("type", "").strip()
                        producer = bottle_data.get("producer", "").strip()
                        
                        # Create a more varied reason based on available data
                        if region and producer:
                            base_reason = f"Try this {bottle_type} from {producer} in {region}"
                        elif region:
                            base_reason = f"Discover {region} whisky with this {bottle_type}"
                        elif producer:
                            base_reason = f"Experience {producer}'s approach to {bottle_type}"
                        elif bottle_type:
                            base_reason = f"Add variety with this {bottle_type}"
                        else:
                            base_reason = "Explore a new addition to your collection"
                            
                        if description := bottle_data.get("description"):
                            # Extract a meaningful snippet from the description
                            snippet = description.strip()[:120]
                            if len(snippet) == 120:
                                snippet += "..."
                            bottle_data["reason"] = f"{base_reason}. {snippet}"
                        else:
                            bottle_data["reason"] = base_reason
                        
                        # Log the RAG fetch
                        bottle_id = bottle_data.get("id")
                        bottle_name = bottle_data.get("name", "")
                        logger.bottle_rag(f"RAG fallback bottle: {bottle_name} (ID: {bottle_id})")
                        
                        # Check for missing data
                        if not bottle_data.get("image_url"):
                            logger.missing_data(f"Missing image_url for fallback bottle: {bottle_name}")
                        if not bottle_data.get("nft_address"):
                            logger.missing_data(f"Missing nft_address for fallback bottle: {bottle_name}")
                        
                        # Cache bottle data for future use
                        cache_key = str(bottle_name) if bottle_name else str(bottle_id)
                        cache_data = {
                            "image_url": bottle_data.get("image_url"),
                            "nft_address": bottle_data.get("nft_address"),
                            "cached_at": time.time()
                        }
                        
                        if bottle_name:
                            self.bottle_cache[bottle_name] = cache_data
                        if bottle_id:
                            self.bottle_cache[str(bottle_id)] = cache_data
                        
                        diverse_recs.append(bottle_data)

        logger.debug(
            f"Recommendations complete: {len(similar_recs)} similar, {len(diverse_recs)} diverse")
        
        # Ensure we have the requested number of recommendations
        similar_recs = similar_recs[:adjusted_num_similar]
        diverse_recs = diverse_recs[:adjusted_num_diverse]
        
        # Check for any recommendations still missing image_url or nft_address
        # This is much more efficient now since we've cached data during fetching
        missing_data_bottles = []
        for rec in [*similar_recs, *diverse_recs]:
            if not rec.get("image_url") or not rec.get("nft_address"):
                missing_data_bottles.append(rec)
        
        # Only enhance recommendations if we still have bottles missing data
        if missing_data_bottles:
            logger.debug(f"Found {len(missing_data_bottles)} bottles still missing image or NFT data")
            await self._enhance_recommendation_data(missing_data_bottles)
        
        return {
            "similar": similar_recs,
            "diverse": diverse_recs
        }

    @ErrorUtils.handle_async_exceptions()
    async def _enhance_recommendation_data(self, recommendations: List[Dict[str, Any]]) -> None:
        """Add image URLs and NFT links to recommendations if not already present."""
        logger.debug(
            f"Enhancing recommendation data for {len(recommendations)} items")

        if not DB_IMPORTS_AVAILABLE:
            logger.warning(
                "Required database modules not available for enhancement")
            return

        bottles_needing_data = []
        for idx, rec in enumerate(recommendations):
            # Skip if both image_url and nft_address are already present and valid
            if "image_url" in rec and rec["image_url"] and "nft_address" in rec:
                continue
                
            bottle_id = rec.get("id")
            bottle_name = rec.get("name", "")
            
            # Check if bottle data is already in cache
            cache_key = str(bottle_name) if bottle_name else str(bottle_id)
            if cache_key and cache_key in self.bottle_cache:
                # Use cached data
                cached_bottle = self.bottle_cache[cache_key]
                
                # Update the recommendation with cached image_url and nft_address
                if "image_url" in cached_bottle and ("image_url" not in rec or not rec["image_url"]):
                    rec["image_url"] = cached_bottle["image_url"]
                    logger.debug(f"Added cached image_url for '{cache_key}'")
                else:
                    if "image_url" not in rec or not rec["image_url"]:
                        logger.missing_data(f"Missing image_url in cache for '{cache_key}'")
                
                if "nft_address" in cached_bottle and ("nft_address" not in rec or not rec["nft_address"]):
                    rec["nft_address"] = cached_bottle["nft_address"]
                    logger.debug(f"Added cached nft_address for '{cache_key}'")
                else:
                    if "nft_address" not in rec or not rec["nft_address"]:
                        logger.missing_data(f"Missing nft_address in cache for '{cache_key}'")
            else:
                # Need to fetch this bottle's data
                bottles_needing_data.append((idx, bottle_id, bottle_name))
                logger.bottle_rag(f"Bottle needs enhancement: {bottle_name} (ID: {bottle_id})")
        
        # If no bottles need fetching, we're done
        if not bottles_needing_data:
            return
            
        # Fetch bottle data for any bottles not in cache
        async with await DatabaseUtils.get_connection() as conn:
            for idx, bottle_id, bottle_name in bottles_needing_data:
                rec = recommendations[idx]
                
                if bottle_name:
                    # Use direct name lookup instead of RAG for more accurate image matching
                    bottle_data = await DatabaseUtils.get_exact_bottle_data_by_name(conn, bottle_name)
                    
                    if bottle_data:
                        # Update with image_url and nft_address from exact name match
                        image_url = bottle_data.get("image_url")
                        nft_address = bottle_data.get("nft_address")
                        
                        if image_url and ("image_url" not in rec or not rec["image_url"]):
                            rec["image_url"] = image_url
                            logger.debug(f"Added image_url for bottle '{bottle_name}' via direct name lookup: {image_url}")
                        else:
                            logger.missing_data(f"No image_url found for bottle '{bottle_name}' via direct name lookup")
                            
                        if nft_address and ("nft_address" not in rec or not rec["nft_address"]):
                            rec["nft_address"] = nft_address
                            logger.debug(f"Added nft_address for bottle '{bottle_name}' via direct name lookup")
                        else:
                            logger.missing_data(f"No nft_address found for bottle '{bottle_name}' via direct name lookup")
                        
                        # Cache the result
                        cache_key = str(bottle_name) if bottle_name else str(bottle_id)
                        cache_data = {
                            "image_url": image_url,
                            "nft_address": nft_address,
                            "cached_at": time.time()
                        }
                        
                        self.bottle_cache[cache_key] = cache_data
                        
                        # If we have a name, cache by name too
                        if bottle_name:
                            self.bottle_cache[bottle_name] = cache_data
                        if bottle_id:
                            self.bottle_cache[str(bottle_id)] = cache_data
                        # We found data by name, continue to next bottle
                        continue
                
                # Fall back to ID lookup if name lookup failed or if we only have ID
                if bottle_id:
                    query = """
                        SELECT id, name, image_url, nft_address 
                        FROM bottles 
                        WHERE id = %s OR baxus_class_id = %s
                        LIMIT 1
                    """
                    params = (bottle_id, str(bottle_id))
                    
                    async with conn.cursor() as cur:
                        await cur.execute(query, params)
                        result = await cur.fetchone()
                        
                        if result:
                            # Update the recommendation with image_url and nft_address if available
                            image_url = result.get("image_url")
                            nft_address = result.get("nft_address")
                            
                            if image_url and ("image_url" not in rec or not rec["image_url"]):
                                rec["image_url"] = image_url
                                logger.debug(f"Added image_url for bottle ID {bottle_id}: {image_url}")
                            else:
                                logger.missing_data(f"No image_url found for bottle ID {bottle_id}")
                            
                            if nft_address and ("nft_address" not in rec or not rec["nft_address"]):
                                rec["nft_address"] = nft_address
                                logger.debug(f"Added nft_address for bottle ID {bottle_id}")
                            else:
                                logger.missing_data(f"No nft_address found for bottle ID {bottle_id}")
                            
                            # Cache the fetched data for future use
                            cache_key = str(bottle_name) if bottle_name else str(bottle_id)
                            cache_data = {
                                "image_url": image_url,
                                "nft_address": nft_address,
                                "cached_at": time.time()
                            }
                            
                            self.bottle_cache[cache_key] = cache_data
                            
                            # If we have a name, cache by name too
                            bottle_name = result.get("name")
                            if bottle_name:
                                self.bottle_cache[bottle_name] = cache_data
                        else:
                            logger.missing_data(f"No data found for bottle ID: {bottle_id}, name: {bottle_name}")

        # Verify all bottles have image URLs after enhancement
        missing_images = [rec.get("name", "Unknown") for rec in recommendations if "image_url" not in rec or not rec["image_url"]]
        if missing_images:
            logger.missing_data(f"After enhancement, {len(missing_images)} bottles still missing images: {', '.join(missing_images[:5])}")
        else:
            logger.debug("All recommendations now have image_url data")
            
        # Verify all bottles have NFT addresses after enhancement
        missing_nft = [rec.get("name", "Unknown") for rec in recommendations if "nft_address" not in rec or not rec["nft_address"]]
        if missing_nft:
            logger.missing_data(f"After enhancement, {len(missing_nft)} bottles still missing NFT addresses: {', '.join(missing_nft[:5])}")
        
        logger.debug(f"Successfully enhanced recommendation data using direct database lookups")

    @ErrorUtils.handle_async_exceptions(default_return={})
    async def get_bottle_knowledge(self, bottle_name: str) -> Dict[str, Any]:
        """Retrieve knowledge about a specific bottle using RAG from the pgvector database."""
        logger.debug(f"Retrieving knowledge for bottle: {bottle_name}")

        if not DB_IMPORTS_AVAILABLE:
            logger.error("Required database modules not available")
            return {}

        # Check if bottle data is already cached
        cache_key = bottle_name.lower()
        if cache_key in self.bottle_cache:
            logger.debug(f"Using cached data for bottle: {bottle_name}")
            return self.bottle_cache[cache_key]

        # First try to find by exact name match
        bottle_data = await BottleUtils.get_bottle_by_id_or_name(
            bottle_name=bottle_name
        )

        if bottle_data:
            # Verify image URL and NFT address
            if not bottle_data.get("image_url"):
                logger.missing_data(f"Database entry missing image_url for exact match: {bottle_name}")
            if not bottle_data.get("nft_address"):
                logger.missing_data(f"Database entry missing nft_address for exact match: {bottle_name}")
            
            # Cache the result for future use
            self.bottle_cache[cache_key] = bottle_data
            return bottle_data

        # If no exact match, try to find similar bottles using the vector embedding
        # Generate an embedding for the bottle name
        embedding = BottleUtils.get_embedding(bottle_name)

        if embedding:
            async with await DatabaseUtils.get_connection() as conn:
                similar_bottles = await DatabaseUtils.search_by_similarity(
                    conn,
                    embedding=embedding,
                    limit=3
                )

                if similar_bottles:
                    # Return the most similar bottle as the main result, with other similar bottles included
                    result = {
                        **similar_bottles[0],
                        "similar_bottles": similar_bottles[1:] if len(similar_bottles) > 1 else []
                    }
                    
                    # Verify image URL and NFT address
                    if not result.get("image_url"):
                        logger.missing_data(f"Similar bottle missing image_url: {result.get('name', bottle_name)}")
                    if not result.get("nft_address"):
                        logger.missing_data(f"Similar bottle missing nft_address: {result.get('name', bottle_name)}")
                    
                    # Cache the result for future use
                    best_match_name = result.get("name", "").lower()
                    if best_match_name:
                        self.bottle_cache[best_match_name] = result
                        
                    # Also cache using the original query for future lookups
                    self.bottle_cache[cache_key] = result
                    
                    return result

        # If we reach here, no bottle was found
        logger.warning(f"No bottle found in database for: {bottle_name}")
        return {}

    async def _format_conversation_history(self, conversation_history=None):
        """Format conversation history into a consistent format."""
        if conversation_history is None:
            return []

        formatted_history = []
        for message in conversation_history:
            if message["role"] in ["user", "human"]:
                formatted_history.append(
                    {"role": "human", "content": message["content"]})
            elif message["role"] in ["assistant", "ai"]:
                formatted_history.append(
                    {"role": "ai", "content": message["content"]})
        return formatted_history

    async def _prepare_conversation_context(self, user_input, username=None):
        """Prepare the conversation context and bottle recommendations.
        
        Returns:
            Tuple containing: (context_string, bottle_recommendations_list)
        """
        # Initialize context
        rag_context = ""
        bottle_recommendations = []

        # Get bottle knowledge
        logger.info("TOOL: Using bottle name extraction")
        bottle_names = await self.extract_bottle_names(user_input)
        for bottle_name in bottle_names:
            logger.info(f"TOOL: Retrieving bottle knowledge for '{bottle_name}'")
            knowledge = await self.get_bottle_knowledge(bottle_name)
            if knowledge:
                # Validate image URL is present, log if missing
                if not knowledge.get("image_url"):
                    logger.missing_data(f"Missing image_url for detected bottle: {bottle_name}")
                else:
                    logger.debug(f"Found valid image_url for bottle '{bottle_name}': {knowledge.get('image_url')}")
                    
                # Validate NFT address is present, log if missing
                if not knowledge.get("nft_address"):
                    logger.missing_data(f"Missing nft_address for detected bottle: {bottle_name}")
                
                rag_context += f"\nInformation about {bottle_name}:\n"
                rag_context += json.dumps(knowledge, indent=2)
                rag_context += "\n"

        # Get user collection data if username provided
        collection_context = ""
        if username:
            logger.debug(f"Getting data for user: {username}")
            
            # Get profile and recommendations
            logger.info(f"TOOL: Fetching user profile for '{username}'")
            profile = await self.get_user_profile(username)
            
            logger.info(f"TOOL: Generating bottle recommendations for '{username}'")
            recommendations = await self.get_recommendations(
                username, num_similar=3, num_diverse=2, diversity_ratio=0.4
            )

            # Create context about the user's collection
            taste_profile = profile.get("taste_profile", {})
            collection_context = f"User {username}'s collection information:\n"
            collection_context += f"- Dominant flavors: {', '.join(taste_profile.get('dominant_flavors', ['Unknown']))}\n"
            collection_context += f"- Favorite type: {taste_profile.get('favorite_type', 'Unknown')}\n"
            collection_context += f"- Favorite region: {taste_profile.get('favorite_region', 'Unknown')}\n"

            # Add recommendations with rich data
            collection_context += "\nRecommended bottles for this user based on their collection:\n"

            # Process similar bottles
            collection_context += "Similar to their collection:\n"
            for bottle in recommendations.get("similar", [])[:3]:
                # Verify bottle has necessary data for display
                if not bottle.get("image_url"):
                    logger.missing_data(f"Recommendation missing image_url: {bottle.get('name', 'Unknown')}")
                    
                bottle_recommendations.append(bottle)
                collection_context += f"- **{bottle.get('name', 'Unknown')}** ({bottle.get('type', 'Unknown')})\n"
                if 'image_url' in bottle and bottle['image_url']:
                    collection_context += f"  EXACT Image URL: {bottle['image_url']}\n"
                    collection_context += f"  ![{bottle.get('name', 'Bottle image')}]({bottle['image_url']})\n"
                if 'nft_address' in bottle and bottle['nft_address']:
                    collection_context += f"  NFT Address: {bottle['nft_address']}\n"
                    collection_context += f"  [View on BAXUS](https://baxus.co/asset/{bottle['nft_address']})\n"

            # Process diverse bottles
            collection_context += "\nTo diversify their collection:\n"
            for bottle in recommendations.get("diverse", [])[:3]:
                # Verify bottle has necessary data for display
                if not bottle.get("image_url"):
                    logger.missing_data(f"Diverse recommendation missing image_url: {bottle.get('name', 'Unknown')}")
                    
                bottle_recommendations.append(bottle)
                collection_context += f"- **{bottle.get('name', 'Unknown')}** ({bottle.get('type', 'Unknown')})\n"
                if 'image_url' in bottle and bottle['image_url']:
                    collection_context += f"  EXACT Image URL: {bottle['image_url']}\n"
                    collection_context += f"  ![{bottle.get('name', 'Bottle image')}]({bottle['image_url']})\n"
                if 'nft_address' in bottle and bottle['nft_address']:
                    collection_context += f"  NFT Address: {bottle['nft_address']}\n"
                    collection_context += f"  [View on BAXUS](https://baxus.co/asset/{bottle['nft_address']})\n"

        # Combine all context
        context = rag_context
        if collection_context:
            context += "\n" + collection_context if context else collection_context

        return context, bottle_recommendations

    async def _prepare_messages(self, user_input, formatted_history, context):
        """Prepare the message list for the LLM."""
        messages = []

        # Add system message with context
        system_message = self.SYSTEM_PROMPT
        if context:
            system_message += f"\n\nCONTEXT INFORMATION:\n{context}"
        
        messages.append(SystemMessage(content=system_message))

        # Add conversation history
        for msg in formatted_history:
            if msg["role"] == "human":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                messages.append(AIMessage(content=msg["content"]))

        # Add current message
        messages.append(HumanMessage(content=user_input))
        
        return messages

    @ErrorUtils.handle_async_exceptions(default_return="I apologize, but I encountered a technical issue. Please try asking your question again.")
    async def chat(self, user_input, conversation_history=None, username=None):
        """Process a user message and return Bob's response."""
        logger.debug(f"Processing chat: '{user_input[:50]}...'")

        # Format conversation history
        formatted_history = await self._format_conversation_history(conversation_history)

        # Prepare context and recommendations
        context, bottle_recommendations = await self._prepare_conversation_context(user_input, username)

        # Extract detected bottle names from the user input
        detected_bottle_names = await self.extract_bottle_names(user_input)
        
        # Get detailed information for detected bottles
        detected_bottles = []
        for bottle_name in detected_bottle_names:
            bottle_data = await self.get_bottle_knowledge(bottle_name)
            if bottle_data:
                detected_bottles.append(bottle_data)
                logger.debug(f"Added detailed data for detected bottle: {bottle_name}")

        # Validate image URLs for both detected bottles and recommendations
        if detected_bottles:
            detected_bottles = await self.validate_bottle_image_urls(detected_bottles)
        if bottle_recommendations:
            bottle_recommendations = await self.validate_bottle_image_urls(bottle_recommendations)

        # Try to use the conversation graph first
        try:
            # Initialize state
            state = AgentState()
            state.username = username
            state.messages = formatted_history + [{"role": "human", "content": user_input}]
            if context:
                state.context.append(context)
            if bottle_recommendations:
                state.recommendations = bottle_recommendations
            if detected_bottles:
                state.detected_bottles = detected_bottles
                # Add detected bottles to recommendations if they're not already there
                bottle_names_in_recs = {rec.get("name", "").lower() for rec in bottle_recommendations}
                for bottle in detected_bottles:
                    if bottle.get("name", "").lower() not in bottle_names_in_recs:
                        state.recommendations.append(bottle)
                        logger.debug(f"Added detected bottle to recommendations: {bottle.get('name')}")

            # Process through the agent graph
            logger.info("TOOL: Using conversation graph for response generation")
            final_state = self.conversation_graph.invoke(state)
            if hasattr(final_state, 'final_answer') and final_state.final_answer:
                response = final_state.final_answer
                logger.debug(f"Chat response (via graph): '{response[:100]}...' [Length: {len(response)}]")
                return response
        except Exception as e:
            logger.error(f"Graph error: {e}")
            # Continue to fallback

        # Fall back to direct model call
        logger.info("TOOL: Using direct LLM call for response generation")
        messages = await self._prepare_messages(user_input, formatted_history, context)
        response = self.llm.invoke(messages)
        response_content = response.content
        logger.debug(f"Chat response (via direct LLM): '{response_content[:100]}...' [Length: {len(response_content)}]")
        return response_content

    @ErrorUtils.handle_async_exceptions(default_return=[])
    async def extract_bottle_names(self, text: str) -> List[str]:
        """Extract potential bottle names from user input for RAG retrieval."""
        logger.debug(f"Extracting bottle names from: '{text[:50]}...'")

        # Use the LLM to extract bottle names with specific guidance
        prompt = f"""
        You are a whisky expert. Extract any whisky bottle names mentioned in the following text.
        Take into account that bottle names often have specific formats like:
        - Brand + Age Statement: "Macallan 18"
        - Brand + Special Edition: "Ardbeg Uigeadail"
        - Full formal names: "The Balvenie DoubleWood 12 Year Old"
        - Vintage releases: "Springbank 1997"
        - Special series: "Blanton's Straight From The Barrel"

        If a user mentions a general category like "bourbon" or "scotch" without a specific bottle name, DO NOT include it.
        
        Return ONLY a JSON array of strings with the specific bottle names found, or an empty array if none are detected.
        Do not include any other text in your response, just the JSON array.
        
        Text: {text}
        """

        logger.info("TOOL: Using LLM for enhanced bottle name extraction")
        response = await self.llm.ainvoke(
            [SystemMessage(content="You are a whisky expert that extracts bottle names from text."),
             HumanMessage(content=prompt)]
        )

        # Parse the response to extract bottle names
        content = response.content
        try:
            # Try to find a JSON array in the response
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                bottle_names = json.loads(json_match.group(0))
                if isinstance(bottle_names, list):
                    logger.info(f"LLM extraction found {len(bottle_names)} bottle names: {', '.join(bottle_names)}")
                    return bottle_names
        except Exception as e:
            logger.warning(f"Failed to parse LLM bottle extraction result: {e}")
            pass

        # Fallback to an extended list of common whisky brands
        logger.info("TOOL: Falling back to common brand matching for bottle extraction")
        common_brands = [
            # Scotch
            "Macallan", "Glenfiddich", "Glenlivet", "Lagavulin", "Ardbeg", "Laphroaig",
            "Balvenie", "Highland Park", "Talisker", "Bowmore", "Oban", "Dalmore",
            "Glenmorangie", "Glenfarclas", "Springbank", "Bruichladdich", "Bunnahabhain",
            "Aberlour", "Caol Ila", "Clynelish", "Mortlach", "GlenDronach", "Auchentoshan",
            
            # Bourbon and American Whiskey
            "Buffalo Trace", "Eagle Rare", "Weller", "Blanton's", "Pappy Van Winkle",
            "Maker's Mark", "Woodford Reserve", "Jack Daniel's", "Jim Beam", "Wild Turkey",
            "Four Roses", "Knob Creek", "Booker's", "Baker's", "Basil Hayden's",
            "Old Forester", "Angel's Envy", "Michter's", "Jefferson's", "Bulleit",
            "Elijah Craig", "Evan Williams", "Old Grand-Dad", "George T. Stagg", "W.L. Weller",
            "Colonel E.H. Taylor", "Sazerac", "1792", "Old Fitzgerald", "Henry McKenna",
            
            # Irish
            "Jameson", "Redbreast", "Green Spot", "Yellow Spot", "Bushmills",
            "Teeling", "Tullamore D.E.W.", "Powers", "Midleton", "Connemara",
            
            # Japanese
            "Hibiki", "Yamazaki", "Nikka", "Hakushu", "Yoichi", "Miyagikyo",
            "Taketsuru", "Akashi", "Chichibu", "Fuji", "Kirin", "Mars Shinshu",
            
            # World Whisky
            "Kavalan", "Amrut", "Paul John", "Starward", "Sullivans Cove",
            "Penderyn", "Crown Royal", "Canadian Club", "Lot 40"
        ]

        found_bottles = []
        for brand in common_brands:
            # Look for exact brand mention or brand followed by additional words (likely a specific expression)
            pattern = r'\b' + re.escape(brand) + r'(\s+[a-zA-Z0-9\s\'\-]+)?\b'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                # Get the full matched bottle name
                full_match = match.group(0)
                found_bottles.append(full_match)
        
        if found_bottles:
            logger.info(f"Brand matching found {len(found_bottles)} bottle names: {', '.join(found_bottles)}")
        else:
            logger.info("No bottle names found through brand matching")
        
        return found_bottles

    @ErrorUtils.handle_async_exceptions()
    async def validate_bottle_image_urls(self, bottles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and ensure consistency between bottles and their image URLs.
        This helps prevent incorrect image URLs from being shown for bottles.
        
        Args:
            bottles: List of bottle data dictionaries
            
        Returns:
            Validated list of bottle data with consistent image URLs
        """
        if not bottles:
            return []
        
        logger.debug(f"Validating image URLs for {len(bottles)} bottles")
        validated_bottles = []
        
        # First build a reference mapping of bottle names to their proper image URLs
        # This can help detect inconsistencies across the dataset
        reference_map = {}
        
        for bottle in bottles:
            bottle_name = bottle.get("name", "").lower()
            if not bottle_name:
                continue
            
            image_url = bottle.get("image_url")
            if image_url:
                # If this bottle name already exists in our reference map with a different URL,
                # log a warning as this indicates potential inconsistency
                if bottle_name in reference_map and reference_map[bottle_name] != image_url:
                    logger.warning(f"Inconsistent image URLs for bottle '{bottle_name}': {reference_map[bottle_name]} vs {image_url}")
                
                reference_map[bottle_name] = image_url
        
        # Check if we have any bottles in the cache that aren't in our current list
        for cached_name, cached_data in self.bottle_cache.items():
            if isinstance(cached_data, dict) and "image_url" in cached_data and cached_name.lower() not in reference_map:
                reference_map[cached_name.lower()] = cached_data["image_url"]
        
        # Now process each bottle to ensure consistency
        for bottle in bottles:
            bottle_name = bottle.get("name", "").lower()
            if not bottle_name:
                # Skip bottles without names
                validated_bottles.append(bottle)
                continue
            
            # Check if this bottle has an image URL
            if not bottle.get("image_url"):
                # If we have a reference for this bottle name, use that image URL
                if bottle_name in reference_map:
                    bottle["image_url"] = reference_map[bottle_name]
                    logger.debug(f"Added missing image URL for '{bottle_name}' from reference map")
            
            # If bottle has an image URL, make sure it's consistent with our reference map
            elif bottle_name in reference_map and bottle.get("image_url") != reference_map[bottle_name]:
                logger.warning(f"Correcting inconsistent image URL for '{bottle_name}'")
                # Use the reference map URL as the canonical URL for this bottle
                bottle["image_url"] = reference_map[bottle_name]
            
            # Add to validated list
            validated_bottles.append(bottle)
        
        logger.debug(f"Completed validation of {len(validated_bottles)} bottles")
        return validated_bottles

    async def chat_stream(self, user_input, conversation_history=None, username=None):
        """Process a user message and stream Bob's response incrementally."""
        logger.debug(f"Processing streaming chat: '{user_input[:50]}...'")
        
        full_response = ""  # To collect the entire response for logging

        try:
            # Format history and prepare context
            formatted_history = await self._format_conversation_history(conversation_history)
            context, bottle_recommendations = await self._prepare_conversation_context(user_input, username)
            
            # Extract detected bottle names from the user input
            detected_bottle_names = await self.extract_bottle_names(user_input)
            
            # Get detailed information for detected bottles
            detected_bottles = []
            for bottle_name in detected_bottle_names:
                bottle_data = await self.get_bottle_knowledge(bottle_name)
                if bottle_data:
                    detected_bottles.append(bottle_data)
                    logger.debug(f"Added detailed data for detected bottle: {bottle_name}")
            
            # Validate image URLs for both detected bottles and recommendations
            if detected_bottles:
                detected_bottles = await self.validate_bottle_image_urls(detected_bottles)
            if bottle_recommendations:
                bottle_recommendations = await self.validate_bottle_image_urls(bottle_recommendations)
            
            # Try to use the conversation graph with streaming first
            try:
                # Initialize state
                state = AgentState()
                state.username = username
                state.messages = formatted_history + [{"role": "human", "content": user_input}]
                if context:
                    state.context.append(context)
                if bottle_recommendations:
                    state.recommendations = bottle_recommendations
                if detected_bottles:
                    state.detected_bottles = detected_bottles
                    # Add detected bottles to recommendations if they're not already there
                    bottle_names_in_recs = {rec.get("name", "").lower() for rec in bottle_recommendations}
                    for bottle in detected_bottles:
                        if bottle.get("name", "").lower() not in bottle_names_in_recs:
                            state.recommendations.append(bottle)
                            logger.debug(f"Added detected bottle to recommendations: {bottle.get('name')}")

                # Process through the agent graph with streaming
                logger.info("TOOL: Using conversation graph with streaming for response generation")
                async for chunk in self.conversation_graph.astream(state):
                    if chunk:
                        full_response += chunk  # Collect the response
                        yield f"data: {json.dumps({'content': chunk})}\n\n"
            
            except Exception as e:
                logger.error(f"Graph streaming error: {e}")
                # Fall back to direct LLM streaming
                logger.info("TOOL: Falling back to direct LLM streaming for response generation")
                messages = await self._prepare_messages(user_input, formatted_history, context)
                
                async for chunk in self.llm.astream(messages):
                    if hasattr(chunk, 'content') and chunk.content:
                        chunk_content = chunk.content
                        full_response += chunk_content  # Collect the response
                        yield f"data: {json.dumps({'content': chunk_content})}\n\n"
        
        except Exception as e:
            logger.error(f"Error in chat stream: {e}", exc_info=True)
            error_message = 'I apologize, but I encountered a technical issue. Please try again.'
            full_response = error_message
            yield f"data: {json.dumps({'content': error_message})}\n\n"
        
        # Signal the end of the stream
        yield f"data: {json.dumps({'end_of_stream': True})}\n\n"
        
        # Log the complete response after streaming is done
        logger.debug(f"Chat stream response: '{full_response[:100]}...' [Length: {len(full_response)}]")

    def setup_agent(self):
        """Set up the LangChain agent with conversation memory and tools."""
        logger.debug("Setting up Bob's agent capabilities")
        # System prompt for Bob's personality and capabilities
        system_prompt = self.SYSTEM_PROMPT

        # Create the LangChain agent
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="history"),
            SystemMessage(
                content="Additional context for your response:\n{context}"),
            HumanMessage(content="{input}")
        ])

        # Create a simple chain for conversation
        self.chain = prompt | self.llm | StrOutputParser()
        logger.debug("Conversation chain created")


async def main():
    """Main function to demonstrate Bob's capabilities."""
    # Load environment variables
    load_dotenv()
    logger.debug("Starting Bob main function")

    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(
        description="Bob - BAXUS Whisky Recommendation AI Agent")
    parser.add_argument("--username", type=str,
                        help="Username to analyze")
    parser.add_argument("--similar", type=int, default=3,
                        help="Number of similar bottle recommendations")
    parser.add_argument("--diverse", type=int, default=2,
                        help="Number of diverse bottle recommendations")
    parser.add_argument("--diversity-ratio", type=float, default=0.4,
                        help="Ratio of diversity in recommendations (0-1)")
    parser.add_argument("--chat", type=str,
                        help="Chat with Bob using this message")
    args = parser.parse_args()
    logger.debug(f"Command line arguments: {args}")

    # Initialize Bob
    try:
        logger.info("Initializing Bob")
        bob = Bob()

        # If username is provided, analyze their collection
        if args.username:
            logger.info(f"Analyzing collection for username: {args.username}")

            # Get user profile
            logger.debug("Getting user profile")
            profile = await bob.get_user_profile(args.username)
            print("\nUser Profile:")
            print(json.dumps(profile, indent=2))

            # Get recommendations
            logger.debug("Getting recommendations")
            recommendations = await bob.get_recommendations(
                args.username,
                num_similar=args.similar,
                num_diverse=args.diverse,
                diversity_ratio=args.diversity_ratio
            )

            print("\nRecommendations:")
            print("\nSimilar bottles:")
            for rec in recommendations.get("similar", []):
                print(f"- **{rec['name']}** ({rec['type']})")
                if "image_url" in rec:
                    print(f"  ![{rec['name']}]({rec['image_url']})")
                if "nft_address" in rec and rec['nft_address']:
                    print(f"  [View on BAXUS](https://baxus.co/asset/{rec['nft_address']})")

            print("\nDiverse options:")
            for rec in recommendations.get("diverse", []):
                print(f"- **{rec['name']}** ({rec['type']}): {rec.get('reason', '')}")
                if "image_url" in rec:
                    print(f"  ![{rec['name']}]({rec['image_url']})")
                if "nft_address" in rec and rec['nft_address']:
                    print(f"  [View on BAXUS](https://baxus.co/asset/{rec['nft_address']})")
        else:
            logger.debug("No username provided, using example")

        # Chat with Bob if requested
        if args.chat:
            logger.debug(f"Processing chat request: {args.chat}")
            print("\nChat response:")
            response = await bob.chat(args.chat)
            print(f"Bob: {response}")

    except Exception as e:
        logger.error(f"Error in main function: {e}", exc_info=True)
        print(f"Failed to initialize Bob: {e}")

    logger.debug("Bob main function completed")


if __name__ == "__main__":
    logger.info("Starting Bob application")
    asyncio.run(main())
    logger.info("Bob application completed")
