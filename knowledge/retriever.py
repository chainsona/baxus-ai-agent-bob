"""
Knowledge Retriever Module
--------------------------
Retrieval system to enhance Bob's knowledge using pgvector RAG.
"""

import os
import logging
from typing import List, Dict, Any

# Import centralized utilities
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

class BobKnowledgeRetriever:
    """Retrieval system to enhance Bob's knowledge using pgvector RAG."""

    def __init__(self):
        """Initialize the knowledge retriever."""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.error("OPENAI_API_KEY environment variable is missing")
            raise ValueError("OPENAI_API_KEY environment variable is required")
        logger.debug("Initialized RAG knowledge retriever")

    @ErrorUtils.handle_async_exceptions(default_return=[])
    async def get_relevant_bottles(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve bottles relevant to the query using semantic search."""
        logger.debug(f"Retrieving bottles relevant to: '{query}'")
        
        # Generate embedding for the query
        query_embedding = BottleUtils.get_embedding(query)
        
        async with await DatabaseUtils.get_connection() as conn:
            return await DatabaseUtils.search_by_similarity(
                conn,
                embedding=query_embedding,
                limit=limit
            )

    def format_knowledge(self, bottles: List[Dict[str, Any]]) -> str:
        """Format retrieved bottle information into a knowledge string."""
        if not bottles:
            return """
            As a whisky expert, I don't have specific bottle data that matches this query,
            but I can still provide information based on my general knowledge of whisky types,
            production methods, flavor profiles, and recommendations.
            """

        knowledge = "Here is relevant information about bottles that may help answer the query:\n\n"

        # Group bottles by type for better organization
        bottles_by_type = {}
        for bottle in bottles:
            bottle_type = bottle.get('type', 'Unknown')
            if bottle_type not in bottles_by_type:
                bottles_by_type[bottle_type] = []
            bottles_by_type[bottle_type].append(bottle)

        # First provide a summary of types found
        types_found = list(bottles_by_type.keys())
        if len(types_found) > 1:
            knowledge += f"Found information about {len(bottles)} bottles across {len(types_found)} types: {', '.join(types_found)}.\n\n"

        # Now list bottle details grouped by type
        for bottle_type, type_bottles in bottles_by_type.items():
            if len(bottles_by_type) > 1:
                knowledge += f"== {bottle_type} ==\n"

            for i, bottle in enumerate(type_bottles, 1):
                knowledge += f"{i}. {bottle.get('name', 'Unknown')}"

                if bottle.get('producer'):
                    knowledge += f" by {bottle.get('producer')}"

                if bottle_type == 'Unknown':
                    if bottle.get('type'):
                        knowledge += f" ({bottle.get('type')})"
                knowledge += "\n"

                # Add image URL if available - use Markdown format
                if bottle.get('image_url'):
                    knowledge += f"   ![{bottle.get('name', 'Bottle image')}]({bottle.get('image_url')})\n"

                # Add marketplace link if NFT address is available - use Markdown format
                if bottle.get('nft_address'):
                    marketplace_url = f"https://baxus.co/asset/{bottle.get('nft_address')}"
                    knowledge += f"   [View on Marketplace]({marketplace_url})\n"

                if bottle.get('description'):
                    knowledge += f"   Description: {bottle.get('description')}\n"

                if bottle.get('region') and bottle.get('country'):
                    knowledge += f"   Origin: {bottle.get('region')}, {bottle.get('country')}\n"
                elif bottle.get('country'):
                    knowledge += f"   Country: {bottle.get('country')}\n"

                if bottle.get('abv'):
                    knowledge += f"   ABV: {bottle.get('abv')}%\n"

                if bottle.get('age_statement'):
                    knowledge += f"   Age: {bottle.get('age_statement')}\n"

                if bottle.get('total_score'):
                    knowledge += f"   Rating: {bottle.get('total_score')}/100\n"

                knowledge += "\n"

            if len(bottles_by_type) > 1:
                knowledge += "\n"

        return knowledge 