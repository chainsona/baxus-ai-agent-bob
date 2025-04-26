"""
Bottle Utilities Module
----------------------
Utility functions for working with bottle data and embeddings.
"""

import os
from typing import List, Dict, Any, Optional
import openai

# Import centralized utilities
from utils.db_utils import DatabaseUtils
from utils.error_utils import ErrorUtils
from utils.logging_utils import get_logger

# Set up logging
logger = get_logger(__name__)

class BottleUtils:
    """Utility functions for working with bottle data and embeddings."""

    @staticmethod
    def get_db_connection_string() -> str:
        """Get the database connection string from environment variables.
        
        Note: This method is maintained for backward compatibility.
        New code should use DatabaseUtils.get_db_connection_string() instead.
        """
        return DatabaseUtils.get_db_connection_string()

    @staticmethod
    @ErrorUtils.handle_exceptions(default_return=[])
    def get_embedding(text: str) -> List[float]:
        """Get embedding for text using OpenAI's embedding model."""
        logger.debug(f"Getting embedding for text: {text[:50]}...")
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        logger.debug(
            f"Embedding generated with dimensions: {len(response.data[0].embedding)}")
        return response.data[0].embedding

    @staticmethod
    def format_bottle_result(result: Dict[str, Any]) -> str:
        """Format a bottle result for display."""
        name = result.get("name", "Unknown")
        producer = result.get("producer", "")
        type_name = result.get("type", "")
        abv = result.get("abv", "")
        price = result.get("price", "")

        formatted = f"{name}"
        if producer:
            formatted += f" from {producer}"
        if type_name:
            formatted += f" ({type_name})"
        if abv:
            formatted += f", {abv}% ABV"
        if price:
            formatted += f", ${price}"

        logger.debug(f"Formatted bottle result: {formatted}")
        return formatted

    @staticmethod
    def format_bottles_info(bottles: List[Dict[str, Any]]) -> str:
        """Format bottle information into a readable string."""
        if not bottles:
            return "No relevant bottles found."

        formatted_info = []
        for bottle in bottles:
            # Format bottle details
            details = [
                f"Name: {bottle.get('name', 'Unknown')}",
                f"Type: {bottle.get('type', 'Unknown')}",
                f"Producer: {bottle.get('producer', 'Unknown')}"
            ]

            if bottle.get('age_statement'):
                details.append(f"Age: {bottle.get('age_statement')}")

            if bottle.get('abv'):
                details.append(f"ABV: {bottle.get('abv')}%")

            if bottle.get('country'):
                location = bottle.get('country')
                if bottle.get('region'):
                    location += f", {bottle.get('region')}"
                details.append(f"Origin: {location}")

            # Add description if available
            if bottle.get('description'):
                details.append(f"Description: {bottle.get('description')}")

            # Add image URL if available - use Markdown format
            if bottle.get('image_url'):
                details.append(
                    f"![{bottle.get('name', 'Bottle image')}]({bottle.get('image_url')})")

            # Add marketplace link if available - use Markdown format
            if bottle.get('nft_address'):
                marketplace_url = f"https://baxus.co/asset/{bottle.get('nft_address')}"
                details.append(f"[View on Marketplace]({marketplace_url})")

            # Format with similarity score if available
            bottle_info = "\n".join(details)
            if 'similarity' in bottle:
                similarity_percentage = round(
                    bottle.get('similarity', 0) * 100, 1)
                bottle_info += f"\nRelevance: {similarity_percentage}%"

            formatted_info.append(bottle_info)

        return "\n\n".join(formatted_info)
    
    @staticmethod
    @ErrorUtils.handle_async_exceptions(default_return={})
    async def get_bottle_by_id_or_name(
        bottle_id: Optional[Any] = None,
        bottle_name: Optional[str] = None,
        fields: str = "*"
    ) -> Dict[str, Any]:
        """Get bottle data by ID or name."""
        async with await DatabaseUtils.get_connection() as conn:
            results = await DatabaseUtils.find_bottle_by_id_or_name(
                conn, bottle_id, bottle_name, fields
            )
            
            if results:
                result = results[0]
                # Remove embedding if present
                if 'embedding' in result:
                    del result['embedding']
                return result
                
        return {} 