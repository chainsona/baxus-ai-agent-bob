#!/usr/bin/env python3
"""
Utility module with shared functions for working with bottle vector data.
"""

import logging
import os
import sys
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Database connection parameters
DB_HOST = os.environ.get('PGVECTOR_HOST', 'localhost')
DB_PORT = os.environ.get('PGVECTOR_PORT', '5452')
DB_NAME = os.environ.get('PGVECTOR_DB', 'baxus')
DB_USER = os.environ.get('PGVECTOR_USER', 'baxus')
DB_PASSWORD = os.environ.get('PGVECTOR_PASSWORD', 'baxuspwd')

# OpenAI API key for generating embeddings
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable is not set")
    sys.exit(1)

# Vector dimension for OpenAI embeddings
VECTOR_DIM = 1536  # OpenAI's text-embedding-3-small dimension

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


def get_embedding(text: str) -> List[float]:
    """Generate embeddings for the given text using OpenAI API."""
    if not text:
        return [0.0] * VECTOR_DIM

    try:
        # Truncate text if it's too long (OpenAI has token limits)
        truncated_text = text[:8000]

        # Get embedding from OpenAI
        response = client.embeddings.create(
            input=truncated_text,
            model="text-embedding-3-small"
        )

        # Extract embedding from response
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        # Return zero vector as fallback
        return [0.0] * VECTOR_DIM


def generate_bottle_embedding(bottle: Dict[str, Any]) -> List[float]:
    """Generate embedding for a bottle by combining name, description, and other relevant fields.

    Uses structured prompt formatting and field weighting to improve semantic representation.
    """
    # Extract all potentially relevant fields with fallbacks
    name = bottle.get("name", "")
    description = bottle.get("description", "")
    bottle_type = bottle.get("type", "")
    producer = bottle.get("producer", "")
    series = bottle.get("series", "")
    country = bottle.get("country", "")
    region = bottle.get("region", "")
    abv = bottle.get("abv", "")
    age_statement = bottle.get("age_statement", "")
    baxus_class_name = bottle.get("baxus_class_name", "")

    # Create a structured prompt format with field labels and weightings
    # This helps the embedding model understand the structure of the data
    structured_parts = []

    # Primary identifiers (most important for searching)
    if name:
        structured_parts.append(f"NAME: {name}")
        # Important terms repeated for emphasis/weighting
        structured_parts.append(f"TITLE: {name}")

    # Key categorization fields
    if bottle_type:
        structured_parts.append(f"TYPE: {bottle_type}")
    if producer:
        structured_parts.append(f"PRODUCER: {producer}")
    if series:
        structured_parts.append(f"SERIES: {series}")

    # Geographic information
    location_parts = []
    if region:
        location_parts.append(region)
    if country:
        location_parts.append(country)
    if location_parts:
        structured_parts.append(f"ORIGIN: {', '.join(location_parts)}")

    # Technical specifications
    specs = []
    if abv:
        specs.append(f"{abv}% ABV")
    if age_statement:
        specs.append(f"{age_statement}")
    if specs:
        structured_parts.append(f"SPECIFICATIONS: {' '.join(specs)}")

    # Classification information
    if baxus_class_name:
        structured_parts.append(f"CLASSIFICATION: {baxus_class_name}")

    # Full description (less weighted but included for completeness)
    if description:
        # Truncate long descriptions to avoid token limits
        desc = description[:1000] + \
            "..." if len(description) > 1000 else description
        structured_parts.append(f"DESCRIPTION: {desc}")

    # Add domain context to help with understanding
    structured_parts.append(
        "DOMAIN: Fine spirits, whiskey, bourbon, scotch, liquor, collectible bottles")

    # Combine all parts with newlines for better structure recognition
    text_to_embed = "\n".join(structured_parts)

    # If no text is available, return zero vector
    if not text_to_embed.strip():
        logger.warning(
            f"No text to embed for bottle: {bottle.get('name', 'Unnamed')}")
        return [0.0] * VECTOR_DIM

    # Generate embedding
    return get_embedding(text_to_embed)


def get_db_connection_string() -> str:
    """Get the database connection string."""
    return f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"


def format_bottle_result(bottle: tuple) -> str:
    """Format a bottle database result for display."""
    id, name, description, type_, producer, series, abv, age, country, region, similarity = bottle

    # Format the output string
    lines = [f"{name} [{similarity:.4f}]"]

    if producer:
        lines.append(f"Producer: {producer}")
    if type_:
        lines.append(f"Type: {type_}")
    if series:
        lines.append(f"Series: {series}")
    if abv:
        lines.append(f"ABV: {abv}%")
    if age:
        lines.append(f"Age: {age} years")
    if country or region:
        location = ", ".join(filter(None, [region, country]))
        lines.append(f"Origin: {location}")
    if description:
        # Truncate long descriptions
        desc = description[:200] + \
            "..." if len(description) > 200 else description
        lines.append(f"Description: {desc}")

    return "\n   ".join(lines)
