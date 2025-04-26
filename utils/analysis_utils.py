"""
Analysis Utilities Module
------------------------
Functions for analyzing user collections and generating insights.
"""

import os
import numpy as np
from typing import Dict, Any, List

from utils.error_utils import ErrorUtils
from utils.logging_utils import get_logger

# Set up logging
logger = get_logger(__name__)

# Flavor profile mappings for different spirits and whisky types
FLAVOR_PROFILES_BY_TYPE = {
  "Bourbon": {"sweet": 0.8, "woody": 0.7, "spicy": 0.5, "vanilla": 0.6, "caramel": 0.7, "oak": 0.6},
  "Scotch": {"smoky": 0.8, "woody": 0.6, "fruity": 0.4, "peaty": 0.7, "malty": 0.6, "oak": 0.5},
  "Rye": {"spicy": 0.9, "woody": 0.5, "sweet": 0.3, "peppery": 0.8, "dry": 0.6, "herbal": 0.5},
  "Irish": {"fruity": 0.7, "sweet": 0.6, "woody": 0.4, "smooth": 0.8, "grassy": 0.5, "vanilla": 0.4},
  "Japanese": {"fruity": 0.6, "woody": 0.7, "sweet": 0.5, "floral": 0.6, "honey": 0.5, "delicate": 0.8},
  "Canadian": {"sweet": 0.7, "spicy": 0.4, "woody": 0.6, "smooth": 0.7, "caramel": 0.5, "light": 0.6},
  "Canadian Whisky": {"sweet": 0.7, "spicy": 0.4, "woody": 0.6, "smooth": 0.7, "caramel": 0.5, "light": 0.6},
  "Whiskey": {"sweet": 0.6, "woody": 0.6, "spicy": 0.4, "vanilla": 0.5, "oak": 0.5, "caramel": 0.4},
  "Blended": {"sweet": 0.6, "fruity": 0.5, "woody": 0.5, "balanced": 0.8, "smooth": 0.6, "honey": 0.4},
  "Single Malt": {"malty": 0.8, "woody": 0.6, "fruity": 0.5, "complex": 0.7, "rich": 0.6, "oak": 0.5},
  "Single Malt Scotch Whisky": {"malty": 0.8, "woody": 0.6, "fruity": 0.5, "complex": 0.7, "rich": 0.6, "oak": 0.5, "smoky": 0.6},
  "Tennessee": {"sweet": 0.7, "charcoal": 0.8, "woody": 0.6, "vanilla": 0.5, "caramel": 0.6, "smooth": 0.5},
  "American": {"sweet": 0.6, "woody": 0.6, "spicy": 0.4, "vanilla": 0.5, "caramel": 0.5, "balanced": 0.6},
  "Rum": {"sweet": 0.9, "tropical": 0.7, "molasses": 0.8, "vanilla": 0.6, "caramel": 0.7, "oak": 0.4},
  "Tequila": {"agave": 0.9, "earthy": 0.7, "peppery": 0.6, "citrus": 0.5, "sweet": 0.4, "herbaceous": 0.6},
  "Gin": {"juniper": 0.9, "botanical": 0.8, "citrus": 0.7, "floral": 0.6, "spicy": 0.5, "herbal": 0.8},
  "Vodka": {"clean": 0.9, "neutral": 0.9, "smooth": 0.7, "grain": 0.5, "subtle": 0.8, "light": 0.7},
  "Cognac": {"fruity": 0.8, "sweet": 0.7, "woody": 0.6, "rich": 0.7, "floral": 0.5, "vanilla": 0.6}
}

# Mapping from detailed flavors to simplified categories
FLAVOR_SIMPLIFICATION = {
    # Sweet category
    "sweet": "sweet",
    "vanilla": "sweet",
    "caramel": "sweet",
    "honey": "sweet",
    "molasses": "sweet",
    "tropical": "sweet",
    "agave": "sweet",
    
    # Woody category
    "woody": "woody",
    "oak": "woody",
    "charcoal": "woody",
    "rich": "woody",
    "malty": "woody",
    
    # Spicy category
    "spicy": "spicy",
    "peppery": "spicy",
    "herbal": "spicy",
    "herbaceous": "spicy",
    "juniper": "spicy",
    "earthy": "spicy",
    
    # Smoky category
    "smoky": "smoky",
    "peaty": "smoky",
    
    # Fruity category
    "fruity": "fruity",
    "citrus": "fruity",
    "tropical": "fruity",
    
    # Smooth category
    "smooth": "smooth",
    "delicate": "smooth",
    "balanced": "smooth",
    "subtle": "smooth",
    "clean": "smooth",
    "light": "smooth",
    "neutral": "smooth",
    "complex": "smooth",
    
    # Floral category
    "floral": "floral",
    "botanical": "floral",
    "grassy": "floral"
}

# Core flavor categories
CORE_FLAVORS = ["sweet", "woody", "spicy", "smoky", "fruity", "smooth", "floral"]

@ErrorUtils.handle_exceptions()
def generate_taste_profile(username: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a taste profile for the user based on their collection."""
    logger.debug(f"Generating taste profile for user: {username}")

    # Initialize flavor profiles with all possible flavors
    detailed_flavor_profiles = {}
    
    # Collect all possible flavors from the profiles
    for type_flavors in FLAVOR_PROFILES_BY_TYPE.values():
        for flavor in type_flavors:
            if flavor not in detailed_flavor_profiles:
                detailed_flavor_profiles[flavor] = 0
    
    # Initialize simplified flavor profiles
    simplified_flavors = {flavor: 0.0 for flavor in CORE_FLAVORS}
    
    # Get raw collection data
    raw_data = analysis.get("raw_data", [])
    
    # Initialize counters for bottle types and regions
    bottle_types = {}
    regions = {}
    total_bottles = len(raw_data)
    
    # Process each bottle in the collection
    for item in raw_data:
        if "product" not in item or not isinstance(item["product"], dict):
            continue
            
        product = item["product"]
        
        # Extract bottle type and spirit
        bottle_type = product.get("type")
        spirit = product.get("spirit_type") or product.get("spirit")
        region = product.get("region")
        
        # Count bottle types
        if bottle_type:
            bottle_types[bottle_type] = bottle_types.get(bottle_type, 0) + 1
        elif spirit:
            bottle_types[spirit] = bottle_types.get(spirit, 0) + 1
            
        # Count regions
        if region:
            regions[region] = regions.get(region, 0) + 1
            
        # Add flavor profiles from both type and spirit
        weight = 1.0 / total_bottles if total_bottles > 0 else 0
        
        # Apply flavor mappings from bottle type
        if bottle_type and bottle_type in FLAVOR_PROFILES_BY_TYPE:
            for flavor, score in FLAVOR_PROFILES_BY_TYPE[bottle_type].items():
                detailed_flavor_profiles[flavor] = detailed_flavor_profiles.get(flavor, 0) + score * weight
                
        # Apply flavor mappings from spirit if no bottle type or different mapping
        elif spirit and spirit in FLAVOR_PROFILES_BY_TYPE:
            for flavor, score in FLAVOR_PROFILES_BY_TYPE[spirit].items():
                detailed_flavor_profiles[flavor] = detailed_flavor_profiles.get(flavor, 0) + score * weight

    # Store type and region data in analysis for other functions
    analysis["types"] = bottle_types
    analysis["regions"] = regions

    # Determine favorite type and region
    favorite_type = max(bottle_types.items(), key=lambda x: x[1])[0] if bottle_types else "Unknown"
    favorite_region = max(regions.items(), key=lambda x: x[1])[0] if regions else "Unknown"

    # Simplify detailed flavors to core categories
    for detailed_flavor, score in detailed_flavor_profiles.items():
        if detailed_flavor in FLAVOR_SIMPLIFICATION:
            simplified_category = FLAVOR_SIMPLIFICATION[detailed_flavor]
            simplified_flavors[simplified_category] += score
    
    # Keep values in 0-1 range
    for flavor in simplified_flavors:
        simplified_flavors[flavor] = min(simplified_flavors[flavor], 1.0)

    # Determine top flavors
    sorted_flavors = sorted(
        simplified_flavors.items(), key=lambda x: x[1], reverse=True)
    
    # Select top flavors with score > 0.2
    top_flavors = [flavor for flavor, score in sorted_flavors
                   if score > 0.2]

    return {
        "dominant_flavors": top_flavors,
        "flavor_profiles": simplified_flavors,
        "favorite_type": favorite_type,
        "favorite_region": favorite_region
    }

@ErrorUtils.handle_exceptions()
def calculate_investment_stats(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate investment statistics for a user's collection."""
    logger.debug("Calculating investment statistics")

    # Extract bottle data with valid price information
    bottle_count = 0
    price_count = 0
    total_value = 0
    min_price = float('inf')
    max_price = 0
    prices_by_type = {}

    # Process raw data if available
    raw_data = analysis.get("raw_data", [])
    for item in raw_data:
        bottle_count += 1

        # Extract price and type from the product data
        price = None
        bottle_type = None
        if "product" in item and isinstance(item["product"], dict):
            price = item["product"].get("average_msrp") or item["product"].get("shelf_price") or item["product"].get("price")
            bottle_type = item["product"].get("type") or item["product"].get("spirit_type") or item["product"].get("spirit")

        # Process price data if available
        if price is not None:
            try:
                price_value = float(price)
                price_count += 1
                total_value += price_value
                min_price = min(min_price, price_value)
                max_price = max(max_price, price_value)

                # Add to type-specific stats
                if bottle_type:
                    if bottle_type not in prices_by_type:
                        prices_by_type[bottle_type] = []
                    prices_by_type[bottle_type].append(price_value)
            except (ValueError, TypeError):
                logger.debug(f"Invalid price value: {price}")

    # Calculate statistics
    avg_bottle_value = total_value / price_count if price_count > 0 else 100

    # Estimate value ranges for bottles without price data
    default_min = 50  # Minimum estimate per bottle
    default_max = 150  # Maximum estimate per bottle
    
    if price_count == 0:
        estimated_low = bottle_count * default_min
        estimated_high = bottle_count * default_max
    else:
        min_price = min_price if min_price != float('inf') else default_min
        estimated_low = (min_price * price_count) + ((bottle_count - price_count) * (min_price * 0.8 if min_price != float('inf') else default_min))
        estimated_high = (max_price * price_count) + ((bottle_count - price_count) * (max_price * 1.2 if max_price > 0 else default_max))

    # Calculate type-specific statistics
    type_stats = _calculate_type_statistics(prices_by_type)

    return {
        "estimated_value": {
            "low": round(estimated_low, 2),
            "high": round(estimated_high, 2),
            "average": round(avg_bottle_value, 2) if price_count > 0 else 0,
            "total": round(total_value, 2)
        },
        "bottle_count": bottle_count,
        "bottles_with_price": price_count,
        "price_range": {
            "min": round(min_price, 2) if min_price != float('inf') else 0,
            "max": round(max_price, 2)
        },
        "value_by_type": type_stats
    }

def _calculate_type_statistics(prices_by_type: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """Calculate statistics for each bottle type."""
    type_stats = {}
    for bottle_type, prices in prices_by_type.items():
        if prices:
            type_stats[bottle_type] = {
                "count": len(prices),
                "total_value": round(sum(prices), 2),
                "avg_value": round(sum(prices) / len(prices), 2),
                "min_value": round(min(prices), 2),
                "max_value": round(max(prices), 2)
            }
    return type_stats

@ErrorUtils.handle_exceptions()
def extract_collection_kpis(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key performance indicators for the collection."""
    logger.debug("Extracting collection KPIs")

    # Extract key metrics from the analysis
    bottle_count = len(analysis.get("raw_data", []))
    types = analysis.get("types", {})
    regions = analysis.get("regions", {})

    # Calculate diversity score - higher is more diverse
    type_diversity = len(types) / max(1, bottle_count) * 10
    region_diversity = len(regions) / max(1, bottle_count) * 10

    # Overall collection diversity score
    diversity_score = (type_diversity + region_diversity) / 2

    return {
        "bottle_count": bottle_count,
        "type_count": len(types),
        "region_count": len(regions),
        "diversity_score": round(diversity_score, 1),
        "types_distribution": types,
        "regions_distribution": regions
    } 