
"""
Script to clean Baxus listings data and output a normalized version.
"""
import json
import os
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path


def load_listings(file_path: str) -> List[Dict[str, Any]]:
    """Load listings data from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Handle both single listing and array of listings
    if isinstance(data, dict) and '_source' in data:
        return [data['_source']]
    elif isinstance(data, list):
        return [item.get('_source', item) for item in data]
    else:
        # If it's a single asset/listing without _source
        if isinstance(data, dict):
            return [data]
        return []


def safe_float(value: Any) -> Optional[float]:
    """Safely convert a value to float, returning None if invalid."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def clean_listing(listing: Dict[str, Any]) -> Dict[str, Any]:
    """Clean and normalize a single listing."""
    cleaned = {}

    # Extract basic fields
    cleaned["name"] = listing.get("name", "")
    cleaned["description"] = listing.get("description", "")
    cleaned["spirit_type"] = listing.get(
        "spiritType") or listing.get("type", "")
    cleaned["image_url"] = listing.get("imageUrl", "")
    cleaned["animation_url"] = listing.get("animationUrl", "")
    cleaned["status"] = listing.get("status", "")
    cleaned["nft_address"] = listing.get("id") or listing.get("nftAddress", "")
    cleaned["is_listed"] = bool(listing.get("isListed", False))
    cleaned["listed_price"] = safe_float(listing.get("price"))

    # Handle dates
    if "listedDate" in listing:
        cleaned["listed_date"] = listing["listedDate"]

    if "lastHeartbeat" in listing:
        cleaned["last_heartbeat"] = listing["lastHeartbeat"]

    # Extract attributes - handle both nested and flat structures
    # First check for nested attributes
    attributes = {}
    if "attributes" in listing and isinstance(listing["attributes"], dict):
        attributes = listing["attributes"]

    # Now handle flat structure fields and merge with any from attributes
    # Producer info
    cleaned["producer"] = attributes.get(
        "Producer", "") or listing.get("producer", "")
    cleaned["producer_type"] = attributes.get(
        "Producer Type", "") or listing.get("producerType", "")
    cleaned["series"] = attributes.get(
        "Series", "") or listing.get("series", "")

    # ABV
    abv_value = attributes.get("ABV") or listing.get("abv")
    cleaned["abv"] = safe_float(abv_value)

    # Age
    age = attributes.get("Age", "") or listing.get("age", "")
    if age:
        cleaned["age_years"] = safe_float(age)
        cleaned["age_statement"] = str(age)

    # Origin
    cleaned["country"] = attributes.get(
        "Country", "") or listing.get("country", "")
    cleaned["region"] = attributes.get(
        "Region", "") or listing.get("region", "")

    # Years
    cleaned["year_bottled"] = attributes.get(
        "Year Bottled", "") or listing.get("yearBottled", "")
    cleaned["year_distilled"] = attributes.get(
        "Year Distilled", "") or listing.get("yearDistilled", "")

    # Packaging
    cleaned["packaging"] = attributes.get(
        "Packaging", "") or listing.get("packaging", "")
    cleaned["package_shot"] = attributes.get(
        "PackageShot", False) or listing.get("packageShot", False)

    # Baxus class info
    cleaned["baxus_class_id"] = attributes.get(
        "Baxus Class ID", "") or listing.get("baxusClassId", "")
    cleaned["baxus_class_name"] = attributes.get(
        "Baxus Class Name", "") or listing.get("baxusClassName", "")

    # Size and volume
    size = attributes.get("Size", "") or listing.get("size", "")
    cleaned["size"] = size

    # Try to extract numeric volume and unit
    if size and "ml" in size.lower():
        volume = safe_float(
            ''.join(c for c in size if c.isdigit() or c == '.'))
        if volume:
            cleaned["size"] = int(volume)
            cleaned["size_unit"] = "ml"

    return cleaned


def clean_listings(listings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean all listings."""
    return [clean_listing(listing) for listing in listings]


def save_as_csv(data: List[Dict[str, Any]], output_file: Path) -> None:
    """Save data as CSV file."""
    if not data:
        print("No data to save as CSV")
        return

    # Get all unique keys from all items
    fieldnames = set()
    for item in data:
        fieldnames.update(item.keys())

    fieldnames = sorted(list(fieldnames))

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"Saved CSV data to {output_file}")


def main():
    # Paths
    input_file = Path("data/baxus-listings.json")
    output_json = Path("data/cleaned-listings.json")
    output_csv = Path("data/cleaned-listings.csv")

    # Check if using alternate input file from command line
    import sys
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])
        # Generate output filenames based on input
        stem = input_file.stem
        output_json = Path(f"data/cleaned-{stem}.json")
        output_csv = Path(f"data/cleaned-{stem}.csv")

    # Check if smaller sample file exists and use that for testing
    sample_file = Path("data/baxus-listing-sample.json")
    if not input_file.exists() and sample_file.exists():
        print(f"Using sample file: {sample_file}")
        input_file = sample_file

    # Load and clean data
    print(f"Loading data from {input_file}...")
    listings = load_listings(str(input_file))
    print(f"Loaded {len(listings)} listings")

    cleaned_listings = clean_listings(listings)
    print(f"Cleaned {len(cleaned_listings)} listings")

    # Create output directory if it doesn't exist
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # Save cleaned data as JSON
    with open(output_json, 'w') as f:
        json.dump(cleaned_listings, f, indent=2)

    print(f"Saved JSON data to {output_json}")

    # Save cleaned data as CSV
    save_as_csv(cleaned_listings, output_csv)


if __name__ == "__main__":
    main()
