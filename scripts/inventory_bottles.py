#!/usr/bin/env python3
"""
Script to inventory bottles with duplicate prevention and extract data from Baxus listings.
"""
import json
import os
import csv
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set
import hashlib
from difflib import SequenceMatcher
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Get project root directory
PROJECT_ROOT = Path(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = PROJECT_ROOT / "data"

# Function to set debug logging


def enable_debug_logging():
    """Enable debug logging for detailed output."""
    logger.setLevel(logging.DEBUG)
    # Also set the root logger to debug
    logging.getLogger().setLevel(logging.DEBUG)
    logger.debug("Debug logging enabled")


class BottleInventory:
    """Class to manage a bottle inventory with duplicate prevention."""

    def __init__(self, inventory_path: Optional[Path] = None):
        """Initialize the bottle inventory.

        Args:
            inventory_path: Path to load/save inventory data
        """
        self.inventory_path = inventory_path or (
            DATA_DIR / "bottles.json")
        self.bottles: List[Dict[str, Any]] = []
        self.bottle_hashes: Set[str] = set()

        # Create data directory if it doesn't exist
        self.inventory_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing inventory if it exists
        if self.inventory_path.exists():
            self.load_inventory()

    def load_inventory(self) -> None:
        """Load inventory from file."""
        try:
            with open(self.inventory_path, 'r', encoding='utf-8') as f:
                self.bottles = json.load(f)
                # Rebuild the hash set for duplicate checking
                self.bottle_hashes = {self._create_bottle_hash(
                    bottle) for bottle in self.bottles}
            logger.info(
                f"Loaded {len(self.bottles)} bottles from {self.inventory_path}")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Error loading inventory: {e}")
            self.bottles = []
            self.bottle_hashes = set()

    def save_inventory(self) -> None:
        """Save inventory to file."""
        self.inventory_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.inventory_path, 'w', encoding='utf-8') as f:
                json.dump(self.bottles, f, indent=2, ensure_ascii=False)
            logger.info(
                f"Saved {len(self.bottles)} bottles to {self.inventory_path}")
        except Exception as e:
            logger.error(f"Error saving inventory: {e}")

    def _create_bottle_hash(self, bottle: Dict[str, Any]) -> str:
        """Create a hash for duplicate detection.

        Uses important fields to create a unique identifier.
        """
        # Create a unique fingerprint from key bottle attributes
        key_attrs = [
            str(bottle.get('name', '')),
            str(bottle.get('producer', '')),
            str(bottle.get('spirit_type', '')),
            str(bottle.get('series', '')),
            str(bottle.get('abv', '')),
            str(bottle.get('age_years', '')),
            str(bottle.get('year_bottled', '')),
            str(bottle.get('year_distilled', '')),
            str(bottle.get('size', '')),
            str(bottle.get('baxus_class_id', ''))
        ]

        # Join attributes and create hash
        fingerprint = '|'.join(key_attrs)
        return hashlib.md5(fingerprint.encode()).hexdigest()

    def add_bottle(self, bottle_data: Dict[str, Any]) -> bool:
        """Add a bottle to the inventory if not a duplicate.

        Args:
            bottle_data: The bottle data to add

        Returns:
            bool: True if bottle was added, False if it was a duplicate
        """
        # Validate required fields
        if not bottle_data.get('name'):
            logger.warning("Bottle missing required 'name' field, skipping")
            return False

        # Normalize data structure
        normalized_bottle = self._normalize_bottle_data(bottle_data)

        # Check for duplicates
        bottle_hash = self._create_bottle_hash(normalized_bottle)
        if bottle_hash in self.bottle_hashes:
            logger.info(
                f"Duplicate bottle found: {normalized_bottle.get('name')} - skipping")
            return False

        # Add to inventory
        self.bottles.append(normalized_bottle)
        self.bottle_hashes.add(bottle_hash)
        logger.info(f"Added bottle: {normalized_bottle.get('name')}")
        return True

    def _extract_numeric_size(self, size_str: str) -> Optional[float]:
        """Extract numeric size from a string like '750ml' or '750 ml'.

        Handles various formats like:
        - 750ml
        - 750 ml
        - 750 ML
        - 750ml ml (duplicated unit)
        """
        if not size_str:
            return None

        try:
            # Remove any duplicate units (like "750ml ml")
            cleaned_str = size_str.lower().replace("ml ml", "ml")

            # Check for space between number and unit
            if " ml" in cleaned_str:
                parts = cleaned_str.split(" ml", 1)
                numeric_part = parts[0].strip()
            # Check for attached ml
            elif "ml" in cleaned_str:
                numeric_part = cleaned_str.replace("ml", "").strip()
            else:
                # Just extract digits and decimal points
                numeric_part = ''.join(
                    c for c in cleaned_str if c.isdigit() or c == '.')

            if numeric_part:
                return float(numeric_part)
            return None
        except (ValueError, TypeError):
            return None

    def _normalize_bottle_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize bottle data to ensure consistent structure."""
        # First extract attributes if present to make them available for normalization
        attributes = {}
        if "attributes" in data and isinstance(data["attributes"], dict):
            attributes = data["attributes"]

        # Log incoming data keys to debug
        logger.debug(f"Normalizing data with keys: {list(data.keys())}")
        if 'animationUrl' in data:
            logger.debug(
                f"animationUrl present in data: {data['animationUrl']}")

        normalized = {
            "name": data.get("name") or attributes.get("Name", ""),
            "description": data.get("description", ""),
            "type": data.get("spirit_type") or data.get("spiritType") or data.get("type") or
            attributes.get("Type") or attributes.get("Spirit Type", ""),
            "image_url": data.get("image_url") or data.get("imageUrl", ""),
            "animation_url": data.get("animation_url") or data.get("animationUrl") or attributes.get("Animation URL", ""),
            "producer": data.get("producer") or attributes.get("Producer", ""),
            # "producer_type": data.get("producer_type") or data.get("producerType") or
            # attributes.get("Producer Type", ""),
            "series": data.get("series") or attributes.get("Series", ""),
            "abv": self._safe_float(data.get("abv") or data.get("ABV") or attributes.get("ABV")),
            # "age_years": self._safe_float(data.get("age_years") or data.get("age") or
            #                               attributes.get("Age")),
            "age_statement": data.get("age_statement") or str(data.get("age") or
                                                              attributes.get("Age", "")),
            "country": data.get("country") or attributes.get("Country", ""),
            "region": data.get("region") or attributes.get("Region", ""),
            "year_bottled": data.get("year_bottled") or data.get("yearBottled") or
            attributes.get("Year Bottled", ""),
            "year_distilled": data.get("year_distilled") or data.get("yearDistilled") or
            attributes.get("Year Distilled", ""),
            # "packaging": data.get("packaging") or attributes.get("Packaging", ""),
            # "package_shot": data.get("package_shot") or data.get("packageShot") or
            # attributes.get("PackageShot", False),
            "baxus_class_id": data.get("baxus_class_id") or data.get("baxusClassId") or
            attributes.get("Baxus Class ID", ""),
            "baxus_class_name": data.get("baxus_class_name") or data.get("baxusClassName") or
            attributes.get("Baxus Class Name", ""),
            "size": data.get("size") or attributes.get("Size", ""),
            "nft_address": data.get("nftAddress") or data.get("id"),

        }

        # Try to extract size and normalize to milliliters
        size_str = str(normalized["size"]).lower(
        ) if normalized["size"] else ""
        if size_str:
            # Convert to numeric milliliters
            if "ml" in size_str or "milliliter" in size_str:
                size_value = self._extract_numeric_size(size_str)
                if size_value:
                    normalized["size"] = int(size_value)
            elif "l" in size_str or "liter" in size_str:
                size_value = self._extract_numeric_size(size_str)
                if size_value:
                    # Convert to ml
                    normalized["size"] = int(size_value * 1000)
            elif "oz" in size_str or "ounce" in size_str:
                size_value = self._extract_numeric_size(size_str)
                if size_value:
                    # Convert to ml (1 oz ≈ 29.57 ml)
                    normalized["size"] = int(size_value * 29.57)
            elif size_str.replace('.', '', 1).isdigit():
                # If just a number, assume it's milliliters
                normalized["size"] = int(float(size_str))

        # Set default size if missing or invalid
        if not normalized["size"] or not isinstance(normalized["size"], (int, float)):
            normalized["size"] = 750  # Default size

        # Remove any empty string values to keep the data clean
        return {k: v for k, v in normalized.items() if v != ""}

    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert a value to float, returning None if invalid."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def import_from_json(self, file_path: Path) -> int:
        """Import bottles from a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            int: Number of bottles added
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle both array and object formats
            bottles_to_import = []
            if isinstance(data, list):
                bottles_to_import = data
            elif isinstance(data, dict):
                if "_source" in data:
                    bottles_to_import = [data["_source"]]
                else:
                    bottles_to_import = [data]

            added_count = 0
            for bottle in bottles_to_import:
                if self.add_bottle(bottle):
                    added_count += 1

            return added_count

        except Exception as e:
            logger.error(f"Error importing from {file_path}: {e}")
            return 0

    def import_from_csv(self, file_path: Path) -> int:
        """Import bottles from a CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            int: Number of bottles added
        """
        try:
            bottles_to_import = []
            with open(file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                bottles_to_import = list(reader)

            added_count = 0
            for bottle in bottles_to_import:
                if self.add_bottle(bottle):
                    added_count += 1

            return added_count

        except Exception as e:
            logger.error(f"Error importing from {file_path}: {e}")
            return 0

    def export_to_csv(self, output_path: Path) -> bool:
        """Export inventory to CSV.

        Args:
            output_path: Path to save the CSV file

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.bottles:
            logger.warning("No bottles to export")
            return False

        try:
            # Get all unique keys
            fieldnames = set()
            for bottle in self.bottles:
                fieldnames.update(bottle.keys())
            fieldnames = sorted(list(fieldnames))

            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.bottles)

            logger.info(
                f"Exported {len(self.bottles)} bottles to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False

    def __len__(self) -> int:
        """Get the number of bottles in inventory."""
        return len(self.bottles)

    def _clean_name(self, name):
        """Clean bottle names for better matching."""
        if not name:
            return ""
        # Convert to lowercase
        name = name.lower()
        # Remove special characters, multiple spaces
        name = re.sub(r'[^\w\s]', ' ', name)
        name = re.sub(r'\s+', ' ', name)
        # Remove year designations and common words that might differ between datasets
        name = re.sub(r'\b\d{4}\b', '', name)  # Remove years
        name = re.sub(r'\byear\b', '', name)   # Remove "year" word
        name = re.sub(r'\bbottled\b', '', name)  # Remove "bottled" word
        name = re.sub(r'\brelease\b', '', name)  # Remove "release" word
        name = re.sub(r'\bedition\b', '', name)  # Remove "edition" word
        # Remove common terms that cause matching issues
        name = re.sub(r'\blimited\b', '', name)  # Remove "limited" word
        name = re.sub(r'\bsingle\b', '', name)   # Remove "single" word
        name = re.sub(r'\bbarrel\b', '', name)   # Remove "barrel" word
        name = re.sub(r'\bcask\b', '', name)     # Remove "cask" word
        name = re.sub(r'\bstrength\b', '', name)  # Remove "strength" word
        name = re.sub(r'\breserve\b', '', name)  # Remove "reserve" word
        name = re.sub(r'\bspecial\b', '', name)  # Remove "special" word
        # Normalize common variations
        name = re.sub(r'whisky', 'whiskey', name)  # Standardize whisky/whiskey
        # Treat similar sizes as close for matching
        name = re.sub(r'700\s*ml', '750ml', name)
        name = re.sub(r'720\s*ml', '750ml', name)
        name = re.sub(r'1\s*l', '1000ml', name)  # Convert 1L to 1000ml
        name = re.sub(r'1\.75\s*l', '1750ml', name)  # Convert 1.75L to 1750ml
        return name.strip()

    def _normalize_size(self, size):
        """Normalize bottle size to a numeric value in milliliters."""
        if not size:
            return 750  # Default size if missing

        # If already a number, just return it
        if isinstance(size, (int, float)):
            return int(size)

        # Convert size to string if needed
        size_str = str(size).lower()

        # Handle common size variations
        if any(x in size_str for x in ['700ml', '700 ml', '70cl', '70 cl']):
            return 750  # Treat 700ml as 750ml for matching
        if any(x in size_str for x in ['720ml', '720 ml', '72cl', '72 cl']):
            return 750  # Treat 720ml as 750ml for matching
        if any(x in size_str for x in ['1l', '1 l', '1ltr', '1 ltr', '100cl', '100 cl', '1000ml', '1000 ml']):
            return 1000  # Normalize 1L variants
        if any(x in size_str for x in ['1.75l', '1.75 l', '175cl', '175 cl', '1750ml', '1750 ml']):
            return 1750  # Normalize 1.75L variants

        # Extract numeric part
        if "ml" in size_str or "milliliter" in size_str:
            match = re.search(r'(\d+(?:\.\d+)?)', size_str)
            if match:
                return int(float(match.group(1)))
        elif "l" in size_str or "liter" in size_str:
            match = re.search(r'(\d+(?:\.\d+)?)', size_str)
            if match:
                return int(float(match.group(1)) * 1000)
        elif "oz" in size_str or "ounce" in size_str:
            match = re.search(r'(\d+(?:\.\d+)?)', size_str)
            if match:
                return int(float(match.group(1)) * 29.57)
        elif re.search(r'^\d+(?:\.\d+)?$', size_str):
            # Plain number - assume milliliters
            return int(float(size_str))

        # If we can extract any number, assume it's milliliters
        match = re.search(r'(\d+)', size_str)
        if match:
            return int(match.group(1))

        return 750  # Default if no numeric part found

    def _similarity_score(self, str1, str2):
        """Calculate string similarity using SequenceMatcher."""
        return SequenceMatcher(None, str1, str2).ratio()

    def enrich_from_csv(self, csv_file_path: Path, save_enriched: bool = False) -> int:
        """Enrich bottles data with information from a CSV dataset.

        Args:
            csv_file_path: Path to the CSV file with bottle data
            save_enriched: Whether to save the enriched data after enrichment

        Returns:
            int: Number of bottles that were enriched
        """
        if not csv_file_path.exists():
            logger.error(f"CSV dataset not found: {csv_file_path}")
            return 0

        # Load CSV dataset
        csv_data = []
        logger.info(f"Loading data from {csv_file_path}")
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                csv_data.append(row)

        logger.info(f"Found {len(csv_data)} entries in CSV dataset")

        # Create lookup dictionary from CSV data
        csv_lookup = {}

        for entry in csv_data:
            # Create a key using name and size for lookup
            clean_entry_name = self._clean_name(entry['name'])
            size_key = self._normalize_size(entry['size'])
            lookup_key = f"{clean_entry_name}_{size_key}"
            csv_lookup[lookup_key] = entry

        # Track statistics
        matched_count = 0
        unmatched_count = 0
        exact_matches = 0
        fuzzy_matches = 0

        # List of keys that should be stored as float (prices)
        float_keys = ['abv', 'avg_msrp', 'fair_price', 'shelf_price', 'price',
                      'retail_price', 'market_price', 'secondary_price',
                      'auction_price', 'value']

        # Enrich bottles data
        logger.info("Enriching bottles data...")
        for bottle in self.bottles:
            if not bottle.get('name'):
                unmatched_count += 1
                continue

            # Clean and normalize bottle name and size
            clean_bottle_name = self._clean_name(bottle['name'])
            bottle_size = self._normalize_size(bottle.get('size', 750))
            lookup_key = f"{clean_bottle_name}_{bottle_size}"

            # Try exact match first
            if lookup_key in csv_lookup:
                entry = csv_lookup[lookup_key]
                # Add all columns from CSV entry
                for key, value in entry.items():
                    # Don't overwrite existing values unless they're empty
                    if key not in bottle or not bottle[key]:
                        # Handle empty values
                        if not value or value.strip() == '':
                            bottle[key] = None
                        # Handle numeric values
                        elif isinstance(value, str) and value.replace('.', '', 1).isdigit():
                            try:
                                # Store prices as float, other numeric values as int
                                if any(price_term in key.lower() for price_term in float_keys):
                                    bottle[key] = float(value)
                                else:
                                    # Use int only if the value is a whole number
                                    if '.' in value:
                                        val = float(value)
                                        bottle[key] = int(
                                            val) if val.is_integer() else val
                                    else:
                                        bottle[key] = int(value)
                            except (ValueError, TypeError):
                                bottle[key] = None
                        else:
                            bottle[key] = value
                matched_count += 1
                exact_matches += 1
                continue

            # Try fuzzy matching if exact match fails
            else:
                best_match = None
                best_score = 0

                for key, entry in csv_lookup.items():
                    # Extract size from the key
                    entry_size = key.split('_')[-1]
                    # Only match if sizes are comparable
                    if int(entry_size) == bottle_size:
                        entry_name = self._clean_name(entry['name'])
                        score = self._similarity_score(
                            clean_bottle_name, entry_name)
                        # Using stricter threshold as requested
                        if score > 0.95 and score > best_score:
                            best_score = score
                            best_match = entry

                if best_match:
                    # Add all columns from the best match
                    for key, value in best_match.items():
                        # Don't overwrite existing values unless they're empty
                        if key not in bottle or not bottle[key]:
                            # Handle empty values
                            if not value or value.strip() == '':
                                bottle[key] = None
                            # Handle numeric values
                            elif isinstance(value, str) and value.replace('.', '', 1).isdigit():
                                try:
                                    # Store prices as float, other numeric values as int
                                    if any(price_term in key.lower() for price_term in float_keys):
                                        bottle[key] = float(value)
                                    else:
                                        # Use int only if the value is a whole number
                                        if '.' in value:
                                            val = float(value)
                                            bottle[key] = int(
                                                val) if val.is_integer() else val
                                        else:
                                            bottle[key] = int(value)
                                except (ValueError, TypeError):
                                    bottle[key] = None
                            else:
                                bottle[key] = value
                    matched_count += 1
                    fuzzy_matches += 1
                else:
                    # If we reach here, no match was found
                    unmatched_count += 1

        # Log detailed results
        logger.info(
            f"Enrichment complete. Total matched: {matched_count}, Unmatched: {unmatched_count}")
        logger.info(
            f"Match breakdown - Exact: {exact_matches}, Fuzzy: {fuzzy_matches}")
        logger.info(f"Total bottles processed: {len(self.bottles)}")

        # Save enriched data if requested
        if save_enriched:
            self.save_inventory()
            logger.info(f"Saved enriched inventory to {self.inventory_path}")

        return matched_count


def load_baxus_listings(file_path: Path) -> List[Dict[str, Any]]:
    """Load Baxus listings data from JSON file.

    Args:
        file_path: Path to the JSON file with Baxus listings

    Returns:
        List of listings data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Normalize data to a list
        if isinstance(data, dict):
            data = [data]

        # Extract _source data from each listing
        listings = []
        for item in data:
            if isinstance(item, dict):
                listing_data = {}

                # Get the core data - either from _source or the item itself
                if "_source" in item:
                    # Preserve all fields from _source
                    listing_data = item["_source"].copy()

                    # Log important fields for debugging
                    logger.debug(
                        f"Original _source fields: {list(listing_data.keys())}")
                    if 'animationUrl' in listing_data:
                        logger.debug(
                            f"Found animationUrl in _source: {listing_data['animationUrl']}")
                else:
                    # If no _source, use the entire item
                    listing_data = item.copy()

                    # Log important fields for debugging
                    logger.debug(
                        f"Original item fields: {list(listing_data.keys())}")
                    if 'animationUrl' in listing_data:
                        logger.debug(
                            f"Found animationUrl in item: {listing_data['animationUrl']}")

                # Process attributes if present
                if "attributes" in listing_data and isinstance(listing_data["attributes"], dict):
                    # We'll keep the original attributes dict for reference
                    # but also bring attributes to the top level for easier access
                    for attr_key, attr_value in listing_data["attributes"].items():
                        # Only add if not already present at top level
                        if attr_key not in listing_data:
                            listing_data[attr_key] = attr_value

                listings.append(listing_data)

        logger.info(f"Loaded {len(listings)} Baxus listings from {file_path}")
        return listings

    except Exception as e:
        logger.error(f"Error loading Baxus listings from {file_path}: {e}")
        return []


def resolve_path(path_str: str) -> Path:
    """Resolve a path, converting relative paths to absolute when needed.

    Args:
        path_str: The path string to resolve

    Returns:
        Resolved Path object
    """
    path = Path(path_str)
    if not path.is_absolute():
        if not path.exists():
            # Try relative to data directory
            data_path = DATA_DIR / path
            if data_path.exists():
                return data_path
        # Default to relative from project root
        return PROJECT_ROOT / path
    return path


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Bottle inventory management script")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # General inventory arguments
    parser.add_argument("--inventory-file", type=str, default=str(DATA_DIR / "bottles.json"),
                        help=f"Path to inventory file (default: {DATA_DIR / 'bottles.json'})")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")

    # Import from JSON command
    import_json_parser = subparsers.add_parser(
        "import-json", help="Import bottles from JSON file")
    import_json_parser.add_argument(
        "file", type=str, help="Path to JSON file to import")

    # Import from CSV command
    import_csv_parser = subparsers.add_parser(
        "import-csv", help="Import bottles from CSV file")
    import_csv_parser.add_argument(
        "file", type=str, help="Path to CSV file to import")

    # Export to CSV command
    export_csv_parser = subparsers.add_parser(
        "export-csv", help="Export inventory to CSV file")
    export_csv_parser.add_argument(
        "file", type=str, help="Path to save CSV export")

    # Extract Baxus listings command
    baxus_parser = subparsers.add_parser(
        "extract-baxus", help="Extract data from Baxus listings")
    baxus_parser.add_argument(
        "file", type=str, help="Path to Baxus listings JSON file")
    baxus_parser.add_argument("--export-csv", type=str,
                              help="Export resulting inventory to CSV file")
    baxus_parser.add_argument("--csv", type=str,
                              help="Path to CSV file to enrich bottle data (e.g., 501 Bottle Dataset.csv)")

    # Enrich from CSV command
    enrich_parser = subparsers.add_parser(
        "enrich-csv", help="Enrich bottles with data from CSV file")
    enrich_parser.add_argument(
        "file", type=str, help="Path to CSV file with bottle data (e.g., 501 Bottle Dataset.csv)")
    enrich_parser.add_argument("--output", type=str,
                               help="Save enriched inventory to a different file")

    args = parser.parse_args()

    # Enable debug logging if requested
    if hasattr(args, 'debug') and args.debug:
        enable_debug_logging()

    # Initialize inventory
    inventory_path = resolve_path(args.inventory_file)
    inventory = BottleInventory(inventory_path)
    logger.info(f"Starting with {len(inventory)} bottles in inventory")

    # Process commands
    if args.command == "import-json":
        import_path = resolve_path(args.file)
        if import_path.exists():
            added = inventory.import_from_json(import_path)
            logger.info(f"Added {added} bottles from {import_path}")
        else:
            logger.error(f"Import file not found: {import_path}")

    elif args.command == "import-csv":
        import_path = resolve_path(args.file)
        if import_path.exists():
            added = inventory.import_from_csv(import_path)
            logger.info(f"Added {added} bottles from {import_path}")
        else:
            logger.error(f"Import file not found: {import_path}")

    elif args.command == "export-csv":
        export_path = resolve_path(args.file)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        inventory.export_to_csv(export_path)

    elif args.command == "extract-baxus":
        listings_path = resolve_path(args.file)
        if not listings_path.exists():
            logger.error(f"Baxus listings file not found: {listings_path}")
            return

        listings = load_baxus_listings(listings_path)
        if not listings:
            logger.error("No valid Baxus listings data found")
            return

        # Add listings to inventory
        added_count = 0
        for listing in listings:
            if inventory.add_bottle(listing):
                added_count += 1

        logger.info(f"Added {added_count} bottles from Baxus listings")

        # Enrich with CSV data if provided
        if hasattr(args, 'csv') and args.csv:
            enrich_path = resolve_path(args.csv)
            if enrich_path.exists():
                matched = inventory.enrich_from_csv(
                    enrich_path, save_enriched=True)
                logger.info(f"Enriched {matched} bottles from {enrich_path}")
            else:
                logger.error(f"CSV enrichment file not found: {enrich_path}")

        # Export to CSV if requested
        if hasattr(args, 'export_csv') and args.export_csv:
            export_path = resolve_path(args.export_csv)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            inventory.export_to_csv(export_path)
            logger.info(f"Exported inventory to {export_path}")

    elif args.command == "enrich-csv":
        enrich_path = resolve_path(args.file)
        if not enrich_path.exists():
            logger.error(f"CSV file not found: {enrich_path}")
            return

        # Enrich the inventory
        matched = inventory.enrich_from_csv(enrich_path, save_enriched=True)
        logger.info(f"Enriched {matched} bottles from {enrich_path}")

        # Export to a different file if requested
        if hasattr(args, 'output') and args.output:
            output_path = resolve_path(args.output)
            # Create a copy of the inventory with a different path
            output_inventory = BottleInventory(output_path)
            output_inventory.bottles = inventory.bottles.copy()
            output_inventory.bottle_hashes = inventory.bottle_hashes.copy()
            output_inventory.save_inventory()
            logger.info(f"Saved enriched inventory to {output_path}")

    # Save inventory after all operations
    inventory.save_inventory()

    logger.info(f"Finished with {len(inventory)} bottles in inventory")


if __name__ == "__main__":
    main()
