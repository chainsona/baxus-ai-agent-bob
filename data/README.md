# Baxus Listings Data

This directory contains data files related to Baxus listings.

## Files

- `baxus-listings.json` - Original Baxus listings data (nested structure format)
- `baxus-listing-sample.json` - Single listing sample for testing (nested structure format)
- `baxus-asset-sample.json` - Single asset sample for testing (flat structure format)
- `cleaned-listings.json` - Cleaned and normalized JSON data
- `cleaned-listings.csv` - Cleaned data in CSV format for easy import into spreadsheets

## Data Cleaning Script

The root directory contains a script `clean_listings.py` that processes Baxus listing data and outputs both JSON and CSV formats with a normalized structure.

### Usage

```bash
# Process the default file (data/baxus-listings.json)
./clean_listings.py

# Process a specific file
./clean_listings.py data/baxus-asset-sample.json
```

When processing a specific file, the output files will be named based on the input filename:
- `data/cleaned-{input-name}.json`
- `data/cleaned-{input-name}.csv`

### Features

- Normalizes data structure
- Handles both nested attribute format and flat structure format
- Extracts nested attributes to a flat structure
- Handles type conversions (e.g., strings to numbers)
- Standardizes field naming using snake_case
- Parses size fields to extract volume measurements
- Outputs both JSON and CSV formats

### Supported Input Formats

The script handles two main data formats:

1. **Nested attribute format** - where most fields are in an "attributes" object
2. **Flat structure format** - where fields are at the root level with camelCase naming

### Output Data Structure

The cleaned data includes the following fields:

- `id` - Unique identifier (from nftAddress)
- `name` - Name of the listing
- `description` - Description of the item
- `spirit_type` - Type of spirit (bourbon, scotch, etc.)
- `image_url` - URL to the item image
- `animation_url` - URL to any animation
- `price` - Listing price
- `status` - Status of the listing
- `owner_address` - Address of the owner
- `buyer_address` - Address of the buyer (if sold)
- `is_listed` - Whether the item is currently listed
- `listed_date` - Date when the item was listed
- `last_heartbeat` - Last heartbeat timestamp
- `producer` - Producer/distillery name
- `producer_type` - Type of producer
- `series` - Series name
- `abv` - Alcohol by volume percentage
- `age_years` - Age in years (if available)
- `age_statement` - Age statement string
- `country` - Country of origin
- `region` - Region of origin
- `year_bottled` - Year bottled
- `year_distilled` - Year distilled
- `packaging` - Packaging type (bottle, etc.)
- `package_shot` - Whether it's a package shot
- `baxus_class_id` - Baxus class identifier
- `baxus_class_name` - Baxus class name
- `size` - Size string (e.g., "750 ml")
- `volume_ml` - Volume in milliliters 