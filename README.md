# 🥃 BAXUS BOB - Expert Whisky AI Agent

Bob is an AI-powered recommendation system for the BAXUS whisky platform. It analyzes users' virtual whisky collections to provide personalized recommendations and insights.

## 🚀 Live Demo

[Try the BOB AI API](https://baxus-bob-api.maikers.com/recommendations/carriebaxus?similar=2&diverse=3&diversity_ratio=0.4)

## 🌐 UI Repository

For the frontend implementation that interacts with this API, visit the [BAXUS BOB UI repository](https://github.com/chainsona/baxus-ai-agent-bob-ui).

The UI provides a user-friendly interface for:
- Viewing personalized whisky recommendations
- Chatting with Bob about whisky
- Exploring your collection insights
- Searching for new bottles to try or collect


## Features

- 🔍 **Collection Analysis**: Analyzes a user's whisky collection to extract insights about their preferences and taste profile
- 🎯 **Personalized Recommendations**: Suggests both similar bottles to what users already enjoy and options to diversify their collection
- 👤 **User Profiles**: Generates taste profiles based on collection analysis
- 📊 **Investment Analysis**: Provides collection value estimates and breakdowns by whisky type
- 💬 **Conversational Interface**: Chat with Bob for whisky recommendations and whisky knowledge
- 🔎 **Semantic Search**: Search for bottles using natural language descriptions

## Tech Stack

- 🐍 **Backend**: Python with FastAPI
- 🗄️ **Database**: PostgreSQL with pgvector extension for vector similarity search
- 🧠 **AI Models**: OpenAI GPT-4 for conversation, text-embedding-3-small for embeddings
- 🔄 **Orchestration**: LangChain and LangGraph for AI agent workflows
- 🐳 **Deployment**: Docker and Kubernetes for containerization and orchestration
- 🛠️ **Development**: Poetry for dependency management

## API Endpoints

### Health Check

Check system health status.

```
GET /health
```

Response:

```json
{
  "status": "healthy",
  "service": "Bob AI Agent",
  "version": "1.0.0",
  "components": {
    "api": "healthy",
    "database": "healthy"
  }
}
```

### User Profile

Get a user's profile including taste preferences and collection stats.

```
GET /profile/{username}
```

Response:

```json
{
  "username": "user123",
  "taste_profile": {
    "dominant_flavors": ["sweet", "woody", "spicy"],
    "flavor_profiles": {
      "sweet": 75,
      "smoky": 30,
      "spicy": 60,
      "fruity": 25,
      "woody": 65,
      "smooth": 40,
      "floral": 15
    },
    "favorite_type": "Bourbon",
    "favorite_region": "Kentucky"
  },
  "collection": {
    "stats": {
      "bottle_count": 10,
      "type_count": 3,
      "region_count": 4,
      "diversity_score": 7.5,
      "types_distribution": {
        "Bourbon": 6,
        "Scotch": 3,
        "Rye": 1
      },
      "regions_distribution": {
        "Kentucky": 5,
        "Scotland": 3,
        "Tennessee": 1,
        "Canada": 1
      }
    },
    "investment": {
      "estimated_value": {
        "low": 500,
        "high": 1500,
        "average": 1000,
        "total": 10000
      },
      "bottle_count": 10,
      "bottles_with_price": 8,
      "price_range": {
        "min": 30,
        "max": 200
      },
      "value_by_type": {
        "Bourbon": {
          "count": 6,
          "total_value": 6000,
          "average_price": 1000
        },
        "Scotch": {
          "count": 3,
          "total_value": 3000,
          "average_price": 1000
        },
        "Rye": {
          "count": 1,
          "total_value": 1000,
          "average_price": 1000
        }
      }
    },
    "bottles": [
      {
        "id": 123456,
        "name": "Heaven Hill Bottled In Bond 7 Year",
        "image_url": "https://assets.baxus.co/123456/123456.jpg",
        "type": "Bourbon",
        "spirit": "Bourbon",
        "price": 47.19
      }
    ]
  }
}
```

### Recommendations

Get personalized bottle recommendations for a user.

```
GET /recommendations/{username}?similar=5&diverse=3&diversity_ratio=0.4
```

Query Parameters:

- `similar`: Number of similar recommendations to return (default: 5)
- `diverse`: Number of diverse recommendations to return (default: 3)
- `diversity_ratio`: Ratio between 0 and 1 that controls how different the diverse recommendations are (default: 0.4). Higher values:
  - Prioritizes bottles that are more different from the user's collection
  - At values > 0.7, strongly prioritizes completely new types and regions
  - At values > 0.9, focuses almost exclusively on introducing new experiences

Response:

```json
{
  "similar": [
    {
      "id": 123,
      "name": "Eagle Rare 10 Year",
      "description": "A delicate, complex bourbon...",
      "type": "Bourbon",
      "producer": "Buffalo Trace",
      "region": "Kentucky",
      "image_url": "https://assets.baxus.co/123/123.jpg",
      "nft_address": "0x123abc...",
      "similarity": 0.92,
      "reason": "Similar to bottles in your collection",
      "flavor_profile": {
        "sweet": 0.75,
        "woody": 0.65,
        "spicy": 0.6,
        "smoky": 0.3,
        "fruity": 0.25,
        "smooth": 0.4,
        "floral": 0.15
      }
    }
  ],
  "diverse": [
    {
      "id": 456,
      "name": "Lagavulin 16",
      "description": "A powerful, smoky Islay single malt...",
      "type": "Scotch",
      "producer": "Lagavulin",
      "region": "Islay",
      "image_url": "https://assets.baxus.co/456/456.jpg",
      "nft_address": "0x456def...",
      "reason": "Adds Scotch to diversify your collection",
      "flavor_profile": {
        "sweet": 0.25,
        "woody": 0.4,
        "spicy": 0.35,
        "smoky": 0.9,
        "fruity": 0.15,
        "smooth": 0.6,
        "floral": 0.1
      }
    }
  ]
}
```

### Chat

Chat with Bob the whisky expert.

```
POST /chat
```

Request:

```json
{
  "message": "What makes bourbon different from other whiskeys?",
  "username": "user123",
  "conversation_history": [
    {
      "role": "user",
      "content": "Hello Bob!"
    },
    {
      "role": "assistant",
      "content": "Hello! I'm Bob, your whisky expert. How can I help you today?"
    }
  ],
  "stream": false
}
```

Response:

```json
{
  "response": "Bourbon has a few key requirements that set it apart: it must be made in the United States, the mash bill must contain at least 51% corn, it must be aged in new charred oak barrels, and it must be distilled to no more than 160 proof (80% ABV) and entered into the barrel for aging at no more than 125 proof (62.5% ABV). The corn-heavy mash bill gives bourbon its characteristic sweetness compared to other whiskeys."
}
```

For streaming responses, set `stream: true` in the request to receive a text/event-stream response.

### Search

Search for bottles similar to a query.

```
POST /search
```

Request:

```json
{
  "query": "sweet bourbon with caramel",
  "limit": 5
}
```

Response: Array of matching bottles with similarity scores and flavor profiles.

## Installation & Setup

### Requirements

- Python 3.10+
- PostgreSQL with pgvector extension
- OpenAI API key

### Environment Variables

Create a `.env` file with:

```
OPENAI_API_KEY=your_openai_api_key
PGVECTOR_HOST=localhost
PGVECTOR_PORT=5432
PGVECTOR_DB=baxus
PGVECTOR_USER=postgres
PGVECTOR_PASSWORD=postgres
BAXUS_API_URL=https://services.baxus.co/api
LOG_LEVEL=INFO
```

### Installation

1. Clone the repository
2. Install dependencies:

```
poetry install
```

### Running

Start the API server:

```
python api.py
```

For development with hot-reload:

```
DEBUG=true python api.py
```

## Data Ingestion with Docker Compose

After deploying the Bob services with Docker Compose, you need to populate the database with bottle data. This section guides you through the data ingestion process.

### 1. Start Docker Compose

First, ensure the Docker Compose services are up and running:

```bash
cd docker
docker compose up -d
```

This will start:

- PostgreSQL with pgvector extension at port 5452
- pgAdmin web interface at port 5050 (accessible at http://localhost:5050)

### 2. Prepare Your Data

Place your bottle data files in the `data/` directory:

- **Required**: `bottles.json` - Primary bottle data file (use `data/bottles.json` as the default location)
- **Optional**: `501 Bottle Dataset.csv` - Additional metadata for enriching bottle data

### 3. Run the Ingestion Process

Before running the ingestion process, ensure you have:

- Set up your environment variables in the `.env` file, particularly the `OPENAI_API_KEY`

The ingestion script supports the following options:

```bash
# Basic ingestion from bottles.json
python scripts/ingest_bottles_to_pgvector.py

# Ingestion with enrichment from CSV data
python scripts/ingest_bottles_to_pgvector.py --bottles data/bottles.json --csv "data/501 Bottle Dataset.csv"
```

### 4. Monitor the Ingestion Process

The ingestion process will:

1. Set up the database schema with the pgvector extension
2. Generate embeddings for each bottle using OpenAI's text-embedding-3-small model
3. Ingest the bottle data in batches
4. Create vector indexes for similarity searches

You'll see progress logs indicating the number of bottles processed and when each batch is committed.

### 5. Verify Database Population

After ingestion completes, you can verify the data was properly loaded:

1. Access pgAdmin at http://localhost:5050
   - Username: admin@example.com
   - Password: admin
2. Connect to the PostgreSQL server:
   - Host: pgvector (or localhost if connecting outside Docker)
   - Port: 5432 (or 5452 if connecting from host)
   - Username: baxus
   - Password: baxuspwd
   - Database: baxus
3. Run a query to check the bottles table:
   ```sql
   SELECT COUNT(*) FROM bottles;
   ```

### 6. Troubleshooting

If you encounter issues during ingestion:

- **Connection errors**: Ensure the PostgreSQL container is running and accessible
- **OpenAI API errors**: Check your API key is correctly set in the .env file
- **Memory issues**: The script processes bottles in batches to avoid memory problems, but you may need to reduce batch sizes for very large datasets

Once the data ingestion is complete, Bob will be able to provide accurate recommendations and insights based on the bottle data.

## Usage

Connect to the API from your frontend application or test with curl:

```bash
# Get a user profile
curl http://localhost:8000/profile/username

# Get recommendations
curl http://localhost:8000/recommendations/username?similar=5&diverse=3&diversity_ratio=0.4

# Chat with Bob
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What whisky should I try next?"}'

# Search for bottles
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "sweet bourbon", "limit": 5}'
```

## Technical Architecture

- **Vector Database**: Uses PostgreSQL with pgvector for similarity searches
- **Conversational AI**: Built with LangChain and LangGraph
- **Embeddings**: Uses OpenAI's text-embedding-3-small model
- **Recommendation Engine**: Hybrid recommender that combines vector similarity with knowledge-based diversification

## BAXUS User Data Format

Bob can analyze user data exported from the BAXUS app. The expected format is:

```json
[
  {
    "id": 123456,
    "product": {
      "id": 13266,
      "name": "Heaven Hill Bottled In Bond 7 Year",
      "spirit": "Bourbon",
      "proof": 100,
      "shelf_price": 47.19
    }
  },
  {
    "id": 123457,
    "product": {
      "id": 24961,
      "name": "Rare Perfection 14 Year",
      "spirit": "Canadian Whisky",
      "proof": 100.7,
      "shelf_price": 179.39
    }
  }
]
```

The most important fields are the product IDs, which Bob uses to match against the bottle database.
