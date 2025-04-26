# Bob - BAXUS Whisky Recommendation AI Agent

Bob is an AI agent that specializes in whisky recommendations. He analyzes users' virtual bar collections in the BAXUS ecosystem to provide personalized bottle recommendations and collection insights.

## Features

- **Collection Analysis**: Identifies patterns in user preferences (regions, styles, price points, etc.)
- **Recommendation Engine**:
  - Suggests bottles similar to the user's existing collection
  - Recommends diverse bottles to expand the user's collection
  - Provides personalized recommendations within similar price ranges
- **Conversational Interface**: Chat with Bob to get recommendations and whisky insights
- **User Data File Support**: Import your collection directly from BAXUS JSON export files

## Technical Architecture

- **Vector Database**: Uses PostgreSQL with pgvector for similarity searches
- **Conversational AI**: Built with LangChain and LangGraph
- **Embeddings**: Uses OpenAI's text-embedding-3-small model

## Requirements

- Python 3.9+
- PostgreSQL 14+ with pgvector extension
- OpenAI API key

## Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/baxus-bob.git
cd baxus-bob
```

2. **Install dependencies**

```bash
# Using Poetry (recommended)
poetry install

# Or with pip
pip install -r requirements.txt
```

3. **Configure environment variables**

```bash
cp .env.example .env
# Edit .env with your database credentials and OpenAI API key
```

4. **Setup the database**

First, create a PostgreSQL database and ensure the pgvector extension is installed:

```bash
createdb baxus
psql -d baxus -c 'CREATE EXTENSION IF NOT EXISTS vector;'
```

Then run the setup script to create the schema and load data:

```bash
python setup_database.py --csv /path/to/bottles.csv
```

## Usage

### Command-line Interface

#### Analyzing your collection

```bash
# Using directly specified bottles
python bob.py

# Using a BAXUS user.json file
python bob.py --user-json /path/to/user.json
```

#### Custom recommendations

```bash
# Control the number of recommendations
python bob.py --user-json /path/to/user.json --similar 5 --diverse 3

# Ask Bob a question
python bob.py --chat "What kind of whisky should I try if I like smoky flavors?"
```

### In your own code

```python
import asyncio
from bob import Bob

async def main():
    # Initialize Bob
    bob = Bob()

    # Example 1: Using direct data
    user_data = [
        {"product": {"id": 164, "name": "Buffalo Trace", "spirit": "Bourbon"}},
        {"product": {"id": 2848, "name": "Lagavulin 16", "spirit": "Scotch"}}
    ]

    analysis = await bob.analyze_user_bar(user_data)
    recommendations = await bob.get_recommendations(user_data)

    # Example 2: Using a JSON file
    file_analysis = await bob.analyze_user_bar_from_file("path/to/user.json")
    file_recommendations = await bob.get_recommendations_from_file("path/to/user.json")

    # Chat with Bob
    response = await bob.chat("What kind of whisky should I try if I like smoky flavors?")
    print(f"Bob: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Web API

To run Bob as a web service:

```bash
python api.py
```

This starts a web server with the following endpoints:

- `POST /analyze` - Analyze a user's collection
- `POST /upload/analyze` - Analyze a user's collection from an uploaded JSON file
- `POST /recommend` - Get recommendations for a user
- `POST /upload/recommend` - Get recommendations from an uploaded JSON file
- `POST /chat` - Chat with Bob
- `POST /search` - Search for bottles similar to a query

### Example: Upload a user.json file

```bash
# Using curl to upload a file for analysis
curl -X POST "http://localhost:8000/upload/analyze" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/user.json"

# To get recommendations from a file
curl -X POST "http://localhost:8000/upload/recommend" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/user.json" \
  -F "num_similar=3" \
  -F "num_diverse=2"
```

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

## License

MIT

## Acknowledgements

- [OpenAI](https://openai.com/) for embedding and LLM services
- [LangChain](https://github.com/langchain-ai/langchain) and [LangGraph](https://github.com/langchain-ai/langgraph) for the agent framework
- [pgvector](https://github.com/pgvector/pgvector) for vector similarity in PostgreSQL

## Docker Setup

The Docker configuration files are located in the `docker` directory:

```
docker/
├── docker-compose.yml   # Docker Compose configuration
└── pgadmin-config/      # PgAdmin configuration
    └── servers.json     # PgAdmin servers configuration
```

To start the Docker containers:

```bash
docker-compose -f docker/docker-compose.yml -p baxus-bob up -d
```

This will start:

- PostgreSQL database with pgvector extension at port 5452
- PgAdmin web interface at http://localhost:5050 (admin@example.com / admin)
  - PgAdmin will automatically connect to the database without prompting for a password
