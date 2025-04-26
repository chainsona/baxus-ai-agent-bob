"""
Bob - Whisky Recommendation AI Agent
-----------------------------------
An AI assistant that analyzes users' virtual bars within the BAXUS ecosystem
to provide personalized bottle recommendations and collection insights.

Bob uses pgvector for similarity searches and LangChain/LangGraph for the conversational interface.

Dependencies:
    pip install langchain langchain-openai langgraph psycopg python-dotenv pandas numpy
"""

import os
import json
import logging
import asyncio
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
from dotenv import load_dotenv

# Database and vector search
import psycopg
from psycopg.rows import dict_row

# LangChain components
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph import END, MessageGraph

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler("bob_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.debug("Environment variables loaded")


class BottleUtils:
    """Utility functions for working with bottle data and embeddings."""

    @staticmethod
    def get_db_connection_string() -> str:
        """Get the database connection string from environment variables."""
        host = os.getenv("PGVECTOR_HOST", "localhost")
        port = os.getenv("PGVECTOR_PORT", "5432")
        dbname = os.getenv("PGVECTOR_DB", "baxus")
        user = os.getenv("PGVECTOR_USER", "postgres")
        password = os.getenv("PGVECTOR_PASSWORD", "postgres")

        connection_string = f"postgres://{user}:{password}@{host}:{port}/{dbname}"
        logger.debug(f"DB connection string created for host: {host}")
        return connection_string

    @staticmethod
    def get_embedding(text: str) -> List[float]:
        """Get embedding for text using OpenAI's embedding model."""
        import openai

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


class BottleRecommender:
    """Core recommendation engine for whisky bottles."""

    def __init__(self, csv_path: Optional[str] = None):
        """Initialize the recommender with optional dataset."""
        self.df = None
        if csv_path and os.path.exists(csv_path):
            logger.debug(f"Loading bottle data from CSV: {csv_path}")
            self.df = pd.read_csv(csv_path)
            # Clean and prepare the data
            self._prepare_data()
            logger.debug(f"Loaded {len(self.df)} bottles from CSV")
        else:
            logger.debug("No CSV data provided or file not found")

    def _prepare_data(self):
        """Clean and prepare the bottle dataset."""
        if self.df is not None:
            logger.debug("Preparing bottle dataset")
            # Fix missing values
            self.df["proof"] = self.df["proof"].fillna(self.df["abv"] * 2)
            self.df["abv"] = self.df["abv"].fillna(self.df["proof"] / 2)
            self.df["spirit_type"] = self.df["spirit_type"].fillna("Unknown")

            # Ensure numeric columns
            numeric_cols = ["avg_msrp", "price", "total_score",
                            "vote_count", "bar_count"]
            for col in numeric_cols:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(
                        self.df[col], errors="coerce").fillna(0)

            logger.debug(
                f"Data preparation complete. Missing values: {self.df.isna().sum().sum()}")

    async def search_similar_bottles(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for bottles similar to the query using pgvector similarity."""
        try:
            logger.debug(
                f"Searching for bottles similar to: '{query}' (limit: {limit})")
            # Generate embedding for the query
            query_embedding = BottleUtils.get_embedding(query)

            async with await psycopg.AsyncConnection.connect(BottleUtils.get_db_connection_string()) as conn:
                # Use dict_row for easier result handling
                conn.row_factory = dict_row
                async with conn.cursor() as cur:
                    # Perform similarity search using cosine distance
                    logger.debug("Executing similarity search query")
                    await cur.execute("""
                        SELECT 
                            id, name, description, type, producer, series, 
                            abv, age_statement, country, region,
                            1 - (embedding <=> %s::vector) AS similarity
                        FROM bottles
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """, (query_embedding, query_embedding, limit))

                    results = await cur.fetchall()
                    logger.debug(f"Found {len(results)} similar bottles")
                    return results

        except Exception as e:
            logger.error(f"Error searching bottles: {e}", exc_info=True)
            return []

    async def recommend_from_bottle_ids(
        self,
        bottle_ids: List[int],
        limit: int = 5,
        price_range: Optional[Tuple[float, float]] = None
    ) -> List[Dict[str, Any]]:
        """Recommend bottles based on a list of bottle IDs the user likes."""
        try:
            if not bottle_ids:
                logger.debug("No bottle IDs provided for recommendation")
                return []

            # Format for SQL IN clause
            bottle_ids_str = ','.join(str(id) for id in bottle_ids)
            logger.debug(
                f"Recommending bottles based on {len(bottle_ids)} bottle IDs")

            async with await psycopg.AsyncConnection.connect(BottleUtils.get_db_connection_string()) as conn:
                conn.row_factory = dict_row
                async with conn.cursor() as cur:
                    # First get embeddings for the liked bottles
                    logger.debug("Fetching embeddings for liked bottles")
                    await cur.execute(f"""
                        SELECT id, embedding
                        FROM bottles
                        WHERE id IN ({bottle_ids_str})
                    """)

                    liked_bottles = await cur.fetchall()
                    if not liked_bottles:
                        logger.debug(
                            "No embeddings found for the provided bottle IDs")
                        return []

                    # Average the embeddings of liked bottles to create a profile vector
                    embeddings = [bottle["embedding"]
                                  for bottle in liked_bottles]

                    # Convert embeddings to numeric arrays if they're strings
                    embeddings = self._convert_embeddings(embeddings)
                    logger.debug(
                        f"Embeddings shape: {embeddings[0].shape if embeddings else 'no embeddings'}")

                    avg_embedding = np.mean(embeddings, axis=0).tolist()
                    logger.debug(
                        f"Created average embedding from {len(embeddings)} bottle embeddings")

                    # Get price information for price filtering if needed
                    price_filter = ""
                    price_params = [avg_embedding, avg_embedding, limit]

                    if price_range:
                        min_price, max_price = price_range
                        price_filter = "AND price BETWEEN %s AND %s"
                        price_params = [
                            avg_embedding, avg_embedding, min_price, max_price, limit]
                        logger.debug(
                            f"Applying price filter: {min_price} to {max_price}")

                    # Find similar bottles excluding the ones they already like
                    logger.debug("Executing recommendation query")
                    await cur.execute(f"""
                        SELECT 
                            id, name, description, type, producer, series, 
                            abv, age_statement, country, region,
                            1 - (embedding <=> %s::vector) AS similarity
                        FROM bottles
                        WHERE id NOT IN ({bottle_ids_str})
                        {price_filter}
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """, price_params)

                    results = await cur.fetchall()
                    logger.debug(f"Found {len(results)} recommended bottles")
                    return results

        except Exception as e:
            logger.error(f"Error recommending bottles: {e}", exc_info=True)
            return []

    async def recommend_diverse_bottles(
        self,
        bottle_ids: List[int],
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Recommend bottles that would diversify a collection."""
        try:
            if not bottle_ids:
                logger.debug(
                    "No bottle IDs provided for diverse recommendations")
                return []

            bottle_ids_str = ','.join(str(id) for id in bottle_ids)
            logger.debug(
                f"Finding diverse recommendations for {len(bottle_ids)} bottle IDs")

            async with await psycopg.AsyncConnection.connect(BottleUtils.get_db_connection_string()) as conn:
                conn.row_factory = dict_row
                async with conn.cursor() as cur:
                    # First analyze what types the user already has
                    logger.debug("Analyzing user's existing bottle types")
                    await cur.execute(f"""
                        SELECT type, COUNT(*) as type_count
                        FROM bottles
                        WHERE id IN ({bottle_ids_str})
                        GROUP BY type
                    """)

                    user_types = await cur.fetchall()
                    user_type_names = [t["type"].lower() for t in user_types]
                    logger.debug(
                        f"User has {len(user_type_names)} different bottle types: {user_type_names}")

                    # Find types that are underrepresented or missing
                    logger.debug("Executing diverse recommendation query")
                    await cur.execute(f"""
                        SELECT 
                            id, name, description, type, producer, series, 
                            abv, age_statement, country, region
                        FROM bottles
                        WHERE id NOT IN ({bottle_ids_str})
                        AND (
                            LOWER(type) NOT IN ({','.join(f"'{t}'" for t in user_type_names) or "''"})
                            OR region NOT IN (
                                SELECT region FROM bottles 
                                WHERE id IN ({bottle_ids_str}) AND region IS NOT NULL
                            )
                        )
                        ORDER BY vote_count DESC
                        LIMIT %s
                    """, (limit,))

                    results = await cur.fetchall()
                    logger.debug(
                        f"Found {len(results)} diverse bottle recommendations")

                    # Add a reason for each recommendation
                    for result in results:
                        if result["type"].lower() not in user_type_names:
                            result["reason"] = f"Adds {result['type']} to diversify your collection"
                        else:
                            result["reason"] = f"Introduces a new region ({result['region']}) to your collection"

                    return results

        except Exception as e:
            logger.error(
                f"Error recommending diverse bottles: {e}", exc_info=True)
            return []

    # Helper function to convert string embeddings to numeric arrays
    def _convert_embeddings(self, embeddings: List[Any]) -> List[np.ndarray]:
        """Convert embeddings from string format to numeric arrays if needed."""
        if not embeddings:
            return []

        # Check if conversion is needed
        if isinstance(embeddings[0], str):
            logger.debug("Converting string embeddings to numeric arrays")
            # If embeddings are returned as strings, convert them to numeric arrays
            numeric_embeddings = []
            for emb in embeddings:
                # Parse string representation of vector to numeric array
                if isinstance(emb, str):
                    # Clean the string and split into values
                    clean_emb = emb.strip('[]()').replace(' ', '')
                    values = [float(val)
                              for val in clean_emb.split(',') if val]
                    numeric_embeddings.append(values)
                else:
                    numeric_embeddings.append(emb)
            embeddings = numeric_embeddings
            logger.debug(
                f"Converted {len(embeddings)} embeddings to numeric format")

        # Ensure all embeddings are properly formatted as numeric arrays
        return [np.array(emb, dtype=np.float32) for emb in embeddings]

    async def analyze_collection(self, bottle_ids: List[int]) -> Dict[str, Any]:
        """Analyze a user's collection to provide insights."""
        try:
            if not bottle_ids:
                logger.debug("No bottles in collection to analyze")
                return {"error": "No bottles in collection to analyze"}

            bottle_ids_str = ','.join(str(id) for id in bottle_ids)
            logger.debug(
                f"Analyzing collection with {len(bottle_ids)} bottles")

            async with await psycopg.AsyncConnection.connect(BottleUtils.get_db_connection_string()) as conn:
                conn.row_factory = dict_row
                async with conn.cursor() as cur:
                    # Get all bottle data
                    logger.debug("Fetching bottle data for analysis")
                    await cur.execute(f"""
                        SELECT *
                        FROM bottles
                        WHERE id IN ({bottle_ids_str})
                    """)

                    bottles = await cur.fetchall()
                    logger.debug(
                        f"Retrieved {len(bottles)} bottles for analysis")

                    # Calculate statistics
                    types_count = {}
                    regions_count = {}
                    abvs = []

                    for bottle in bottles:
                        # Count types
                        bottle_type = bottle.get("type", "Unknown")
                        types_count[bottle_type] = types_count.get(
                            bottle_type, 0) + 1

                        # Count regions
                        region = bottle.get("region", "Unknown")
                        regions_count[region] = regions_count.get(
                            region, 0) + 1

                        # Track prices and ABVs - remove price check as it doesn't exist
                        if bottle.get("abv"):
                            abvs.append(float(bottle["abv"]))

                    # Calculate summary statistics
                    avg_price = 0  # Set to 0 since price column doesn't exist
                    avg_abv = round(np.mean(abvs), 1) if abvs else 0

                    # Find favorite type and region
                    favorite_type = max(types_count.items(), key=lambda x: x[1])[
                        0] if types_count else "Unknown"
                    favorite_region = max(regions_count.items(), key=lambda x: x[1])[
                        0] if regions_count else "Unknown"

                    logger.debug(
                        f"Analysis complete: {len(types_count)} types, {len(regions_count)} regions")
                    return {
                        "bottle_count": len(bottles),
                        "types": types_count,
                        "regions": regions_count,
                        "avg_price": avg_price,
                        "avg_abv": avg_abv,
                        "favorite_type": favorite_type,
                        "favorite_region": favorite_region
                    }

        except Exception as e:
            logger.error(f"Error analyzing collection: {e}", exc_info=True)
            return {"error": f"Failed to analyze collection: {str(e)}"}


class Bob:
    """
    Bob - The BAXUS whisky recommendation agent

    An AI assistant that analyzes users' virtual bars within the BAXUS ecosystem
    to provide personalized bottle recommendations and collection insights.
    """

    def __init__(self):
        """Initialize Bob with the necessary components."""
        logger.debug("Initializing Bob")
        self.recommender = BottleRecommender()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.error("OPENAI_API_KEY environment variable is missing")
            raise ValueError("OPENAI_API_KEY environment variable is required")

        # Initialize LangChain components
        logger.debug("Setting up LangChain components")
        self.llm = ChatOpenAI(temperature=0.7)
        self.setup_agent()
        logger.debug("Bob initialization complete")

    def setup_agent(self):
        """Set up the LangChain agent with conversation memory and tools."""
        logger.debug("Setting up Bob's agent capabilities")
        # System prompt for Bob's personality and capabilities
        system_prompt = """
        You are Bob, a whisky expert AI assistant for the BAXUS app. 
        Your main job is to analyze users' virtual whisky collections and provide personalized recommendations.
        
        You are incredibly knowledgeable about the whisky industry, with expertise in:
        - Different whisky types (Bourbon, Scotch, Rye, etc.)
        - Regions and distilleries
        - Flavor profiles, aging, and production methods
        - Price points and value
        
        When making recommendations, consider:
        1. The user's existing collection
        2. Their preferences in terms of types, regions, and price points
        3. Both similar bottles to what they already enjoy and options that diversify their collection
        
        Keep your tone friendly, enthusiastic about whisky, and conversational. Use your expertise to educate
        users about their recommendations when appropriate.
        """

        # Create the LangChain agent
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="history"),
            HumanMessage(content="{input}")
        ])

        # Create a simple chain for conversation
        self.chain = prompt | self.llm | StrOutputParser()
        logger.debug("Conversation chain created")

        # Create a more advanced graph using LangGraph
        # Commenting out the graph setup for now as it's causing issues
        # We'll use the simpler chain approach instead
        # self.setup_graph()

    def setup_graph(self):
        """Set up the LangGraph for more complex interactions."""
        logger.debug("Setting up LangGraph (disabled)")
        # Define the agent state

        class AgentState:
            def __init__(self):
                self.messages = []
                self.user_bottles = []
                self.current_analysis = {}

        # Simplified version - not using the full graph functionality for now
        # as it's causing compatibility issues
        # We'll just use the simple conversation chain instead
        self.conversation_graph = None

    @staticmethod
    def load_user_data_from_file(file_path: str) -> List[Dict[str, Any]]:
        """Load user data from a JSON file."""
        try:
            logger.debug(f"Loading user data from file: {file_path}")
            if not os.path.exists(file_path):
                logger.error(f"User data file not found: {file_path}")
                return []

            with open(file_path, 'r') as f:
                user_data = json.load(f)

            if not isinstance(user_data, list):
                logger.error(
                    f"User data file has invalid format: expected a list of items")
                return []

            logger.info(f"Loaded {len(user_data)} bottles from user data file")
            return user_data

        except json.JSONDecodeError:
            logger.error(
                f"Failed to parse user data file as JSON: {file_path}")
            return []
        except Exception as e:
            logger.error(f"Error loading user data: {e}", exc_info=True)
            return []

    async def analyze_user_bar(self, user_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze a user's bar data to extract bottle IDs and preferences."""
        logger.debug(f"Analyzing user bar with {len(user_data)} items")
        bottle_ids = []
        for item in user_data:
            product = item.get("product", {})
            if product_id := product.get("id"):
                bottle_ids.append(product_id)

        logger.debug(f"Extracted {len(bottle_ids)} bottle IDs from user data")
        # Get collection analysis
        analysis = await self.recommender.analyze_collection(bottle_ids)
        return {
            "bottle_ids": bottle_ids,
            "analysis": analysis
        }

    async def analyze_user_bar_from_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a user's bar data from a JSON file."""
        logger.debug(f"Analyzing user bar from file: {file_path}")
        user_data = self.load_user_data_from_file(file_path)
        if not user_data:
            logger.error(f"No user data loaded from file: {file_path}")
            return {"error": f"Failed to load user data from file: {file_path}"}

        return await self.analyze_user_bar(user_data)

    async def get_recommendations(
        self,
        user_data: List[Dict[str, Any]],
        num_similar: int = 3,
        num_diverse: int = 2
    ) -> Dict[str, Any]:
        """Get both similar and diverse recommendations for a user's collection."""
        logger.debug(
            f"Getting recommendations for user data with {len(user_data)} items")
        # Extract bottle IDs from user data
        bottle_ids = []
        for item in user_data:
            product = item.get("product", {})
            if product_id := product.get("id"):
                bottle_ids.append(product_id)

        logger.debug(
            f"Extracted {len(bottle_ids)} bottle IDs for recommendations")
        # Get similar recommendations
        logger.debug(f"Requesting {num_similar} similar recommendations")
        similar_recs = await self.recommender.recommend_from_bottle_ids(bottle_ids, limit=num_similar)

        # Get diverse recommendations
        logger.debug(f"Requesting {num_diverse} diverse recommendations")
        diverse_recs = await self.recommender.recommend_diverse_bottles(bottle_ids, limit=num_diverse)

        logger.debug(
            f"Recommendations complete: {len(similar_recs)} similar, {len(diverse_recs)} diverse")
        return {
            "similar": similar_recs,
            "diverse": diverse_recs
        }

    async def get_recommendations_from_file(
        self,
        file_path: str,
        num_similar: int = 3,
        num_diverse: int = 2
    ) -> Dict[str, Any]:
        """Get recommendations based on a user's collection loaded from a JSON file."""
        logger.debug(f"Getting recommendations from file: {file_path}")
        user_data = self.load_user_data_from_file(file_path)
        if not user_data:
            logger.error(f"No user data loaded from file: {file_path}")
            return {"error": f"Failed to load user data from file: {file_path}"}

        return await self.get_recommendations(user_data, num_similar, num_diverse)

    async def chat(self, user_input: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """Process a user message and return Bob's response."""
        logger.debug(f"Processing chat: '{user_input[:50]}...'")
        if conversation_history is None:
            conversation_history = []

        # Format the conversation history for LangChain
        formatted_history = []
        for message in conversation_history:
            if message["role"] == "user":
                formatted_history.append(
                    HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                formatted_history.append(AIMessage(content=message["content"]))

        logger.debug(
            f"Conversation history has {len(formatted_history)} messages")
        # Process the message
        response = await self.chain.ainvoke({
            "history": formatted_history,
            "input": user_input
        })
        logger.debug(f"Generated response: '{response[:50]}...'")

        return response


async def main():
    """Main function to demonstrate Bob's capabilities."""
    # Load environment variables
    load_dotenv()
    logger.debug("Starting Bob main function")

    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(
        description="Bob - BAXUS Whisky Recommendation AI Agent")
    parser.add_argument("--user-data", type=str,
                        help="Path to user.json file with collection data")
    parser.add_argument("--similar", type=int, default=3,
                        help="Number of similar bottle recommendations")
    parser.add_argument("--diverse", type=int, default=2,
                        help="Number of diverse bottle recommendations")
    parser.add_argument("--chat", type=str,
                        help="Chat with Bob using this message")
    args = parser.parse_args()
    logger.debug(f"Command line arguments: {args}")

    # Initialize Bob
    try:
        logger.info("Initializing Bob")
        bob = Bob()

        # If user JSON file is provided, use it
        if args.user_data:
            logger.info(f"Using user data from file: {args.user_data}")

            # Analyze the collection
            logger.debug("Analyzing user collection")
            analysis = await bob.analyze_user_bar_from_file(args.user_data)
            print("\nCollection Analysis:")
            print(json.dumps(analysis, indent=2))

            # Get recommendations
            logger.debug("Getting recommendations")
            recommendations = await bob.get_recommendations_from_file(
                args.user_data,
                num_similar=args.similar,
                num_diverse=args.diverse
            )

            print("\nRecommendations:")
            print("\nSimilar bottles:")
            for rec in recommendations.get("similar", []):
                print(f"- {rec['name']} ({rec['type']})")

            print("\nDiverse options:")
            for rec in recommendations.get("diverse", []):
                print(
                    f"- {rec['name']} ({rec['type']}): {rec.get('reason', '')}")
        else:
            # Example user data (simplified)
            logger.debug("Using example user data")
            user_data = [
                {"product": {"id": 164, "name": "Buffalo Trace", "spirit": "Bourbon"}},
                {"product": {"id": 2848, "name": "Lagavulin 16", "spirit": "Scotch"}}
            ]

            # Analyze the user's collection
            logger.debug("Analyzing example collection")
            analysis = await bob.analyze_user_bar(user_data)
            print("Collection Analysis:")
            print(json.dumps(analysis, indent=2))

            # Get recommendations
            logger.debug("Getting recommendations for example data")
            recommendations = await bob.get_recommendations(user_data)
            print("\nRecommendations:")
            print("Similar bottles:")
            for rec in recommendations["similar"]:
                print(f"- {rec['name']} ({rec['type']})")

            print("\nDiverse options:")
            for rec in recommendations["diverse"]:
                print(
                    f"- {rec['name']} ({rec['type']}): {rec.get('reason', '')}")

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
