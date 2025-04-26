"""
agent.py – whisky recommendation engine + KG explanations
--------------------------------------------------------
This module builds a **hybrid recommender** that combines ANN vector search
with knowledge‑graph path explanations.  It now also supports real user‑data
payloads (the JSON list sent by the Baxus backend) so you can feed a user's
bar inventory directly and get personalised picks with a single call.

Quick start (Python ≥3.9)::

    from agent import WhiskeyAgent
    import json, pathlib

    agent = WhiskeyAgent("501 Bottle Dataset.csv")
    user_json = json.loads(pathlib.Path("sample_user.json").read_text())
    recs = agent.recommend_from_user_data(user_json, k=5)
    for bid, score, name in recs:
        print(bid, score, name)

Dependencies
~~~~~~~~~~~~
``pip install faiss-cpu pandas numpy graphiti-core openai scikit-learn``

Graphiti docs: https://github.com/getzep/graphiti
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Vector search -------------------------------------------------------------
try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "faiss-cpu is required – install with `pip install faiss-cpu`.\n" "Original error: " + str(exc)) from exc

# Knowledge‑graph -----------------------------------------------------------
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodicNode, EpisodeType

# Embeddings (OpenAI preferred, TF‑IDF fallback) ----------------------------
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

EMBED_DIM = 384  # OpenAI text-embedding-3-small dimension
NUMERIC_COLS = [
    "abv",
    "avg_msrp",
    "shelf_price",
    "total_score",
    "vote_count",
    "bar_count",
]


# =====================================================================
# Main class
# =====================================================================
class WhiskeyAgent:
    """Vector ANN recommender backed by a lightweight knowledge graph."""

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def __init__(
        self,
        csv_path: str,
        graph: Optional[Graphiti] = None,
        openai_api_key: Optional[str] = None,
    ) -> None:
        self.df = self._load_and_clean(csv_path)
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_client = None
        if self.openai_api_key:
            if OpenAI is None:
                raise RuntimeError(
                    "`openai` Python package missing – run `pip install openai`. ")
            # Initialize the client only once
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.vectors = self._build_vectors()
        self.index = self._build_index()
        self.graph = graph or self._init_graph()

    # ------------------------------------------------------------------
    # Data prep
    # ------------------------------------------------------------------
    @staticmethod
    def _load_and_clean(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        # repair proof / ABV completeness
        df["proof"] = df["proof"].fillna(df["abv"] * 2)
        df["abv"] = df["abv"].fillna(df["proof"] / 2)
        # Fix inplace warning by using assignment instead
        df["spirit_type"] = df["spirit_type"].fillna("Unknown")
        # ensure numeric
        for col in ["avg_msrp", "shelf_price", "total_score", "vote_count", "bar_count"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def _embed(self, texts: List[str]) -> np.ndarray:
        """Return a (n, EMBED_DIM) float32 matrix of L2‑normalised vectors."""
        if self.openai_api_key:
            # batch up to 2048 tokens safely – 100 inputs chunk keeps us clear
            out: List[List[float]] = []

            for i in range(0, len(texts), 100):
                chunk = texts[i: i + 100]
                # Use the new API format for embeddings
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=chunk
                )
                out.extend([data.embedding for data in response.data])
            arr = np.asarray(out, dtype=np.float32)
        else:
            # Simple TF‑IDF mean vector fallback (dim capped / padded)
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

            tfidf = TfidfVectorizer(stop_words="english").fit(texts)
            mat = tfidf.transform(texts).toarray().astype(np.float32)
            if mat.shape[1] >= EMBED_DIM:
                arr = mat[:, :EMBED_DIM]
            else:
                pad = np.zeros(
                    (mat.shape[0], EMBED_DIM - mat.shape[1]), dtype=np.float32)
                arr = np.hstack([mat, pad])
        faiss.normalize_L2(arr)
        return arr

    def _build_vectors(self) -> np.ndarray:
        df = self.df
        # numeric z‑score
        num = df[NUMERIC_COLS].values.astype(np.float32)
        num = (num - num.mean(axis=0)) / (num.std(axis=0) + 1e-8)

        # categorical one‑hot: spirit_type
        cat = pd.get_dummies(df["spirit_type"], dtype=np.float32)
        cat_arr = cat.values.astype(np.float32)

        # text embedding – bottle name
        text_vec = self._embed(df["name"].tolist())

        full = np.hstack([num, cat_arr, text_vec]).astype(np.float32)
        faiss.normalize_L2(full)
        return full

    # ------------------------------------------------------------------
    # FAISS index
    # ------------------------------------------------------------------
    def _build_index(self) -> faiss.Index:
        dim = self.vectors.shape[1]
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efSearch = 64
        index.add(self.vectors)
        return index

    # ------------------------------------------------------------------
    # Knowledge Graph
    # ------------------------------------------------------------------
    def _init_graph(self) -> Graphiti:
        # Initialize a new Graphiti instance
        # Note: These connection values should be passed in or retrieved from environment variables
        g = Graphiti("bolt://localhost:7687", "neo4j", "password")

        # Add bottles as episodes to the graph
        async def add_bottles_to_graph():
            for _, row in self.df.iterrows():
                bottle_data = {
                    "id": str(int(row["id"])),
                    "name": row["name"],
                    "spirit_type": row["spirit_type"],
                    "brand_id": str(int(row["brand_id"])) if not math.isnan(row.get("brand_id", math.nan)) else "0"
                }

                # Add additional attributes if they exist in the dataframe
                for col in ["abv", "proof", "avg_msrp", "shelf_price", "total_score"]:
                    if col in row and not pd.isna(row[col]):
                        bottle_data[col] = row[col]

                # In a real implementation, you would add each bottle to the graph:
                # await g.add_episode(
                #     name=bottle_data["name"],
                #     episode_body=json.dumps(bottle_data),
                #     source=EpisodeType.json,
                #     source_description="Whisky bottle data",
                #     reference_time=datetime.now(timezone.utc),
                # )

        # Note: Since the main class is synchronous, we can't actually run the async code
        # directly, but this shows how you would structure the code
        # In a real implementation, you might:
        # 1. Make the _init_graph method async
        # 2. Run the async code with asyncio.run() or similar
        # 3. Or create a synchronous wrapper around the graphiti_core API

        return g

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _price_score(price: float, pref: Optional[float]) -> float:
        if not pref or price <= 0:
            return 1.0
        return max(0.0, 1 - abs(math.log(price / pref)) / 3)

    def _pop_score(self, votes: float) -> float:
        max_vote = float(self.df["vote_count"].max()) or 1.0
        return math.log1p(votes) / math.log1p(max_vote)

    # ==================================================================
    # PUBLIC API
    # ==================================================================
    def recommend(
        self,
        liked_ids: List[int],
        price_pref: Optional[float] = None,
        k: int = 10,
    ) -> List[Tuple[int, float]]:
        """Recommend based on bottle IDs already present in dataset."""
        df = self.df
        if liked_ids:
            # keep only those that actually exist in dataset
            idx = df[df["id"].isin(liked_ids)].index
            if len(idx) == 0:
                # cold start fallback later
                idx = None
        else:
            idx = None

        if idx is not None and len(idx):
            profile_vec = self.vectors[idx].mean(axis=0)
            if price_pref is None:
                price_pref = float(df.loc[idx, "shelf_price"].mean())
        else:
            # cold‑start: centroid of top‑pop bottles
            pop_idx = df["vote_count"].nlargest(20).index
            profile_vec = self.vectors[pop_idx].mean(axis=0)

        # L2 normalise query
        faiss.normalize_L2(profile_vec.reshape(1, -1))
        dist, ind = self.index.search(profile_vec.reshape(1, -1), 50)

        suggestions: List[Tuple[int, float]] = []
        for i, d in zip(ind[0], dist[0]):
            bid = int(df.iloc[i]["id"])
            if bid in liked_ids:
                continue
            sim = 1 - d / 2  # cosine dist → similarity [0,1]
            price = float(df.iloc[i]["shelf_price"].item())
            pop = float(df.iloc[i]["vote_count"].item())
            score = (
                0.6 * sim
                + 0.25 * self._price_score(price, price_pref)
                + 0.15 * self._pop_score(pop)
            )
            suggestions.append((bid, score))
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:k]

    # ------------------------------------------------------------------
    # NEW: recommend from Baxus‑style JSON payload ----------------------
    # ------------------------------------------------------------------
    def recommend_from_user_data(
        self,
        user_items: List[Dict],
        k: int = 10,
    ) -> List[Tuple[int, float, str]]:
        """Public helper that accepts the raw JSON list provided by the client.

        Parameters
        ----------
        user_items: list of dicts
            Each item is expected to match the structure shown in the example:
            {
              "id": ...,  # bar row id (irrelevant here)
              "product": {"id": 13266, "shelf_price": 47.19, ...}
            }
        k: int, default 10 – number of recs
        Returns
        -------
        list of tuples  (product_id, score, product_name)
        """
        # extract product ids present in our dataset
        liked_ids: List[int] = []
        prices: List[float] = []
        for item in user_items:
            prod = item.get("product", {})
            pid = prod.get("id")
            if pid is None:
                continue
            if pid in set(self.df["id"].values):
                liked_ids.append(pid)
                if p := prod.get("shelf_price"):
                    prices.append(float(p))
        price_pref = float(np.mean(prices)) if prices else None

        recs = self.recommend(liked_ids, price_pref=price_pref, k=k)
        # enrich with names so downstream caller doesn't need df lookup
        enriched = [
            (
                bid,
                round(score, 3),
                self.df.loc[self.df["id"] == bid, "name"].values[0],
            )
            for bid, score in recs
        ]
        return enriched

    # ------------------------------------------------------------------
    # Graph explanation (optional)
    # ------------------------------------------------------------------
    def explain(self, source_id: int, target_id: int, depth: int = 4) -> List[Tuple[str, str, str]]:
        if not self.graph:
            return []

        # For graphiti_core, we need to use the search functionality instead
        # of the direct shortest_path method that was available in the original graphiti
        try:
            # Search for connections between the source and target bottles
            # This is a placeholder implementation - the actual implementation
            # would depend on graphiti_core's specific API for path finding
            results = []
            # Note: In a real implementation, you would use graphiti_core's
            # search functionality to find paths between nodes
            # For example:
            # results = await self.graph.search(f"Find connections between bottle {source_id} and bottle {target_id}")
            return results
        except Exception as e:
            print(f"Error finding path: {e}")
            return []

    # ------------------------------------------------------------------
    # NEW: recommend diversifying bottles ----------------------------------
    # ------------------------------------------------------------------
    def recommend_diverse(
        self,
        user_items: List[Dict],
        k: int = 5,
        price_tolerance: float = 1.5,
    ) -> List[Tuple[int, float, str, str]]:
        """Recommend bottles that would diversify a collection.

        This method analyzes the user's existing collection and suggests bottles
        that would complement it by filling gaps in terms of spirit types,
        flavor profiles, price ranges, and popularity.

        Parameters
        ----------
        user_items: list of dicts
            Each item is expected to match the structure shown in the example:
            {
              "id": ...,  # bar row id (irrelevant here)
              "product": {"id": 13266, "shelf_price": 47.19, "spirit": "Bourbon", ...}
            }
        k: int, default 5 – number of recommendations
        price_tolerance: float, default 1.5 – multiplier for price range

        Returns
        -------
        list of tuples (product_id, score, product_name, reason)
            reason is a string explaining why this bottle diversifies the collection
        """
        df = self.df

        # Extract user collection data
        liked_ids: List[int] = []
        user_spirits: Dict[str, int] = {}
        user_prices: List[float] = []

        for item in user_items:
            prod = item.get("product", {})
            pid = prod.get("id")
            if pid is None:
                continue

            # Track spirit types in collection
            # Use 'spirit' from the user data (not spirit_type) since that's how it's structured in user.json
            spirit = prod.get("spirit")
            if spirit:
                # Normalize spirit type to match our dataset
                spirit = self._normalize_spirit_type(spirit)
                user_spirits[spirit] = user_spirits.get(spirit, 0) + 1

            # Track prices
            if p := prod.get("shelf_price"):
                user_prices.append(float(p))

            # Track bottle IDs
            if pid in set(df["id"].values):
                liked_ids.append(pid)

        # Debug logging
        print(f"User spirits: {user_spirits}")

        # Calculate statistics about the user's collection
        if user_prices:
            avg_price = float(np.mean(user_prices))
            # Set reasonable floor
            min_price = max(10, avg_price / price_tolerance)
            max_price = avg_price * price_tolerance
        else:
            avg_price = 50.0  # Default if no prices available
            min_price = 20.0
            max_price = 100.0

        # Get all possible spirit types from our dataset
        all_spirit_types = set(self._normalize_spirit_type(s)
                               for s in df["spirit_type"].unique())

        # Identify what's missing or underrepresented in the user's collection
        user_spirit_types = set(user_spirits.keys())

        # These are spirit types the user doesn't have at all
        completely_missing = all_spirit_types - user_spirit_types

        # Group similar spirit types for better diversity
        spirit_categories = {
            "Whisky": ["Bourbon", "Rye", "Scotch", "Canadian Whisky", "Irish Whiskey", "Japanese Whisky", "American Whiskey", "Tennessee Whiskey", "Whisky"],
            "Rum": ["Rum", "Rhum"],
            "Brandy": ["Brandy", "Cognac", "Armagnac"],
            "Agave": ["Tequila", "Mezcal"],
            "Gin": ["Gin", "London Dry Gin"],
            "Vodka": ["Vodka"],
            "Other": ["Liqueur", "Amaro", "Vermouth", "Absinthe"]
        }

        # Map spirit types to categories
        spirit_to_category = {}
        for category, spirits in spirit_categories.items():
            for spirit in spirits:
                spirit_to_category[spirit] = category

        # Count categories in user collection
        user_categories = {}
        for spirit, count in user_spirits.items():
            category = spirit_to_category.get(spirit, "Other")
            user_categories[category] = user_categories.get(
                category, 0) + count

        # Identify which categories the user has
        categories_in_collection = set(user_categories.keys())

        # Find completely missing categories (higher priority)
        missing_categories = set(
            spirit_categories.keys()) - categories_in_collection

        # All categories ordered by diversity priority
        # 1. Missing categories
        # 2. Categories with only 1 bottle
        # 3. All other categories
        category_priority = []

        # Add missing categories first
        for cat in missing_categories:
            category_priority.append((cat, 3))  # Priority 3 (highest)

        # Then categories with only 1 bottle
        for cat, count in user_categories.items():
            if count == 1:
                category_priority.append((cat, 2))  # Priority 2 (medium)

        # Finally, add remaining categories with lower priority
        for cat in spirit_categories:
            if cat not in missing_categories and user_categories.get(cat, 0) > 1:
                category_priority.append((cat, 1))  # Priority 1 (lowest)

        # Sort by priority (higher number = higher priority)
        category_priority.sort(key=lambda x: x[1], reverse=True)

        # Debug logging
        print(f"User categories: {user_categories}")
        print(f"Category priority: {category_priority}")

        # Find spirits in missing or underrepresented categories
        priority_spirits = {}  # Map spirit to its priority
        for category, priority in category_priority:
            for spirit in all_spirit_types:
                if spirit_to_category.get(spirit, "Other") == category:
                    # Spirits from missing categories or categories with just 1 bottle get higher priority
                    # Within same category, prioritize completely missing spirit types
                    if spirit in completely_missing:
                        priority_spirits[spirit] = priority + \
                            0.5  # Bonus for completely missing
                    else:
                        priority_spirits[spirit] = priority

        # Debug logging
        print(f"Completely missing spirits: {completely_missing}")
        print(
            f"Priority spirits (with weights): {[(s, p) for s, p in sorted(priority_spirits.items(), key=lambda x: x[1], reverse=True)[:10]]}")

        # Score all bottles in our dataset
        candidates = []

        for _, row in df.iterrows():
            bid = int(row["id"])

            # Skip bottles they already have
            if bid in liked_ids:
                continue

            name = row["name"]
            spirit = self._normalize_spirit_type(row["spirit_type"])
            price = float(row["shelf_price"])
            popularity = float(row["vote_count"])
            category = spirit_to_category.get(spirit, "Other")

            # Skip if price is outside tolerance range
            if not (min_price <= price <= max_price):
                continue

            # Base diversity score (0-1)
            # Higher if the spirit has higher priority
            if spirit in priority_spirits:
                # Normalize priority to 0-1 scale (3.5 is max possible priority)
                raw_priority = priority_spirits[spirit]
                diversity_score = min(1.0, raw_priority / 3.5)

                # Boost non-Whisky categories for true diversity
                if category != "Whisky" and "Whisky" in user_categories:
                    diversity_score = min(
                        1.0, diversity_score * 1.2)  # 20% boost

                # Generate appropriate reason
                if spirit in completely_missing:
                    reason = f"Adds {spirit} - new type to your collection"
                elif category in missing_categories:
                    reason = f"Introduces {category} spirits to your collection"
                elif user_categories.get(category, 0) == 1:
                    reason = f"Expands your {category} selection"
                else:
                    reason = f"Adds variety to your {category} collection"
            else:
                # Lower priority for spirits not in our priority list
                diversity_score = 0.2
                if price < avg_price * 0.7:
                    reason = f"Affordable {spirit} option"
                elif price > avg_price * 1.3:
                    reason = f"Premium {spirit} to upgrade collection"
                else:
                    reason = f"Quality {spirit} with good value"

            # Price match factor - prefer bottles within budget but diverse pricing
            price_factor = self._price_score(price, avg_price)

            # Popularity factor - include some hidden gems, but ensure quality
            pop_factor = self._pop_score(popularity)

            # Calculate final score - emphasize diversity more
            final_score = (0.7 * diversity_score + 0.15 *
                           price_factor + 0.15 * pop_factor)

            candidates.append(
                (bid, final_score, name, reason, spirit, category))

        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Ensure category diversity in recommendations
        final_recommendations = []
        category_counts = {}
        spirit_counts = {}

        # First pass - take top candidates while ensuring category diversity
        for bid, score, name, reason, spirit, category in candidates:
            # Limit categories to ensure diversity
            if category_counts.get(category, 0) < 2:
                # Also limit spirits within category
                if spirit_counts.get(spirit, 0) < 2:
                    final_recommendations.append((bid, score, name, reason))
                    category_counts[category] = category_counts.get(
                        category, 0) + 1
                    spirit_counts[spirit] = spirit_counts.get(spirit, 0) + 1

                    # Once we have enough recommendations, stop
                    if len(final_recommendations) >= k:
                        break

        # If we don't have enough recommendations, add more from candidates
        if len(final_recommendations) < k:
            for bid, score, name, reason, spirit, category in candidates:
                if any(bid == rec[0] for rec in final_recommendations):
                    continue

                final_recommendations.append((bid, score, name, reason))
                if len(final_recommendations) >= k:
                    break

        return final_recommendations[:k]

    # Helper to normalize spirit types across dataset and user data
    def _normalize_spirit_type(self, spirit_type: str) -> str:
        """Normalize spirit type names to handle variations."""
        if not spirit_type:
            return "Unknown"

        # Convert to lowercase for comparison
        spirit = spirit_type.lower()

        # Map common variations
        if "bourbon" in spirit:
            return "Bourbon"
        elif "rye" in spirit:
            return "Rye"
        elif "scotch" in spirit:
            return "Scotch"
        elif "canadian" in spirit or "canada" in spirit:
            return "Canadian Whisky"
        elif "irish" in spirit:
            return "Irish Whiskey"
        elif "japanese" in spirit or "japan" in spirit:
            return "Japanese Whisky"
        elif "tennessee" in spirit:
            return "Tennessee Whiskey"
        elif "whisky" in spirit or "whiskey" in spirit:
            if "american" in spirit:
                return "American Whiskey"
            else:
                return "Whisky"
        elif "rum" in spirit:
            return "Rum"
        elif "gin" in spirit:
            if "london dry" in spirit:
                return "London Dry Gin"
            else:
                return "Gin"
        elif "vodka" in spirit:
            return "Vodka"
        elif "tequila" in spirit:
            return "Tequila"
        elif "mezcal" in spirit:
            return "Mezcal"
        elif "cognac" in spirit:
            return "Cognac"
        elif "brandy" in spirit:
            return "Brandy"
        elif "liqueur" in spirit:
            return "Liqueur"
        else:
            # Keep original if no match
            return spirit_type

    # ------------------------------------------------------------------
    # NEW: recommend from Baxus‑style JSON payload with diversity option
    # ------------------------------------------------------------------
    def recommend_from_user_data_with_diversity(
        self,
        user_items: List[Dict],
        k: int = 10,
        diversity_ratio: float = 0.3,
    ) -> List[Tuple[int, float, str, str]]:
        """Provide recommendations with a mix of similar and diverse bottles.

        Parameters
        ----------
        user_items: list of dicts - user's collection
        k: int, default 10 - total recommendations to return 
        diversity_ratio: float, default 0.3 - portion of recommendations that should be diverse

        Returns
        -------
        list of tuples (product_id, score, product_name, reason)
        """
        # Calculate how many of each type to return
        diverse_count = max(1, int(k * diversity_ratio))
        similar_count = k - diverse_count

        # Get similar recommendations
        similar_recs = self.recommend_from_user_data(
            user_items, k=similar_count)

        # Get diverse recommendations
        diverse_recs = self.recommend_diverse(user_items, k=diverse_count)

        # Combine results
        result = []

        # Add similar recommendations with reason
        for bid, score, name in similar_recs:
            result.append((bid, score, name, "Based on your preferences"))

        # Add diverse recommendations
        for bid, score, name, reason in diverse_recs:
            # Check if this recommendation is already in similar_recs
            if not any(bid == s[0] for s in similar_recs):
                result.append((bid, score, name, reason))

        # Sort by score
        result.sort(key=lambda x: x[1], reverse=True)
        return result


# ----------------------------------------------------------------------
# CLI DEMO
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys
    import pathlib

    parser = argparse.ArgumentParser(description="Whisky recommender demo")
    parser.add_argument(
        "--csv", default="501 Bottle Dataset.csv", help="Bottle dataset CSV")
    parser.add_argument(
        "--user-json", help="Path to Baxus‑style user JSON file; if omitted, falls back to --likes")
    parser.add_argument("--likes", nargs="*", type=int,
                        default=[164, 2848], help="Liked bottle IDs (dataset IDs)")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of recommendations")
    parser.add_argument("--diverse", action="store_true",
                        help="Include diverse recommendations")
    parser.add_argument("--diversity-ratio", type=float, default=0.4,
                        help="Ratio of diverse recommendations (0.0-1.0)")
    args = parser.parse_args()

    agent = WhiskeyAgent(args.csv)

    if args.user_json:
        data = json.loads(pathlib.Path(args.user_json).read_text())

        if args.diverse:
            # Use the diversity-aware recommendation method
            recs = agent.recommend_from_user_data_with_diversity(
                data,
                k=args.k,
                diversity_ratio=args.diversity_ratio
            )

            print("\nWhisky Recommendations (with diversity):")
            for bid, score, name, reason in recs:
                print(f"{bid:6}  {score:.3f}  {name:40} | {reason}")

        else:
            # Standard recommendations
            recs = agent.recommend_from_user_data(data, k=args.k)

            print("\nStandard Recommendations:")
            for bid, score, name in recs:
                print(f"{bid:6}  {score:.3f}  {name}")
    else:
        # Basic recommendation from IDs
        recs_raw = agent.recommend(args.likes, k=args.k)
        recs = [
            (
                bid,
                round(score, 3),
                agent.df.loc[agent.df["id"] == bid, "name"].values[0],
            )
            for bid, score in recs_raw
        ]

        print("\nRecommendations:")
        for bid, score, name in recs:
            print(f"{bid:6}  {score:.3f}  {name}")
