#!/usr/bin/env python3
"""
API for Bob AI Agent - BAXUS Whisky Recommendation System
--------------------------------------------------------
This module provides a FastAPI web service that exposes Bob's capabilities:
- Collection analysis
- Personalized recommendations
- Conversational interface
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Body, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import Bob
from bob import Bob

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Bob - BAXUS Whisky Recommendation AI Agent",
    description="API for analyzing whisky collections and providing personalized recommendations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    # In production, restrict this to your frontend domain
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get the Bob instance


async def get_bob():
    try:
        bob = Bob()
        yield bob
    except Exception as e:
        logger.error(f"Error initializing Bob: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to initialize recommendation engine: {str(e)}")

# Pydantic models for request/response validation


class BottleProduct(BaseModel):
    id: int
    name: Optional[str] = None
    spirit: Optional[str] = None
    shelf_price: Optional[float] = None
    age_statement: Optional[str] = None


class BarItem(BaseModel):
    id: Optional[int] = None
    product: BottleProduct


class UserCollection(BaseModel):
    items: List[BarItem] = Field(...,
                                 description="List of bottles in the user's collection")


class RecommendationRequest(BaseModel):
    username: str = Field(...,
                          description="Username of the user")
    num_similar: int = Field(
        3, description="Number of similar recommendations to return")
    num_diverse: int = Field(
        2, description="Number of diverse recommendations to return")


class FlavorProfile(BaseModel):
    sweet: Optional[float] = Field(None, description="Sweetness level (0-1)")
    woody: Optional[float] = Field(None, description="Woodiness level (0-1)")
    spicy: Optional[float] = Field(None, description="Spiciness level (0-1)")
    smoky: Optional[float] = Field(None, description="Smokiness level (0-1)")
    fruity: Optional[float] = Field(None, description="Fruitiness level (0-1)")
    smooth: Optional[float] = Field(None, description="Smoothness level (0-1)")
    floral: Optional[float] = Field(None, description="Floral level (0-1)")


class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message to Bob")
    username: Optional[str] = Field(
        None, description="Username for collection-based recommendations")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        None, description="Previous messages in the conversation")
    stream: bool = Field(False, description="Whether to stream the response")


class ChatResponse(BaseModel):
    response: str = Field(..., description="Bob's response to the user")


class FileAnalysisParams(BaseModel):
    num_similar: int = Field(
        3, description="Number of similar recommendations to return")
    num_diverse: int = Field(
        2, description="Number of diverse recommendations to return")


class InvestmentStats(BaseModel):
    estimated_value: Dict[str, float] = Field(
        ..., description="Estimated value ranges (low, high, average, total)")
    bottle_count: int = Field(...,
                              description="Total number of bottles in collection")
    bottles_with_price: int = Field(...,
                                    description="Number of bottles that have price data")
    price_range: Dict[str, float] = Field(...,
                                          description="Price range information (min, max)")
    value_by_type: Dict[str, Dict[str, Any]] = Field(
        ..., description="Value statistics broken down by spirit type")


# API Routes


@app.get("/")
async def root():
    return {"message": "Welcome, I'm Bob, the BAXUS Whisky Expert AI Agent"}


@app.get("/health")
async def health_check(bob: Bob = Depends(get_bob)):
    """Health check endpoint for monitoring and service status."""
    health_status = {
        "status": "healthy",
        "service": "Bob AI Agent",
        "version": app.version,
        "components": {
            "api": "healthy"
        }
    }
    
    # Check database connection by using a method that requires database access
    try:
        # Use the recommender's search method which should access the database
        # Limit to 1 result to minimize overhead
        await bob.recommender.search_similar_bottles("test", 1)
        health_status["components"]["database"] = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_status["components"]["database"] = "unhealthy"
        health_status["status"] = "degraded"
    
    # Return 500 status code if any component is unhealthy
    if health_status["status"] == "degraded":
        raise HTTPException(status_code=500, detail=health_status)
    
    return health_status


@app.get("/profile/{username}")
async def get_user_profile(
    username: str,
    bob: Bob = Depends(get_bob)
):
    """Get a user's profile including taste preferences, collection stats, and investment analysis."""
    try:
        profile = await bob.get_user_profile(username)
        return profile
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get user profile: {str(e)}")


@app.get("/recommendations/{username}")
async def get_recommendations(
    username: str,
    similar: int = 5,
    diverse: int = 3,
    diversity_ratio: float = 0.4,
    bob: Bob = Depends(get_bob)
):
    """Get personalized bottle recommendations for a user.

    Args:
        username: Username to get recommendations for
        similar: Number of similar recommendations to return
        diverse: Number of diverse recommendations to return
        diversity_ratio: Ratio between 0 and 1 that controls the balance of diversity
                         Higher values increase diversity in recommendations
    """
    try:
        recommendations = await bob.get_recommendations(
            username,
            num_similar=similar,
            num_diverse=diverse,
            diversity_ratio=diversity_ratio
        )
        
        # Add flavor profiles to each recommended bottle
        for category in ['similar', 'diverse']:
            for bottle in recommendations.get(category, []):
                bottle_type = bottle.get('type') or bottle.get('spirit_type')
                bottle['flavor_profile'] = await bob.recommender.get_bottle_flavor_profile(bottle_type)
                
        return recommendations
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(
            status_code=500, detail=f"Recommendation failed: {str(e)}")


@app.post("/chat", response_model=None)
async def chat_with_bob(
    request: ChatRequest = Body(...),
    bob: Bob = Depends(get_bob)
):
    """Chat with Bob the whisky expert."""
    print(f"Chat request: {request}")
    try:
        if request.stream:
            return StreamingResponse(
                bob.chat_stream(request.message, request.conversation_history, request.username),
                media_type="text/event-stream"
            )
        else:
            response = await bob.chat(request.message, request.conversation_history, request.username)
            print(f"Chat response: {response}")
            return {"response": response}
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.post("/search")
async def search_bottles(
    query: str = Body(..., embed=True),
    limit: int = Body(5, embed=True),
    bob: Bob = Depends(get_bob)
):
    """Search for bottles similar to the query."""
    try:
        results = await bob.recommender.search_similar_bottles(query, limit)
        
        # Add flavor profiles to each search result
        for bottle in results:
            bottle_type = bottle.get('type') or bottle.get('spirit_type')
            bottle['flavor_profile'] = await bob.recommender.get_bottle_flavor_profile(bottle_type)
            
        return results
    except Exception as e:
        logger.error(f"Error searching bottles: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    # Run the FastAPI app with uvicorn
    uvicorn.run("api:app", host=host, port=port, reload=debug)
