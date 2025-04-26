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
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import Bob
from bob import Bob

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
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
    allow_origins=["*"],  # In production, restrict this to your frontend domain
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
        raise HTTPException(status_code=500, detail=f"Failed to initialize recommendation engine: {str(e)}")

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
    items: List[BarItem] = Field(..., description="List of bottles in the user's collection")

class RecommendationRequest(BaseModel):
    items: List[BarItem] = Field(..., description="List of bottles in the user's collection")
    num_similar: int = Field(3, description="Number of similar recommendations to return")
    num_diverse: int = Field(2, description="Number of diverse recommendations to return")

class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message to Bob")
    conversation_history: Optional[List[Dict[str, str]]] = Field(None, description="Previous messages in the conversation")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Bob's response to the user")

class FileAnalysisParams(BaseModel):
    num_similar: int = Field(3, description="Number of similar recommendations to return")
    num_diverse: int = Field(2, description="Number of diverse recommendations to return")

# API Routes
@app.get("/")
async def root():
    return {"message": "Welcome to Bob, the BAXUS whisky recommendation AI agent"}

@app.post("/analyze")
async def analyze_collection(
    collection: UserCollection = Body(...),
    bob: Bob = Depends(get_bob)
):
    """Analyze a user's whisky collection and provide insights."""
    try:
        analysis = await bob.analyze_user_bar(collection.items)
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing collection: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/upload/analyze")
async def analyze_collection_from_file(
    file: UploadFile = File(...),
    bob: Bob = Depends(get_bob)
):
    """Analyze a user's whisky collection from an uploaded JSON file."""
    try:
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{file.filename}"
        try:
            contents = await file.read()
            with open(temp_file_path, "wb") as f:
                f.write(contents)
            
            # Analyze the collection
            analysis = await bob.analyze_user_bar_from_file(temp_file_path)
            return analysis
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error analyzing collection from file: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/recommend")
async def get_recommendations(
    request: RecommendationRequest = Body(...),
    bob: Bob = Depends(get_bob)
):
    """Get personalized bottle recommendations for a user."""
    try:
        recommendations = await bob.get_recommendations(
            request.items,
            num_similar=request.num_similar,
            num_diverse=request.num_diverse
        )
        return recommendations
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

@app.post("/upload/recommend")
async def recommend_from_file(
    file: UploadFile = File(...),
    num_similar: int = Form(3),
    num_diverse: int = Form(2),
    bob: Bob = Depends(get_bob)
):
    """Get personalized bottle recommendations from an uploaded JSON file."""
    try:
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{file.filename}"
        try:
            contents = await file.read()
            with open(temp_file_path, "wb") as f:
                f.write(contents)
            
            # Get recommendations
            recommendations = await bob.get_recommendations_from_file(
                temp_file_path,
                num_similar=num_similar,
                num_diverse=num_diverse
            )
            return recommendations
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error getting recommendations from file: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bob(
    request: ChatRequest = Body(...),
    bob: Bob = Depends(get_bob)
):
    """Chat with Bob the whisky expert."""
    try:
        response = await bob.chat(request.message, request.conversation_history)
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