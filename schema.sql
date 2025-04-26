-- PostgreSQL schema for BAXUS whisky recommendation system
-- This schema includes pgvector extension for similarity searches

-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS pgvector;

-- Bottles table with vector embeddings
CREATE TABLE IF NOT EXISTS bottles (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    type VARCHAR(100),
    producer VARCHAR(255),
    series VARCHAR(255),
    abv DECIMAL(5,2),
    proof DECIMAL(5,2),
    age_statement VARCHAR(50),
    country VARCHAR(100),
    region VARCHAR(100),
    shelf_price DECIMAL(10,2),
    avg_msrp DECIMAL(10,2),
    total_score DECIMAL(5,2),
    vote_count INTEGER DEFAULT 0,
    bar_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    embedding vector(384)  -- For text-embedding-3-small
);

-- Brands table
CREATE TABLE IF NOT EXISTS brands (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    country VARCHAR(100),
    website VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Add foreign key to bottles
ALTER TABLE bottles ADD COLUMN IF NOT EXISTS brand_id INTEGER REFERENCES brands(id);

-- User virtual bars
CREATE TABLE IF NOT EXISTS user_bars (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    bottle_id INTEGER REFERENCES bottles(id),
    added_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    rating DECIMAL(3,1),
    UNIQUE(user_id, bottle_id)
);

-- User wishlist
CREATE TABLE IF NOT EXISTS user_wishlist (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    bottle_id INTEGER REFERENCES bottles(id),
    added_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    UNIQUE(user_id, bottle_id)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_bottles_type ON bottles(type);
CREATE INDEX IF NOT EXISTS idx_bottles_region ON bottles(region);
CREATE INDEX IF NOT EXISTS idx_bottles_price ON bottles(shelf_price);
CREATE INDEX IF NOT EXISTS idx_user_bars_user_id ON user_bars(user_id);
CREATE INDEX IF NOT EXISTS idx_user_wishlist_user_id ON user_wishlist(user_id);

-- Create a GIN index on the embedding vectors for faster similarity searches
CREATE INDEX IF NOT EXISTS idx_bottles_embedding ON bottles USING ivfflat (embedding vector_l2_ops)
WITH (lists = 100);  -- Adjust the number of lists based on dataset size

-- Helper function to update the timestamp when a record is updated
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Add trigger to bottles table for updated_at
CREATE TRIGGER update_bottles_updated_at
BEFORE UPDATE ON bottles
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column(); 