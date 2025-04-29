"""
Conversation Graph Module
------------------------
Tracks user conversation state and history for Bob.
"""

import json
import uuid
from typing import Dict, List, Optional, Tuple, Any
import datetime

from utils.logging_utils import get_logger

# Set up logging
logger = get_logger(__name__)

class ConversationNode:
    """Represents a single interaction in the conversation."""
    
    def __init__(self, 
                 node_id: str = None, 
                 content: str = "", 
                 node_type: str = "user", 
                 timestamp: Optional[datetime.datetime] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a conversation node.
        
        Args:
            node_id: Unique identifier for the node
            content: The text content of the node
            node_type: Either "user" or "assistant"
            timestamp: When this node was created
            metadata: Additional data to store with the node
        """
        self.node_id = node_id or str(uuid.uuid4())
        self.content = content
        self.node_type = node_type
        self.timestamp = timestamp or datetime.datetime.now()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            "id": self.node_id,
            "content": self.content,
            "type": self.node_type,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationNode':
        """Create node from dictionary representation."""
        try:
            timestamp = datetime.datetime.fromisoformat(data.get("timestamp"))
        except (ValueError, TypeError):
            timestamp = datetime.datetime.now()
            
        return cls(
            node_id=data.get("id"),
            content=data.get("content", ""),
            node_type=data.get("type", "user"),
            timestamp=timestamp,
            metadata=data.get("metadata", {})
        )


class ConversationGraph:
    """
    Manages the conversation history as a graph of interactions.
    Currently implemented as a simple linear conversation for MVP.
    """
    
    def __init__(self, llm=None, session_id: str = None):
        """
        Initialize conversation graph.
        
        Args:
            llm: Language model for processing
            session_id: Unique identifier for this conversation
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.nodes: List[ConversationNode] = []
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.datetime.now().isoformat(),
            "last_updated": datetime.datetime.now().isoformat()
        }
        self.llm = llm
        logger.debug(f"Created new conversation graph with session_id: {self.session_id}")
    
    def _build_enhanced_bottle_prompt(self, bottle_data=None, detected_bottles=None) -> str:
        """
        Build enhanced prompt with detailed bottle information.
        
        Args:
            bottle_data: List of bottle dictionaries with detailed information
            detected_bottles: List of bottles specifically detected in user queries
        
        Returns:
            Enhanced prompt string with bottle-specific instructions
        """
        # Check if we have any bottles to work with
        has_detected = detected_bottles and isinstance(detected_bottles, list) and len(detected_bottles) > 0
        has_recommendations = bottle_data and isinstance(bottle_data, list) and len(bottle_data) > 0
        
        if not has_detected and not has_recommendations:
            return ""
        
        # Create a specialized bottle prompt
        bottle_prompt = "\n\nWHISKY BOTTLE DATA:\n"
        
        # Create a direct mapping of bottle names to their image URLs for reference
        image_url_map = {}
        nft_address_map = {}
        
        # Process detected bottles first (higher priority)
        if has_detected:
            bottle_prompt += "\n## DETECTED BOTTLES IN USER QUERY\n"
            bottle_prompt += "The user has specifically mentioned these bottles. Prioritize addressing them in your response:\n\n"
            
            for bottle in detected_bottles:
                bottle_name = bottle.get("name", "Unknown Bottle")
                bottle_type = bottle.get("type", "")
                bottle_region = bottle.get("region", "")
                bottle_country = bottle.get("country", "")
                bottle_desc = bottle.get("description", "")
                bottle_producer = bottle.get("producer", "")
                bottle_spirit = bottle.get("spirit_type", "")
                bottle_price = bottle.get("shelf_price", "")
                
                # Store the exact image URL for this bottle in the mapping
                if bottle.get("image_url"):
                    image_url_map[bottle_name.lower()] = bottle.get("image_url")
                
                # Store the NFT address for this bottle
                if bottle.get("nft_address"):
                    nft_address_map[bottle_name.lower()] = bottle.get("nft_address")
                
                bottle_prompt += f"### {bottle_name}\n"
                
                # Add bottle metadata in a formatted way
                metadata_parts = []
                if bottle_type:
                    metadata_parts.append(f"Type: {bottle_type}")
                if bottle_region:
                    metadata_parts.append(f"Region: {bottle_region}")
                if bottle_country:
                    metadata_parts.append(f"Country: {bottle_country}")
                if bottle_producer:
                    metadata_parts.append(f"Producer: {bottle_producer}")
                if bottle_spirit:
                    metadata_parts.append(f"Spirit: {bottle_spirit}")
                if bottle_price:
                    metadata_parts.append(f"Price: ${bottle_price}")
                
                if metadata_parts:
                    bottle_prompt += f"{' | '.join(metadata_parts)}\n"
                
                # Include a description if available
                if bottle_desc:
                    # Limit description length for prompt space efficiency
                    if len(bottle_desc) > 300:
                        bottle_desc = bottle_desc[:297] + "..."
                    bottle_prompt += f"Description: {bottle_desc}\n"
                
                # Include image and NFT info for proper rendering with special emphasis on exact image URL
                if bottle.get("image_url"):
                    bottle_prompt += f"EXACT Image URL: {bottle.get('image_url')}\n"
                    bottle_prompt += f"![{bottle_name}]({bottle.get('image_url')})\n"
                else:
                    bottle_prompt += f"NOTE: No image URL available for this bottle. DO NOT create one or use another bottle's image.\n"
                
                if bottle.get("nft_address"):
                    bottle_prompt += f"NFT Address: {bottle.get('nft_address')}\n"
                    bottle_prompt += f"[View on BAXUS](https://baxus.co/asset/{bottle.get('nft_address')})\n"
                else:
                    bottle_prompt += f"NOTE: No NFT address available for this bottle. DO NOT include a BAXUS link.\n"
                
                # Add a separator between bottles
                bottle_prompt += "\n"
        
        # Add recommended bottles if available
        if has_recommendations:
            # Filter out bottles that were already covered in detected_bottles to avoid duplication
            filtered_recommendations = bottle_data
            if has_detected:
                detected_names = {bottle.get("name", "").lower() for bottle in detected_bottles}
                filtered_recommendations = [rec for rec in bottle_data if rec.get("name", "").lower() not in detected_names]
            
            if filtered_recommendations:
                bottle_prompt += "\n## RECOMMENDED BOTTLES\n"
                bottle_prompt += "Consider including these bottles in your response when relevant:\n\n"
                
                for bottle in filtered_recommendations:
                    bottle_name = bottle.get("name", "Unknown Bottle")
                    bottle_type = bottle.get("type", "")
                    bottle_region = bottle.get("region", "")
                    bottle_reason = bottle.get("reason", "")
                    
                    # Store the exact image URL for this bottle in the mapping
                    if bottle.get("image_url"):
                        image_url_map[bottle_name.lower()] = bottle.get("image_url")
                    
                    # Store the NFT address for this bottle
                    if bottle.get("nft_address"):
                        nft_address_map[bottle_name.lower()] = bottle.get("nft_address")
                    
                    bottle_prompt += f"### {bottle_name}\n"
                    if bottle_type and bottle_region:
                        bottle_prompt += f"Type: {bottle_type} | Region: {bottle_region}\n"
                    elif bottle_type:
                        bottle_prompt += f"Type: {bottle_type}\n"
                    elif bottle_region:
                        bottle_prompt += f"Region: {bottle_region}\n"
                    
                    if bottle_reason:
                        bottle_prompt += f"Recommendation reason: {bottle_reason}\n"
                    
                    # Include image and NFT info for proper rendering with special emphasis on exact image URL
                    if bottle.get("image_url"):
                        bottle_prompt += f"EXACT Image URL: {bottle.get('image_url')}\n"
                        bottle_prompt += f"![{bottle_name}]({bottle.get('image_url')})\n"
                    else:
                        bottle_prompt += f"NOTE: No image URL available for this bottle. DO NOT create one or use another bottle's image.\n"
                    
                    if bottle.get("nft_address"):
                        bottle_prompt += f"NFT Address: {bottle.get('nft_address')}\n"
                        bottle_prompt += f"[View on BAXUS](https://baxus.co/asset/{bottle.get('nft_address')})\n"
                    else:
                        bottle_prompt += f"NOTE: No NFT address available for this bottle. DO NOT include a BAXUS link.\n"
                    
                    # Add a separator between bottles
                    bottle_prompt += "\n"
        
        # Add a reference table mapping bottle names to their exact image URLs
        if image_url_map:
            bottle_prompt += "\n## EXACT IMAGE URL MAPPING\n"
            bottle_prompt += "CRITICAL: Use ONLY these exact image URLs for each specific bottle. DO NOT mix them up or substitute any URLs:\n\n"
            
            for bottle_name, image_url in image_url_map.items():
                bottle_prompt += f"- {bottle_name.title()}: `{image_url}`\n"
        
        # Add a reference table mapping bottle names to their NFT addresses
        if nft_address_map:
            bottle_prompt += "\n## EXACT NFT ADDRESS MAPPING\n"
            bottle_prompt += "CRITICAL: Use ONLY these exact NFT addresses for each specific bottle. DO NOT mix them up or substitute any addresses:\n\n"
            
            for bottle_name, nft_address in nft_address_map.items():
                bottle_prompt += f"- {bottle_name.title()}: `{nft_address}`\n"
        
        # Add guidance on how to use this bottle data with stronger emphasis on image accuracy
        bottle_prompt += """
## CRITICAL DATA INTEGRITY INSTRUCTIONS
When discussing bottles in your response:
1. NEVER fabricate or make up any bottle data - only use what's provided from the database
2. If image URL is missing for a bottle, DO NOT include an image for that bottle
3. If NFT address is missing for a bottle, DO NOT include a BAXUS link for that bottle
4. NEVER mix up images or NFT addresses between different bottles
5. NEVER modify URLs or addresses - use them EXACTLY as provided

## CRITICAL IMAGE INSTRUCTIONS
When displaying bottle images in your response:
1. NEVER mix up images between different bottles - this is the #1 priority
2. ALWAYS use the EXACT image URL provided for each specific bottle
3. If discussing 'Macallan 18', ONLY use the Macallan 18 image, not any other Macallan or whisky image
4. Do not modify, edit, or create image URLs - use them EXACTLY as provided in the database
5. Double-check that the image URL you use matches the bottle you're describing
6. If an image URL isn't available for a specific bottle, DO NOT include an image at all
7. NEVER make up or invent image URLs

## NFT/ASSET ADDRESS INSTRUCTIONS
When providing BAXUS links:
1. ONLY include a BAXUS link when a genuine NFT address is available in the database
2. NEVER make up or invent NFT addresses - only use exact addresses from the database
3. Always use the format: [View on BAXUS](https://baxus.co/asset/NFT_ADDRESS)
4. If an NFT address isn't available, DO NOT include a BAXUS link at all

## GENERAL BOTTLE RESPONSE INSTRUCTIONS
When referencing these bottles in your response:
1. ALWAYS provide specific details about the bottle that are relevant to the user's question
2. If the user asks about bottles you detected, ALWAYS prioritize those bottles in your response
3. ONLY use information provided from the database - NEVER make up details
"""
        
        return bottle_prompt
    
    def invoke(self, state):
        """
        Process the agent state through the conversation graph.
        
        Args:
            state: The AgentState containing conversation context and messages
            
        Returns:
            The updated state with final_answer populated
        """
        try:
            logger.debug("Processing agent state through conversation graph")
            
            # Add user message to the graph
            if state.messages and len(state.messages) > 0:
                last_message = state.messages[-1]
                if last_message["role"] == "human":
                    self.add_user_message(last_message["content"])
            
            # Generate response using the context
            context_text = "\n".join(state.context) if state.context else ""
            
            if self.llm:
                # Process the messages through the LLM
                from langchain_core.messages import HumanMessage, SystemMessage
                
                # Base system message
                system_content = "You are Bob, a whisky expert AI assistant for BAXUS."
                
                # Check if we have bottle data to enhance the prompt
                has_detected_bottles = hasattr(state, 'detected_bottles') and state.detected_bottles
                has_recommendations = hasattr(state, 'recommendations') and state.recommendations
                
                if has_detected_bottles or has_recommendations:
                    # Add enhanced bottle-specific instructions
                    bottle_prompt = self._build_enhanced_bottle_prompt(
                        bottle_data=state.recommendations if has_recommendations else None,
                        detected_bottles=state.detected_bottles if has_detected_bottles else None
                    )
                    if bottle_prompt:
                        system_content += bottle_prompt
                        logger.debug("Added enhanced bottle prompt to system message")
                
                messages = [
                    SystemMessage(content=system_content),
                ]
                
                # Add context if available
                if context_text:
                    messages.append(SystemMessage(content=f"Additional context:\n{context_text}"))
                
                # Add the conversation history
                for msg in state.messages:
                    if msg["role"] == "human":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "ai":
                        from langchain_core.messages import AIMessage
                        messages.append(AIMessage(content=msg["content"]))
                
                # Generate response
                response = self.llm.invoke(messages)
                
                # Add assistant response to graph
                self.add_assistant_message(response.content)
                
                # Update state with the final answer
                state.final_answer = response.content
            else:
                logger.warning("No language model provided to conversation graph")
                state.final_answer = "I apologize, but I'm having trouble processing your request right now."
            
            return state
            
        except Exception as e:
            logger.error(f"Error in conversation graph processing: {e}")
            state.final_answer = "I encountered an issue while processing your request. Please try again."
            return state
    
    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a message from the user to the conversation.
        
        Args:
            content: The message text
            metadata: Additional information about the message
            
        Returns:
            The ID of the new node
        """
        node = ConversationNode(
            content=content,
            node_type="user",
            metadata=metadata or {}
        )
        self.nodes.append(node)
        self._update_last_modified()
        logger.debug(f"Added user message to conversation {self.session_id}: {content[:50]}...")
        return node.node_id
    
    def add_assistant_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a message from the assistant to the conversation.
        
        Args:
            content: The message text
            metadata: Additional information about the message
            
        Returns:
            The ID of the new node
        """
        node = ConversationNode(
            content=content,
            node_type="assistant",
            metadata=metadata or {}
        )
        self.nodes.append(node)
        self._update_last_modified()
        logger.debug(f"Added assistant message to conversation {self.session_id}")
        return node.node_id
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the conversation history.
        
        Args:
            limit: Maximum number of messages to return, starting from most recent
            
        Returns:
            List of messages in chronological order
        """
        nodes = self.nodes
        if limit is not None:
            nodes = nodes[-limit:]
        return [node.to_dict() for node in nodes]
    
    def get_last_n_exchanges(self, n: int = 3) -> List[Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]]:
        """
        Get the last N complete exchanges (user message + assistant response).
        
        Args:
            n: Number of exchanges to retrieve
            
        Returns:
            List of tuples, each containing (user_message, assistant_message)
        """
        exchanges = []
        idx = len(self.nodes) - 1
        
        while idx >= 0 and len(exchanges) < n:
            assistant_msg = None
            user_msg = None
            
            # Find assistant message
            if idx >= 0 and self.nodes[idx].node_type == "assistant":
                assistant_msg = self.nodes[idx].to_dict()
                idx -= 1
            
            # Find corresponding user message
            if idx >= 0 and self.nodes[idx].node_type == "user":
                user_msg = self.nodes[idx].to_dict()
                idx -= 1
            
            if assistant_msg or user_msg:
                exchanges.append((user_msg, assistant_msg))
            else:
                idx -= 1
        
        # Reverse to get chronological order
        exchanges.reverse()
        return exchanges
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the entire conversation graph to a dictionary."""
        return {
            "session_id": self.session_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationGraph':
        """Create a conversation graph from a dictionary."""
        graph = cls(session_id=data.get("session_id"))
        graph.metadata = data.get("metadata", {})
        
        for node_data in data.get("nodes", []):
            node = ConversationNode.from_dict(node_data)
            graph.nodes.append(node)
        
        return graph
    
    def to_json(self) -> str:
        """Convert the conversation graph to a JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ConversationGraph':
        """Create a conversation graph from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def _update_last_modified(self) -> None:
        """Update the last modified timestamp."""
        self.metadata["last_updated"] = datetime.datetime.now().isoformat()

    async def astream(self, state):
        """
        Process the agent state through the conversation graph and stream the response.
        
        Args:
            state: The AgentState containing conversation context and messages
            
        Returns:
            An async generator that yields response chunks
        """
        try:
            logger.debug("Processing agent state through conversation graph with streaming")
            
            # Add user message to the graph
            if state.messages and len(state.messages) > 0:
                last_message = state.messages[-1]
                if last_message["role"] == "human":
                    self.add_user_message(last_message["content"])
            
            # Generate response using the context
            context_text = "\n".join(state.context) if state.context else ""
            
            if self.llm:
                # Process the messages through the LLM
                from langchain_core.messages import HumanMessage, SystemMessage
                
                # Base system message
                system_content = "You are Bob, a whisky expert AI assistant for BAXUS."
                
                # Check if we have bottle data to enhance the prompt
                has_detected_bottles = hasattr(state, 'detected_bottles') and state.detected_bottles
                has_recommendations = hasattr(state, 'recommendations') and state.recommendations
                
                if has_detected_bottles or has_recommendations:
                    # Add enhanced bottle-specific instructions
                    bottle_prompt = self._build_enhanced_bottle_prompt(
                        bottle_data=state.recommendations if has_recommendations else None,
                        detected_bottles=state.detected_bottles if has_detected_bottles else None
                    )
                    if bottle_prompt:
                        system_content += bottle_prompt
                        logger.debug("Added enhanced bottle prompt to system message for streaming")
                
                messages = [
                    SystemMessage(content=system_content),
                ]
                
                # Add context if available
                if context_text:
                    messages.append(SystemMessage(content=f"Additional context:\n{context_text}"))
                
                # Add the conversation history
                for msg in state.messages:
                    if msg["role"] == "human":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "ai":
                        from langchain_core.messages import AIMessage
                        messages.append(AIMessage(content=msg["content"]))
                
                # Stream response chunks
                full_response = ""
                async for chunk in self.llm.astream(messages):
                    if hasattr(chunk, 'content') and chunk.content:
                        full_response += chunk.content
                        yield chunk.content
                
                # After streaming is complete, add the full assistant response to the graph
                self.add_assistant_message(full_response)
                
                # Update state with the final answer
                state.final_answer = full_response
            else:
                logger.warning("No language model provided to conversation graph")
                error_message = "I apologize, but I'm having trouble processing your request right now."
                state.final_answer = error_message
                yield error_message
            
        except Exception as e:
            logger.error(f"Error in conversation graph streaming: {e}")
            error_message = "I encountered an issue while processing your request. Please try again."
            state.final_answer = error_message
            yield error_message 