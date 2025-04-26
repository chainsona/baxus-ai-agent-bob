"""
Agent State Module
-----------------
Define the state class used across agent execution.
"""

class AgentState:
    """State tracked across agent execution."""

    def __init__(self):
        self.messages = []
        self.context = []
        self.username = None
        self.collection_data = None
        self.tools_to_call = []
        self.tool_results = []
        self.current_tool = None
        self.final_answer = None
        self.step_count = 0
        self.recommendations = []  # Store detected bottle recommendations
        self.detected_bottles = []  # Store specifically detected bottles from user queries 