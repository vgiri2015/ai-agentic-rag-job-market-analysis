"""Base agent for job market analysis."""
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

class BaseJobAgent:
    """Base class for job market analysis agents."""
    
    def __init__(self, openai_key: str):
        """Initialize base agent."""
        self.llm = ChatOpenAI(
            api_key=openai_key,
            model="gpt-4",
            temperature=0.0
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
    def get_completion(self, messages: List) -> str:
        """Get completion from language model."""
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            self.logger.error(f"Error getting completion: {str(e)}")
            raise
        
    def save_json(self, data: Dict[str, Any], filename: str) -> None:
        """Save data to JSON file."""
        try:
            output_path = self.data_dir / filename
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
                
            self.logger.info(f"Saved data to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            raise
        
    def load_json(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load data from JSON file."""
        try:
            input_path = self.data_dir / filename
            if input_path.exists():
                with open(input_path, "r") as f:
                    return json.load(f)
            return None
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
