"""AI impact analysis agent for job market analysis."""
import logging
from typing import Dict, List, Optional
from pathlib import Path

from .base_agent import BaseJobAgent
from .rag_store import JobMarketRAGStore

logger = logging.getLogger(__name__)

class AIImpactAnalyzerAgent(BaseJobAgent):
    """Agent for analyzing AI's impact on job market."""
    
    def __init__(self, openai_key: str):
        """Initialize the AI impact analyzer agent."""
        super().__init__(openai_key)
        self.rag_store = JobMarketRAGStore(openai_key)
        
    def analyze_ai_impact(self, job_data: List[Dict], tech_analysis: Dict) -> Dict:
        """Analyze AI's impact on job market using RAG."""
        # Add jobs to RAG store if not already added
        self.rag_store.add_jobs(job_data)
        
        # Analyze different aspects of AI impact
        impact_analysis = {
            "ai_skill_requirements": self._analyze_ai_skills(),
            "ai_job_evolution": self._analyze_job_evolution(),
            "ai_tool_adoption": self._analyze_ai_tools(tech_analysis),
            "ai_industry_impact": self._analyze_industry_impact(),
            "future_trends": self._analyze_future_trends()
        }
        
        # Save analysis
        self.save_json(impact_analysis, "ai_impact_analysis.json")
        return impact_analysis
        
    def _analyze_ai_skills(self) -> Dict:
        """Analyze AI-specific skill requirements using RAG."""
        return self.rag_store.analyze_trends("""
Analyze AI-specific skill requirements in job postings:
1. Required AI/ML frameworks and tools
2. Experience levels for AI roles
3. Specialized AI skills (NLP, CV, etc.)
4. Non-technical skills for AI roles
Analyze how jobs are evolving with AI integration:
1. Traditional roles incorporating AI
2. New AI-specific job titles
3. Changes in job responsibilities
4. AI automation impact
Analyze AI tool adoption considering this tech analysis:
{tech_analysis}

Focus on:
1. Popular AI/ML frameworks
2. Cloud AI services
3. AI development tools
4. Industry-specific AI solutions
Analyze AI's impact across different industries:
1. Industry-specific AI adoption
2. Transformation of workflows
3. AI-driven innovation
4. Industry challenges and opportunities
Analyze future AI trends in the job market:
1. Emerging AI technologies
2. Future skill requirements
3. Potential job market changes
4. AI adoption challenges
"""
