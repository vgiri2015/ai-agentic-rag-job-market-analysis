"""Market analysis reporter for job market analysis."""
import logging
import statistics
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

from .base_agent import BaseJobAgent
from .rag_store import JobMarketRAGStore

logger = logging.getLogger(__name__)

class MarketReporterAgent(BaseJobAgent):
    """Agent for generating market analysis reports."""
    
    def __init__(self, openai_key: str):
        """Initialize the market reporter agent."""
        super().__init__(openai_key)
        self.rag_store = JobMarketRAGStore(openai_key)
        
    def generate_report(self, job_data: List[Dict], tech_analysis: Dict) -> Dict:
        """Generate market report using RAG-enhanced analysis."""
        # Add jobs to RAG store if not already added
        self.rag_store.add_jobs(job_data)
        
        # Analyze different market aspects
        market_insights = {
            "salary_trends": self._analyze_salary_trends(job_data),
            "location_analysis": self._analyze_locations(),
            "company_insights": self._analyze_companies(),
            "remote_work_trends": self._analyze_remote_work(),
            "market_demands": self._analyze_market_demands(tech_analysis),
            "industry_trends": self._analyze_industry_trends()
        }
        
        # Save the report
        self.save_json(market_insights, "market_report.json")
        return market_insights
        
    def _analyze_salary_trends(self, job_data: List[Dict]) -> Dict:
        """Analyze salary trends using both statistical and RAG analysis."""
        # Extract and clean salary data
        salaries = []
        for job in job_data:
            salary = job.get('salary')
            if salary and isinstance(salary, (int, float)):
                salaries.append(salary)
        
        # Calculate basic statistics
        stats = {
            "average": statistics.mean(salaries) if salaries else 0,
            "median": statistics.median(salaries) if salaries else 0,
            "min": min(salaries) if salaries else 0,
            "max": max(salaries) if salaries else 0
        }
        
        # Get RAG insights about salary trends
        salary_insights = self.rag_store.analyze_trends("""
        Analyze salary trends in the job market. Consider:
        1. Salary ranges for different experience levels
        2. Industry-specific salary variations
        3. Location-based salary differences
        4. Correlation between skills and compensation
        """)
        
        return {
            "statistics": stats,
            "insights": salary_insights
        }
        
    def _analyze_locations(self) -> Dict:
        """Analyze location-based trends using RAG."""
        return self.rag_store.analyze_trends("""
        Analyze location-based trends in the job market:
        1. Top hiring locations
        2. Regional salary differences
        3. Location-specific skill requirements
        4. Remote work policies by region
        """)
        
    def _analyze_companies(self) -> Dict:
        """Analyze company-specific trends using RAG."""
        return self.rag_store.analyze_trends("""
        Analyze company-related trends:
        1. Top hiring companies
        2. Company size distribution
        3. Industry sector distribution
        4. Company benefits and perks
        """)
        
    def _analyze_remote_work(self) -> Dict:
        """Analyze remote work trends using RAG."""
        return self.rag_store.analyze_trends("""
        Analyze remote work trends:
        1. Percentage of remote positions
        2. Hybrid vs fully remote options
        3. Remote work requirements
        4. Geographic restrictions for remote work
        """)
        
    def _analyze_market_demands(self, tech_analysis: Dict) -> Dict:
        """Analyze market demands combining tech analysis with RAG insights."""
        # Combine tech analysis with job postings for enhanced insights
        return self.rag_store.analyze_trends(f"""
        Analyze market demands considering the following tech analysis:
        {tech_analysis}
        
        Focus on:
        1. High-demand skills and technologies
        2. Emerging role types
        3. Experience level requirements
        4. Industry-specific demands
        """)
        
    def _analyze_industry_trends(self) -> Dict:
        """Analyze broader industry trends using RAG."""
        return self.rag_store.analyze_trends("""
        Analyze broader industry trends:
        1. Growing industries and sectors
        2. Declining or transforming roles
        3. New job titles and roles
        4. Industry-specific technology adoption
        """)
