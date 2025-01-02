"""Technology analysis agent for job market analysis."""
import logging
from typing import Dict, List, Optional
from pathlib import Path

from .base_agent import BaseJobAgent
from .rag_store import JobMarketRAGStore

logger = logging.getLogger(__name__)

class TechAnalyzerAgent(BaseJobAgent):
    """Agent for analyzing technology requirements in job listings."""
    
    def __init__(self, openai_key: str):
        """Initialize the tech analyzer agent."""
        super().__init__(openai_key)
        self.rag_store = JobMarketRAGStore(openai_key)
        
    def analyze_tech_requirements(self, job_data: List[Dict]) -> Dict:
        """Analyze technology requirements from job listings using RAG."""
        # First, add jobs to RAG store for analysis
        self.rag_store.add_jobs(job_data)
        
        # Analyze different aspects using RAG
        analyses = {
            "programming_languages": self._analyze_aspect("What are the most in-demand programming languages? Include frequency and context of usage."),
            "frameworks": self._analyze_aspect("What are the popular frameworks and technologies? Include both frontend and backend frameworks."),
            "cloud_services": self._analyze_aspect("What cloud services and platforms are commonly required? Include specific services and their use cases."),
            "tools": self._analyze_aspect("What development tools, version control systems, and DevOps tools are mentioned?"),
            "soft_skills": self._analyze_aspect("What soft skills and non-technical requirements are emphasized?")
        }
        
        # Get similar job clusters for pattern analysis
        tech_clusters = self._analyze_tech_clusters(job_data)
        
        # Combine all analyses
        tech_analysis = {
            "analyses": analyses,
            "tech_clusters": tech_clusters,
            "total_jobs_analyzed": len(job_data)
        }
        
        # Save analysis
        self.save_json(tech_analysis, "tech_analysis.json")
        return tech_analysis
        
    def _analyze_aspect(self, query: str) -> Dict:
        """Analyze a specific aspect using RAG-enhanced querying."""
        return self.rag_store.analyze_trends(query)
        
    def _analyze_tech_clusters(self, job_data: List[Dict]) -> List[Dict]:
        """Analyze technology clusters using similar job matching."""
        clusters = []
        
        # Sample some jobs as cluster centers
        for job in job_data[:5]:  # Use first 5 jobs as cluster centers
            title = job.get('title', '')
            similar_jobs = self.rag_store.query_similar_jobs(
                f"Find jobs similar to: {title}",
                top_k=3
            )
            
            clusters.append({
                "center_job": title,
                "similar_jobs": similar_jobs,
                "common_technologies": self._extract_common_technologies(similar_jobs)
            })
            
        return clusters
        
    def _extract_common_technologies(self, similar_jobs: List[Dict]) -> Dict:
        """Extract common technologies from a cluster of similar jobs."""
        # Use RAG to analyze common patterns
        jobs_text = "\n".join([job['text'] for job in similar_jobs])
        analysis = self.rag_store.analyze_trends(
            f"What are the common technologies and skills required across these related positions?\n\n{jobs_text}"
        )
        return analysis
