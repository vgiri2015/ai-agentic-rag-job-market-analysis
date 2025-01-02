"""Final report generation agent."""
import json
import logging
import re
from typing import Dict, List, Optional
from .base_agent import BaseJobAgent
from pathlib import Path

logger = logging.getLogger(__name__)

class FinalReporterAgent(BaseJobAgent):
    """Agent for generating comprehensive final reports."""
    
    def _extract_job_essentials(self, job: Dict) -> Dict:
        """Extract only essential information from a job listing to reduce tokens."""
        # Check for AI/ML in title or description
        title = job.get("title", "").lower()
        description = job.get("description", "").lower()
        ai_terms = ["ai", "machine learning", "ml", "deep learning", "neural network", "data scientist"]
        is_ai = any(term in title or term in description for term in ai_terms)
        
        # Check for remote work in extensions or description
        extensions = job.get("extensions", [])
        is_remote = any("remote" in ext.lower() for ext in extensions) or "remote" in description
        
        return {
            "title": job.get("title", ""),
            "salary": self._parse_salary(job.get("salary", ""), description),
            "is_remote": is_remote,
            "is_ai": is_ai
        }
    
    def _parse_salary(self, salary_str: str, description: str) -> Optional[float]:
        """Parse salary from string, looking for both explicit salary and salary ranges."""
        if not salary_str and description:
            # Try to find salary in description
            matches = re.findall(r'\$\s*(\d{2,3}(?:,\d{3})*(?:\.\d{2})?)\s*k?', description.lower())
            if matches:
                # Use the first match
                salary_str = matches[0]
        
        if not salary_str:
            return None
            
        try:
            # Remove common non-numeric characters
            clean_salary = salary_str.replace("$", "").replace(",", "").replace("k", "000")
            
            # Handle ranges by taking the average
            if "-" in clean_salary:
                low, high = map(float, clean_salary.split("-"))
                return (low + high) / 2
            
            return float(clean_salary)
        except (ValueError, TypeError):
            return None
    
    def _analyze_jobs(self, jobs: List[Dict]) -> Dict:
        """Analyze jobs directly without using LLM."""
        stats = {
            "total_jobs": len(jobs),
            "ai_specific_roles": 0,
            "remote_jobs": 0,
            "salary_data": {
                "total_with_salary": 0,
                "ranges": {"0_50k": 0, "50_100k": 0, "100k_plus": 0},
                "average": 0,
                "total_salary": 0
            }
        }
        
        for job in jobs:
            # Count AI roles
            if job["is_ai"]:
                stats["ai_specific_roles"] += 1
            
            # Count remote jobs
            if job["is_remote"]:
                stats["remote_jobs"] += 1
            
            # Analyze salary
            if job["salary"]:
                salary = job["salary"]
                stats["salary_data"]["total_with_salary"] += 1
                stats["salary_data"]["total_salary"] += salary
                
                if salary <= 50000:
                    stats["salary_data"]["ranges"]["0_50k"] += 1
                elif salary <= 100000:
                    stats["salary_data"]["ranges"]["50_100k"] += 1
                else:
                    stats["salary_data"]["ranges"]["100k_plus"] += 1
        
        # Calculate percentages and averages
        total = stats["total_jobs"]
        stats["remote_percentage"] = round((stats["remote_jobs"] / total) * 100, 1) if total > 0 else 0
        stats["ai_percentage"] = round((stats["ai_specific_roles"] / total) * 100, 1) if total > 0 else 0
        
        if stats["salary_data"]["total_with_salary"] > 0:
            stats["salary_data"]["average"] = round(
                stats["salary_data"]["total_salary"] / stats["salary_data"]["total_with_salary"], 2
            )
        
        return stats
    
    def generate_comprehensive_report(
        self,
        job_data: List[Dict],
        tech_analysis: Dict,
        market_report: Dict,
        ai_impact: Dict
    ) -> Dict:
        """Generate comprehensive final report using LLM analysis."""
        logger.info("Generating comprehensive final report")
        
        if not all([job_data, tech_analysis, market_report, ai_impact]):
            raise ValueError("Missing required data for report generation")

        try:
            # Process jobs directly without chunking
            logger.info("Analyzing job data...")
            essential_jobs = [self._extract_job_essentials(job) for job in job_data]
            statistics = self._analyze_jobs(essential_jobs)
            
            # Extract key insights
            insights = {
                "statistics": statistics,
                "tech_trends": tech_analysis.get("emerging_trends", [])[:3],
                "market_trends": market_report.get("market_trends", [])[:3],
                "ai_impact": {
                    "summary": ai_impact.get("impact_summary", ""),
                    "key_skills": ai_impact.get("required_ai_skills", [])[:5]
                }
            }
            
            # Generate the analysis
            analysis_prompt = f"""Generate a focused job market analysis report in this exact JSON format:
            {{
                "executive_summary": [
                    <string: key finding about overall job market>,
                    <string: key finding about AI roles and skills>,
                    <string: key finding about remote work and salary trends>
                ],
                "market_overview": {{
                    "key_findings": [<string: finding about current state>, <string: finding about skills>],
                    "opportunities": [<string: opportunity in AI/ML>, <string: opportunity in work arrangements>]
                }},
                "future_outlook": {{
                    "trends": [<string: tech trend>, <string: workplace trend>],
                    "recommendations": [<string: recommendation for professionals>, <string: recommendation for companies>]
                }}
            }}
            
            Base your analysis on these insights: {json.dumps(insights)}
            Focus on actionable insights and clear trends.
            """
            
            analysis_response = self.get_completion(analysis_prompt)
            try:
                # Try to find JSON object in the text
                start = analysis_response.find('{')
                end = analysis_response.rfind('}')
                if start >= 0 and end >= 0:
                    report = json.loads(analysis_response[start:end+1])
                else:
                    report = json.loads(analysis_response)
            except Exception as e:
                logger.error(f"Error parsing analysis response: {str(e)}")
                report = {"error": "Failed to parse analysis"}
            
            report["statistics"] = statistics
            
            # Generate markdown report
            markdown_prompt = f"""Convert this job market analysis to a clear markdown document.
            Use ## for main sections and ### for subsections.
            Use bullet points (*) for lists.
            Include these sections:
            1. Executive Summary
            2. Key Statistics (include all numbers)
            3. Market Overview
            4. Future Outlook
            
            Format salary and percentage data clearly.
            Report data: {json.dumps(report)}
            """
            
            markdown_response = self.get_completion(markdown_prompt)
            
            # Save reports
            self.save_json(report, "final_report.json")
            
            # Ensure reports directory exists
            Path("reports").mkdir(exist_ok=True)
            
            # Save markdown report
            with open("reports/final_report.md", "w") as f:
                f.write(markdown_response)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating final report: {str(e)}")
            raise
