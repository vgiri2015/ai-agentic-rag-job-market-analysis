"""AI impact analysis agent."""
import json
import logging
from typing import Dict, List
from .base_agent import BaseJobAgent

logger = logging.getLogger(__name__)

class AIImpactAnalyzerAgent(BaseJobAgent):
    """Agent for analyzing AI's impact on job roles."""
    
    @property
    def system_prompt(self) -> str:
        """Get the system prompt for AI impact analysis."""
        return """You are an AI impact analysis expert. Your task is to analyze how AI is impacting job roles and the job market.

        Format your response as a valid JSON object with the following structure:
        {
            "ai_role_analysis": [],  // List of findings about AI's role
            "required_ai_skills": [],  // List of required AI skills
            "impact_on_traditional_roles": [],  // List of impacts
            "future_adaptations": []  // List of needed adaptations
        }
        """

    def analyze_ai_impact(
        self,
        job_data: List[Dict],
        tech_analysis: Dict,
        market_report: Dict
    ) -> Dict:
        """Analyze the impact of AI on job roles."""
        logger.info("Analyzing AI impact on job roles")
        
        # Process job data in batches
        batch_size = 10  # Reduced batch size
        num_batches = (len(job_data) + batch_size - 1) // batch_size
        all_impacts = []
        
        for i in range(num_batches):
            logger.info(f"Processing batch {i + 1} of {num_batches}")
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(job_data))
            batch_jobs = job_data[start_idx:end_idx]
            
            # Simplify job data to essential fields
            simplified_jobs = []
            for job in batch_jobs:
                simplified_job = {
                    "title": job.get("title", ""),
                    "company_name": job.get("company_name", ""),
                    "location": job.get("location", ""),
                    "description": job.get("description", "")[:300]  # Truncate description
                }
                simplified_jobs.append(simplified_job)
            
            # Create a summary of tech analysis and market report
            tech_summary = {
                "top_technologies": tech_analysis.get("top_technologies", [])[:5],  # Reduced to top 5
                "emerging_trends": tech_analysis.get("emerging_trends", [])[:3]  # Reduced to top 3
            }
            
            market_summary = {
                "key_trends": market_report.get("key_trends", [])[:3],  # Reduced to top 3
                "market_insights": market_report.get("market_insights", [])[:3]  # Reduced to top 3
            }
            
            prompt = f"""Please analyze the impact of AI on these job roles based on the provided data.

Job Data (Batch {i + 1} of {num_batches}):
{json.dumps(simplified_jobs, indent=2)}

Technology Analysis Summary:
{json.dumps(tech_summary, indent=2)}

Market Report Summary:
{json.dumps(market_summary, indent=2)}

Please provide an analysis focusing on:
1. AI's role in these positions
2. Required AI skills and knowledge
3. Impact on traditional responsibilities
4. Future trends and adaptations needed

Format the response as a JSON object with these keys:
{{
    "ai_role_analysis": [],  // List of findings about AI's role
    "required_ai_skills": [],  // List of required AI skills
    "impact_on_traditional_roles": [],  // List of impacts
    "future_adaptations": []  // List of needed adaptations
}}"""
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            try:
                response = self.get_completion(messages)
                impact_analysis = json.loads(response)
                all_impacts.append(impact_analysis)
            except Exception as e:
                logger.error(f"Error analyzing AI impact: {str(e)}")
                raise
        
        # Combine all impacts
        combined_impact = {
            "ai_role_analysis": [],
            "required_ai_skills": [],
            "impact_on_traditional_roles": [],
            "future_adaptations": []
        }
        
        for impact in all_impacts:
            combined_impact["ai_role_analysis"].extend(impact.get("ai_role_analysis", []))
            combined_impact["required_ai_skills"].extend(impact.get("required_ai_skills", []))
            combined_impact["impact_on_traditional_roles"].extend(impact.get("impact_on_traditional_roles", []))
            combined_impact["future_adaptations"].extend(impact.get("future_adaptations", []))
        
        # Remove duplicates while preserving order
        for key in combined_impact:
            combined_impact[key] = list(dict.fromkeys(combined_impact[key]))
        
        # Save the analysis
        self.save_json(combined_impact, "ai_impact_analysis.json")
        
        return combined_impact
