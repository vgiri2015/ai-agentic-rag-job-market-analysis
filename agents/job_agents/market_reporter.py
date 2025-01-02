"""Market report generation agent."""
import json
import logging
from typing import Dict, List
from .base_agent import BaseJobAgent

logger = logging.getLogger(__name__)

class MarketReporterAgent(BaseJobAgent):
    """Agent for generating market analysis reports."""
    
    def generate_report(self, job_data: List[Dict], tech_analysis: Dict) -> Dict:
        """Generate market analysis report."""
        logger.info("Generating market analysis report")
        try:
            # Process jobs in batches of 20 to avoid context length limits
            batch_size = 20
            all_analyses = []
            
            for i in range(0, len(job_data), batch_size):
                batch = job_data[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1} of {(len(job_data) + batch_size - 1)//batch_size}")
                
                # Prepare simplified job data
                simplified_batch = []
                for job in batch:
                    desc = job.get("description", "")
                    # Take first 500 characters of description
                    truncated_desc = desc[:500] + "..." if len(desc) > 500 else desc
                    
                    simplified_job = {
                        "title": job.get("title", "Unknown Title"),
                        "company": job.get("company_name", "Unknown Company"),
                        "location": job.get("location", "Unknown Location"),
                        "description": truncated_desc
                    }
                    simplified_batch.append(simplified_job)
                
                messages = [
                    {"role": "system", "content": """You are a job market analyst. Your task is to analyze job postings and identify:
                    1. Salary trends and compensation patterns
                    2. Required experience levels and qualifications
                    3. Industry sectors and company types
                    4. Remote work and location preferences
                    5. Benefits and perks
                    
                    Format your response as a JSON object with these keys:
                    {
                        "salary_insights": {"range": count, ...},
                        "experience_requirements": {"level": count, ...},
                        "industry_sectors": {"sector": count, ...},
                        "work_arrangements": {"type": count, ...},
                        "benefits_perks": {"benefit": count, ...}
                    }
                    
                    IMPORTANT: All count values must be integers."""},
                    {"role": "user", "content": f"Analyze these job postings and provide market insights: {json.dumps(simplified_batch)}"}
                ]

                completion = self.llm.invoke(messages)
                try:
                    analysis = json.loads(completion.content)
                    all_analyses.append(analysis)
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON from completion: {str(e)}")
                    logger.error(f"Raw completion: {completion.content}")
                    continue
            
            # Combine all analyses
            combined_analysis = {
                "salary_insights": {},
                "experience_requirements": {},
                "industry_sectors": {},
                "work_arrangements": {},
                "benefits_perks": {}
            }
            
            def ensure_int(val):
                """Convert value to int if possible, otherwise return 0"""
                try:
                    return int(val)
                except (ValueError, TypeError):
                    return 0
            
            for analysis in all_analyses:
                # Combine salary insights
                for salary, count in analysis.get("salary_insights", {}).items():
                    combined_analysis["salary_insights"][salary] = \
                        combined_analysis["salary_insights"].get(salary, 0) + ensure_int(count)
                
                # Combine experience requirements
                for exp, count in analysis.get("experience_requirements", {}).items():
                    combined_analysis["experience_requirements"][exp] = \
                        combined_analysis["experience_requirements"].get(exp, 0) + ensure_int(count)
                
                # Combine industry sectors
                for sector, count in analysis.get("industry_sectors", {}).items():
                    combined_analysis["industry_sectors"][sector] = \
                        combined_analysis["industry_sectors"].get(sector, 0) + ensure_int(count)
                
                # Combine work arrangements
                for arr, count in analysis.get("work_arrangements", {}).items():
                    combined_analysis["work_arrangements"][arr] = \
                        combined_analysis["work_arrangements"].get(arr, 0) + ensure_int(count)
                
                # Combine benefits and perks
                for benefit, count in analysis.get("benefits_perks", {}).items():
                    combined_analysis["benefits_perks"][benefit] = \
                        combined_analysis["benefits_perks"].get(benefit, 0) + ensure_int(count)
            
            # Sort all dictionaries by value in descending order
            for key in combined_analysis:
                combined_analysis[key] = dict(sorted(
                    combined_analysis[key].items(),
                    key=lambda x: x[1],
                    reverse=True
                ))
            
            # Add tech analysis summary
            combined_analysis["tech_requirements"] = {
                "top_skills": dict(list(tech_analysis.get("technical_skills", {}).items())[:10]),
                "top_stacks": dict(list(tech_analysis.get("tech_stacks", {}).items())[:10]),
                "emerging_trends": tech_analysis.get("emerging_trends", [])[:5],
                "top_education": dict(list(tech_analysis.get("education_requirements", {}).items())[:5])
            }
            
            # Save the analysis
            self.save_json(combined_analysis, "market_report.json")
            
            return combined_analysis

        except Exception as e:
            logger.error(f"Error generating market report: {str(e)}")
            raise
