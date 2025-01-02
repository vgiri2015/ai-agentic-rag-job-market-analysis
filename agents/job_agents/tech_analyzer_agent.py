"""Tech stack analysis agent."""
import json
import logging
from typing import Dict, List
from .base_agent import BaseJobAgent

logger = logging.getLogger(__name__)

class TechAnalyzerAgent(BaseJobAgent):
    """Agent for analyzing technology requirements in job postings."""
    
    def analyze_tech_requirements(self, job_data: List[Dict]) -> Dict:
        """Analyze technology requirements from job data."""
        logger.info("Analyzing technology requirements")
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
                        "description": truncated_desc
                    }
                    simplified_batch.append(simplified_job)
                
                messages = [
                    {"role": "system", "content": """You are a technology requirements analyzer. Your task is to analyze job postings and identify:
                    1. Required technical skills and tools
                    2. Common technology stacks
                    3. Emerging technology trends
                    4. Education and certification requirements
                    
                    Format your response as a JSON object with these keys:
                    {
                        "technical_skills": {"skill_name": frequency_count, ...},
                        "tech_stacks": {"stack_name": frequency_count, ...},
                        "emerging_trends": ["trend1", "trend2", ...],
                        "education_requirements": {"requirement": frequency_count, ...}
                    }"""},
                    {"role": "user", "content": f"Analyze these job postings and provide insights about technology requirements: {json.dumps(simplified_batch)}"}
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
                "technical_skills": {},
                "tech_stacks": {},
                "emerging_trends": set(),
                "education_requirements": {}
            }
            
            for analysis in all_analyses:
                # Combine technical skills
                for skill, count in analysis.get("technical_skills", {}).items():
                    combined_analysis["technical_skills"][skill] = \
                        combined_analysis["technical_skills"].get(skill, 0) + count
                
                # Combine tech stacks
                for stack, count in analysis.get("tech_stacks", {}).items():
                    combined_analysis["tech_stacks"][stack] = \
                        combined_analysis["tech_stacks"].get(stack, 0) + count
                
                # Combine emerging trends (using set to avoid duplicates)
                combined_analysis["emerging_trends"].update(analysis.get("emerging_trends", []))
                
                # Combine education requirements
                for req, count in analysis.get("education_requirements", {}).items():
                    combined_analysis["education_requirements"][req] = \
                        combined_analysis["education_requirements"].get(req, 0) + count
            
            # Convert emerging_trends set back to list
            combined_analysis["emerging_trends"] = list(combined_analysis["emerging_trends"])
            
            # Sort dictionaries by value in descending order
            combined_analysis["technical_skills"] = dict(sorted(
                combined_analysis["technical_skills"].items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            combined_analysis["tech_stacks"] = dict(sorted(
                combined_analysis["tech_stacks"].items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            combined_analysis["education_requirements"] = dict(sorted(
                combined_analysis["education_requirements"].items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            # Save the analysis
            self.save_json(combined_analysis, "tech_analysis.json")
            
            return combined_analysis

        except Exception as e:
            logger.error(f"Error in tech analysis: {str(e)}")
            raise
