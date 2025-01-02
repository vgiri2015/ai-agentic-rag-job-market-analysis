"""Final report generation agent."""
import json
import logging
from typing import Dict, List
from .base_agent import BaseJobAgent
from pathlib import Path

logger = logging.getLogger(__name__)

class FinalReporterAgent(BaseJobAgent):
    """Agent for generating comprehensive final reports."""
    
    def generate_comprehensive_report(
        self,
        job_data: List[Dict],
        tech_analysis: Dict,
        market_report: Dict,
        ai_impact: Dict
    ) -> Dict:
        """Generate comprehensive final report."""
        logger.info("Generating comprehensive final report")
        
        if not all([job_data, tech_analysis, market_report, ai_impact]):
            raise ValueError("Missing required data for report generation")
            
        # Create report prompt
        system_prompt = "You are a job market analysis expert."
        user_prompt = """Generate a comprehensive report on the job market analysis. Include:
        1. Executive Summary
        2. Job Market Overview
        3. Technology Landscape
        4. AI Impact Analysis
        5. Future Trends and Recommendations
        
        Use the following data:
        Job Data: {job_data}
        Tech Analysis: {tech_analysis}
        Market Report: {market_report}
        AI Impact: {ai_impact}
        
        Format the response as a JSON object with these sections as keys.
        """
        
        try:
            # Get report from language model
            response = self.get_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt.format(
                    job_data=json.dumps(job_data[:10]),  # Sample of jobs
                    tech_analysis=json.dumps(tech_analysis),
                    market_report=json.dumps(market_report),
                    ai_impact=json.dumps(ai_impact)
                )
            )
            
            report = json.loads(response)
            
            # Save report
            self.save_json(report, "final_report.json")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating final report: {str(e)}")
            raise

    def _combine_agent_data(self, job_data: Dict, tech_analysis: Dict, market_report: Dict, ai_impact: Dict) -> Dict:
        """Combine data from all agents for comprehensive analysis."""
        # Ensure we're working with lists
        def get_list_items(data, key=None, max_items=15):
            try:
                if isinstance(data, (list, tuple)):
                    items = list(data)
                elif isinstance(data, dict) and key:
                    items = list(data.get(key, []))
                else:
                    items = []
                
                return items[:max_items] if items else []
            except Exception as e:
                logger.warning(f"Error processing items for {key}: {str(e)}")
                return []

        # Process AI impact data
        ai_impact_data = {
            "ai_role_analysis": get_list_items(ai_impact, "ai_role_analysis"),
            "required_ai_skills": get_list_items(ai_impact, "required_ai_skills"),
            "impact_on_roles": get_list_items(ai_impact, "impact_on_traditional_roles"),
            "future_adaptations": get_list_items(ai_impact, "future_adaptations")
        }
        
        # Process job market data
        job_market_data = {
            "job_postings": get_list_items(job_data),
            "skills_required": get_list_items(job_data, "skills_required"),
            "job_trends": get_list_items(job_data, "trends")
        }
        
        # Process technology analysis
        tech_analysis_data = {
            "tech_stacks": get_list_items(tech_analysis, "tech_stacks"),
            "emerging_technologies": get_list_items(tech_analysis, "emerging_tech"),
            "tool_requirements": get_list_items(tech_analysis, "tools")
        }
        
        # Process market insights
        market_insights_data = {
            "market_trends": get_list_items(market_report, "trends"),
            "skill_demands": get_list_items(market_report, "skills"),
            "industry_shifts": get_list_items(market_report, "shifts")
        }

        return {
            "job_market_data": job_market_data,
            "technology_analysis": tech_analysis_data,
            "market_insights": market_insights_data,
            "ai_impact_analysis": ai_impact_data
        }

    def _generate_comprehensive_report(self, combined_data: Dict) -> Dict:
        """Generate a comprehensive report with detailed AI skills and trends analysis."""
        system_prompt = """You are a leading AI technology and job market expert. Analyze the provided data and create a detailed report covering AI skills, trends, and transformations across different roles and sectors."""
        
        # Split the analysis into smaller chunks
        sections = [
            {
                "name": "current_ai_skills",
                "title": "Current AI Skills for Developers",
                "data": combined_data.get("ai_impact_analysis", {}),
                "focus": ["LLM frameworks", "AI/ML development", "Cloud services", "Vector databases", "Infrastructure"]
            },
            {
                "name": "leadership_skills",
                "title": "Leadership AI Skills",
                "data": combined_data.get("market_insights", {}),
                "focus": ["Strategy", "Team structure", "Ethics", "Risk management", "ROI"]
            },
            {
                "name": "data_engineering_trends",
                "title": "Data Engineering/Science Trends",
                "data": combined_data.get("technology_analysis", {}),
                "focus": ["Data architecture", "Analytics", "MLOps", "Governance"]
            },
            {
                "name": "essential_ai_coding",
                "title": "Essential AI Coding Skills",
                "data": combined_data.get("job_market_data", {}),
                "focus": ["LLM integration", "RAG", "Fine-tuning", "Prompt engineering", "Testing"]
            },
            {
                "name": "saas_ai_revolution",
                "title": "AI Revolution in SaaS",
                "data": combined_data.get("market_insights", {}),
                "focus": ["Architecture", "Design patterns", "Pricing", "Integration", "UX"]
            },
            {
                "name": "rag_frameworks",
                "title": "RAG Frameworks and Tools",
                "data": combined_data.get("technology_analysis", {}),
                "focus": ["LlamaIndex", "LangChain", "Vector stores", "Optimization"]
            },
            {
                "name": "agentic_ai_skills",
                "title": "Agentic AI Skills",
                "data": combined_data.get("ai_impact_analysis", {}),
                "focus": ["Architectures", "Tool integration", "Planning", "Multi-agent", "State management"]
            },
            {
                "name": "developer_upskilling",
                "title": "Software Developer Upskilling",
                "data": combined_data.get("job_market_data", {}),
                "focus": ["AI concepts", "Infrastructure", "Testing", "Security", "Workflows"]
            },
            {
                "name": "big_data_evolution",
                "title": "Big Data Evolution",
                "data": combined_data.get("technology_analysis", {}),
                "focus": ["AI platforms", "Processing", "Feature engineering", "Training", "Quality"]
            }
        ]
        
        final_report = {}
        
        for section in sections:
            try:
                user_prompt = f"""Based on the provided data, analyze the {section["title"]} focusing on:
                {chr(10).join(f"- {item}" for item in section["focus"])}
                
                Use this data for analysis:
                {json.dumps(section["data"], indent=2)}
                
                Format the response as a JSON object with these keys:
                - current_state
                - required_skills
                - implementation_strategies
                - future_trends
                - recommendations
                
                Keep the response concise and actionable."""

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                response = self.get_completion(messages)
                final_report[section["name"]] = json.loads(response)
                
            except Exception as e:
                logger.error(f"Error processing section {section['name']}: {str(e)}")
                final_report[section["name"]] = {
                    "error": f"Failed to process section: {str(e)}",
                    "current_state": [],
                    "required_skills": [],
                    "implementation_strategies": [],
                    "future_trends": [],
                    "recommendations": []
                }
        
        return final_report

    def _generate_markdown_report(self, report_data: Dict) -> str:
        """Convert the report data to markdown format."""
        sections = [
            ("Current AI Skills", "current_ai_skills"),
            ("Leadership AI Skills", "leadership_skills"),
            ("Data Engineering Trends", "data_engineering_trends"),
            ("Essential AI Coding Skills", "essential_ai_coding"),
            ("AI Revolution in SaaS", "saas_ai_revolution"),
            ("RAG Frameworks and Tools", "rag_frameworks"),
            ("Agentic AI Skills", "agentic_ai_skills"),
            ("Software Developer Upskilling", "developer_upskilling"),
            ("Big Data Evolution", "big_data_evolution")
        ]
        
        markdown = "# AI Impact on Job Market Analysis Report\n\n"
        
        for title, key in sections:
            section_data = report_data.get(key, {})
            if not section_data:
                continue
                
            markdown += f"## {title}\n\n"
            
            if "current_state" in section_data:
                markdown += "### Current State\n"
                for item in section_data["current_state"]:
                    markdown += f"- {item}\n"
                markdown += "\n"
                
            if "required_skills" in section_data:
                markdown += "### Required Skills\n"
                for item in section_data["required_skills"]:
                    markdown += f"- {item}\n"
                markdown += "\n"
                
            if "implementation_strategies" in section_data:
                markdown += "### Implementation Strategies\n"
                for item in section_data["implementation_strategies"]:
                    markdown += f"- {item}\n"
                markdown += "\n"
                
            if "future_trends" in section_data:
                markdown += "### Future Trends\n"
                for item in section_data["future_trends"]:
                    markdown += f"- {item}\n"
                markdown += "\n"
                
            if "recommendations" in section_data:
                markdown += "### Recommendations\n"
                for item in section_data["recommendations"]:
                    markdown += f"- {item}\n"
                markdown += "\n"
        
        return markdown

    def generate_report(self, job_data: Dict, tech_analysis: Dict, market_report: Dict, ai_impact: Dict) -> Dict:
        """Generate final report using data from all agents."""
        logger.info("Generating final report from combined agent data")
        
        if not all([job_data, tech_analysis, market_report, ai_impact]):
            raise ValueError("Missing required data for report generation")
            
        try:
            # Combine data from all agents
            combined_data = self._combine_agent_data(job_data, tech_analysis, market_report, ai_impact)
            
            # Generate comprehensive report
            report = self._generate_comprehensive_report(combined_data)
            
            # Save JSON report
            self.save_json(report, "final_report.json")
            
            # Generate and save markdown report
            markdown_report = self._generate_markdown_report(report)
            markdown_path = Path("data") / "final_report.md"
            markdown_path.write_text(markdown_report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating final report: {str(e)}")
            raise
