"""Final Reporter Agent for generating comprehensive job market analysis reports.

This agent synthesizes data from various sources to generate a detailed report with:
1. Key Statistics and Rankings
   - Job market overview
   - Top 10 AI jobs with salary ranges
   - Top AI skills by demand
   - Top AI development tools
   - Vector database usage
   - LLM usage statistics
   - Market projections

2. Detailed Analysis Sections
   - Executive summary
   - Technical skills landscape
   - Market dynamics
   - AI impact assessment
   - Strategic recommendations

The report format is optimized for readability and actionable insights,
with specific statistics and percentages to support all findings.
"""
import logging
from typing import Dict, List, Optional
from pathlib import Path
import json
from datetime import datetime

from .base_agent import BaseJobAgent
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

class FinalReporterAgent(BaseJobAgent):
    """Agent responsible for generating comprehensive job market reports."""
    
    def __init__(self, openai_key: str):
        """Initialize the final reporter agent."""
        super().__init__(openai_key)
        self.summarizer = ChatOpenAI(
            api_key=openai_key,
            model="gpt-3.5-turbo",
            temperature=0.0
        )
        
    def _chunk_data(self, data: Dict) -> List[Dict]:
        """Split data into smaller chunks."""
        chunks = []
        max_size = 50000  # Reduced size to handle token limits better
        
        if isinstance(data, list):
            # Handle list input
            current_chunk = []
            current_size = 0
            
            for item in data:
                item_str = json.dumps(item)
                item_size = len(item_str)
                
                if item_size > max_size:
                    # If a single item is too large, add it as its own chunk
                    # Try to extract key information
                    if isinstance(item, dict):
                        summary_item = {
                            k: str(v)[:max_size//2] if isinstance(v, str) else v 
                            for k, v in item.items()
                        }
                        chunks.append([summary_item])
                    else:
                        chunks.append([str(item)[:max_size]])
                else:
                    if current_size + item_size > max_size:
                        # Start a new chunk
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = [item]
                        current_size = item_size
                    else:
                        current_chunk.append(item)
                        current_size += item_size
            
            if current_chunk:
                chunks.append(current_chunk)
                
        else:
            # Handle dictionary input
            current_chunk = {}
            current_size = 0
            
            for key, value in data.items():
                value_str = json.dumps(value)
                value_size = len(value_str)
                
                if value_size > max_size:
                    # If a single value is too large, split it further
                    if isinstance(value, list):
                        # Split list into smaller chunks
                        chunk_size = len(value) // ((value_size // max_size) + 1)
                        for i in range(0, len(value), chunk_size):
                            chunk = {key: value[i:i + chunk_size]}
                            chunks.append(chunk)
                    elif isinstance(value, str):
                        # For large strings, take first portion
                        chunks.append({key: value[:max_size]})
                    else:
                        # For other types, convert to string and truncate
                        chunks.append({key: str(value)[:max_size]})
                else:
                    if current_size + value_size > max_size:
                        # Start a new chunk
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = {key: value}
                        current_size = value_size
                    else:
                        current_chunk[key] = value
                        current_size += value_size
            
            if current_chunk:
                chunks.append(current_chunk)
        
        return chunks
        
    def _extract_key_points(self, data: Dict) -> str:
        """Extract key points from analysis data."""
        chunks = self._chunk_data(data)
        summaries = []
        
        # Process each chunk with focused prompts
        for chunk in chunks:
            chunk_str = str(chunk)
            
            # Code assist analysis
            code_assist_prompt = """Analyze the adoption and impact of AI code assistance tools.
            Focus on:
            1. Development tools evolution and adoption rates
            2. Developer productivity metrics
            3. Popular tools and market share
            4. ROI and efficiency gains
            
            Data to analyze:
            """ + chunk_str
            
            code_assist_analysis = self.get_completion(code_assist_prompt)
            summaries.append(code_assist_analysis)
            
            # Business domain analysis
            business_domain_prompt = """Analyze AI technology adoption across business domains.
            Focus on:
            1. Industry-specific adoption rates
            2. Implementation areas and success rates
            3. ROI and impact metrics
            4. Key trends and challenges
            
            Data to analyze:
            """ + chunk_str
            
            business_domain_analysis = self.get_completion(business_domain_prompt)
            summaries.append(business_domain_analysis)
        
        # Combine summaries with shorter context
        combined_prompt = """Synthesize these findings into a cohesive analysis:
        
        Code Assistant Analysis:
        {}
        
        Business Domain Analysis:
        {}
        """.format("\n".join(summaries[::2]), "\n".join(summaries[1::2]))
        
        return self.get_completion(combined_prompt)
        
    def _analyze_tech_landscape(self, tech_summary: str) -> str:
        """Analyze technical skills landscape."""
        MARKET_SUMMARY_PROMPT = """Analyze the market trends with detailed statistics focusing on:

1. Market Size and Growth:
   - Total AI market value
   - Growth rate by sector
   - Regional distribution
   - Investment trends
   - Future projections

2. Top 10 AI Companies by:
   - Market share
   - Innovation index
   - Employee count
   - Revenue growth
   - AI patent filings

3. LLM Market Analysis:
   - Market leaders and share %
   - Pricing models comparison
   - Open source vs proprietary
   - Deployment preferences
   - Cost analysis

4. RAG Implementation Stats:
   - Adoption rates by industry
   - Popular frameworks (%)
   - Vector DB market share
   - Implementation costs
   - ROI metrics

5. AI Tools Market Share:
   - Development platforms
   - Cloud services
   - Vector databases
   - MLOps tools
   - Monitoring solutions

Include market size, growth rates, and adoption percentages.
"""

        prompt = f"""
        {MARKET_SUMMARY_PROMPT}
        
        {tech_summary}
        
        Create a brief section (max 200 words) covering:
        1. Most in-demand skills (top 2)
        2. Emerging technologies (top 2)
        3. Key skill gaps (top 2)
        
        Format the response in markdown with bullet points.
        """
        
        messages = [
            SystemMessage(content="You are an expert technical skills analyst."),
            HumanMessage(content=prompt)
        ]
        
        return self.get_completion(messages)
        
    def _analyze_market_dynamics(self, market_summary: str) -> str:
        """Analyze market dynamics."""
        AI_IMPACT_PROMPT = """Analyze the AI impact with specific metrics focusing on:

1. Workforce Transformation:
   - Jobs automated (%)
   - New roles created
   - Skill transition rates
   - Salary impact
   - Training requirements

2. LLM Integration Impact:
   - Productivity gains
   - Cost savings
   - Development speedup
   - Quality improvements
   - Resource optimization

3. RAG Implementation ROI:
   - Efficiency gains
   - Cost reduction
   - Knowledge access improvement
   - Error reduction
   - User satisfaction

4. Agentic AI Metrics:
   - Automation levels
   - Task completion rates
   - Accuracy improvements
   - Cost efficiency
   - Integration success rates

Include before/after comparisons and specific improvement metrics.
"""

        prompt = f"""
        {AI_IMPACT_PROMPT}
        
        {market_summary}
        
        Create a brief section (max 200 words) covering:
        1. Key salary trends (top 2)
        2. Geographic insights (top 2)
        3. Industry highlights (top 2)
        
        Format the response in markdown with bullet points.
        """
        
        messages = [
            SystemMessage(content="You are an expert market analyst."),
            HumanMessage(content=prompt)
        ]
        
        return self.get_completion(messages)
        
    def _analyze_ai_impact(self, ai_summary: str) -> str:
        """Analyze AI impact."""
        DETAILED_SECTIONS_PROMPT = """Generate detailed sections with specific statistics covering:

1. AI Technology Stack Analysis:
   - Usage percentages
   - Performance metrics
   - Cost comparisons
   - Integration success rates
   - User satisfaction scores

2. Implementation Metrics:
   - Deployment times
   - Success rates
   - Common challenges
   - Resolution approaches
   - Best practices

3. ROI Analysis:
   - Cost savings
   - Productivity gains
   - Quality improvements
   - Time savings
   - Resource optimization

Include specific numbers and comparisons.
"""

        prompt = f"""
        {DETAILED_SECTIONS_PROMPT}
        
        {ai_summary}
        
        Create a brief section (max 200 words) covering:
        1. Current AI adoption trends (top 2)
        2. Future projections (top 2)
        3. Critical skill shifts (top 2)
        
        Format the response in markdown with bullet points.
        """
        
        messages = [
            SystemMessage(content="You are an expert AI impact analyst."),
            HumanMessage(content=prompt)
        ]
        
        return self.get_completion(messages)
        
    def _generate_detailed_sections(
        self,
        tech_summary: str,
        market_summary: str,
        ai_summary: str
    ) -> str:
        """Generate detailed sections from summaries."""
        tech_section = self._analyze_tech_landscape(tech_summary)
        market_section = self._analyze_market_dynamics(market_summary)
        ai_section = self._analyze_ai_impact(ai_summary)
        
        return f"""## Technical Skills Landscape
{tech_section}

## Market Dynamics
{market_section}

## AI Impact Assessment
{ai_section}"""

    def _generate_recommendations(
        self,
        tech_summary: str,
        market_summary: str,
        ai_summary: str
    ) -> str:
        """Generate strategic recommendations."""
        prompt = f"""Based on these summaries:

Technical: {tech_summary}
Market: {market_summary}
AI Impact: {ai_summary}

Generate strategic recommendations for:
1. Job seekers
2. Employers
3. Educational institutions

Include specific, actionable items with timeframes where relevant.
Format in markdown with clear sections."""
        
        return self.get_completion(prompt)

    def _generate_executive_summary(
        self,
        tech_summary: str,
        market_summary: str,
        ai_summary: str
    ) -> str:
        """Generate executive summary."""
        prompt = f"""Based on these detailed analyses:

Technical: {tech_summary}
Market: {market_summary}
AI Impact: {ai_summary}

Generate a concise executive summary that:
1. Highlights key findings
2. Emphasizes critical trends
3. Notes important recommendations

Keep it brief but comprehensive. Format in markdown."""
        
        return self.get_completion(prompt)

    def _analyze_tech_landscape(self, summary: str) -> str:
        """Analyze technical landscape from summary."""
        prompt = f"""Based on this summary of technical requirements and skills:

{summary}

Generate a detailed analysis of the technical skills landscape, focusing on:
1. Most in-demand skills
2. Emerging technologies
3. Key skill gaps

Format in markdown with clear sections and bullet points."""
        
        return self.get_completion(prompt)

    def _analyze_market_dynamics(self, summary: str) -> str:
        """Analyze market dynamics from summary."""
        prompt = f"""Based on this market analysis summary:

{summary}

Generate a detailed analysis of market dynamics, focusing on:
- Key salary trends
- Geographic insights
- Industry highlights

Format in markdown with clear sections and bullet points."""
        
        return self.get_completion(prompt)

    def _analyze_ai_impact(self, summary: str) -> str:
        """Analyze AI impact from summary."""
        prompt = f"""Based on this AI impact analysis summary:

{summary}

Generate a detailed analysis of AI's impact, focusing on:
- Current AI adoption trends
- Future projections
- Critical skill shifts

Format in markdown with clear sections and bullet points."""
        
        return self.get_completion(prompt)

    SKILLS_ANALYSIS_PROMPT = """Analyze and list the top hands-on technical skills for each AI job category, and provide career transition advice:

1. AI Software Development
2. Front End Development
3. Back End Development
4. Full Stack Development
5. AI Product Management
6. AI Code Assistant skills
7. DevOps
8. Cloud Systems
9. Computer Vision
10. NLP
11. UI/UX Design
12. Data Science
13. Data Engineering
14. AI Security
15. LLM Models
16. Vector database
17. Generative AI Image Generation
18. Chatbots
19. Agent frameworks
20. Open Source models
21. Kids/Teenagers/College Students
22. Technical Support Professionals

For each category:
1. Technical Skills:
   - List the most in-demand technical skills
   - Focus on specific tools, languages, and frameworks
   - Include version control and collaboration tools
   - Add relevant certifications if applicable

2. Career Transition Advice:
   - Explain how to leverage AI tools in their current role
   - Suggest integration paths with AI technologies
   - Recommend learning paths and resources
   - Highlight opportunities for AI adoption

Format as markdown with category headers, skills bullets, and transition advice section.

Example format:
## [Category Name]

### Key Technical Skills
- [Skills list]

### Career Transition to AI
[Specific advice on how professionals in this field can adopt AI tools and transition to AI-enhanced roles]

Note: For Data Engineers specifically, emphasize:
- Adopting AI-based code development tools (Copilot, Codeium)
- Using Agent frameworks to revamp data engineering pipelines
- Understanding end-to-end AI app development and cloud deployment
- Transitioning to Agentic RAG frameworks
"""

    def _analyze_skills(self, data: Dict) -> str:
        """Analyze skills and provide career transition advice for different AI job categories."""
        prompt = self.SKILLS_ANALYSIS_PROMPT
        return self.get_completion(prompt)

    def generate_comprehensive_report(
        self,
        tech_analysis: Dict,
        market_analysis: Dict,
        ai_impact_analysis: Dict,
        timestamp: Optional[str] = None
    ) -> str:
        """Generate comprehensive report with all analyses."""
        try:
            logger.info("Generating comprehensive report...")
            
            # Get existing analyses
            tech_summary = self._extract_key_points(tech_analysis)
            market_summary = self._extract_key_points(market_analysis)
            ai_summary = self._extract_key_points(ai_impact_analysis)
            
            # Generate detailed sections
            detailed_sections = self._generate_detailed_sections(
                tech_summary,
                market_summary,
                ai_summary
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                tech_summary,
                market_summary,
                ai_summary
            )
            
            # Generate executive summary
            exec_summary = self._generate_executive_summary(
                tech_summary,
                market_summary,
                ai_summary
            )
            
            # Generate skills analysis and career transition advice
            skills_analysis = self._analyze_skills(tech_analysis)
            
            # Use provided timestamp or generate new one
            report_timestamp = timestamp or datetime.now().isoformat()
            
            # Combine all sections
            report = f"""# Job Market Analysis Report
            
**Executive Summary**

{exec_summary}

{detailed_sections}

{recommendations}

## Technical Skills and Career Transition Guide

{skills_analysis}

Report generated on: {report_timestamp}"""
            
            logger.info("Generated comprehensive report")
            return report
            
        except Exception as e:
            error_msg = f"Error generating report: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
