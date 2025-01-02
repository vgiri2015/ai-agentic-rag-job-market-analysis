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
        TECH_SUMMARY_PROMPT = """Analyze the technical requirements and skills from the data and create a comprehensive summary with specific statistics and rankings:

1. Job Market Overview:
   - Total number of AI job postings analyzed
   - Year-over-year growth in AI jobs
   - Distribution by job type
   - Remote vs on-site ratio
   - Average salary ranges by role

2. Top 10 AI Jobs by Demand:
   - Machine Learning Engineer
   - AI Research Scientist
   - Data Scientist
   - AI Product Manager
   - MLOps Engineer
   - AI Solutions Architect
   - NLP Engineer
   - Computer Vision Engineer
   - AI Ethics Officer
   - AI Infrastructure Engineer
   Include salary ranges and YoY growth for each.

3. Top AI Skills by Demand (with usage %):
   - Programming Languages (Python, Julia, etc.)
   - ML Frameworks (PyTorch, TensorFlow, etc.)
   - Cloud Platforms (AWS, Azure, GCP)
   - MLOps Tools
   - Database Technologies

4. LLM Models and Frameworks:
   - Most used LLM models
   - Popular fine-tuning approaches
   - Deployment platforms
   - Cost considerations
   - Performance metrics

5. RAG Technology Stack:
   - Vector Databases (usage %)
   - Embedding Models
   - Document Processing Tools
   - Search Optimization Tools
   - Integration Frameworks

6. Top AI Development Tools:
   - LangChain usage statistics
   - LlamaIndex adoption rates
   - Vector store platforms
   - Model serving tools
   - Development frameworks

7. Agentic AI Components:
   - Agent frameworks
   - Planning systems
   - Tool integration platforms
   - Orchestration solutions
   - State management tools

Include specific numbers, percentages, and growth rates wherever possible.
"""

        chunks = self._chunk_data(data)
        summaries = []
        
        for chunk in chunks:
            prompt = f"""
            {TECH_SUMMARY_PROMPT}
            
            {chunk}
            
            Focus on:
            1. Main findings (max 2)
            2. Statistical highlights (max 2)
            3. Critical trends (max 2)
            
            Format as a very brief bullet-point list. Total response should be under 150 words.
            """
            
            messages = [
                SystemMessage(content="You are a data analyst expert at extracting key insights."),
                HumanMessage(content=prompt)
            ]
            
            try:
                response = self.summarizer.invoke(messages)
                summaries.append(response.content)
            except Exception as e:
                logger.error(f"Error processing chunk: {str(e)}")
                continue
        
        # Combine summaries
        combined_summary = "\n\n".join(summaries)
        
        # Create final summary of summaries if needed
        if len(summaries) > 1:
            prompt = f"""
            Combine these summaries into a single coherent list of key points:
            
            {combined_summary}
            
            Focus on eliminating redundancy and keeping only the most important points.
            Format as a very brief bullet-point list. Total response should be under 200 words.
            """
            
            messages = [
                SystemMessage(content="You are a data analyst expert at synthesizing insights."),
                HumanMessage(content=prompt)
            ]
            
            try:
                response = self.summarizer.invoke(messages)
                return response.content
            except Exception as e:
                logger.error(f"Error combining summaries: {str(e)}")
                return combined_summary
                
        return combined_summary
        
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
        
    def _generate_recommendations(
        self,
        tech_section: str,
        market_section: str,
        ai_section: str
    ) -> str:
        """Generate strategic recommendations."""
        RECOMMENDATIONS_PROMPT = """Generate strategic recommendations with specific metrics:

1. Technology Selection:
   - Cost-benefit analysis
   - Implementation timeline
   - Resource requirements
   - Risk assessment
   - Success metrics

2. Skill Development:
   - Learning paths
   - Time investment
   - Certification costs
   - Career impact
   - Salary potential

3. Implementation Strategy:
   - Phasing approach
   - Resource allocation
   - Timeline planning
   - Risk mitigation
   - Success criteria

Include specific timelines and success metrics.
"""

        prompt = f"""
        {RECOMMENDATIONS_PROMPT}
        
        Technical Analysis:
        {tech_section}
        
        Market Analysis:
        {market_section}
        
        AI Impact:
        {ai_section}
        
        Create brief, actionable recommendations (max 2 each) for:
        1. Job seekers
        2. Employers
        3. Educational institutions
        
        Format the response in markdown with bullet points.
        Total response should be under 250 words.
        """
        
        messages = [
            SystemMessage(content="You are an expert career strategist."),
            HumanMessage(content=prompt)
        ]
        
        return self.get_completion(messages)
        
    def _generate_executive_summary(
        self,
        tech_section: str,
        market_section: str,
        ai_section: str,
        recommendations: str
    ) -> str:
        """Generate executive summary."""
        EXECUTIVE_SUMMARY_PROMPT = """Create a data-driven executive summary highlighting:

1. Market Overview:
   - Total market size
   - Growth rates
   - Key players
   - Major trends
   - Future projections

2. Technology Impact:
   - Adoption rates
   - Success metrics
   - Cost implications
   - ROI analysis
   - Risk factors

3. Strategic Direction:
   - Priority actions
   - Timeline
   - Resource needs
   - Success criteria
   - Risk mitigation

Include key statistics and metrics throughout.
"""

        prompt = f"""
        {EXECUTIVE_SUMMARY_PROMPT}
        
        Technical Skills:
        {tech_section}
        
        Market Dynamics:
        {market_section}
        
        AI Impact:
        {ai_section}
        
        Recommendations:
        {recommendations}
        
        The summary must be under 200 words and highlight only the most 
        critical findings and implications. Focus on actionable insights.
        
        Format the response in markdown.
        """
        
        messages = [
            SystemMessage(content="You are an expert business analyst."),
            HumanMessage(content=prompt)
        ]
        
        return self.get_completion(messages)
        
    def generate_comprehensive_report(
        self,
        tech_analysis: Dict,
        market_analysis: Dict,
        ai_impact_analysis: Dict,
        timestamp: str
    ) -> str:
        """Generate a comprehensive report combining all analyses."""
        logger.info("Generating comprehensive report...")
        
        try:
            # First, extract key points using GPT-3.5-turbo with chunking
            tech_summary = self._extract_key_points(tech_analysis)
            logger.info("Generated tech summary")
            
            market_summary = self._extract_key_points(market_analysis)
            logger.info("Generated market summary")
            
            ai_summary = self._extract_key_points(ai_impact_analysis)
            logger.info("Generated AI impact summary")
            
            # Generate detailed sections from summaries using GPT-4
            tech_section = self._analyze_tech_landscape(tech_summary)
            market_section = self._analyze_market_dynamics(market_summary)
            ai_section = self._analyze_ai_impact(ai_summary)
            logger.info("Generated detailed sections")
            
            # Generate recommendations based on sections
            recommendations = self._generate_recommendations(
                tech_section,
                market_section,
                ai_section
            )
            logger.info("Generated recommendations")
            
            # Generate executive summary
            executive_summary = self._generate_executive_summary(
                tech_section,
                market_section,
                ai_section,
                recommendations
            )
            logger.info("Generated executive summary")
            
            # Combine all sections into final report
            report = f"""# Job Market Analysis Report
            
{executive_summary}

## Technical Skills Landscape
{tech_section}

## Market Dynamics
{market_section}

## AI Impact Assessment
{ai_section}

## Strategic Recommendations
{recommendations}

Report generated on: {timestamp}
"""
            
            # Save report
            report_path = self.data_dir / "final_report.md"
            with open(report_path, "w") as f:
                f.write(report)
                
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise
