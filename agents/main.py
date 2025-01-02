"""Main entry point for job market analysis workflow."""
import os
import json
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, TypedDict
from datetime import datetime

from dotenv import load_dotenv

from .job_agents.job_data_collector_agent import JobDataCollectorAgent
from .job_agents.tech_analyzer_agent import TechAnalyzerAgent
from .job_agents.market_reporter import MarketReporterAgent
from .job_agents.ai_impact_analyzer import AIImpactAnalyzerAgent
from .job_agents.final_reporter import FinalReporterAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class JobMarketState(TypedDict):
    """State management for job market analysis workflow."""
    job_data: Optional[List[Dict]]
    tech_analysis: Optional[Dict]
    market_report: Optional[Dict]
    ai_impact: Optional[Dict]
    final_report: Optional[Dict]
    error: Optional[str]

class JobMarketWorkflow:
    """Orchestrates the job market analysis workflow."""
    
    def __init__(self, force_new_collection: bool = False):
        """Initialize workflow."""
        logger.info("Initializing JobMarketWorkflow")
        
        # Load environment variables
        self.serpapi_key = os.getenv("SERPAPI_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        
        if not self.serpapi_key or not self.openai_key:
            raise ValueError("Missing required API keys")
            
        # Initialize agents
        self.collector = JobDataCollectorAgent(self.serpapi_key, self.openai_key)
        self.analyzer = TechAnalyzerAgent(self.openai_key)
        self.reporter = MarketReporterAgent(self.openai_key)
        self.impact_analyzer = AIImpactAnalyzerAgent(self.openai_key)
        self.final_reporter = FinalReporterAgent(self.openai_key)
        
        self.force_new_collection = force_new_collection
        logger.info("Workflow initialized")
    
    async def collect_data(self) -> List[Dict]:
        """Collect job data."""
        logger.info("Collecting job data")
        try:
            if not self.force_new_collection and os.path.exists("data/job_data.json"):
                logger.info("Loading existing job data")
                with open("data/job_data.json", 'r') as f:
                    return json.load(f)
            
            job_data = await self.collector.collect_jobs()
            os.makedirs("data", exist_ok=True)
            with open("data/job_data.json", 'w') as f:
                json.dump(job_data, f, indent=2)
            return job_data
        except Exception as e:
            logger.error(f"Error collecting data: {str(e)}")
            raise
    
    async def analyze_tech(self, job_data: List[Dict]) -> Dict:
        """Analyze tech requirements."""
        logger.info("Analyzing tech requirements")
        try:
            if not self.force_new_collection and os.path.exists("data/tech_analysis.json"):
                logger.info("Loading existing tech analysis")
                with open("data/tech_analysis.json", 'r') as f:
                    return json.load(f)
            
            tech_analysis = await self.analyzer.analyze_tech_requirements(job_data)
            with open("data/tech_analysis.json", 'w') as f:
                json.dump(tech_analysis, f, indent=2)
            return tech_analysis
        except Exception as e:
            logger.error(f"Error analyzing tech: {str(e)}")
            raise
    
    async def generate_market_report(self, job_data: List[Dict], tech_analysis: Dict) -> Dict:
        """Generate market report."""
        logger.info("Generating market report")
        try:
            if not self.force_new_collection and os.path.exists("data/market_report.json"):
                logger.info("Loading existing market report")
                with open("data/market_report.json", 'r') as f:
                    return json.load(f)
            
            market_report = await self.reporter.generate_report(job_data, tech_analysis)
            with open("data/market_report.json", 'w') as f:
                json.dump(market_report, f, indent=2)
            return market_report
        except Exception as e:
            logger.error(f"Error generating market report: {str(e)}")
            raise
    
    async def analyze_ai_impact(self, job_data: List[Dict], tech_analysis: Dict, market_report: Dict) -> Dict:
        """Analyze AI impact."""
        logger.info("Analyzing AI impact")
        try:
            if not self.force_new_collection and os.path.exists("data/ai_impact_analysis.json"):
                logger.info("Loading existing AI impact analysis")
                with open("data/ai_impact_analysis.json", 'r') as f:
                    return json.load(f)
            
            ai_impact = await self.impact_analyzer.analyze_ai_impact(job_data, tech_analysis, market_report)
            with open("data/ai_impact_analysis.json", 'w') as f:
                json.dump(ai_impact, f, indent=2)
            return ai_impact
        except Exception as e:
            logger.error(f"Error analyzing AI impact: {str(e)}")
            raise
    
    async def generate_final_report(self, job_data: List[Dict], tech_analysis: Dict, market_report: Dict, ai_impact: Dict) -> Dict:
        """Generate final report."""
        logger.info("Generating final report")
        try:
            os.makedirs("reports", exist_ok=True)
            final_report = self.final_reporter.generate_comprehensive_report(
                job_data, tech_analysis, market_report, ai_impact
            )
            
            # Save workflow state
            with open("data/workflow_state.json", "w") as f:
                json.dump({
                    "job_data": job_data,
                    "tech_analysis": tech_analysis,
                    "market_report": market_report,
                    "ai_impact": ai_impact,
                    "final_report": final_report,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
            
            return final_report
        except Exception as e:
            logger.error(f"Error generating final report: {str(e)}")
            raise
    
    async def run(self) -> Dict:
        """Run the complete workflow."""
        try:
            # Ensure data directory exists
            os.makedirs("data", exist_ok=True)
            
            # Run workflow steps
            job_data = await self.collect_data()
            tech_analysis = await self.analyze_tech(job_data)
            market_report = await self.generate_market_report(job_data, tech_analysis)
            ai_impact = await self.analyze_ai_impact(job_data, tech_analysis, market_report)
            final_report = await self.generate_final_report(job_data, tech_analysis, market_report, ai_impact)
            
            logger.info("Workflow completed successfully!")
            return final_report
        except Exception as e:
            logger.error(f"Error in workflow: {str(e)}")
            raise

async def main():
    """Main workflow for job market analysis."""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Run job market analysis workflow")
        parser.add_argument("--force-new", action="store_true", help="Force new data collection")
        parser.add_argument("--report-only", action="store_true", help="Only generate final report using existing data")
        args = parser.parse_args()
        
        workflow = JobMarketWorkflow(force_new_collection=args.force_new)
        
        if args.report_only:
            logger.info("Generating report from existing data")
            # Load existing data
            with open("data/job_data.json", 'r') as f:
                job_data = json.load(f)
            with open("data/tech_analysis.json", 'r') as f:
                tech_analysis = json.load(f)
            with open("data/market_report.json", 'r') as f:
                market_report = json.load(f)
            with open("data/ai_impact_analysis.json", 'r') as f:
                ai_impact = json.load(f)
            
            # Generate final report only
            final_report = await workflow.generate_final_report(
                job_data, tech_analysis, market_report, ai_impact
            )
        else:
            # Run complete workflow
            final_report = await workflow.run()
        
        # Print summary
        stats = final_report.get("statistics", {})
        print("\nKey Statistics:")
        print(f"- Total Jobs: {stats.get('total_jobs', 0)}")
        print(f"- AI Roles: {stats.get('ai_specific_roles', 0)}")
        print(f"- Remote Work: {stats.get('remote_percentage', 0)}%")
        
        return final_report
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
