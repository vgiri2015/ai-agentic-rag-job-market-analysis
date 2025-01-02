"""Main entry point for job market analysis workflow."""
import os
import sys
import json
import logging
import asyncio
import argparse
from typing import Dict, List, Optional, TypedDict
from pathlib import Path
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
        self.data_dir = Path("data")
        logger.info("Workflow initialized")
    
    async def collect_data(self) -> List[Dict]:
        """Collect job data."""
        logger.info("Collecting job data")
        try:
            if not self.force_new_collection and (self.data_dir / "job_data.json").exists():
                logger.info("Loading existing job data")
                with open(self.data_dir / "job_data.json", 'r') as f:
                    return json.load(f)
            
            job_data = await self.collector.collect_jobs()
            self.data_dir.mkdir(exist_ok=True)
            with open(self.data_dir / "job_data.json", 'w') as f:
                json.dump(job_data, f, indent=2)
            return job_data
        except Exception as e:
            logger.error(f"Error collecting data: {str(e)}")
            raise
    
    async def analyze_tech(self, job_data: List[Dict]) -> Dict:
        """Analyze tech requirements."""
        logger.info("Analyzing tech requirements")
        try:
            if not self.force_new_collection and (self.data_dir / "tech_analysis.json").exists():
                logger.info("Loading existing tech analysis")
                with open(self.data_dir / "tech_analysis.json", 'r') as f:
                    return json.load(f)
            
            tech_analysis = await self.analyzer.analyze_tech_requirements(job_data)
            with open(self.data_dir / "tech_analysis.json", 'w') as f:
                json.dump(tech_analysis, f, indent=2)
            return tech_analysis
        except Exception as e:
            logger.error(f"Error analyzing tech: {str(e)}")
            raise
    
    async def generate_market_report(self, job_data: List[Dict], tech_analysis: Dict) -> Dict:
        """Generate market report."""
        logger.info("Generating market report")
        try:
            if not self.force_new_collection and (self.data_dir / "market_report.json").exists():
                logger.info("Loading existing market report")
                with open(self.data_dir / "market_report.json", 'r') as f:
                    return json.load(f)
            
            market_report = await self.reporter.generate_report(job_data, tech_analysis)
            with open(self.data_dir / "market_report.json", 'w') as f:
                json.dump(market_report, f, indent=2)
            return market_report
        except Exception as e:
            logger.error(f"Error generating market report: {str(e)}")
            raise
    
    async def analyze_ai_impact(self, job_data: List[Dict], tech_analysis: Dict, market_report: Dict) -> Dict:
        """Analyze AI impact."""
        logger.info("Analyzing AI impact")
        try:
            if not self.force_new_collection and (self.data_dir / "ai_impact_analysis.json").exists():
                logger.info("Loading existing AI impact analysis")
                with open(self.data_dir / "ai_impact_analysis.json", 'r') as f:
                    return json.load(f)
            
            ai_impact = await self.impact_analyzer.analyze_ai_impact(job_data, tech_analysis, market_report)
            with open(self.data_dir / "ai_impact_analysis.json", 'w') as f:
                json.dump(ai_impact, f, indent=2)
            return ai_impact
        except Exception as e:
            logger.error(f"Error analyzing AI impact: {str(e)}")
            raise
    
    async def generate_final_report(
        self,
        tech_analysis: Dict,
        market_analysis: Dict,
        ai_impact_analysis: Dict,
        timestamp: str
    ) -> Dict:
        """Generate final report from all analyses."""
        try:
            logger.info("Generating final report")
            final_report = self.final_reporter.generate_comprehensive_report(
                tech_analysis,
                market_analysis,
                ai_impact_analysis,
                timestamp
            )
            
            # Save report to file
            report_path = self.data_dir / "final_report.md"
            with open(report_path, "w") as f:
                f.write(final_report)
            
            # Return report data
            return {
                "report": final_report,
                "statistics": {
                    "timestamp": timestamp,
                    "status": "completed"
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating final report: {str(e)}")
            raise
            
    async def run(self) -> Dict:
        """Run the complete workflow."""
        try:
            logger.info("Starting workflow")
            
            # Load job data
            with open(self.data_dir / "job_data.json", "r") as f:
                job_data = json.load(f)
            
            # Run analysis pipeline
            tech_analysis = await self.analyze_tech(job_data)
            market_report = await self.generate_market_report(job_data, tech_analysis)
            ai_impact = await self.analyze_ai_impact(job_data, tech_analysis, market_report)
            
            # Generate final report
            final_report = await self.generate_final_report(
                tech_analysis,
                market_report,
                ai_impact,
                datetime.now().isoformat()
            )
            
            logger.info("Workflow completed successfully!")
            return final_report
            
        except Exception as e:
            logger.error(f"Error in workflow: {str(e)}")
            raise
            
async def main() -> Dict:
    """Main entry point."""
    try:
        # Initialize workflow
        workflow = JobMarketWorkflow()
        logger.info("Workflow initialized")
        
        # Check if we should only generate report
        if len(sys.argv) > 1 and sys.argv[1] == "--report-only":
            logger.info("Generating report from existing data")
            
            # Load existing analysis results
            with open(workflow.data_dir / "tech_analysis.json", "r") as f:
                tech_analysis = json.load(f)
            with open(workflow.data_dir / "market_report.json", "r") as f:
                market_report = json.load(f)
            with open(workflow.data_dir / "ai_impact_analysis.json", "r") as f:
                ai_impact = json.load(f)
            
            # Generate final report only
            final_report = await workflow.generate_final_report(
                tech_analysis,
                market_report,
                ai_impact,
                datetime.now().isoformat()
            )
        else:
            # Run complete workflow
            final_report = await workflow.run()
        
        # Print summary
        stats = final_report.get("statistics", {})
        print("\nKey Statistics:")
        print(f"- Timestamp: {stats.get('timestamp', '')}")
        print(f"- Status: {stats.get('status', '')}")
        
        return final_report
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
