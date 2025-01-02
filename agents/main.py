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
from langgraph.graph import StateGraph, END

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
        serpapi_key = os.getenv("SERPAPI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if not serpapi_key or not openai_key:
            raise ValueError("Missing required API keys")
            
        # Initialize agents
        self.collector = JobDataCollectorAgent(serpapi_key, openai_key)
        self.analyzer = TechAnalyzerAgent(openai_key)
        self.reporter = MarketReporterAgent(openai_key)
        self.impact_analyzer = AIImpactAnalyzerAgent(openai_key)
        self.final_reporter = FinalReporterAgent(openai_key)
        
        # Setup workflow graph
        self.workflow = self.setup_workflow()
        self.force_new_collection = force_new_collection
        
        logger.info("Workflow graph setup completed")
        
    def setup_workflow(self) -> StateGraph:
        """Setup workflow graph."""
        workflow = StateGraph(JobMarketState)
        
        # Add nodes
        workflow.add_node("collect_data", self.collect_data)
        workflow.add_node("analyze_tech", self.analyze_tech)
        workflow.add_node("generate_market_report", self.generate_market_report)
        workflow.add_node("analyze_ai_impact", self.analyze_ai_impact)
        workflow.add_node("generate_final_report", self.generate_final_report)
        
        # Add edges
        workflow.add_edge("collect_data", "analyze_tech")
        workflow.add_edge("analyze_tech", "generate_market_report")
        workflow.add_edge("generate_market_report", "analyze_ai_impact")
        workflow.add_edge("analyze_ai_impact", "generate_final_report")
        workflow.add_edge("generate_final_report", END)
        
        # Set entry point
        workflow.set_entry_point("collect_data")
        
        return workflow.compile()
        
    async def collect_data(self, state: JobMarketState) -> JobMarketState:
        """Collect job data."""
        logger.info("Collecting job data")
        job_data_file = "data/job_data.json"
        
        if os.path.exists(job_data_file) and not self.force_new_collection:
            logger.info("Using existing job data from file")
            with open(job_data_file, 'r') as f:
                job_data = json.load(f)
        else:
            logger.info("Collecting new job data from API")
            job_data = self.collector.collect_jobs(self.force_new_collection)
        
        state["job_data"] = job_data
        return state
            
    async def analyze_tech(self, state: JobMarketState) -> JobMarketState:
        """Analyze tech requirements."""
        logger.info("Analyzing tech requirements")
        try:
            if not state["job_data"]:
                raise ValueError("No job data available")
                
            tech_analysis = self.analyzer.analyze_tech_requirements(state["job_data"])
            state["tech_analysis"] = tech_analysis
            return state
        except Exception as e:
            logger.error(f"Error analyzing tech requirements: {str(e)}")
            state["error"] = f"Tech analysis failed: {str(e)}"
            return state
            
    async def generate_market_report(self, state: JobMarketState) -> JobMarketState:
        """Generate market report."""
        logger.info("Generating market report")
        try:
            if not state["job_data"] or not state["tech_analysis"]:
                raise ValueError("Missing required data for market report")
                
            market_report = self.reporter.generate_report(
                state["job_data"],
                state["tech_analysis"]
            )
            state["market_report"] = market_report
            return state
        except Exception as e:
            logger.error(f"Error generating market report: {str(e)}")
            state["error"] = f"Market report generation failed: {str(e)}"
            return state
            
    async def analyze_ai_impact(self, state: JobMarketState) -> JobMarketState:
        """Analyze AI impact."""
        logger.info("Analyzing AI impact")
        try:
            if not all([state["job_data"], state["tech_analysis"], state["market_report"]]):
                raise ValueError("Missing required data for AI impact analysis")
                
            ai_impact = self.impact_analyzer.analyze_ai_impact(
                state["job_data"],
                state["tech_analysis"],
                state["market_report"]
            )
            state["ai_impact"] = ai_impact
            return state
        except Exception as e:
            logger.error(f"Error analyzing AI impact: {str(e)}")
            state["error"] = f"AI impact analysis failed: {str(e)}"
            return state
            
    async def generate_final_report(self, state: JobMarketState) -> JobMarketState:
        """Generate final report."""
        logger.info("Generating final report")
        try:
            if not all([
                state["job_data"],
                state["tech_analysis"],
                state["market_report"],
                state["ai_impact"]
            ]):
                raise ValueError("Missing required data for report generation")
                
            final_report = self.final_reporter.generate_comprehensive_report(
                state["job_data"],
                state["tech_analysis"],
                state["market_report"],
                state["ai_impact"]
            )
            state["final_report"] = final_report
            return state
        except Exception as e:
            logger.error(f"Error generating final report: {str(e)}")
            state["error"] = f"Final report generation failed: {str(e)}"
            return state
            
    async def run(self) -> Dict:
        """Run workflow."""
        logger.info("Starting job market analysis workflow")
        try:
            initial_state = {
                "job_data": None,
                "tech_analysis": None,
                "market_report": None,
                "ai_impact": None,
                "final_report": None,
                "error": None
            }
            return await self.workflow.ainvoke(initial_state)
        except Exception as e:
            logger.error(f"Workflow ended due to error: {str(e)}")
            raise

async def main():
    """Main workflow for job market analysis."""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Run job market analysis workflow")
        parser.add_argument("--force-new", action="store_true", help="Force new data collection")
        args = parser.parse_args()
        
        logger.info("Initializing JobMarketWorkflow")
        workflow = JobMarketWorkflow()
        
        # Get API keys from environment
        serpapi_key = os.getenv("SERPAPI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if not serpapi_key or not openai_key:
            raise ValueError("Missing required API keys. Please set SERPAPI_API_KEY and OPENAI_API_KEY environment variables.")
        
        # Initialize agents
        job_collector = JobDataCollectorAgent(serpapi_key=serpapi_key, openai_key=openai_key)
        ai_impact_analyzer = AIImpactAnalyzerAgent(openai_key=openai_key)
        final_reporter = FinalReporterAgent(openai_key=openai_key)
        
        logger.info("Loading existing data files")
        
        # Load job data
        with open("data/job_data.json", 'r') as f:
            job_data = json.load(f)
            
        # Load tech analysis
        with open("data/tech_analysis.json", 'r') as f:
            tech_analysis = json.load(f)
            
        # Load market report
        with open("data/market_report.json", 'r') as f:
            market_report = json.load(f)
        
        # Load existing AI impact analysis if available, otherwise generate new one
        try:
            logger.info("Loading existing AI impact analysis")
            with open("data/ai_impact_analysis.json", 'r') as f:
                ai_impact = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info("Existing AI impact analysis not found, generating new one")
            try:
                ai_impact = ai_impact_analyzer.analyze_ai_impact(job_data, tech_analysis, market_report)
            except Exception as e:
                logger.error(f"Error analyzing AI impact: {str(e)}")
                raise
            
        logger.info("Generating final report")
        try:
            final_report = final_reporter.generate_report(job_data, tech_analysis, market_report, ai_impact)
        except Exception as e:
            logger.error(f"Error generating final report: {str(e)}")
            raise
        
        # Save final state
        output_path = Path("data") / "workflow_state.json"
        with open(output_path, "w") as f:
            json.dump({
                "job_data": job_data,
                "tech_analysis": tech_analysis,
                "market_report": market_report,
                "ai_impact": ai_impact,
                "final_report": final_report
            }, f, indent=2)
            
        logger.info("Analysis complete!")
        return final_report
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
