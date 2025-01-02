"""Generate final report from AI impact analysis."""
import json
import logging
import os
from dotenv import load_dotenv
from agents.job_agents.final_reporter import FinalReporterAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Generate final report from existing AI impact analysis."""
    try:
        # Load environment variables
        load_dotenv()
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if not openai_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable")
        
        # Load AI impact data
        logger.info("Loading AI impact analysis data")
        with open("data/ai_impact_analysis.json", "r") as f:
            ai_impact = json.load(f)
        
        # Generate report
        logger.info("Generating final report")
        final_reporter = FinalReporterAgent(openai_key)
        final_report = final_reporter.generate_report(ai_impact)
        
        logger.info("Report generated successfully!")
        return final_report
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise

if __name__ == "__main__":
    main()
