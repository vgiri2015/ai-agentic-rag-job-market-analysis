"""Generate comprehensive job market report."""
import json
import os
from dotenv import load_dotenv
from agents.job_agents.final_reporter import FinalReporterAgent

def main():
    """Generate the report."""
    try:
        # Load environment variables
        load_dotenv()
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Create reports directory if it doesn't exist
        os.makedirs("reports", exist_ok=True)
        
        # Load data
        with open('data/job_data.json') as f:
            job_data = json.load(f)
        with open('data/tech_analysis.json') as f:
            tech_analysis = json.load(f)
        with open('data/market_report.json') as f:
            market_report = json.load(f)
        with open('data/ai_impact_analysis.json') as f:
            ai_impact = json.load(f)
        
        print("Data loaded successfully!")
        print(f"Found {len(job_data)} jobs to analyze")
        
        # Create agent and generate report
        agent = FinalReporterAgent(openai_key=openai_key)
        report = agent.generate_comprehensive_report(
            job_data,
            tech_analysis,
            market_report,
            ai_impact
        )
        
        print("\nReport generated successfully!")
        print(f"Report saved to: reports/final_report.md")
        print("\nKey Statistics:")
        stats = report.get("statistics", {})
        print(f"- Total Jobs: {stats.get('total_jobs', 0)}")
        print(f"- AI Roles: {stats.get('ai_specific_roles', 0)}")
        print(f"- Remote Work: {stats.get('remote_percentage', 0)}%")

    except Exception as e:
        print(f"Error generating report: {str(e)}")

if __name__ == "__main__":
    main()
