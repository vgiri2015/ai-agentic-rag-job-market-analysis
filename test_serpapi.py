from serpapi import GoogleSearch
import os
from dotenv import load_dotenv
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
serpapi_key = os.getenv("SERPAPI_API_KEY")

if not serpapi_key:
    raise ValueError("SERPAPI_API_KEY not found in environment variables")

print(f"Using API key: {serpapi_key}")

# Test search with minimal parameters
params = {
    "api_key": serpapi_key,
    "engine": "google_jobs",
    "q": "AI engineer",  # Simple, focused query
    "location": "United States",  # Broader location
    "google_domain": "google.com",
    "hl": "en"
}

print("\nUsing params:", json.dumps(params, indent=2))

try:
    search = GoogleSearch(params)
    results = search.get_dict()
    
    print("\nFull response:", json.dumps(results, indent=2))
    
    if "error" in results:
        print("\nError:", results["error"])
    elif "jobs_results" in results:
        print(f"\nFound {len(results['jobs_results'])} jobs")
        if results['jobs_results']:
            first_job = results['jobs_results'][0]
            print("\nFirst job details:")
            print(f"Title: {first_job.get('title')}")
            print(f"Company: {first_job.get('company_name')}")
            print(f"Location: {first_job.get('location')}")
    
except Exception as e:
    print(f"\nError occurred: {str(e)}")
