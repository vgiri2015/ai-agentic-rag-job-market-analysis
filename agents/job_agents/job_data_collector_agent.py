"""Job data collection agent."""
import json
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from serpapi import GoogleSearch
from .base_agent import BaseJobAgent, logger

class JobDataCollectorAgent(BaseJobAgent):
    """Agent for collecting job data from SerpAPI."""
    
    def __init__(self, serpapi_key: str, openai_key: str):
        """Initialize the collector agent."""
        super().__init__(openai_key)
        if not serpapi_key:
            raise ValueError("serpapi_key cannot be empty")
        self.serpapi_key = serpapi_key
        
    def load_existing_data(self) -> Optional[List[Dict]]:
        """Load existing job data if available."""
        file_path = self.data_dir / "job_data.json"
        if file_path.exists():
            logger.info(f"Loading existing job data from {file_path}")
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                
                # Ensure data is a list
                if not isinstance(data, list):
                    data = [data]
                
                # Ensure each job has required fields
                valid_jobs = []
                for job in data:
                    if isinstance(job, dict):
                        # Add default values for required fields if missing
                        job_entry = {
                            "title": job.get("title", "Unknown Title"),
                            "company_name": job.get("company_name", "Unknown Company"),
                            "location": job.get("location", "Unknown Location"),
                            "description": job.get("description", "No description available"),
                            "type": job.get("type", "Not specified"),
                            "metadata": {
                                "source": job.get("source", "existing_data"),
                                "last_updated": datetime.now().isoformat()
                            }
                        }
                        valid_jobs.append(job_entry)
                    elif isinstance(job, str):
                        # If job is a string, treat it as a description
                        job_entry = {
                            "title": "Unknown Title",
                            "company_name": "Unknown Company",
                            "location": "Unknown Location",
                            "description": job,
                            "type": "Not specified",
                            "metadata": {
                                "source": "existing_data",
                                "last_updated": datetime.now().isoformat()
                            }
                        }
                        valid_jobs.append(job_entry)
                
                logger.info(f"Loaded and validated {len(valid_jobs)} jobs from existing data")
                
                # Save the validated data back
                self.save_json(valid_jobs, "job_data.json")
                
                return valid_jobs
                
            except Exception as e:
                logger.error(f"Error loading existing data: {str(e)}")
                return None
                
        logger.info("No existing job data found")
        return None
        
    def search_jobs(self, query: str, location: str = None) -> List[Dict]:
        """Search for jobs using SerpAPI."""
        logger.info(f"Searching for {query} in {location}")
        
        params = {
            "api_key": self.serpapi_key,
            "engine": "google_jobs",
            "q": query,
            "google_domain": "google.com",
            "hl": "en"
        }
        
        if location:
            params["location"] = location
            
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            
            logger.info(f"Response keys: {list(results.keys())}")
            
            if "jobs_results" not in results:
                logger.warning(f"No jobs_results found in response. Keys: {list(results.keys())}")
                return []
                
            jobs = results.get("jobs_results", [])
            
            logger.info(f"Found {len(jobs)} jobs under jobs_results")
            if jobs:
                logger.info(f"Sample job: {jobs[0].get('title')} at {jobs[0].get('company_name')}")
                
            return jobs
            
        except Exception as e:
            logger.error(f"Error searching jobs: {str(e)}")
            return []
            
    def collect_jobs(self, force_new: bool = False) -> List[Dict]:
        """Collect job data from SerpAPI."""
        logger.info("Collecting job data")
        
        job_data_file = "data/job_data.json"
        
        # Check if cached data exists and we're not forcing new collection
        if os.path.exists(job_data_file) and not force_new:
            logger.info("Using existing job data from file")
            try:
                with open(job_data_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading cached data: {str(e)}")
                logger.info("Falling back to API collection")
        
        logger.info("Collecting new job data from SerpAPI")
        
        # Initialize collections
        all_jobs = []
        unique_jobs = set()  # Track unique job IDs
        
        # Define roles and locations to search for
        roles = [
            "Software Engineer",
            "AI Engineer",
            "Machine Learning Engineer",
            "Data Scientist",
            "DevOps Engineer",
            "Cloud Engineer",
            "Full Stack Developer",
            "Backend Engineer",
            "Frontend Engineer",
            "Computer Vision Engineer",
            "UI/UX Designer",
            "AI Product Manager"
        ]
        
        locations = [
            "United States",
            "Canada",
            "United Kingdom",
            "Europe",
            "Asia",
            "Australia",
            "New Zealand",
            "Middle East",
            "South America"
        ]
        
        # Search for each role in each location
        for role in roles:
            for location in locations:
                logger.info(f"\n--- Searching for {role} in {location} ---")
                logger.info(f"Searching for {role} in {location}")
                
                # Get jobs for this role and location
                try:
                    params = {
                        "engine": "google_jobs",
                        "q": role,
                        "location": location,
                        "hl": "en",
                        "api_key": os.getenv("SERPAPI_API_KEY")
                    }
                    
                    search = GoogleSearch(params)
                    results = search.get_dict()
                    
                    # Log response structure
                    logger.info(f"Response keys: {list(results.keys())}")
                    
                    if "jobs_results" in results:
                        jobs = results["jobs_results"]
                        logger.info(f"Found {len(jobs)} jobs under jobs_results")
                        
                        if jobs:
                            # Log sample job
                            logger.info(f"Sample job: {jobs[0]['title']} at {jobs[0].get('company_name', 'Unknown Company')}")
                        
                        # Add location to each job
                        for job in jobs:
                            job["search_location"] = location
                            
                            # Generate a unique ID for the job
                            job_id = f"{job.get('company_name', '')}_{job.get('title', '')}_{job.get('location', '')}"
                            
                            # Only add if we haven't seen this job before
                            if job_id not in unique_jobs:
                                unique_jobs.add(job_id)
                                all_jobs.append(job)
                        
                        logger.info(f"Found {len(jobs)} jobs for {role} in {location}")
                        logger.info(f"Total jobs collected so far: {len(all_jobs)}")
                    else:
                        logger.warning(f"No jobs_results found in response. Keys: {list(results.keys())}")
                        logger.warning(f"No jobs found for {role} in {location}")
                
                except Exception as e:
                    logger.error(f"Error searching for {role} in {location}: {str(e)}")
                    continue
                
                # Add a small delay to avoid rate limiting
                time.sleep(1)
        
        logger.info(f"Final collection: {len(all_jobs)} unique jobs")
        
        # Save the collected data
        os.makedirs("data", exist_ok=True)
        self.save_json(all_jobs, "job_data.json")
        
        return all_jobs
