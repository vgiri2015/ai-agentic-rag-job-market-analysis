"""RAG-based document store for job market analysis."""
import logging
from typing import List, Dict, Optional
from pathlib import Path
import faiss

from llama_index.legacy import VectorStoreIndex, Document, ServiceContext
from llama_index.legacy.llms.openai import OpenAI
from llama_index.legacy.embeddings.openai import OpenAIEmbedding
from llama_index.legacy.vector_stores.faiss import FaissVectorStore
from llama_index.legacy.storage.storage_context import StorageContext
from llama_index.legacy.schema import TextNode, NodeWithScore

logger = logging.getLogger(__name__)

class JobMarketRAGStore:
    """RAG-based document store for job market analysis."""
    
    def __init__(self, openai_key: str):
        """Initialize the RAG store with LlamaIndex components."""
        self.openai_key = openai_key
        self.vector_store_dir = Path("data/vector_store")
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize LlamaIndex components
        self.llm = OpenAI(api_key=openai_key, model="gpt-4", temperature=0)
        self.embed_model = OpenAIEmbedding(api_key=openai_key)
        
        # Create service context
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embed_model,
        )
        
        # Initialize or create FAISS index
        self.faiss_index_file = self.vector_store_dir / "faiss.index"
        if self.faiss_index_file.exists():
            # Load existing index
            self.faiss_index = faiss.read_index(str(self.faiss_index_file))
        else:
            # Create new index
            self.faiss_index = faiss.IndexFlatL2(1536)  # OpenAI embedding dimension
            
        # Initialize vector store with FAISS index
        self.vector_store = FaissVectorStore(faiss_index=self.faiss_index)
        
        # Create storage context
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        # Initialize or load index
        self._initialize_index()
        
    def _initialize_index(self):
        """Initialize or load existing vector index."""
        try:
            if self.faiss_index_file.exists():
                logger.info("Loading existing vector store...")
                self.index = VectorStoreIndex.from_vector_store(
                    self.vector_store,
                    service_context=self.service_context
                )
            else:
                logger.info("Creating new vector store...")
                self.index = VectorStoreIndex(
                    [],
                    service_context=self.service_context,
                    storage_context=self.storage_context
                )
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
            
    def add_jobs(self, jobs: List[Dict]):
        """Add job listings to the vector store."""
        documents = []
        
        for job in jobs:
            # Create rich text by combining all job information
            text = f"""
            Title: {job.get('title', '')}
            Company: {job.get('company', '')}
            Location: {job.get('location', '')}
            Salary: {job.get('salary', 'Not specified')}
            Description: {job.get('description', '')}
            Requirements: {job.get('requirements', '')}
            """
            
            # Create document with metadata
            doc = Document(
                text=text,
                metadata={
                    "title": job.get('title', ''),
                    "company": job.get('company', ''),
                    "location": job.get('location', ''),
                    "salary": job.get('salary', ''),
                    "id": job.get('id', '')
                }
            )
            documents.append(doc)
        
        # Insert documents into index
        self.index.insert_documents(documents)
        
        # Save FAISS index
        faiss.write_index(self.faiss_index, str(self.faiss_index_file))
        logger.info(f"Added {len(documents)} jobs to vector store")
        
    def query_similar_jobs(self, query: str, top_k: int = 5) -> List[Dict]:
        """Query similar jobs based on semantic search."""
        query_engine = self.index.as_query_engine(
            similarity_top_k=top_k,
        )
        
        response = query_engine.query(query)
        
        similar_jobs = []
        for node in response.source_nodes:
            similar_jobs.append({
                "score": node.score,
                "metadata": node.metadata,
                "text": node.text
            })
            
        return similar_jobs
        
    def analyze_trends(self, query: str) -> str:
        """Analyze trends using RAG-enhanced prompting."""
        # Create a structured query engine
        query_engine = self.index.as_query_engine(
            similarity_top_k=10,
            response_mode="tree_summarize"
        )
        
        # Add analysis context to the query
        enhanced_query = f"""
        Based on the job market data, analyze the following aspect:
        {query}
        
        Provide insights about:
        1. Common patterns and trends
        2. Statistical observations
        3. Notable outliers or unique cases
        4. Market implications
        """
        
        response = query_engine.query(enhanced_query)
        return str(response)
