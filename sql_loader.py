import os
import logging
from typing import List, Dict, Any, Optional
import sqlalchemy
from sqlalchemy import create_engine, text
from langchain.docstore.document import Document
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class SQLDatabaseLoader:
    def __init__(self, db_url: str):
        """Initialize SQL loader with database connection"""
        self.db_url = db_url
        self.engine = create_engine(db_url)
        
    def execute_query(self, query: str) -> List[Dict]:
        """Execute SQL query and return results"""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(query))
                return [dict(row._mapping) for row in result]
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return []

    def load_table_as_documents(self, table: str) -> List[Document]:
        """Load table data as documents"""
        try:
            # Get table data
            query = f"SELECT * FROM {table}"
            rows = self.execute_query(query)
            
            documents = []
            for row in rows:
                # Convert row to string format
                content = "\n".join(f"{k}: {v}" for k, v in row.items())
                
                # Create metadata
                metadata = {
                    "source": table,
                    "type": "sql",
                    "id": str(row.get('id', '')),
                    "timestamp": datetime.now().isoformat()
                }
                
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} records from {table}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading table {table}: {str(e)}")
            return []

    def load_custom_query(self, query: str, query_name: str) -> List[Document]:
        """Load data using custom SQL query"""
        try:
            rows = self.execute_query(query)
            
            documents = []
            for row in rows:
                content = "\n".join(f"{k}: {v}" for k, v in row.items())
                metadata = {
                    "source": "custom_query",
                    "query_name": query_name,
                    "type": "sql",
                    "timestamp": datetime.now().isoformat()
                }
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} records from query {query_name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error executing query {query_name}: {str(e)}")
            return []

    def direct_query(self, question: str) -> Optional[List[Dict]]:
        """Handle direct SQL queries based on question patterns"""
        patterns = {
            "how many": "SELECT COUNT(*) as count FROM {}",
            "latest": "SELECT * FROM {} ORDER BY created_at DESC LIMIT 1",
            "average": "SELECT AVG({}) as average FROM {}",
            "total": "SELECT SUM({}) as total FROM {}"
        }
        
        for pattern, template in patterns.items():
            if pattern in question.lower():
                try:
                    # Extract table name from question
                    words = question.lower().split()
                    if "from" in words:
                        table_index = words.index("from") + 1
                        table = words[table_index]
                    else:
                        table = words[-1]
                    
                    # Build and execute query
                    query = template.format(table)
                    return self.execute_query(query)
                    
                except Exception as e:
                    logger.error(f"Error in direct query: {str(e)}")
                    return None
        
        return None