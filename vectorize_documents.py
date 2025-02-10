import os
from pathlib import Path
import pandas as pd
import logging
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import (
    PyMuPDFLoader, DirectoryLoader, TextLoader, 
    UnstructuredExcelLoader, CSVLoader
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import chromadb
import numpy as np
from tqdm import tqdm

# Setup logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedDocumentVectorizer:
    def __init__(
        self, 
        base_dir: str = None,
        embedding_model_name: str = "bge-large-en-v1.5",  # Better embedding model
        chunk_size: int = 512,  # Smaller chunks for better context
        chunk_overlap: int = 50
    ):
        self.base_dir = Path(base_dir) if base_dir else Path(os.path.dirname(os.path.abspath(__file__)))
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Define model directories
        self.model_dir = self.base_dir / "Models" / embedding_model_name
        self.clip_model_dir = self.base_dir / "Models/clip-vit-large-patch14"  # Upgraded CLIP model
        self.persist_dir = self.base_dir / "vector_database"
        
        # Environment setup
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["ANONYMIZED_TELEMETRY"] = "False"
        os.environ["CHROMADB_TELEMETRY"] = "False"
        
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize embedding and CLIP models with error handling"""
        try:
            logger.info("Initializing models...")
            
            # Initialize sentence transformer with specific parameters
            self.sentence_model = SentenceTransformer(
                str(self.model_dir),
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # Configure embedding model
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=str(self.model_dir),
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
                encode_kwargs={'normalize_embeddings': True}  # Important for cosine similarity
            )
            
            # Initialize CLIP with improved model
            self.clip_processor = CLIPProcessor.from_pretrained(str(self.clip_model_dir))
            self.clip_model = CLIPModel.from_pretrained(
                str(self.clip_model_dir),
                device_map='auto',
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    def _create_text_splitter(self, document_type: str = "default"):
        """Create appropriate text splitter based on document type"""
        if document_type == "markdown":
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            return MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text content"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        # Basic cleaning
        text = text.replace('\n', ' ').replace('\t', ' ')
        return text

    def load_documents(self, directory: str = "data") -> List[Dict]:
        """Enhanced document loading with better metadata"""
        try:
            loader = DirectoryLoader(
                path=directory,
                glob="*.pdf",
                loader_cls=PyMuPDFLoader,
                show_progress=True
            )
            docs = loader.load()
            
            # Enhance metadata
            for doc in docs:
                doc.metadata.update({
                    "file_type": "pdf",
                    "char_count": len(doc.page_content),
                    "processed_date": pd.Timestamp.now().isoformat()
                })
            
            logger.info(f"Loaded {len(docs)} PDF documents from {directory}")
            return docs
            
        except Exception as e:
            logger.error(f"Error loading PDF documents: {str(e)}")
            return []

    def load_excel(self, directory: str = "data") -> List[Dict]:
        """Load Excel files with enhanced processing"""
        excel_docs = []
        try:
            for file in os.listdir(directory):
                if file.endswith(('.xlsx', '.xls')):
                    loader = UnstructuredExcelLoader(
                        os.path.join(directory, file),
                        mode="elements"
                    )
                    docs = loader.load()
                    excel_docs.extend(docs)
                    
            logger.info(f"Loaded {len(excel_docs)} Excel documents")
            return excel_docs
            
        except Exception as e:
            logger.error(f"Error loading Excel files: {str(e)}")
            return []

    def process_documents(self, data_dir: str = "data"):
        """Process documents with enhanced features"""
        try:
            # Load all document types
            all_documents = []
            
            # Process each document type with progress bar
            for loader_func in [self.load_documents, self.load_excel]:
                docs = loader_func(data_dir)
                all_documents.extend(docs)
            
            logger.info(f"Total documents loaded: {len(all_documents)}")
            
            # Enhanced text splitting
            text_splitter = self._create_text_splitter()
            chunks = []
            
            for doc in tqdm(all_documents, desc="Processing documents"):
                # Preprocess text
                doc.page_content = self._preprocess_text(doc.page_content)
                
                # Split document
                doc_chunks = text_splitter.split_documents([doc])
                chunks.extend(doc_chunks)
            
            # Create vector store with optimized settings
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            
            chroma_settings = chromadb.Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
            
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                persist_directory=str(self.persist_dir),
                collection_name="enhanced_collection",
                collection_metadata={
                    "dimensionality": 1024,  # Updated for bge-large
                    "model_name": "bge-large-en-v1.5",
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap
                },
                client_settings=chroma_settings
            )
            
            logger.info(f"Created vector store with {len(chunks)} chunks")
            return vector_store, self.embedding_model
            
        except Exception as e:
            logger.error(f"Error in document processing: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        vectorizer = EnhancedDocumentVectorizer()
        vector_store, embedding_model = vectorizer.process_documents()
        logger.info("Document processing completed successfully")
    except Exception as e:
        logger.error(f"Failed to process documents: {str(e)}")