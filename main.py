import os
import logging
from pathlib import Path
import streamlit as st
import re
import torch
import tempfile
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from vectorize_documents import EnhancedDocumentVectorizer
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment setup
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMADB_TELEMETRY"] = "False"

# Direct, simple prompt
CHAT_PROMPT = PromptTemplate.from_template("""Based on the provided context and uploaded document(s), answer the following question:

Context: {context}

Question: {question}

Answer:""")

def clean_response(text: str) -> str:
    """Extract only the relevant chatbot answer and remove unnecessary text."""
    if not text:
        return "I'm not sure about that."

    # Use regex to find only the actual answer
    match = re.search(r"Answer:\s*(.*)", text, re.DOTALL)

    if match:
        cleaned_text = match.group(1).strip()
    else:
        cleaned_text = text.strip()

    return cleaned_text

def process_uploaded_document(uploaded_file, vectorizer):
    """Process uploaded document and add to vector store"""
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            temp_dir_path.mkdir(parents=True, exist_ok=True)
            
            # Save uploaded file
            file_path = temp_dir_path / uploaded_file.name
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            logger.info(f"Processing document: {file_path}")
            
            # Process document
            vector_store, _ = vectorizer.process_documents(str(temp_dir_path))
            
            # Store in session state
            st.session_state.current_vectorstore = vector_store
            st.session_state.using_uploaded_doc = True
            st.session_state.chain_initialized = False  # Force reinitialization
            st.session_state.current_file = uploaded_file.name
            
            logger.info("Document processed successfully")
            return True, "Document processed successfully!"
            
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return False, str(e)

def create_sidebar():
    """Create sidebar with model selection and document upload"""
    with st.sidebar:
        st.markdown("### Model Configuration")
        
        # Model selection
        st.selectbox(
            "Current Model",
            ["Llama-3.2-3B-instruct"],
            disabled=True,
            help="Currently using Llama 3.2 3B Instruct model"
        )
        
        st.markdown("---")
        
        # Document upload section
        st.markdown("### Document Upload")
        
        # File uploader with custom styling
        st.markdown("""
            <style>
                .uploadedFile {
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    padding: 10px;
                    margin: 10px 0;
                }
            </style>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=['pdf', 'xlsx', 'xls', 'csv', 'txt'],
            help="Upload documents to process for the chatbot"
        )
        
        if uploaded_file:
            st.markdown('<div class="uploadedFile">', unsafe_allow_html=True)
            st.markdown("#### Document Details")
            st.text(f"Name: {uploaded_file.name}")
            st.text(f"Size: {uploaded_file.size/1024:.2f} KB")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add processing button
            if st.button("Process Document", type="primary", key="process_doc"):
                with st.spinner("Processing document..."):
                    try:
                        vectorizer = EnhancedDocumentVectorizer()
                        success, message = process_uploaded_document(uploaded_file, vectorizer)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
        
        st.markdown("---")
        
        # Database connection section
        st.markdown("### Database Connection")
        db_type = st.selectbox(
            "Select Database Type",
            ["PostgreSQL", "MySQL", "SQLite", "MongoDB"],
            disabled=True
        )
        st.button("Connect to Database", disabled=True)
        
        st.markdown("---")
        
        # Clear conversation button
        if st.button("Clear Conversation", type="primary"):
            st.session_state.chat_history = []
            st.rerun()

class ChatbotInterface:
    def __init__(self):
        self.working_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.llm_model_dir = self.working_dir / "Models/Llama-3.2-3B-instruct"
        self.vectorstore = None
        self.conversational_chain = None

    def setup_vectorstore(self):
        """Initialize vector store"""
        try:
            if st.session_state.get("current_vectorstore"):
                return st.session_state.current_vectorstore
                
            persist_directory = self.working_dir / "vector_database"
            
            from chromadb.config import Settings
            client_settings = Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
            
            self.vectorstore = Chroma(
                persist_directory=str(persist_directory),
                embedding_function=HuggingFaceEmbeddings(
                    model_name=str(self.working_dir / "Models/MiniLM-L6-v2")
                ),
                client_settings=client_settings
            )
            logger.info("Vector store initialized successfully")
            return self.vectorstore
        except Exception as e:
            logger.error(f"Error setting up vector store: {str(e)}")
            raise

    def initialize_chain(self, vectorstore):
        """Initialize the conversation chain with improved pipeline settings"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(self.llm_model_dir))
            model = AutoModelForCausalLM.from_pretrained(
                str(self.llm_model_dir),
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                device_map=None
            )
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            tokenizer.pad_token = tokenizer.eos_token
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=150,
                temperature=0.1,
                top_k=10,
                repetition_penalty=1.2,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                min_new_tokens=10,
                max_time=30,
                device=device
            )
            
            llm = HuggingFacePipeline(pipeline=pipe)
            
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 2},
                search_type="similarity"
            )
            
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                output_key="answer",
                return_messages=True,
                input_key="question"
            )
            
            self.conversational_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                combine_docs_chain_kwargs={"prompt": CHAT_PROMPT},
                return_source_documents=True,
                verbose=False
            )
            
            logger.info("Conversation chain initialized successfully")
            return self.conversational_chain
            
        except Exception as e:
            logger.error(f"Error initializing conversation chain: {str(e)}")
            raise

def initialize_session_state():
    """Initialize Streamlit session state"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = ChatbotInterface()
    if "chain_initialized" not in st.session_state:
        st.session_state.chain_initialized = False
    if "current_vectorstore" not in st.session_state:
        st.session_state.current_vectorstore = None
    if "using_uploaded_doc" not in st.session_state:
        st.session_state.using_uploaded_doc = False
    if "current_file" not in st.session_state:
        st.session_state.current_file = None

def main():
    st.set_page_config(
        page_title="Document Assistant",
        page_icon="📚",
        layout="wide"
    )
    
    # Initialize session state first
    initialize_session_state()
    
    # Create sidebar
    create_sidebar()
    
    # Main chat interface
    main_container = st.container()
    with main_container:
        st.title("📚 Document Assistant")
        if st.session_state.current_file:
            st.caption(f"Currently using: {st.session_state.current_file}")
        st.markdown("---")
        
        # Create columns for chat and sources
        chat_col, source_col = st.columns([2, 1])
        
        with chat_col:
            # Fixed height chat container
            st.markdown("""
                <style>
                    .stChatMessage {
                        padding: 10px;
                        margin: 5px 0;
                        border-radius: 5px;
                    }
                    .chat-container {
                        height: calc(80vh - 100px);
                        overflow-y: auto;
                        margin-bottom: 20px;
                    }
                    .chat-input {
                        position: fixed;
                        bottom: 0;
                        width: 100%;
                        padding: 20px;
                        background: white;
                    }
                </style>
                """, unsafe_allow_html=True)
            
            # Messages container
            messages_container = st.container()
            with messages_container:
                # Display chat history
                for message in st.session_state.chat_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
            
            # Input container at the bottom
            st.markdown("<div style='height: 100px'></div>", unsafe_allow_html=True)
            user_input = st.chat_input("Ask about your documents...")
            
            if user_input:
                # Add user message to history
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                with st.chat_message("user"):
                    st.markdown(user_input)
                
                with st.chat_message("assistant"):
                    try:
                        if not st.session_state.get("using_uploaded_doc"):
                            st.warning("Please upload and process a document first.")
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": "Please upload and process a document first."
                            })
                        else:
                            with st.spinner("Thinking..."):
                                # Get response
                                response = st.session_state.chatbot.conversational_chain.invoke({
                                    "question": user_input,
                                    "chat_history": []
                                })
                                
                                # Process response
                                assistant_response = clean_response(response.get("answer", ""))
                                if not assistant_response:
                                    assistant_response = "I couldn't find relevant information in the document."
                                
                                # Display response
                                st.markdown(assistant_response)
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": assistant_response
                                })
                                
                                # Show sources
                                if response.get("source_documents"):
                                    with source_col:
                                        st.markdown("### Sources")
                                        for i, doc in enumerate(response["source_documents"], 1):
                                            with st.expander(f"Source {i}"):
                                                st.markdown(clean_response(doc.page_content))
                    
                    except Exception as e:
                        st.error("An error occurred while processing your request.")
                        logger.error(f"Error in chat processing: {str(e)}")
                
                # Force streamlit to rerun to update the chat
                st.rerun()

if __name__ == "__main__":
    main()