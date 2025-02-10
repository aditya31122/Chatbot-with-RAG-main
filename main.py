import os
import logging
from pathlib import Path
import streamlit as st
import torch
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
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
CHAT_PROMPT = PromptTemplate.from_template("""Provide a clear and direct answer based on the context provided. If asking about a topic or concept, give a brief definition and key points.

Context: {context}

Question: {question}

Answer:""")

def clean_response(text: str) -> str:
    """Aggressively clean the response text"""
    if not text:
        return "I'm not sure about that."
    
    # Remove token ID messages
    text = text.replace(
        "Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.",
        ""
    )
    
    # Handle greetings specially - only if it's just a greeting
    if text.lower().strip() in ["hi", "hello", "hey", "hi!", "hello!", "hey!"]:
        return "Hello! How can I help you?"
    
    # Remove conversation analysis
    if "User1:" in text or "User2:" in text:
        text = text.split("User1:")[0].strip()
    
    # Remove common artifacts
    artifacts = [
        "Answer the question based only on",
        "Keep your response focused",
        "Question:",
        "Context:",
        "Answer:",
        "Assistant:",
        "Human:",
        "Based on the context provided,",
        "According to the context,",
        "Standalone question:",
        "Follow-up Question:",
        "The original prompt",
        "(Conversation",
        "Based solely",
        "*",
        "Reasoning Skill",
        "Let me help you understand",
        "I'll help you with that",
        "I hope this helps",
        "Let me know if you have any other questions"
    ]
    
    for artifact in artifacts:
        text = text.replace(artifact, "").strip()
    
    # Clean up whitespace and newlines
    text = ' '.join(text.split())
    
    # Remove any remaining conversation analysis
    text = text.split("Conversation continues")[0].strip()
    text = text.split("Conversation ends")[0].strip()
    
    # Keep only the core response
    if len(text) > 300:  # If response is too long
        sentences = text.split('.')
        if sentences:
            # Keep first 2 meaningful sentences
            meaningful = [s for s in sentences if len(s.strip()) > 20][:2]
            text = '. '.join(meaningful) + '.'
    
    return text.strip()
class ChatbotInterface:
    def __init__(self):
        self.working_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.llm_model_dir = self.working_dir / "Models/Llama-3.2-3B-instruct"
        self.vectorstore = None
        self.conversational_chain = None

    def setup_vectorstore(self):
        """Initialize vector store"""
        try:
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
        """Initialize the conversation chain"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(self.llm_model_dir))
            model = AutoModelForCausalLM.from_pretrained(
                str(self.llm_model_dir),
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )

            tokenizer.pad_token = tokenizer.eos_token
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=100,  # Short responses
                temperature=0.3,     # More focused
                top_p=0.85,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            llm = HuggingFacePipeline(pipeline=pipe)
            
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 2},  # Reduced for focus
                search_type="similarity"
            )
            
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                output_key="answer",
                return_messages=True
            )
            
            self.conversational_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                combine_docs_chain_kwargs={
                    "prompt": CHAT_PROMPT,
                    "document_separator": "\n"
                },
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

def main():
    st.set_page_config(
        page_title="Document Assistant",
        page_icon="📚",
        layout="centered"
    )
    
    st.title("📚 Document Assistant")
    
    initialize_session_state()
    
    if not st.session_state.chain_initialized:
        try:
            with st.spinner("Initializing chatbot..."):
                chatbot = st.session_state.chatbot
                vectorstore = chatbot.setup_vectorstore()
                chatbot.initialize_chain(vectorstore)
                st.session_state.chain_initialized = True
        except Exception as e:
            st.error(f"Error initializing the chatbot: {str(e)}")
            return

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input("Ask about your documents...")

    if user_input:
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    response = st.session_state.chatbot.conversational_chain.invoke({
                        "question": user_input
                    })
                    
                    assistant_response = clean_response(response.get("answer", ""))
                    st.markdown(assistant_response)
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": assistant_response
                    })
                    
                    sources = response.get("source_documents", [])
                    if sources:
                        with st.expander("View Sources"):
                            for i, doc in enumerate(sources, 1):
                                st.markdown(f"**Source {i}:**")
                                st.markdown(doc.page_content)
                                st.markdown("---")
                
            except Exception as e:
                st.error("Sorry, there was an error processing your request.")
                logger.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()