import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from fastapi import HTTPException,status
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDocument
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from src.services.document_service import document_service
from src.database import get_db
from src.models.user import User
from src.models.document import Document
from dotenv import load_dotenv

load_dotenv()
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROK_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "document-chat")

class RagService:
    
    def __init__(self):
        logger.info("Initializing RAG service...")
        
        # 1. Initialize Embeddings using Hugging Face Inference API
        self.embeddings = HuggingFaceEndpointEmbeddings(
            huggingfacehub_api_token=HUGGINGFACE_API_KEY,
            task="feature-extraction",
            model="sentence-transformers/all-mpnet-base-v2"
        )
        
        
        # 2. Initialize Text Splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            length_function=len,
        )

    

        # 3. Initialize Pinecone Vector Store
        self._init_pinecone()
        
        logger.info("RAG service initialized successfully.")
    
   
    def _init_pinecone(self):
        logger.info("Initializing Pinecone client...")
        try:
            self.pc = Pinecone(api_key=PINECONE_API_KEY)
            
            # Get the correct dimension for the embedding model
            embedding_dimension = 768
            logger.info(f"Embedding model dimension: {embedding_dimension}")
            
            existing_indexes = self.pc.list_indexes()
            index_names = [index.name for index in existing_indexes]
            
            if PINECONE_INDEX_NAME not in index_names:
                logger.info(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' with dimension {embedding_dimension}...")
                
                self.pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=embedding_dimension,  # Use actual embedding dimension
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                # Wait for index to be ready
                import time
                time.sleep(30)
            else:
                logger.info(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists.")
                
                # Check if existing index has correct dimension
                index_info = self.pc.describe_index(PINECONE_INDEX_NAME)
                existing_dimension = index_info.dimension
                
                if existing_dimension != embedding_dimension:
                    logger.error(f"Dimension mismatch! Index dimension: {existing_dimension}, Embedding dimension: {embedding_dimension}")
                    logger.info("Deleting and recreating index with correct dimension...")
                    
                    # Delete existing index
                    self.pc.delete_index(PINECONE_INDEX_NAME)
                    # Wait for deletion to complete
                    time.sleep(30)
                    
                    # Create new index with correct dimension
                    self.pc.create_index(
                        name=PINECONE_INDEX_NAME,
                        dimension=embedding_dimension,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
                    # Wait for index to be ready
                    time.sleep(30)
            
            index = self.pc.Index(PINECONE_INDEX_NAME)
            self.vectorstore = PineconeVectorStore(
                index=index,
                embedding=self.embeddings,
                text_key="text"
            )
            logger.info(f"Pinecone index '{PINECONE_INDEX_NAME}' is ready.")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise

    def _split_text(self, text: str) -> List[LangchainDocument]:
        if not text.strip():
            logger.warning("Text content is empty during splitting.")
            raise ValueError("Empty text content")
        logger.info("Splitting text into chunks...")
        documents = self.text_splitter.create_documents([text])
        logger.info(f"Split text into {len(documents)} chunks.")
        return documents

    async def add_document(self, document: Document, file_bytes: bytes) -> Dict[str, Any]:
        logger.info(f"Processing document '{document.filename}' for user '{document.user_id}'...")
        try:
            text = await document_service.extract_text(document.filename, file_bytes)
            if not text.strip():
                logger.warning("Extracted text is empty.")
                return {"status": "error", "message": "Empty document content"}
            
            chunks = self._split_text(text)
            # Save preview snippet in DB and set ready status after indexing
            preview = text[:400]
            
            chunk_documents = []
            for i, chunk in enumerate(chunks):
                chunk_doc = LangchainDocument(
                    page_content=chunk.page_content,
                    metadata={
                        "document_id": document.id,
                        "user_id": document.user_id,
                        "filename": document.filename,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                )
                chunk_documents.append(chunk_doc)
            
            logger.info(f"Storing {len(chunk_documents)} chunks in Pinecone...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self.vectorstore.add_documents(chunk_documents))
            logger.info("Document stored successfully in Pinecone.")
            try:
                db = get_db()
                await db.documents.update_one(
                    {"id": document.id, "user_id": document.user_id},
                    {"$set": {"status": "ready", "preview": preview}}
                )
            except Exception as e:
                logger.warning(f"Failed to update document status/preview: {str(e)}")
            
            return {
                "status": "success",
                "chunks_count": len(chunks),
                "document_id": document.id,
                "message": f"Document processed successfully into {len(chunks)} chunks"
            }
        except Exception as e:
            logger.error(f"Failed to process document: {str(e)}")
            return {"status": "error", "message": f"Failed to process document: {str(e)}", "document_id": document.id}

    async def query_question(self, user: User, question: str,api_key:str, document_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        logger.info(f"Querying question for user '{user.id}': {question}")
        try:
            llm_model = ChatGroq(
                groq_api_key=api_key,
                temperature=0.1,
                model_name="llama-3.3-70b-versatile"
            )
            filter_dict = {"user_id": user.id}
            if document_ids:
                filter_dict["document_id"] = {"$in": document_ids}
            
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5, "filter": filter_dict}
            )
            
            prompt_template = """You are a professional AI assistant helping users analyze and understand their documents. 
            Use the following context to provide a comprehensive, well-structured answer to the user's question.
            
            Guidelines:
            - Provide clear, professional responses with proper formatting
            - Use bullet points, numbered lists, or sections when appropriate
            - Be specific and cite relevant information from the context
            - If the context doesn't contain enough information, clearly state what information is missing
            - Structure your response logically with clear headings or sections
            - Be helpful and educational in your explanations
            
            Context: {context}

            Question: {question}
            
            Answer: """
            
            PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm_model,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, qa_chain, {"query": question})
            
            sources = []
            for doc in result.get("source_documents", []):
                sources.append({
                    "document_id": doc.metadata.get("document_id"),
                    "filename": doc.metadata.get("filename"),
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })
            
            # Format the response with sources
            answer = result["result"]
            if sources:
                answer += "\n\n**Sources:**\n"
                for i, source in enumerate(sources, 1):
                    answer += f"{i}. **{source['filename']}** (Chunk #{source['chunk_index']})\n"
            
            logger.info("Query processed successfully.")
            
            # Save chat history to database
            await self._save_chat_history(user, question, answer, api_key)
            
            return {
                "answer": answer,
                "sources": sources,
                "question": question,
                "document_ids": document_ids or "all_user_documents",
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            raise HTTPException(status_code=404,detail=str(e))

    async def _save_chat_history(self, user, question: str, answer: str, api_key: str):
        """Save chat history to database"""
        try:
            from src.database import get_db
            from datetime import datetime
            import uuid
            
            db = get_db()
            chat_id = str(uuid.uuid4())
            
            chat_record = {
                "id": chat_id,
                "user_id": str(user.id),
                "openai_api_key": api_key,
                "search_message": question,
                "response": answer,
                "created_at": datetime.utcnow(),
                "updated_at": None
            }
            
            await db["chat_history"].insert_one(chat_record)
            logger.info(f"Chat history saved for user {user.id}")
            
        except Exception as e:
            logger.error(f"Error saving chat history: {str(e)}")
            # Don't raise exception to avoid breaking the main flow

    async def delete_document_embeddings(self, document_id: str, user_id: str) -> bool:
        logger.info(f"Deleting embeddings for document '{document_id}' of user '{user_id}'...")
        try:
            self.vectorstore._index.delete(filter={"document_id": document_id, "user_id": user_id})
            logger.info("Embeddings deleted successfully.")
            return True
        except Exception as e:
            logger.error(f"Error deleting embeddings: {str(e)}")
            return False

# Create a single global instance
rag_service = RagService()