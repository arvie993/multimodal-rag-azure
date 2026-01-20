"""
Multi-Modal RAG Agent for Azure AI Foundry

This agent provides a chat interface for querying multi-modal content (video, audio, images, PDFs)
indexed in Azure AI Search. It uses Azure OpenAI for embeddings and chat completions.

Can be tested locally or deployed to Azure AI Foundry as an agent.
"""

import os
import json
import requests
from typing import Optional
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

# Load environment variables
load_dotenv()


class MultiModalRAGAgent:
    """
    A RAG agent that can answer questions based on multi-modal content
    indexed in Azure AI Search (video, audio, images, PDFs).
    """

    def __init__(self):
        """Initialize the agent with Azure services configuration."""
        # Azure OpenAI configuration
        self.azure_ai_endpoint = os.getenv("AZURE_AI_ENDPOINT")
        self.embedding_model = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-large")
        self.chat_model = os.getenv("CHAT_MODEL_NAME", "gpt-4o")

        # Azure Search configuration
        self.search_endpoint = os.getenv("SEARCH_SERVICE_ENDPOINT", "").strip().rstrip('/')
        self.search_index = os.getenv("SEARCH_SERVICE_INDEX_NAME", "multi-modal-rag-index")

        # Authentication
        self.auth_mode = os.getenv("AUTH_MODE", "entra").lower()
        self.credential = None
        self.openai_client = None

        # System prompt for the agent
        self.system_prompt = """
You are a professional RAG-based assistant whose context comes exclusively from a database in Azure AI Search.

Always follow these rules:

1. Answer strictly from the context provided. Do not invent, assume, or add details outside the given context.
2. If no relevant information is available in the context, politely say so.
3. Do not include any external links, citations, or references unless they are explicitly present in the context object.
4. Context format: The system will pass a Python list of objects, each containing:
   {
     "chunk": "the content (text, JSON, transcript, or description)",
     "title": "the document title",
     "score": "the relevancy score"
   }
   These are the top matches based on cosine similarity with the user query.
5. Style & tone:
   - Respond in a professional, natural way, as if conversing with a human.
   - Structure answers clearly and concisely.
   - If the context contains a field such as 'url', 'video_url', or 'image_url', include it as a clickable hyperlink in your response.
   - Reference the source document titles when appropriate to help users understand where the information comes from.

Your role is to act like a knowledgeable human assistant who can reference the provided information smoothly and contextually, across any modality (text, image, video, or audio).
"""

        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize Azure OpenAI and authentication clients."""
        if self.auth_mode == "entra":
            self.credential = DefaultAzureCredential()
            token_provider = get_bearer_token_provider(
                self.credential,
                "https://cognitiveservices.azure.com/.default"
            )
            self.openai_client = AzureOpenAI(
                azure_ad_token_provider=token_provider,
                api_version="2024-02-15-preview",
                azure_endpoint=self.azure_ai_endpoint
            )
            print("✅ Agent initialized with Entra ID authentication")
        else:
            api_key = os.getenv("AZURE_AI_API_KEY")
            self.openai_client = AzureOpenAI(
                api_key=api_key,
                api_version="2024-02-15-preview",
                azure_endpoint=self.azure_ai_endpoint
            )
            print("✅ Agent initialized with API Key authentication")

    def _generate_embedding(self, text: str) -> list:
        """Generate vector embedding for the given text."""
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return response.data[0].embedding

    def _search_index(self, query_embedding: list, top_k: int = 5) -> list:
        """Search the Azure AI Search index with vector similarity."""
        url = f"{self.search_endpoint}/indexes/{self.search_index}/docs/search?api-version=2023-11-01"

        # Set up headers based on auth mode
        if self.auth_mode == "entra":
            token = self.credential.get_token("https://search.azure.com/.default").token
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}"
            }
        else:
            headers = {
                "Content-Type": "application/json",
                "api-key": os.getenv("SEARCH_SERVICE_API_KEY")
            }

        body = {
            "count": True,
            "select": "content_text, document_title",
            "vectorQueries": [{
                "vector": query_embedding,
                "k": top_k,
                "fields": "content_embedding",
                "kind": "vector"
            }]
        }

        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()

        results = response.json().get('value', [])
        return [
            {
                "chunk": doc.get('content_text', ''),
                "title": doc.get('document_title', 'Unknown'),
                "score": doc.get('@search.score', 0)
            }
            for doc in results
        ]

    def _generate_response(self, query: str, context: list) -> str:
        """Generate a response using Azure OpenAI with the retrieved context."""
        user_message = f"The user query is: {query}\nThe context is: {json.dumps(context, indent=2)}"

        response = self.openai_client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7
        )

        return response.choices[0].message.content

    def chat(self, user_message: str, top_k: int = 5) -> dict:
        """
        Process a user message and return a response.

        Args:
            user_message: The user's question or query
            top_k: Number of documents to retrieve from the index

        Returns:
            dict with 'response', 'sources', and 'query'
        """
        try:
            # Step 1: Generate embedding for the query
            query_embedding = self._generate_embedding(user_message)

            # Step 2: Search the index
            context = self._search_index(query_embedding, top_k)

            # Step 3: Generate response
            response = self._generate_response(user_message, context)

            return {
                "response": response,
                "sources": context,
                "query": user_message
            }

        except Exception as e:
            return {
                "response": f"I encountered an error while processing your request: {str(e)}",
                "sources": [],
                "query": user_message,
                "error": str(e)
            }


def create_agent() -> MultiModalRAGAgent:
    """Factory function to create a MultiModalRAGAgent instance."""
    return MultiModalRAGAgent()


# ============================================================================
# Interactive CLI for local testing
# ============================================================================

def run_interactive_cli():
    """Run an interactive command-line interface for testing the agent."""
    print("\n" + "=" * 70)
    print("🤖 Multi-Modal RAG Agent - Interactive Mode")
    print("=" * 70)
    print("\nThis agent can answer questions about BMW sustainability content")
    print("including videos, audio, images, and PDF documents.")
    print("\nType 'quit' or 'exit' to end the session.")
    print("Type 'help' for example queries.")
    print("-" * 70)

    agent = create_agent()

    example_queries = [
        "What is BMW's approach to circularity?",
        "Tell me about BMW's forwardism strategy",
        "What are BMW's sustainability initiatives for natural rubber?",
        "Describe BMW's sustainability journey from 1973 to 2030"
    ]

    while True:
        try:
            user_input = input("\n📝 You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            if user_input.lower() == 'help':
                print("\n💡 Example queries you can try:")
                for i, q in enumerate(example_queries, 1):
                    print(f"   {i}. {q}")
                continue

            # Process the query
            print("\n🔍 Searching knowledge base...")
            result = agent.chat(user_input)

            # Display the response
            print(f"\n🤖 Agent: {result['response']}")

            # Show sources
            if result.get('sources'):
                print("\n📚 Sources:")
                for i, source in enumerate(result['sources'][:3], 1):
                    print(f"   {i}. {source['title']} (relevance: {source['score']:.4f})")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


# ============================================================================
# FastAPI Server for Foundry Integration
# ============================================================================

def create_fastapi_app():
    """Create a FastAPI app for serving the agent as an HTTP endpoint."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError:
        print("❌ FastAPI not installed. Run: pip install fastapi uvicorn")
        return None

    app = FastAPI(
        title="Multi-Modal RAG Agent",
        description="A RAG agent for querying multi-modal content (video, audio, images, PDFs)",
        version="1.0.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize the agent
    agent = create_agent()

    class ChatRequest(BaseModel):
        message: str
        top_k: int = 5

    class ChatResponse(BaseModel):
        response: str
        sources: list
        query: str

    @app.get("/")
    async def root():
        return {"status": "healthy", "agent": "Multi-Modal RAG Agent"}

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """Send a message to the RAG agent and get a response."""
        try:
            result = agent.chat(request.message, request.top_k)
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/chat")
    async def api_chat(request: ChatRequest):
        """Alternative endpoint for chat (compatible with various frontends)."""
        return await chat(request)

    return app


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        # Run as HTTP server
        try:
            import uvicorn
            app = create_fastapi_app()
            if app:
                print("\n🚀 Starting Multi-Modal RAG Agent Server...")
                print("   Endpoints:")
                print("   - POST /chat - Send chat messages")
                print("   - GET /health - Health check")
                print("\n")
                uvicorn.run(app, host="0.0.0.0", port=8000)
        except ImportError:
            print("❌ uvicorn not installed. Run: pip install uvicorn")
    else:
        # Run interactive CLI
        run_interactive_cli()