"""
Create Multi-Modal RAG Agent in Azure AI Foundry

This script creates an agent directly via the Azure OpenAI Assistants API,
which will appear in your Azure AI Foundry portal.
"""

import os
import requests
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential

load_dotenv()

# Configuration from your .env
AZURE_ENDPOINT = os.getenv("CONTENT_UNDERSTANDING_ENDPOINT", "https://<your-ai-services>.services.ai.azure.com")
CHAT_MODEL = os.getenv("CHAT_MODEL_NAME", "gpt-4o")
SEARCH_INDEX = os.getenv("SEARCH_SERVICE_INDEX_NAME", "multi-modal-rag-index")
SEARCH_ENDPOINT = os.getenv("SEARCH_SERVICE_ENDPOINT", "https://<your-search>.search.windows.net")

AGENT_NAME = "Multi-Modal RAG Agent"
AGENT_INSTRUCTIONS = """You are a professional RAG-based assistant whose context comes exclusively from a database in Azure AI Search containing multi-modal content.

Always follow these rules:

1. Answer strictly from the context provided. Do not invent, assume, or add details outside the given context.
2. If no relevant information is available in the context, politely say so.
3. The search index contains content from:
   - Videos (transcripts and visual descriptions)
   - Audio (speech transcripts)
   - PDFs (extracted text)
   - Images (visual descriptions)

4. Style & tone:
   - Respond in a professional, natural way, as if conversing with a human.
   - Structure answers clearly and concisely.
   - Reference the source document titles when appropriate.

Your role is to act like a knowledgeable human assistant who can reference the provided information smoothly and contextually, across any modality (text, image, video, or audio)."""


def create_agent():
    """Create the agent via Azure OpenAI Assistants API."""
    
    print("🚀 Creating Multi-Modal RAG Agent...")
    print(f"   Endpoint: {AZURE_ENDPOINT}")
    print(f"   Model: {CHAT_MODEL}")
    
    # Get credential
    credential = DefaultAzureCredential()
    token = credential.get_token("https://cognitiveservices.azure.com/.default").token
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Create assistant via REST API
    api_url = f"{AZURE_ENDPOINT}/openai/assistants?api-version=2024-05-01-preview"
    
    body = {
        "model": CHAT_MODEL,
        "name": AGENT_NAME,
        "instructions": AGENT_INSTRUCTIONS,
        "tools": []  # Tools will be added in portal
    }
    
    print(f"\n📡 Sending request to: {api_url}")
    
    response = requests.post(api_url, headers=headers, json=body)
    
    if response.status_code in [200, 201]:
        agent = response.json()
        print(f"\n✅ Agent created successfully!")
        print(f"   Agent ID: {agent.get('id')}")
        print(f"   Agent Name: {agent.get('name')}")
        print(f"   Model: {agent.get('model')}")
        print(f"\n🎉 Go to Azure AI Foundry portal and refresh to see your agent!")
        print(f"\n📝 Next step: Add Azure AI Search tool in the portal:")
        print(f"   1. Click on your agent '{AGENT_NAME}'")
        print(f"   2. Go to 'Tools' section")
        print(f"   3. Add 'Azure AI Search' tool")
        print(f"   4. Select index: {SEARCH_INDEX}")
        return agent
    else:
        print(f"\n❌ Failed to create agent")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text}")
        return None


if __name__ == "__main__":
    create_agent()