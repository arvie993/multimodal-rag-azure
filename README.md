# Multi-Modal RAG with Azure AI

A complete solution for building a **Retrieval-Augmented Generation (RAG)** system that works across multiple content modalities using **Azure AI Content Understanding** and **Azure AI Search**.

![RAG Data Preparation](Assets/rag_data_prep.png)

## 🎯 Overview

This project demonstrates how to:

1. **Process multi-modal content** (videos, audio, PDFs, images) using Azure AI Content Understanding
2. **Index content** in Azure AI Search with vector embeddings
3. **Build a RAG agent** that can answer questions across all modalities
4. **Deploy to Azure AI Foundry** for testing and production use

### Supported Content Types

| Modality | Example | Processing |
|----------|---------|------------|
| 📄 **PDF** | BMW Sustainable Natural Rubber | Text extraction + chunking |
| 🎥 **Video** | BMW Circularity | Transcript + visual descriptions |
| 🎵 **Audio** | BMW Forwardism | Speech-to-text transcription |
| 🖼️ **Image** | BMW Sustainability Journey | Image verbalization/description |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Modal RAG Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐   ┌─────────────────────┐   ┌──────────────────┐ │
│  │  PDF     │   │ Azure AI Content    │   │                  │ │
│  │  Video   │──▶│ Understanding       │──▶│  Azure AI Search │ │
│  │  Audio   │   │ (Analyze + Embed)   │   │  (Vector Index)  │ │
│  │  Image   │   └─────────────────────┘   └────────┬─────────┘ │
│  └──────────┘                                      │            │
│                                                    ▼            │
│  ┌──────────┐   ┌─────────────────────┐   ┌──────────────────┐ │
│  │   User   │   │   Azure OpenAI      │   │   RAG Agent      │ │
│  │   Query  │──▶│   (GPT-4o)          │◀──│   (Retrieval)    │ │
│  └──────────┘   └─────────────────────┘   └──────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
multimodal-rag-azure/
├── RAG_Data_Preparation.ipynb   # Process & index multi-modal content
├── RAG_in_Action.ipynb          # Test RAG queries with results
├── multimodal_rag_agent.py      # Standalone Python RAG agent
├── create_foundry_agent.py      # Deploy agent to Azure AI Foundry
├── requirements.txt             # Python dependencies
├── index.json                   # Azure AI Search index schema
├── .env.sample                  # Environment variables template
├── Assets/                      # Documentation images
│   ├── rag_data_prep.png
│   └── rag_in_action.png
└── Data/                        # Sample multi-modal content
    ├── BMW_circularity.mp4
    ├── BMW_forwardism.mp3
    ├── BMW_sustainable_natural_rubber.pdf
    └── image.png
```

## 🚀 Getting Started

### Prerequisites

- **Azure Subscription** with the following resources:
  - Azure AI Services (with Content Understanding and OpenAI)
  - Azure AI Search (Basic tier or higher for vector search)
  - Azure Blob Storage
- **Python 3.10+**
- **Azure CLI** (for authentication)

### 1. Clone the Repository

```bash
git clone https://github.com/arvie993/multimodal-rag-azure.git
cd multimodal-rag-azure
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
# Copy the sample file
cp .env.sample .env

# Edit .env with your Azure resource details
```

### 4. Azure Authentication

```bash
# Login to Azure
az login

# Set your subscription (optional)
az account set --subscription "YOUR_SUBSCRIPTION_ID"
```

### 5. Run the Notebooks

1. **RAG_Data_Preparation.ipynb** - Process your content and create the search index
2. **RAG_in_Action.ipynb** - Test queries against your indexed content

## 🤖 Using the RAG Agent

### Option 1: Local Python Agent

```bash
# Interactive CLI mode
python multimodal_rag_agent.py

# Start as API server
python multimodal_rag_agent.py serve
```

### Option 2: Deploy to Azure AI Foundry

```bash
# Create agent in Foundry portal
python create_foundry_agent.py
```

Then in Azure AI Foundry portal:
1. Navigate to your project → Agents
2. Find "Multi-Modal RAG Agent"
3. Add **Azure AI Search** tool with your index
4. Test in the playground!

![RAG in Action](Assets/rag_in_action.png)

## 📊 Example Queries

Once your content is indexed, try these queries:

| Query | Expected Sources |
|-------|------------------|
| "What is BMW's approach to circularity?" | Video + PDF |
| "Tell me about sustainable natural rubber" | PDF (pages 1-2) |
| "What is BMW's forwardism strategy?" | Audio transcript |

## 🔧 Azure Resource Setup

### Required Role Assignments

For Entra ID authentication, assign these roles:

| Principal | Resource | Role |
|-----------|----------|------|
| Your User | AI Services | Cognitive Services OpenAI User |
| Your User | AI Search | Search Index Data Contributor |
| AI Services Identity | AI Search | Search Index Data Reader |
| AI Search Identity | AI Services | Cognitive Services OpenAI User |

### Azure AI Search Index Schema

The index uses the following fields:

| Field | Type | Purpose |
|-------|------|--------|
| `content_id` | String (key) | Unique document identifier |
| `document_title` | String | Source document name |
| `content_text` | String | Extracted/transcribed content |
| `content_embedding` | Vector (3072) | text-embedding-3-large vectors |

## 🛠️ Technologies Used

- **Azure AI Content Understanding** - Multi-modal content analysis
- **Azure AI Search** - Vector search and indexing
- **Azure OpenAI** - GPT-4o for chat, text-embedding-3-large for vectors
- **Azure AI Foundry** - Agent deployment and testing
- **Python** - OpenAI SDK, Azure Identity

## 📚 Learn More

- [Azure AI Content Understanding Documentation](https://learn.microsoft.com/azure/ai-services/content-understanding/)
- [Azure AI Search Vector Search](https://learn.microsoft.com/azure/search/vector-search-overview)
- [Azure OpenAI Service](https://learn.microsoft.com/azure/ai-services/openai/)
- [Azure AI Foundry Agents](https://learn.microsoft.com/azure/ai-studio/concepts/agents)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Built with ❤️ using Azure AI Services**