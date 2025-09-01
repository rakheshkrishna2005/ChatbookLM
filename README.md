# 📰 Chatbook LM - RAG Document Q&A System

A powerful Retrieval-Augmented Generation (RAG) application that allows you to upload PDF documents and ask questions about their content using advanced AI models. Built with Streamlit for a user-friendly interface and powered by state-of-the-art language models.

![Chatbook LM Interface](https://github.com/rakheshkrishna2005/ChatbookLM/blob/main/app.png)

## 🚀 Features

- **📄 PDF Document Upload**: Upload multiple PDF files (up to 200MB per file)
- **🔍 Intelligent Document Indexing**: Automatic text chunking and vector embedding
- **💬 Conversational AI**: Ask questions in natural language about your documents
- **🎯 Source Selection**: Choose which documents to query from
- **📊 Real-time Progress Tracking**: Visual progress bars during indexing
- **🗑️ Document Management**: Upload, index, and delete documents as needed
- **💾 Persistent Chat History**: Maintain conversation context across sessions
- **⚡ Fast Vector Search**: Powered by Weaviate vector database
- **🤖 Advanced AI Models**: Uses Google Gemini 1.5 Flash for generation and Cohere for embeddings

## 🛠️ Tech Stack

### Core Technologies
- **Streamlit** - Web application framework
- **Python 3.12** - Programming language

### AI & ML
- **Google Gemini 1.5 Flash** - Large Language Model for answer generation
- **Cohere Embeddings** - Text embedding model (embed-english-v3.0)
- **LangChain** - Document processing and text splitting

### Database & Storage
- **Weaviate Cloud** - Vector database for similarity search
- **Temporary File System** - PDF processing

### Document Processing
- **PyPDF** - PDF document loading and parsing
- **RecursiveCharacterTextSplitter** - Intelligent text chunking

### Utilities
- **python-dotenv** - Environment variable management
- **Memory Management** - Automatic cleanup and garbage collection

## 📋 Prerequisites

Before running this application, you'll need:

1. **Python 3.12** or higher
2. **API Keys** for the following services:
   - Google AI (Gemini API)
   - Cohere AI
   - Weaviate Cloud

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd rag
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the root directory with your API keys:

```env
GOOGLE_API_KEY=your_google_ai_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
WEAVIATE_API_KEY=your_weaviate_api_key_here
WEAVIATE_URL=your_weaviate_cluster_url_here
```

### 5. Run the Application
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 📖 How to Use

### 1. Upload Documents
- Use the sidebar to upload PDF files
- Supported format: PDF only
- File size limit: 200MB per file
- Multiple files can be uploaded simultaneously

### 2. Index Documents
- Click "📥 Index Sources" to process uploaded documents
- The system will:
  - Split documents into chunks (500 characters with 100 character overlap)
  - Generate embeddings using Cohere
  - Store vectors in Weaviate database
  - Show progress with a visual progress bar

### 3. Select Sources
- Choose which documents to query from the "Source Selection" dropdown
- Multiple sources can be selected for comprehensive answers

### 4. Ask Questions
- Type your question in the chat input
- The system will:
  - Search for relevant document chunks
  - Generate contextual answers using Gemini
  - Display the response in the chat interface

### 5. Manage Documents
- **Delete Sources**: Remove all indexed documents
- **Clear Chat**: Reset conversation history

## 🔧 Technical Architecture

### Document Processing Pipeline
1. **Upload** → PDF files uploaded via Streamlit
2. **Load** → PyPDFLoader extracts text content
3. **Split** → RecursiveCharacterTextSplitter creates chunks
4. **Embed** → Cohere generates vector embeddings
5. **Store** → Weaviate stores vectors with metadata

### Query Processing Pipeline
1. **Query** → User question input
2. **Embed** → Question converted to vector
3. **Search** → Vector similarity search in Weaviate
4. **Retrieve** → Top 5 relevant chunks returned
5. **Generate** → Gemini creates contextual answer

### Memory Management
- Automatic cleanup of database connections
- Garbage collection for memory optimization
- SSL socket cleanup to prevent resource leaks

## 🔄 Development

### Project Structure
```
rag/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (create this)
├── README.md          # This file
└── app.png            # Application screenshot
```

### Key Functions
- `get_client()` - Weaviate connection management
- `load_documents()` - PDF processing
- `index_documents()` - Vector indexing
- `query_vector_db()` - Similarity search
- `generate_answer()` - AI response generation
