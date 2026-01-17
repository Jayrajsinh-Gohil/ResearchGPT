# 📚 AI Research Paper Analyzer

An intelligent web application that leverages AI to automatically analyze and extract key information from research papers in PDF format. The system uses advanced natural language processing with Llama3 LLM and LangChain to provide comprehensive insights including abstracts, summaries, technologies used, challenges, benefits, conclusions, and metadata.

## ✨ Features

- **📄 PDF Upload & Processing**: Upload research papers in PDF format (up to 50MB)
- **🤖 AI-Powered Analysis**: Automated extraction using Llama3 LLM via Ollama
- **📊 Comprehensive Information Extraction**:
  - Abstract extraction or generation
  - Full paper summary in simple language
  - Technologies, tools, and frameworks identification
  - Challenges and limitations analysis
  - Benefits and impact assessment
  - Conclusion extraction
  - Metadata extraction (title, authors, year, etc.)
- **🎨 Modern UI**: Beautiful, responsive web interface with light/dark theme
- **⚡ Performance Metrics**: Real-time processing time tracking for each extraction
- **🔍 Vector Search**: ChromaDB for efficient semantic search across paper content
- **💾 Smart Cleanup**: Automatic temporary file and embeddings cleanup

## 🏗️ Architecture

- **Backend**: Flask web framework with CORS support
- **LLM**: Llama3 via Ollama (local inference)
- **Embeddings**: HuggingFace sentence-transformers (bert-base-nli-mean-tokens)
- **Vector Database**: ChromaDB for document embeddings and retrieval
- **PDF Processing**: PyPDF2 for text extraction
- **Document Processing**: LangChain for text chunking and retrieval chains
- **Frontend**: HTML, CSS, JavaScript with modern UI/UX

## 📋 Prerequisites

- **Python**: 3.13.3 (specified in runtime.txt)
- **Ollama**: Installed and running on localhost:11434
- **RAM**: Minimum 8GB recommended for LLM operations
- **Storage**: Adequate space for temporary embeddings

## 🚀 Installation & Setup

### Option 1: Virtual Environment (Recommended)

#### 1. Clone or Download the Project
```bash
cd "𝗥𝗲𝘀𝗲𝗮𝗿𝗰𝗵𝗚𝗣𝗧"
```

#### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate

# On Linux/Mac:
source .venv/bin/activate
```

#### 3. Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install Flask==3.1.0 flask-cors==6.0.1 PyPDF2==3.0.1
pip install langchain==0.3.27 langchain-core==0.3.76
pip install langchain-community==0.3.29 langchain-huggingface==0.3.1
pip install langchain-ollama==0.3.10 chromadb==1.1.1
pip install sentence-transformers==5.1.0 transformers==4.56.2
pip install torch==2.8.0 requests==2.32.5
pip install huggingface-hub==0.35.0 tokenizers==0.22.1 ollama==0.6.0
```

### Option 2: Conda Environment

#### 1. Create Environment from YAML
```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate 𝗥𝗲𝘀𝗲𝗮𝗿𝗰𝗵𝗚𝗣𝗧
```

### Ollama Setup (Required)

#### 1. Install Ollama
```bash
# Visit and download: https://ollama.ai/download
# Or use package manager on Linux:
curl -fsSL https://ollama.com/install.sh | sh
```

#### 2. Pull Llama3 Model
```bash
ollama pull llama3
```

#### 3. Start Ollama Service
```bash
# This should run in a separate terminal
ollama serve
```

Verify Ollama is running:
```bash
curl http://localhost:11434/api/version
```

## 🎯 Usage

### 1. Start the Application
```bash
# Make sure Ollama is running in another terminal
python app.py
```

The application will start on `http://localhost:5000`

### 2. Access Web Interface
- Open your browser and navigate to: `http://localhost:5000`
- Or for network access: `http://<your-ip>:5000`

### 3. Upload and Analyze
1. Click the upload area or drag-and-drop a PDF research paper
2. Wait for the AI to process the document
3. View extracted information in organized sections:
   - **Abstract**: Paper's abstract or AI-generated summary
   - **Summary**: Concise overview in simple language
   - **Technologies**: Tools, frameworks, and technologies used
   - **Challenges**: Difficulties and limitations identified
   - **Benefits**: Advantages and who benefits from the research
   - **Conclusion**: Key takeaways and final remarks
   - **Metadata**: Author names, publication info, etc.
4. View processing time metrics for performance insights

## 📁 Project Structure

```
𝗥𝗲𝘀𝗲𝗮𝗿𝗰𝗵𝗚𝗣𝗧/
├── app.py                    # Main Flask application
├── templates/
│   ├── index.html           # Main web interface
│   └── indexo1.html         # Alternative interface
├── uploads/                 # Temporary PDF storage (auto-cleaned)
├── Photos/                  # UI assets
├── environment.yml          # Conda environment configuration
├── runtime.txt              # Python version specification
├── README.md                # This file
```

## 🔧 Configuration

### Model Configuration
In [app.py](app.py), you can customize:

```python
# Embedding model (line 43-45)
model_name = "sentence-transformers/bert-base-nli-mean-tokens"
# Alternatives: "sentence-transformers/all-MiniLM-L6-v2"

# LLM model (line 73)
llm = OllamaLLM(model="llama3", base_url="http://localhost:11434")

# Chunk settings (line 65-68)
chunk_size = 800
chunk_overlap = 50

# Retrieval settings (line 157-161)
search_type = "similarity"
k = 100  # Number of chunks to retrieve
```

### Upload Limits
```python
# Maximum file size (line 22)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
```

## 🌐 Network Access

To access from other devices on your network:

1. Find your IP address:
   ```bash
   # Windows
   ipconfig
   
   # Linux/Mac
   ifconfig
   ```

2. Access via: `http://<your-ip>:5000`

3. Ensure firewall allows port 5000

For detailed network setup, see [LOCAL_NETWORK_SETUP.md](LOCAL_NETWORK_SETUP.md)

## ⚙️ How It Works

1. **PDF Upload**: User uploads a research paper PDF
2. **Text Extraction**: PyPDF2 extracts text from all pages
3. **Chunking**: Text split into 800-character chunks with 50-char overlap
4. **Embedding**: HuggingFace model creates vector embeddings
5. **Vector Store**: ChromaDB stores embeddings for semantic search
6. **Retrieval**: Top 100 relevant chunks retrieved for each query
7. **LLM Processing**: Llama3 analyzes chunks and generates answers
8. **Response**: Extracted information returned to frontend
9. **Cleanup**: Temporary files and embeddings automatically deleted

## 📊 Performance

Typical processing times (varies by PDF size and hardware):
- File preprocessing: 2-10 seconds
- Abstract extraction: 5-15 seconds
- Summary generation: 10-25 seconds
- Technology extraction: 8-20 seconds
- Challenges analysis: 8-20 seconds
- Benefits analysis: 8-20 seconds
- Conclusion extraction: 5-15 seconds
- Metadata extraction: 5-15 seconds

**Total**: 50-140 seconds per paper (average: ~90 seconds)

## 🛠️ Troubleshooting

### Ollama Connection Error
```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Start Ollama if not running
ollama serve
```

### Model Not Found
```bash
# Pull the Llama3 model
ollama pull llama3

# List available models
ollama list
```

### Port Already in Use
```python
# Change port in app.py (line 489)
app.run(host='0.0.0.0', port=5001, debug=True)
```

### Memory Issues
- Close other applications
- Use a smaller model if available
- Reduce chunk retrieval: change `k=100` to `k=50` in [app.py](app.py#L157)

### ChromaDB Errors
The app automatically handles cleanup. If issues persist:
```bash
# Manually delete embeddings folders
rm -rf embeddings_*
```

## 🔐 Security Considerations

- **Local Use**: Designed for local/trusted network use
- **File Size Limit**: 50MB maximum to prevent abuse
- **Auto Cleanup**: Uploaded files deleted after processing
- **CORS**: Enabled for all origins (restrict in production)
- **No Authentication**: Add authentication for public deployment

## 📝 Dependencies

### Core Dependencies
- Flask 3.1.0 - Web framework
- PyPDF2 3.0.1 - PDF text extraction
- langchain 0.3.27 - LLM application framework
- chromadb 1.1.1 - Vector database
- sentence-transformers 5.1.0 - Embeddings
- torch 2.8.0 - Deep learning framework
- ollama 0.6.0 - Ollama Python client

### Full Dependency List
See [environment.yml](environment.yml) for complete list with versions

## 🤝 Contributing

Contributions are welcome! To contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is available for educational and research purposes.

## 🙏 Acknowledgments

- **Ollama** - For providing local LLM infrastructure
- **LangChain** - For the powerful LLM application framework
- **HuggingFace** - For sentence transformer models
- **ChromaDB** - For efficient vector storage
- **Meta AI** - For the Llama3 model

## 📮 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed setup
3. Check [LOCAL_NETWORK_SETUP.md](LOCAL_NETWORK_SETUP.md) for network issues

## 🔄 Future Enhancements

- [ ] Support for multiple PDF processing
- [ ] Export results to various formats (PDF, DOCX, JSON)
- [ ] User authentication and session management
- [ ] Database for storing analysis history
- [ ] Support for other document formats
- [ ] Batch processing capabilities
- [ ] API endpoints for programmatic access
- [ ] Custom prompt templates
- [ ] Multiple LLM model support
- [ ] Improved error handling and validation

---

**Built with ❤️ using AI and Open Source Technologies**
