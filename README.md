# Document Summarizer with Q&A

A powerful web application for summarizing documents and answering questions about their content using advanced language models through the OpenRouter API.

![summaaa](https://github.com/user-attachments/assets/5c3c2370-73d4-4df8-8543-29956d5be407)


## Features

- **Document Processing**: Upload and process PDF, DOCX, TXT, Markdown, and HTML files
- **Advanced Summarization**: Generate concise summaries with customizable focus areas and length
- **Interactive Q&A**: Ask questions about your documents and get AI-powered answers
- **Suggested Questions**: Intelligent question suggestions based on document content
- **Multiple LLM Support**: Integration with various language models through OpenRouter
- **Optimized Processing**: Efficient chunking and processing of large documents

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

- Python 3.8 or higher
- OpenRouter API key (sign up at [OpenRouter.ai](https://openrouter.ai))
- Internet connection for API access

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/document-summarizer.git
   cd document-summarizer
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

4. Create an uploads directory for temporary file storage:
   ```bash
   mkdir uploads
   ```

## Configuration

The application can be configured by modifying environment variables in the `.env` file:

- `OPENROUTER_API_KEY`: Your OpenRouter API key (required)
- `UPLOAD_FOLDER`: Directory for temporary file storage (default: `./uploads`)

## Usage

### Starting the Application

Run the application using the provided `run.py` script:

```bash
python run.py
```

This will start the server at http://localhost:8000 and automatically open the web interface in your default browser.

### Command Line Options

The `run.py` script accepts several command line arguments:

```bash
python run.py --help
```

Available options:
- `--host`: Host to bind to (default: `0.0.0.0`)
- `--port`: Port to use (default: `8000`)
- `--no-reload`: Disable auto-reload for development
- `--no-browser`: Don't open browser automatically
- `--check`: Only check configuration and exit

### Using the Web Interface

1. **Select a Model**: Choose a language model from the dropdown or search for specific models
2. **Upload Document**: Upload a supported document (PDF, DOCX, TXT, MD, HTML)
3. **Customize Settings**: 
   - Set summary length (Very Brief to Comprehensive)
   - Select focus areas (Key Points, Methodology, Examples, etc.)
   - Adjust advanced settings if needed
4. **Generate Summary**: Click "Generate Summary" to process the document
5. **View Summary**: The generated summary will be displayed with metadata
6. **Ask Questions**: Use the Q&A section to ask questions about the document
7. **Follow-up Questions**: Explore suggested follow-up questions

## API Endpoints

The application exposes the following API endpoints:

- `GET /`: Web interface
- `GET /api/health`: Health check endpoint
- `GET /models`: List available OpenRouter models
- `POST /summarize`: Upload and summarize a document
- `GET /tasks/{task_id}`: Get status of a summarization task
- `POST /qa`: Ask a question about a document
- `GET /qa/{qa_task_id}`: Get status and answer for a Q&A task
- `GET /suggest-questions/{task_id}`: Get suggested questions for a document

For detailed API documentation, visit `http://localhost:8000/docs` when the server is running.

## Project Structure

```
document-summarizer/
├── document_processor.py    # Handles different document types
├── enhanced_summarizer.py   # Core summarization and QA functionality
├── index.html               # Web UI
├── main.py                  # FastAPI application and endpoints
├── models.py                # Data models and utility functions
├── openrouter_patch.py      # OpenRouter API integration
├── requirements.txt         # Python dependencies
├── run.py                   # Startup script
└── uploads/                 # Temporary file storage
```

### Key Components

- **DocumentProcessor**: Handles various document formats and extracts text
- **EnhancedSummarizer**: Performs document summarization and question answering
- **ChatOpenRouter**: Integration with OpenRouter API for language model access
- **FastAPI Application**: Provides web endpoints and background task processing

## Technical Details

### Document Processing

The application uses various libraries to process different document types:
- PDF files: PyPDF2
- DOCX files: python-docx
- Markdown: markdown + BeautifulSoup
- HTML: BeautifulSoup
- Plain text: Native Python

### Summarization Process

1. Document is uploaded and processed to extract text
2. Text is split into manageable chunks with configurable overlap
3. Each chunk is summarized using the selected language model
4. Chunk summaries are combined to create a comprehensive final summary
5. Custom prompts are generated based on user-selected focus areas

### Question Answering

1. Document text is retained in server memory for the session
2. User questions are processed against the document content
3. Language models determine relevant answers from the document
4. Follow-up questions are suggested based on document content and conversation history

## Troubleshooting

### Common Issues

- **API Key Error**: Ensure your OpenRouter API key is correctly set in the `.env` file
- **Connection Issues**: Check your internet connection and firewall settings
- **Document Processing Errors**: Some complex documents may not process correctly
- **Model Selection**: Different models have various capabilities and cost implications

### Logs

Check the application logs for detailed error information and debugging:

```bash
tail -f document-summarizer.log
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [OpenRouter](https://openrouter.ai) for providing API access to various language models
- [FastAPI](https://fastapi.tiangolo.com/) for the high-performance web framework
- [LangChain](https://langchain.com/) for the language model integration utilities
