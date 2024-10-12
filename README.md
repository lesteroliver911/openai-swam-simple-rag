# openai-swam-simple-rag

This repository demonstrates a simple implementation of OpenAI Swarm for multi-agent orchestration with **Retrieval-Augmented Generation (RAG)**. The system showcases how agents work together to perform tasks like text summarization, sentiment analysis, keyword extraction, and document search using FAISS and OpenAI's language models.

## Features

- **Multi-Agent System**: Utilizes the Swarm framework for orchestrating specialized agents.
- **Text Analysis**: Perform sentiment analysis, summarization, and keyword extraction.
- **Document Search**: Use FAISS for efficient similarity-based document retrieval.
- **Retrieval-Augmented Generation (RAG)**: Enhance document querying with relevant context and OpenAI language models.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/openai-swam-simple-rag.git
    cd openai-swam-simple-rag
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables:

    Create a `.env` file in the root directory and add your OpenAI API key:

    ```bash
    OPENAI_API_KEY=your_openai_api_key
    ```

## Usage

1. Start the application by running:

    ```bash
    python app.py
    ```

2. You'll be prompted to load documents and can interact with the agents via the command-line interface for:

    - Summarization
    - Sentiment analysis
    - Keyword extraction
    - Document search

3. **Commands**:
    - Type `load_documents` to load a text file for analysis.
    - Ask questions like "Summarize this document" or "What is the sentiment of the text?".
    - Type `exit` to quit.

## Example

1. Load a document:
    
    ```bash
    User: load_documents
    Enter the path to the text file: /path/to/your/document.txt
    ```

2. Summarize the loaded document:

    ```bash
    User: Summarize the document
    Text Analyzer: [Generated summary]
    ```

## Agents in Action

The system runs a **Swarm** of agents with specialized tasks. Each agent uses OpenAI's GPT-3.5-turbo model to handle queries such as:

- **Sentiment Analysis**: Analyze the sentiment of a given document.
- **Summarization**: Generate concise summaries of text.
- **Keyword Extraction**: Identify key terms within the content.
- **Document Search**: Retrieve the most relevant sections from the document using FAISS.

## Future Improvements

- Implement additional agent roles for advanced document processing.
- Add web-based front-end for enhanced user experience.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
