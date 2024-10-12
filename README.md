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
    git clone https://github.com/lesteroliver911/openai-swam-simple-rag.git
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

## Sample Output

Here’s an example of how the system works in a typical session:

```bash
(resumerank) lesteroliver@Lesters-MacBook-Air resume-rank % /Users/lesteroliver/Dev/resume-rank/resumerank/bin/python /Users/lesteroliver/Dev/resume-rank/a.py
/Users/lesteroliver/Dev/resume-rank/a.py:32: LangChainDeprecationWarning: Please see the migration guide at: https://python.langchain.com/docs/versions/migrating_memory/
  self.memory = ConversationBufferMemory(return_messages=True)
Starting Final LLM Swarm Text Analyzer
You can ask for sentiment analysis, summarization, keyword extraction, or ask questions about loaded documents.
Type 'load_documents' to load a text file for analysis.
Type 'exit' to quit
User: load_documents
Enter the path to the text file: 10000_char_text.txt
Document loaded successfully. Content length: 143520 characters
User: summerize this document
Text Analyzer: The document discusses the impact of artificial intelligence (AI) on various aspects of society, including healthcare, education, and the future of work. It highlights the potential benefits of AI in improving efficiency, decision-making, and innovation, but also raises concerns about ethical considerations, data privacy, and algorithmic bias. The document emphasizes the importance of collaboration between policymakers, technologists, and citizens to ensure that AI is developed and deployed in a way that benefits all of humanity. It also stresses the need for ongoing dialogue, transparency, and accountability in the development of AI technologies.
User: 
```
