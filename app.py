import os
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import tiktoken

# Load environment variables
load_dotenv()

# Ensure you have set your OpenAI API key in the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

class Agent:
    def __init__(self, name: str, instructions: str, functions: List, model_name: str = "gpt-3.5-turbo"):
        self.name = name
        self.instructions = instructions
        self.functions = functions
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        self.memory = ConversationBufferMemory(return_messages=True)

class Swarm:
    def __init__(self):
        self.agents = {}
        self.vectorstore = None
        self.document_loaded = False
        self.document_content = ""

    def add_agent(self, agent: Agent):
        self.agents[agent.name] = agent

    def run(self, agent_name: str, query: str) -> str:
        agent = self.agents.get(agent_name)
        if not agent:
            return f"Agent {agent_name} not found"

        # Check for summarization request
        if any(word in query.lower() for word in ["summarize", "summary", "summarization"]):
            return summarize_text(agent, query, self)

        # Check for keyword extraction request
        if "keyword" in query.lower() or "extract keywords" in query.lower():
            return extract_keywords(agent, query, self)

        # Check for sentiment analysis request
        if "sentiment" in query.lower():
            return analyze_sentiment(agent, query, self)

        # For all other queries, use the answer_question function
        return answer_question(agent, query, self)

    def chat_with_llm(self, agent: Agent, query: str) -> str:
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", agent.instructions),
            ("human", "{input}"),
        ])
        chain = chat_prompt | agent.llm
        response = chain.invoke({"input": query})
        return response.content

    def load_documents(self, file_path: str):
        try:
            loader = TextLoader(file_path)
            documents = loader.load()
            self.document_content = "\n".join([doc.page_content for doc in documents])
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)
            embeddings = OpenAIEmbeddings()
            self.vectorstore = FAISS.from_documents(texts, embeddings)
            self.document_loaded = True
            print(f"Document loaded successfully. Content length: {len(self.document_content)} characters")
        except Exception as e:
            print(f"Error loading document: {str(e)}")
            self.document_loaded = False

    def search_documents(self, query: str, k: int = 3) -> List[str]:
        if not self.vectorstore:
            return ["No documents loaded. Please load documents first."]
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

def analyze_sentiment(agent: Agent, query: str, swarm: Swarm) -> str:
    if not swarm.document_loaded:
        return "No document has been loaded yet. Please load a document first using the 'load_documents' command."
    
    token_limit = 3000
    content = swarm.document_content[:8000]
    while num_tokens_from_string(content) > token_limit:
        content = content[:int(len(content)*0.9)]
    
    prompt = f"""
    Analyze the sentiment of the following text. Provide a brief explanation of your analysis.
    
    Text: {content}
    
    Sentiment analysis:
    """
    return agent.llm.invoke(prompt).content

def summarize_text(agent: Agent, query: str, swarm: Swarm) -> str:
    if not swarm.document_loaded:
        return "No document has been loaded yet. Please load a document first using the 'load_documents' command."
    
    token_limit = 3000
    content = swarm.document_content[:8000]
    while num_tokens_from_string(content) > token_limit:
        content = content[:int(len(content)*0.9)]
    
    prompt = f"""
    Provide a concise summary of the following text:
    
    {content}
    
    Summary:
    """
    return agent.llm.invoke(prompt).content

def extract_keywords(agent: Agent, query: str, swarm: Swarm) -> str:
    if not swarm.document_loaded:
        return "No document has been loaded yet. Please load a document first using the 'load_documents' command."
    
    token_limit = 3000
    content = swarm.document_content[:8000]
    while num_tokens_from_string(content) > token_limit:
        content = content[:int(len(content)*0.9)]
    
    prompt = f"""
    Extract the main keywords from the following text:
    
    {content}
    
    Keywords:
    """
    return agent.llm.invoke(prompt).content

def answer_question(agent: Agent, query: str, swarm: Swarm) -> str:
    if not swarm.document_loaded:
        return "No document has been loaded yet. Please load a document first using the 'load_documents' command."
    
    # Check if the query is about the document in general
    if "document" in query.lower() or "text" in query.lower():
        token_limit = 3000
        content = swarm.document_content[:8000]
        while num_tokens_from_string(content) > token_limit:
            content = content[:int(len(content)*0.9)]
        
        prompt = f"""
        Answer the following question based on the entire document content:

        Document content:
        {content}

        Question: {query}

        Answer:
        """
    else:
        # For specific questions, use the existing search functionality
        relevant_docs = swarm.search_documents(query)
        context = "\n\n".join(relevant_docs)
        prompt = f"""
        Answer the following question based on the provided context. If the answer is not in the context, say "I don't have enough information to answer this question."

        Context:
        {context}

        Question: {query}

        Answer:
        """
    
    return agent.llm.invoke(prompt).content

# Create text analysis agent
text_analysis_agent = Agent(
    name="Text Analyzer",
    instructions="You are an AI assistant specialized in text analysis. You can perform various tasks such as sentiment analysis, summarization, keyword extraction, and answering questions about given texts.",
    functions=[analyze_sentiment, summarize_text, extract_keywords, answer_question],
    model_name="gpt-3.5-turbo"
)

# Create Swarm
swarm = Swarm()
swarm.add_agent(text_analysis_agent)

# Main demo loop
def run_demo_loop():
    print("Starting Final LLM Swarm Text Analyzer")
    print("You can ask for sentiment analysis, summarization, keyword extraction, or ask questions about loaded documents.")
    print("Type 'load_documents' to load a text file for analysis.")
    print("Type 'exit' to quit")
    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'load_documents':
            file_path = input("Enter the path to the text file: ")
            swarm.load_documents(file_path)
        else:
            try:
                response = swarm.run("Text Analyzer", user_input)
                print(f"Text Analyzer: {response}")
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                print("Please try again with a different query.")

if __name__ == "__main__":
    run_demo_loop()
