# RAG Pipeline for Technical Q&A

This is my final year project. It's a simple Question & Answer system that uses Retrieval-Augmented Generation (RAG) to answer questions about technical documents. I built it using Python, LangChain, and Pinecone.

The main idea is to let a user ask a question in plain English and get an answer from a set of documents.

## What it Does

*   Loads PDF or text documents from a directory.
*   Splits the documents into smaller chunks.
*   Uses OpenAI's models to create embeddings (numerical representations) of the chunks.
*   Stores these embeddings in a Pinecone vector database.
*   When a user asks a question, it finds the most relevant document chunks from Pinecone.
*   It then feeds the question and the relevant chunks to an LLM (like GPT-3.5) to generate an answer.

## How to Run

### Prerequisites

- Python 3.8+
- An OpenAI API key
- A Pinecone API key

### Installation

1.  **Clone the project:**
    ```bash
    git clone https://github.com/yourusername/document-intelligence-rag.git
    cd document-intelligence-rag
    ```

2.  **Install dependencies:**
    It's a good idea to use a virtual environment!
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Set up your API keys:**
    Copy the `.env.example` file to a new file called `.env`.
    ```bash
    cp .env.example .env
    ```
    Then, open `.env` and paste in your API keys.

### Usage

1.  **Add your documents:**
    Place any PDF or TXT files you want to query inside the `docs/` directory.

2.  **Run the main script:**
    This will start an interactive session where you can ask questions.
    ```bash
    python src/main.py
    ```
    The first time you run it, you might need to uncomment the `rag.index_documents(...)` line in `src/main.py` to index your files.

## What I Learned

This was my first time building a full RAG pipeline, and I learned a lot:
- **Embeddings are key:** The quality of the embeddings makes a huge difference in retrieval. I used OpenAI's `text-embedding-3-small` which seemed like a good balance of cost and performance.
- **Prompt Engineering is hard:** Getting the prompt right for the final answer generation took a lot of trial and error. I ended up using a simple Chain-of-Thought style prompt.
- **LangChain is powerful but complex:** There's a lot to learn in LangChain. I used the new LCEL syntax which is really cool for building chains.
- **Vector databases:** I used Pinecone because they have a free tier. It was pretty easy to get started with.

## Future Ideas

- Try to build a simple web interface for it using Flask or FastAPI.
- Experiment with other open-source embedding models instead of OpenAI.
- See if I can get it working with a local vector database like FAISS to avoid needing Pinecone.
- Improve the prompt to handle more complex questions.

## Acknowledgments

- This project was heavily inspired by the tutorials and documentation from [LangChain](https://github.com/langchain-ai/langchain) and [Pinecone](https://www.pinecone.io/).
