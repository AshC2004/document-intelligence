import os
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from rag_chain import TechnicalQAChain, FastTechnicalQAChain


class DocumentIntelligenceRAG:
    def __init__(self, use_fast_mode: bool = True):
        load_dotenv()

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "document-intelligence")

        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))

        self.use_fast_mode = use_fast_mode

        self.doc_processor = DocumentProcessor(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.vector_store_manager = VectorStoreManager(
            api_key=self.pinecone_api_key,
            environment=self.pinecone_environment,
            index_name=self.index_name,
            embedding_model=self.embedding_model
        )

        self.qa_chain = None

    def index_documents(self, docs_directory: str, glob_pattern: str = "**/*.pdf"):
        print("Starting document indexing...")

        print("\n1. Processing and chunking documents...")
        chunks = self.doc_processor.process_documents(docs_directory, glob_pattern)

        if not chunks:
            print("No new documents found to index.")
            return

        print("\n2. Creating Pinecone index (if it doesn't exist)...")
        self.vector_store_manager.create_index()

        print("\n3. Adding documents to Pinecone...")
        self.vector_store_manager.add_documents(chunks)

        print("\nIndexing complete!")

    def initialize_qa_chain(self):
        vectorstore = self.vector_store_manager.get_vectorstore()

        if self.use_fast_mode:
            print("Initializing Fast QA Chain...")
            self.qa_chain = FastTechnicalQAChain(vectorstore=vectorstore)
        else:
            print("Initializing Standard QA Chain...")
            self.qa_chain = TechnicalQAChain(
                vectorstore=vectorstore,
                model_name=self.llm_model
            )

    def query(self, question: str) -> dict:
        if self.qa_chain is None:
            self.initialize_qa_chain()
        result = self.qa_chain.query(question)
        return result

    def stream_query(self, question: str):
        if self.qa_chain is None:
            self.initialize_qa_chain()

        print(f"\nQuestion: {question}\n")
        print("Answer: ", end="", flush=True)

        for chunk in self.qa_chain.stream_query(question):
            print(chunk, end="", flush=True)
        print("\n")

def main():
    rag = DocumentIntelligenceRAG(use_fast_mode=True)
    print("\n" + "="*60)
    print("Document Q&A System")
    print("="*60)
    print("Ask a question about your documents. Type 'quit' to exit.")

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if not question:
            continue
        rag.query(question)

if __name__ == "__main__":
    main()
