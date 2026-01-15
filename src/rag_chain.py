import time
from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


class TechnicalQAChain:
    """The main RAG chain for answering technical questions."""

    CHAIN_OF_THOUGHT_TEMPLATE = """
    You are a technical expert assistant. Answer the question using the provided context documents.
    Use chain-of-thought reasoning:
    1. First, identify the key technical concepts in the question
    2. Then, analyze the relevant information from the context
    3. Finally, provide a clear, accurate answer
    
    Context Documents: {context}
    Question: {question}
    Technical Analysis:Let me break this down step by step:
    1. Key Concepts: [Identify the main technical concepts in the question]
    2. Relevant Information: [Extract and analyze relevant details from the context]
    3. Answer: [Provide a clear, comprehensive answer]
    Please provide your response following this chain-of-thought structure.
    """
    def __init__(
        self,
        vectorstore,
        model_name: str = "gpt-4-turbo-preview",
        temperature: float = 0,
        k: int = 4  # Number of docs to retrieve
    ):
        self.vectorstore = vectorstore
        self.docs_to_retrieve = k

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            streaming=True,
            max_tokens=1000
        )

        self.prompt = ChatPromptTemplate.from_template(self.CHAIN_OF_THOUGHT_TEMPLATE)

        self.chain = self._build_chain()

    def _format_docs(self, docs: List[Document]) -> str:
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content.strip()
            formatted.append(f"Document {i} (Source: {source}):\n{content}")
        return "\n\n".join(formatted)

    def _build_chain(self):
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.docs_to_retrieve}
        )

        chain = (
            {
                "context": retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def query(self, question: str) -> Dict[str, any]:
        start_time = time.time()
        docs = self.vectorstore.similarity_search(question, k=self.docs_to_retrieve)
        answer = self.chain.invoke(question)
        end_time = time.time()
        total_time = end_time - start_time

        print(f"\nQuestion: {question}")
        print(f"\nAnswer:\n{answer}")
        print(f"\nRetrieved {len(docs)} relevant documents.")

        return {
            "answer": answer,
            "latency": total_time,
            "documents": docs,
        }

    def stream_query(self, question: str):
        for chunk in self.chain.stream(question):
            yield chunk

class FastTechnicalQAChain(TechnicalQAChain):
    """A faster version of the chain that uses a simpler prompt and model."""

    FAST_COT_TEMPLATE = """
    Answer this technical question using the context.
    Context: {context}
    Question: {question}
    Answer:
    """

    def __init__(self, vectorstore, **kwargs):
        kwargs['model_name'] = 'gpt-3.5-turbo'
        kwargs['k'] = 3  # Retrieve fewer documents
        super().__init__(vectorstore, **kwargs)

        self.prompt = ChatPromptTemplate.from_template(self.FAST_COT_TEMPLATE)
        self.chain = self._build_chain()
