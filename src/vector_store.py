import time
from typing import List, Optional
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


class VectorStoreManager:
    def __init__(
        self,
        api_key: str,
        environment: str,
        index_name: str,
        embedding_model: str = "text-embedding-3-small"
    ):
        self.index_name = index_name
        self.pc = Pinecone(api_key=api_key)

        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            chunk_size=1000
        )

    def create_index(self, dimension: int = 1536, metric: str = "cosine"):
        existing_indexes = [index.name for index in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            print(f"Creating index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
            print(f"Index {self.index_name} created successfully")
        else:
            print(f"Index {self.index_name} already exists")

    def add_documents(self, documents: List[Document], batch_size: int = 100) -> PineconeVectorStore:
        print(f"Adding {len(documents)} documents to Pinecone...")

        vectorstore = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=self.embeddings,
            index_name=self.index_name,
            batch_size=batch_size
        )

        print("Documents added successfully")
        return vectorstore

    def get_vectorstore(self) -> PineconeVectorStore:
        return PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None
    ) -> List[Document]:
    
        vectorstore = self.get_vectorstore()
        return vectorstore.similarity_search(query, k=k, filter=filter)

    def delete_index(self):
        self.pc.delete_index(self.index_name)
        print(f"Index {self.index_name} deleted")
