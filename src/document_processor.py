"""Document processing and chunking for RAG pipeline."""

import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader
)
from langchain.schema import Document


class DocumentProcessor:
    """Handles document loading and chunking for the RAG pipeline."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        """Load a single PDF file."""
        loader = PyPDFLoader(file_path)
        return loader.load()

    def load_text(self, file_path: str) -> List[Document]:
        """Load a single text file."""
        loader = TextLoader(file_path)
        return loader.load()

    def load_directory(self, directory_path: str, glob_pattern: str = "**/*.pdf") -> List[Document]:
        """
        Load all documents from a directory.

        Args:
            directory_path: Path to directory containing documents
            glob_pattern: Pattern to match files (default: PDF files)
        """
        if glob_pattern.endswith('.pdf'):
            loader = DirectoryLoader(
                directory_path,
                glob=glob_pattern,
                loader_cls=PyPDFLoader
            )
        else:
            loader = DirectoryLoader(
                directory_path,
                glob=glob_pattern,
                loader_cls=TextLoader
            )
        return loader.load()

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunked documents
        """
        return self.text_splitter.split_documents(documents)

    def process_documents(self, directory_path: str, glob_pattern: str = "**/*.pdf") -> List[Document]:
        """
        Complete pipeline: load and chunk documents.

        Args:
            directory_path: Path to directory containing documents
            glob_pattern: Pattern to match files

        Returns:
            List of chunked documents ready for embedding
        """
        documents = self.load_directory(directory_path, glob_pattern)
        chunks = self.chunk_documents(documents)

        print(f"Loaded {len(documents)} documents")
        print(f"Created {len(chunks)} chunks")

        return chunks
