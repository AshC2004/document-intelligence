"""
A few examples to show how to use the RAG system.
You can run them one by one to see what the script does.
"""

import os
import sys
import time

# Add src to path so we can import from our own modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import DocumentIntelligenceRAG


def example_1_index_documents():
    """Example 1: Index documents into the vector store."""
    print("="*60)
    print("Example 1: Indexing Documents")
    print("="*60)

    rag = DocumentIntelligenceRAG(use_fast_mode=True)

    # Index all text files in the docs directory
    rag.index_documents("docs", "**/*.txt")

    print("\n✓ Documents indexed successfully!")


def example_2_single_query():
    """Example 2: Single query with timing."""
    print("\n" + "="*60)
    print("Example 2: Single Technical Query")
    print("="*60)

    rag = DocumentIntelligenceRAG(use_fast_mode=True)

    question = "What authentication method does the API use?"

    result = rag.query(question, verbose=True)

    print(f"\nLatency: {result['latency']:.3f}s")
    print(f"Sub-2s: {'✓ Yes' if result['latency'] < 2.0 else '✗ No'}")


def example_3_multiple_queries():
    """Example 3: Multiple queries with latency tracking."""
    print("\n" + "="*60)
    print("Example 3: Multiple Queries with Latency Tracking")
    print("="*60)

    rag = DocumentIntelligenceRAG(use_fast_mode=True)

    questions = [
        "What are the main components of the microservices architecture?",
        "How does rate limiting work in the API?",
        "What is the database backup frequency?",
        "Explain the CI/CD pipeline stages"
    ]

    latencies = []

    for i, question in enumerate(questions, 1):
        print(f"\n{'─'*60}")
        print(f"Query {i}: {question}")
        print('─'*60)

        result = rag.query(question, verbose=False)

        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nLatency: {result['latency']:.3f}s")

        latencies.append(result['latency'])

    # Summary
    print("\n" + "="*60)
    print("Latency Summary")
    print("="*60)
    avg_latency = sum(latencies) / len(latencies)
    print(f"Average latency across {len(questions)} questions: {avg_latency:.3f}s")


def example_4_streaming():
    """Example 4: Streaming response for better UX."""
    print("\n" + "="*60)
    print("Example 4: Streaming Response")
    print("="*60)

    rag = DocumentIntelligenceRAG(use_fast_mode=True)

    question = "What technologies are used for monitoring?"

    rag.stream_query(question)


def example_5_interactive():
    """Example 5: Interactive Q&A session."""
    print("\n" + "="*60)
    print("Example 5: Interactive Technical Q&A")
    print("="*60)

    rag = DocumentIntelligenceRAG(use_fast_mode=True)

    print("\nStarting interactive session...")
    print("Ask me anything about the docs! Type 'quit' to exit.")

    while True:
        question = input("\n> ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("Bye!")
            break

        if not question:
            continue

        result = rag.query(question, verbose=False)
        print(f"\nAnswer:\n{result['answer']}")


if __name__ == "__main__":
    # Run all examples sequentially
    example_1_index_documents()
    time.sleep(1)
    example_2_single_query()
    time.sleep(1)
    example_3_multiple_queries()
    time.sleep(1)
    example_4_streaming()

    # To run the interactive example, uncomment the line below
    # example_5_interactive()
