# Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies (1 min)

```bash
pip install -r requirements.txt
```

### 2. Configure Environment (2 min)

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```env
OPENAI_API_KEY=sk-your-key-here
PINECONE_API_KEY=your-key-here
PINECONE_ENVIRONMENT=us-east-1-aws
```

### 3. Index Sample Documents (1 min)

```python
from src.main import DocumentIntelligenceRAG

rag = DocumentIntelligenceRAG(use_fast_mode=True)
rag.index_documents("docs", "**/*.txt")
```

### 4. Query the System (1 min)

```python
result = rag.query("What authentication method does the API use?")
print(result['answer'])
```

## Key Commands

```bash
# Run interactive mode
python src/main.py

# Run example 1 (indexing)
python example_usage.py 1

# Run example 2 (single query)
python example_usage.py 2

# Run example 5 (interactive Q&A)
python example_usage.py 5
```

## API Keys Setup

### Get OpenAI API Key
1. Go to https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Copy the key to `.env`

### Get Pinecone API Key
1. Go to https://app.pinecone.io/
2. Create a free account
3. Go to "API Keys" section
4. Copy the key and environment to `.env`

## Testing Sub-2s Latency

```python
rag = DocumentIntelligenceRAG(use_fast_mode=True)
result = rag.query("Your question here", verbose=True)

if result['latency'] < 2.0:
    print(f"✓ Sub-2s: {result['latency']:.3f}s")
```

## Customization

### Use Better Quality (Slower)
```python
rag = DocumentIntelligenceRAG(use_fast_mode=False)
```

### Change Models in .env
```env
# Faster, cheaper
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-3.5-turbo

# Better quality
EMBEDDING_MODEL=text-embedding-3-large
LLM_MODEL=gpt-4-turbo-preview
```

## Troubleshooting

**Error: "No module named 'src'"**
```bash
# Run from project root directory
cd document-intelligence-rag
python example_usage.py 2
```

**Error: "Index not found"**
```bash
# Index documents first
python example_usage.py 1
```

**Slow responses (>2s)**
- Use `use_fast_mode=True`
- Set `LLM_MODEL=gpt-3.5-turbo` in `.env`
- Reduce `retrieval_k` in `src/rag_chain.py`

## Next Steps

1. Add your own documents to `docs/` folder
2. Run indexing: `python example_usage.py 1`
3. Start querying: `python example_usage.py 5`
4. Customize prompts in `src/rag_chain.py`
5. Adjust configuration in `.env`

---

Ready to use! Check `README.md` for detailed documentation.
