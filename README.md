# Adaptive RAG System

An intelligent Retrieval-Augmented Generation (RAG) system that dynamically adjusts its retrieval parameters based on query types and performance feedback. This system enhances the quality of responses by adapting to different types of questions and learning from evaluation metrics.

## Key Features

- Dynamic document retrieval adjustment based on query complexity
- Automatic query type classification (Factual, Conceptual, Analytical)
- Real-time performance evaluation and feedback loop
- Adaptive k-parameter tuning for optimal context retrieval
- Comprehensive response evaluation metrics

## Requirements

- Python 3.8+
- OpenAI API key
- FAISS for efficient similarity search
- NumPy for numerical operations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ferrerallan/adaptive-rag.git
cd adaptive-rag
```

2. Install required packages:
```bash
pip install openai faiss-cpu numpy python-dotenv
```

3. Configure environment variables:
Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_KEY=your-api-key-here
```

## How It Works

The Adaptive RAG system operates through several sophisticated mechanisms:

1. **Document Embedding**: When documents are added to the system, they are converted into embeddings using OpenAI's text-embedding-ada-002 model and stored in a FAISS index for efficient similarity search.

2. **Query Classification**: Each incoming question is automatically classified into one of three types:
   - Factual: Questions requiring specific facts or data
   - Conceptual: Questions requiring broader understanding
   - Analytical: Questions requiring synthesis of multiple pieces of information

3. **Dynamic Parameter Adjustment**: The system adjusts its retrieval parameters (k) based on:
   - Query type classification
   - Historical performance feedback
   - Response evaluation metrics

4. **Response Generation**: The system:
   - Retrieves relevant documents based on the current k value
   - Generates responses using GPT-4
   - Evaluates response quality using multiple metrics
   - Adjusts parameters for future queries based on performance

## Usage

Here's a basic example of how to use the system:

```python
from adaptive_rag import AdaptiveRAG
import os
from dotenv import load_dotenv

# Initialize the system
load_dotenv()
rag = AdaptiveRAG(os.getenv("OPENAI_KEY"))

# Add documents to the knowledge base
documents = [
    "Python is a high-level programming language known for its simplicity.",
    "Python was created by Guido van Rossum in 1991.",
    "Python is widely used in data science and machine learning."
]
rag.add_documents(documents)

# Query the system
result = rag.query("When was Python created?")

# Access the results
print(f"Response: {result['response']}")
print(f"Query type: {result['query_type']}")
print(f"Context used: {result['context']}")
print(f"Evaluation: {result['evaluation']}")
```

## Performance Metrics

The system evaluates responses using three key metrics:

1. **Relevance** (1-10): How well the response addresses the question
2. **Context Usage** (1-10): How effectively the system uses the provided context
3. **Accuracy** (1-10): The factual correctness of the response

These metrics are used to automatically adjust the system's parameters for optimal performance.

## Advanced Configuration

The system allows customization of several parameters:

- `k_range`: Minimum and maximum number of documents to retrieve (default: 2-5)
- `current_k`: Initial number of documents to retrieve (default: 3)
- `dimension`: Embedding dimension (default: 1536 for Ada-002)

Example of custom configuration:

```python
rag = AdaptiveRAG(api_key)
rag.k_range = (3, 7)  # Customize document retrieval range
rag.current_k = 4     # Set initial k value
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

