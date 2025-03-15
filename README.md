# Personal Knowledge AI Assistant

## Overview
Knowledge Navigator is an intelligent assistant that interfaces with your personal knowledge management system, transforming your digital notes into an interactive AI-powered experience. By leveraging advanced language models, it helps you extract, analyze, and utilize information from your existing knowledge base.

## What You'll Learn
The project is based on a series of articles by Paul Lusztin and the book LLM Engineer's Handbook and covers a wide range of modern practices in artificial intelligence and software engineering:

### AI Architecture & MLOps
- Design and implement systems using the FTI (Flexible, Transparent, Iterative) architecture
- Apply MLOps best practices including data registries, model registries, and experiment tracking
- Use Crawl4AI for efficient web crawling and Markdown normalization
- Implement LLM-based quality scoring mechanisms
- Generate and manage summarization datasets through distillation techniques

### Model Development & Deployment
- Fine-tune Llama models using Unsloth and Comet
- Deploy models to Hugging Face serverless Dedicated Endpoints
- Implement advanced RAG (Retrieval-Augmented Generation) algorithms featuring:
  - Contextual retrieval
  - Hybrid search systems
  - MongoDB vector search integration

### Agent Development & Tools
- Build multi-tool agents using Hugging Face's smolagents framework
- Implement LLMOps best practices including:
  - Prompt monitoring
  - RAG evaluation using Opik
- Integrate pipeline orchestration using ZenML
- Track artifacts and metadata effectively

### Software Engineering
- Modern Python project management using uv and ruff
- Implementation of software engineering best practices
- Efficient metadata and artifact tracking
- Version control and collaboration workflows

## Features
- **Notion Integration**: Seamlessly connects with your Notion workspace to access and process your existing notes and documents
- **Contextual Understanding**: Analyzes your provided content to generate relevant and accurate responses
- **Smart Query Processing**: Intelligently interprets your questions and retrieves information from your personal knowledge base
- **Natural Language Interaction**: Communicate with your knowledge base using everyday language
- **Privacy-First Approach**: Your data remains secure and private, processed locally when possible

## Flow of the data

1. Collect raw Notion documents in Markdown format.

2. Crawl each link in the Notion documents and normalize them in Markdown.

3. Store a snapshot of the data in a NoSQL database.

4. For fine-tuning, filter the documents more strictly to narrow the data to only high-quality samples.

5. Use the high-quality samples to distillate a summarization instruction dataset, which we store in a data registry.

6. Using the generated dataset, fine-tune an open-source LLM, which will be save in a model registry.

7. In parallel, use a different filter threshold for RAG to narrow down the documents to medium to high-quality samples.

8. Chunk, embed, plus other advanced RAG preprocessing steps to optimize the retrieval of the documents.

9. Load the embedded chunks and their metadata in a vector database.

10. Leveraging the vector database, we use semantic search to retrieve the top K most relevant chunks relative to a user query.

## AI Assistant Architecture

1. The data pipelines

2. The feature pipelines

3. The training pipeline

4. The inference pipelines

5. The observability pipeline
