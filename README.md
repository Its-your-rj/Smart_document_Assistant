# Smart Document Assistant

## Overview

Smart Document Assistant is an interactive Streamlit application that enables users to upload multiple documents (PDF, DOCX, TXT), ask natural language questions about their content, and receive intelligent, context-aware answers and summaries. The project leverages modern NLP and deep learning techniques, including Large Language Models (LLMs), Transformers, Retrieval-Augmented Generation (RAG), vector databases, embeddings, and prompt engineering.

---

## Core Concepts & Technologies

1. **Large Language Models (LLMs)**
   - Utilizes transformer-based models (Hugging Face's BART for summarization, RoBERTa for question answering).
   - Capable of understanding and generating human-like text responses.

2. **Transformers**
   - Employs Hugging Face's `transformers` library for state-of-the-art NLP tasks.
   - Models used: `facebook/bart-large-cnn` for summarization, `deepset/roberta-base-squad2` for QA.

3. **Retrieval-Augmented Generation (RAG)**
   - Combines semantic search (retrieval) with generative models to answer queries based on relevant document chunks.
   - Ensures responses are grounded in the actual content of uploaded documents.

4. **Vector Databases**
   - Uses FAISS for efficient similarity search over document embeddings.
   - Enables fast retrieval of semantically similar text segments.

5. **Embeddings**
   - Generates vector representations of text chunks using `sentence-transformers/all-MiniLM-L6-v2`.
   - Facilitates semantic search and matching between user queries and document content.

6. **Prompt Engineering**
   - Designs prompts for summarization and topic extraction to guide LLMs for more accurate and relevant outputs.

7. **Model Evaluation Frameworks**
   - Implements basic evaluation via user feedback and conversation history.
   - Can be extended for more formal evaluation metrics (accuracy, relevance, etc.).

---

## Sequential Workflow

1. **Document Upload**
   - Users upload one or more documents in PDF, DOCX, or TXT format via the Streamlit sidebar.

2. **Text Extraction**
   - The app reads and extracts text from each document using PyPDF2 (PDF), python-docx (DOCX), or standard decoding (TXT).

3. **Text Chunking**
   - Extracted text is split into manageable chunks using LangChain's `RecursiveCharacterTextSplitter` for efficient processing.

4. **Embeddings Generation**
   - Each chunk is converted into a vector embedding using Hugging Face's Sentence Transformers.

5. **Vector Store Creation**
   - All embeddings are stored in a FAISS vector database for fast similarity search.

6. **Semantic Search**
   - When a user asks a question, the app retrieves the most relevant chunks using semantic similarity.

7. **Response Generation**
   - For fact-based queries, the QA model (`roberta-base-squad2`) extracts direct answers from relevant chunks.
   - For open-ended or summary-type queries, the summarization model (`bart-large-cnn`) generates comprehensive summaries or topic lists.

8. **Conversation History**
   - All questions and answers are stored and displayed, allowing users to track their interactions.

---

## Learning Outcomes

- **Understand the end-to-end workflow of a modern NLP application.**
- **Gain hands-on experience with LLMs and transformer models for QA and summarization.**
- **Learn how to implement Retrieval-Augmented Generation (RAG) for document Q&A.**
- **Explore the use of vector databases (FAISS) for semantic search.**
- **Apply prompt engineering techniques to guide LLM outputs.**
- **Integrate multiple NLP tools and frameworks (LangChain, Hugging Face, Streamlit) in a single project.**
- **Evaluate and improve model responses based on user queries and feedback.**

---




## Example Use Cases

- Summarize lengthy research papers or reports.
- Extract key findings or topics from multiple documents.
- Ask specific questions and get direct answers from document content.
- Compare and analyze information across several files.

---


---

## License

This project is licensed under the MIT License.

---
