# Smart Document Assistant 🧠📄

A Streamlit-based application that turns your documents (PDF, DOCX, TXT) into an interactive question-answering and summarization assistant. This repository contains a lightweight yet powerful end-to-end demonstration of a modern Retrieval-Augmented Generation (RAG) system. It showcases document ingestion, text extraction, chunking, semantic embeddings, vector search, and answer generation using open-source Hugging Face models.

![Demo GIF](https://your-link-to-a-demo-gif.com/demo.gif)
*(**Note**: It is highly recommended to record a short GIF of your application in action and replace the link above.)*

---

## Table of Contents

-   [Project Overview](#project-overview)
-   [Key Features](#key-features)
-   [Architecture & Data Flow (Step-by-step)](#architecture--data-flow-step-by-step)
-   [Core ML / AI Concepts (Explained)](#core-ml--ai-concepts-explained)
    -   [Transformers & LLMs](#transformers--llms)
    -   [Pipelines (Hugging Face)](#pipelines-hugging-face-pipeline)
    -   [Embeddings & Semantic Search](#embeddings--semantic-search)
    -   [Vector Databases (Chroma)](#vector-databases-chroma)
    -   [Text Chunking & Overlap](#text-chunking--overlap)
    -   [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
-   [Project Structure and Important Files](#project-structure-and-important-files)
-   [Detailed Component Documentation (Code Walkthrough)](#detailed-component-documentation-code-walkthrough)
    -   [UI Layer (Streamlit)](#ui-layer-streamlit)
    -   [Model Loading & Caching](#model-loading--caching)
    -   [EnhancedDocumentProcessor](#enhanceddocumentprocessor)
    -   [AdvancedSummarizer](#advancedsummarizer)
    -   [SmartDocumentQA](#smartdocumentqa)
-   [Installation & Running Locally](#installation--running-locally)
-   [Configuration & Environment Variables](#configuration--environment-variables)
-   [Performance, Costs & Best Practices](#performance-costs--best-practices)
-   [Extending the Project / Roadmap](#extending-the-project--roadmap)
-   [Learning Outcomes](#learning-outcomes)
-   [Troubleshooting & FAQ](#troubleshooting--faq)
-   [License & Acknowledgements](#license--acknowledgements)

---

## Project Overview

The Smart Document Assistant is a developer-focused demo showing how to build an interactive document understanding assistant. The core problem it solves is making large, unstructured text documents accessible and "queryable" through natural language. Users upload documents via a simple web UI. The application then processes these documents to build a searchable knowledge base, allowing the user to ask questions and receive context-aware answers or summaries.

This demonstration uses local Hugging Face transformer pipelines (**BART** for summarization, a SQuAD-style **RoBERTa** for QA) and a small sentence-transformer embedding model (**all-MiniLM-L6-v2**), making it possible to run entirely on local hardware without reliance on external APIs.

---

## Key Features

-   **Multi-Format Document Ingestion**: Handles PDF, DOCX, and TXT files seamlessly.
-   **Robust Text Extraction**: Employs dedicated libraries for each file type to ensure high-quality text extraction.
-   **Intelligent Text Chunking**: Uses `RecursiveCharacterTextSplitter` to intelligently break down large texts while preserving context through chunk overlap.
-   **Semantic Search**: Goes beyond simple keyword matching by using sentence embeddings to find the most contextually relevant information.
-   **Dual AI Capabilities**:
    -   **Extractive Question-Answering**: Pinpoints and extracts precise answers from the text.
    -   **Abstractive Summarization**: Generates concise summaries of relevant document sections.
-   **Optimized User Experience**: Features a polished Streamlit UI, session-state preservation to remember conversation history, and intelligent model caching to eliminate wait times on reloads.
-   **Resilient Fallbacks**: Includes logic to handle model loading failures and defaults to a simpler search mechanism if semantic search encounters issues.

---

## Architecture & Data Flow (Step-by-step)

The application follows a logical, step-by-step pipeline from data ingestion to answer generation:

1.  **Upload**: The user uploads one or more files in the Streamlit sidebar UI.
2.  **Extraction**: Each file is read, and the `EnhancedDocumentProcessor` class extracts raw text using file-format-specific logic:
    -   **PDF**: `PyPDF2.PdfReader` iterates through pages and calls `.extract_text()`.
    -   **DOCX**: `python-docx` parses the document object model to extract text from paragraphs.
    -   **TXT**: A simple `.read()` and `.decode('utf-8')` is performed.
3.  **Chunking**: The extracted text is passed to LangChain's `RecursiveCharacterTextSplitter`, which creates chunks of approximately 1000 characters with an overlap of 150 characters. Each chunk is tagged with its source document name (e.g., `[Document: report.pdf]`).
4.  **Embedding & Indexing**: The `HuggingFaceEmbeddings` class (using the `sentence-transformers/all-MiniLM-L6-v2` model) converts each text chunk into a 384-dimensional vector. These embeddings are then stored (indexed) in an in-memory **Chroma** vector database.
5.  **QA System Initialization**: An instance of the `SmartDocumentQA` class is created. This class holds the vector store and loads the QA and summarization models into memory (leveraging Streamlit's cache).
6.  **Query Handling**: When a user submits a question, the `generate_answer` method is triggered:
    -   The user's question is first converted into an embedding vector.
    -   A **similarity search** is performed in Chroma to find the top-k most relevant text chunks.
    -   **Logic Branch**:
        -   If the query contains keywords like "summary" or "overview", the retrieved chunks are sent to the summarization model.
        -   Otherwise, the chunks are passed to the question-answering model to extract a direct answer. If this model fails or returns a low-confidence answer, the system falls back to summarizing the chunks.
7.  **Display**: The final answer is added to the session's conversation history and displayed in the UI in a reverse chronological feed.

---

## Core ML / AI Concepts (Explained)

### Transformers & LLMs

**Transformers** are a revolutionary neural network architecture that relies on a mechanism called **self-attention** to weigh the importance of different words in a sequence. This allows them to build a deep contextual understanding of language. Modern **Large Language Models (LLMs)** like GPT and LLaMA are built using this architecture. In this project, we use smaller, task-specific Transformer models from Hugging Face:
-   **`facebook/bart-large-cnn`**: A model fine-tuned specifically for summarizing text.
-   **`deepset/roberta-base-squad2`**: A model fine-tuned for extractive question-answering on the SQuAD2 dataset.

### Pipelines (Hugging Face pipeline)

The Hugging Face `pipeline` is a high-level API that dramatically simplifies the process of using pre-trained models. It abstracts away the complex steps of tokenization (converting text to numbers), model inference (feeding numbers through the model), and post-processing (converting model output back to human-readable text). We use it to load our QA and summarization models with a single line of code.

### Embeddings & Semantic Search

**Embeddings** are vector representations of text where semantic meaning is encoded as direction and distance. Our chosen model (`all-MiniLM-L6-v2`) is excellent at creating sentence-level embeddings. This enables **Semantic Search**, where we find documents based on meaning, not just keyword overlap. For example, a search for "How much money did we make?" will match chunks containing "The company's profit was..." because their embeddings are close in vector space.

### Vector Databases (Chroma)

A **Vector Database** is a specialized database designed to efficiently store and search through millions of embedding vectors. **Chroma** is a lightweight, open-source vector store that is perfect for local development and demos. It allows for blazing-fast similarity searches, which is the foundational step for our RAG system.

### Text Chunking & Overlap

LLMs have a fixed context window (a limit on how much text they can see at once). To process large documents, we must split them into smaller **chunks**. We use an **overlap** (e.g., 150 characters) between consecutive chunks to ensure that a sentence or idea that spans a chunk boundary isn't completely lost. `RecursiveCharacterTextSplitter` is a smart method that tries to split on natural boundaries (like paragraphs or sentences) first before resorting to a hard character cut.

### Retrieval-Augmented Generation (RAG)

**RAG** is the core pattern that makes this application "smart." It addresses a fundamental limitation of LLMs: they don't know about your private data. RAG enhances the LLM by providing it with relevant information on the fly. The process is:
1.  **Retrieve**: When a user asks a question, we first retrieve relevant document chunks using semantic search in our Chroma vector database.
2.  **Augment**: We then "augment" the user's original question by prepending the retrieved chunks as context.
3.  **Generate**: Finally, we send this combined prompt (context + question) to the LLM, which generates an answer based *only* on the information provided.

---



---

## Detailed Component Documentation (Code Walkthrough)

### UI Layer (Streamlit)

-   **`main()` function**: Initializes the page configuration (`st.set_page_config`), applies custom CSS, and manages the session state (`st.session_state`).
-   **`create_sidebar()`**: Manages all sidebar elements, including the `st.file_uploader` and the "Process Documents" and "Clear Chat" buttons. This is the main user input area for documents.
-   **State Management**: `st.session_state` is used heavily to persist the `qa_system`, `conversation_history`, and `processed_files` list across page re-runs. This is crucial for preventing the entire application from resetting on every button click.

### Model Loading & Caching

-   **`@st.cache_resource`**: This Streamlit decorator is the key to performance. It tells Streamlit to run the decorated function (e.g., `load_qa_model`) only once. The result (the loaded model) is stored in a cache. On subsequent re-runs, Streamlit simply retrieves the model from the cache instead of re-downloading and loading it, saving significant time.
-   **Fallback Logic**: The `load_summarizer_model` function includes a `try...except` block. If the preferred, larger BART model fails to load (e.g., due to memory constraints), it gracefully falls back to a smaller, more efficient DistilBART model.

### `EnhancedDocumentProcessor`

This class is a simple, focused utility for data extraction. Its purpose is to isolate the logic for handling different file types.
-   `extract_text_from_pdf()`: Uses `PyPDF2` to read a PDF file object.
-   `extract_text_from_docx()`: Uses `python-docx` to read a DOCX file object.
-   `extract_text_from_txt()`: A straightforward method to read a plain text file.

### `AdvancedSummarizer`

This class wraps the summarization pipeline and contains logic for handling long texts.
-   `generate_contextual_summary()`: If a text is too long for the model, it first chunks the text, summarizes each chunk individually, and then runs a final summarization pass on the combined summaries to produce a coherent final output.
-   `extract_key_topics()`: A clever use of the summarization model. By prompting it to "Summarize the main topics...", it can produce a list of key themes from a document.

### `SmartDocumentQA`

This is the main orchestrator class that ties everything together.
-   **`__init__()`**: The constructor initializes all components: it organizes the text chunks, loads the AI models (via the cached functions), and sets up the Chroma vector store by calling `setup_embeddings()`.
-   **`setup_embeddings()`**: This method creates the `HuggingFaceEmbeddings` object and uses the `Chroma.from_texts()` utility to automatically process all text chunks and build the searchable vector index.
-   **`generate_answer()`**: This is the core reasoning engine. It first checks for summarization keywords. If none are found, it performs the RAG process: it calls `search_documents()` to retrieve context and then passes that context to the QA model. It includes a confidence threshold (`result['score'] > 0.1`) to filter out low-quality answers.

---

## Installation & Running Locally

```

### 3. Install Dependencies
Create a `requirements.txt` file with the content below, then run the pip install command.

**`requirements.txt`**:
```text
streamlit
python-dotenv
PyPDF2
python-docx
langchain
langchain-community
sentence-transformers
chromadb
transformers
torch
nltk
accelerate
```

**Installation Command**:
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App
```bash
streamlit run app.py
```

---

## Configuration & Environment Variables

This project uses `python-dotenv` to load environment variables, though none are required for the current local-only setup. If you were to extend this to use an API-based model (like OpenAI), you would create a `.env` file in the root directory:

```
# .env file
OPENAI_API_KEY="your_api_key_here"
```

---

## Performance, Costs & Best Practices

-   **Performance**: The app's performance depends on your hardware (CPU/GPU). The initial model download and caching will take time and require an internet connection. Subsequent runs will be much faster. Using a GPU (`device=0`) will significantly speed up inference.
-   **Costs**: This project is **100% free** to run locally as it relies on open-source models. If extended to use commercial APIs, you would incur costs based on token usage.
-   **Best Practices**:
    -   **Caching**: Using `@st.cache_resource` is essential for responsive Streamlit apps that load large objects like models.
    -   **Virtual Environments**: Always use a virtual environment to manage dependencies and avoid conflicts.
    -   **Error Handling**: The code includes `try...except` blocks for model loading and document parsing, which is crucial for a robust user experience.

---

## Extending the Project / Roadmap

-   [ ] **Integrate a Generative LLM**: Replace the extractive QA model with a powerful generative model (like a local LLaMA or an API like GPT-4) to provide more conversational, synthesized answers.
-   [ ] **Metadata Filtering**: Store metadata (e.g., file name, page number) with the vectors in Chroma to allow users to filter their search to specific documents.
-   [ ] **Dockerize the Application**: Create a `Dockerfile` to containerize the app and its dependencies for easy, reproducible deployment.
-   [ ] **Asynchronous Processing**: For very large documents, move the chunking and embedding process to a background task to keep the UI responsive.

---

## Learning Outcomes

By building and studying this project, you will gain practical experience with:
-   **Implementing a full RAG pipeline** from ingestion to generation.
-   **Using open-source models** from Hugging Face for core NLP tasks.
-   The critical role of **text embeddings** and **vector databases** in modern AI.
-   Building interactive and performant AI applications with **Streamlit**.

---

## Troubleshooting & FAQ

-   **Error `numpy.dtype size changed`**: This is a common dependency conflict. Your environment is likely corrupted. The best solution is to delete your virtual environment, recreate it, and reinstall the dependencies from `requirements.txt`.
-   **First run is very slow**: This is expected. The application needs to download several gigabytes of model files from Hugging Face. Subsequent runs will be fast due to caching.
-   **Application is using a lot of RAM**: The loaded models are large and reside in memory. Ensure your machine has at least 16GB of RAM for a smooth experience, especially with the larger BART model.

---

