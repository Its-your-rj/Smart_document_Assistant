import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings # Corrected import path
from langchain_community.vectorstores import Chroma
from io import BytesIO
from transformers import pipeline
import re
import torch
from datetime import datetime
import textwrap

# Download NLTK data if available
try:
    import nltk
    nltk.download('punkt', quiet=True)
except ImportError:
    st.warning("NLTK not found. Using basic text splitting. For best results, run: pip install nltk")

# --- UI Styling ---
enhanced_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem; border-radius: 12px; margin-bottom: 2rem;
        color: white; text-align: center; box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 { font-size: 2.5rem; font-weight: 700; }
    .question-box {
        background: #e3f2fd; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;
        border-left: 4px solid #2196f3; box-shadow: 0 4px 12px rgba(33, 150, 243, 0.15);
    }
    .answer-box-container {
        background: #f3e5f5; padding: 2rem; border-radius: 10px; margin: 1rem 0;
        border-left: 4px solid #9c27b0; box-shadow: 0 4px 12px rgba(156, 39, 176, 0.15);
        line-height: 1.6;
    }
    .document-item {
        background: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
        border: 1px solid #dee2e6;
    }
    .alert-info {
        background: #e7f3ff; border: 1px solid #b8daff; border-radius: 8px;
        padding: 1rem; margin: 1rem 0; color: #0c5460;
    }
    .stDeployButton, #MainMenu { display: none; }
</style>
"""

# --- Model Loading with Caching ---
# IMPROVEMENT: Caching models prevents them from being reloaded on every interaction,
# significantly speeding up the app after the first run.

@st.cache_resource
def load_summarizer_model():
    """Loads a summarization model with a fallback."""
    try:
        st.info("Loading summarization model (BART)...")
        model = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if torch.cuda.is_available() else -1,
        )
        st.success("Summarization model loaded!")
        return model
    except Exception as e:
        st.warning(f"BART model failed: {e}. Trying fallback (DistilBART)...")
        try:
            model = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",
                device=-1,
            )
            st.success("Fallback summarization model loaded.")
            return model
        except Exception as e2:
            st.error(f"Could not load any summarization models. Error: {e2}")
            return None

@st.cache_resource
def load_qa_model():
    """Loads a question-answering model."""
    try:
        model = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=0 if torch.cuda.is_available() else -1
        )
        return model
    except Exception as e:
        st.warning(f"Could not load QA model: {e}")
        return None

# --- Core Classes ---

class AdvancedSummarizer:
    def __init__(self):
        self.summarizer_model = load_summarizer_model()

    def generate_contextual_summary(self, text, max_length=150, min_length=40):
        if not self.summarizer_model:
            return "Summarization model is not available."
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        chunks = text_splitter.split_text(text)
        summaries = []
        for chunk in chunks:
            try:
                summary = self.summarizer_model(
                    chunk,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    truncation=True
                )[0]['summary_text']
                summaries.append(summary)
            except Exception:
                continue
        if not summaries:
            return ' '.join(text.split()[:100]) + "..."
        
        aggregated = " ".join(summaries)
        
        # If the aggregated summary is already short, just return it.
        if len(aggregated.split()) <= max_length:
            return aggregated
            
        try:
            final_summary = self.summarizer_model(
                aggregated,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )[0]['summary_text']
            return final_summary
        except Exception:
            return aggregated

    def extract_key_topics(self, text):
        if not self.summarizer_model:
            return ["Topic extraction model is not available."]
        
        # Using a simpler prompt for better compatibility with summarization models
        prompt = f"Summarize the main topics in the following text: {text[:2000]}"
        try:
            result = self.summarizer_model(prompt, max_length=100, min_length=20, do_sample=False)[0]['summary_text']
            # Simple sentence splitting as a proxy for topics
            topics = [topic.strip() for topic in result.split('.') if topic.strip()]
            return topics if topics else ["Could not determine specific topics."]
        except Exception as e:
            st.warning(f"Topic extraction failed: {e}")
            return ["Topic extraction failed."]

class EnhancedDocumentProcessor:
    def extract_text_from_pdf(self, pdf_file):
        try:
            pdf_file.seek(0)
            reader = PdfReader(BytesIO(pdf_file.read()))
            return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        except Exception as e:
            st.error(f"Error reading PDF {pdf_file.name}: {e}")
            return ""

    def extract_text_from_docx(self, docx_file):
        try:
            docx_file.seek(0)
            doc = docx.Document(BytesIO(docx_file.read()))
            return "\n".join(para.text for para in doc.paragraphs if para.text.strip())
        except Exception as e:
            st.error(f"Error reading DOCX {docx_file.name}: {e}")
            return ""

    def extract_text_from_txt(self, txt_file):
        try:
            txt_file.seek(0)
            return txt_file.read().decode('utf-8')
        except Exception as e:
            st.error(f"Error reading TXT {txt_file.name}: {e}")
            return ""

class SmartDocumentQA:
    def __init__(self, text_chunks, document_info):
        self.chunks = text_chunks
        self.doc_info = document_info
        self.vectorstore = None
        self.summarizer = AdvancedSummarizer()
        self.qa_model = load_qa_model() # Using the cached model loader
        self.documents_dict = self._organize_chunks_by_document()
        self.setup_embeddings()
        
    def _organize_chunks_by_document(self):
        documents = {}
        for chunk in self.chunks:
            doc_name = "Uploaded Document"
            match = re.search(r'\[Document: ([^\]]+)\]', chunk)
            if match:
                doc_name = match.group(1)
            
            clean_chunk = re.sub(r'\[Document: [^\]]+\]', '', chunk).strip()
            if doc_name not in documents:
                documents[doc_name] = []
            if clean_chunk:
                documents[doc_name].append(clean_chunk)
        return documents

    def setup_embeddings(self):
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            if self.chunks:
                self.vectorstore = Chroma.from_texts(texts=self.chunks, embedding=embeddings)
        except Exception as e:
            st.warning(f"Could not setup semantic search: {e}. Falling back to keyword search.")

    def get_document_summary(self):
        if not self.documents_dict:
            return "No documents available to summarize."
        summaries = []
        for doc_name, doc_chunks in self.documents_dict.items():
            full_text = " ".join(doc_chunks)
            word_count = len(full_text.split())
            
            summary_parts = [f"## 📄 Summary for: **{doc_name}**", f"*~{word_count:,} words*"]
            
            executive_summary = self.summarizer.generate_contextual_summary(full_text, max_length=200)
            summary_parts.extend(["### Executive Summary", executive_summary])
            
            key_topics = self.summarizer.extract_key_topics(full_text)
            summary_parts.append("### Key Topics")
            summary_parts.append("\n".join([f"- {topic}" for topic in key_topics]))
            
            summaries.append("\n\n".join(summary_parts))
        return "\n\n---\n\n".join(summaries)

    def search_documents(self, question, max_results=4):
        if self.vectorstore:
            try:
                results = self.vectorstore.similarity_search(question, k=max_results)
                return [doc.page_content for doc in results]
            except Exception:
                pass # Fallback to keyword search if similarity search fails
        
        # Keyword search fallback
        question_words = set(question.lower().split())
        return sorted(self.chunks, key=lambda c: len(question_words.intersection(c.lower().split())), reverse=True)[:max_results]
    
    # BUG FIX: Combined the two `generate_answer` methods into one.
    def generate_answer(self, question):
        summary_keywords = ['summarize', 'summary', 'overview', 'main points', 'key findings', 'topics']
        if any(keyword in question.lower() for keyword in summary_keywords):
            return self.get_document_summary()
            
        relevant_chunks = self.search_documents(question, max_results=5)
        if not relevant_chunks:
            return "I couldn't find relevant information in the documents to answer this question."
        
        # Use QA model if available and confident
        if self.qa_model:
            best_answer = ""
            best_score = -1
            for chunk in relevant_chunks:
                try:
                    result = self.qa_model(question=question, context=chunk)
                    if result['score'] > best_score and result['score'] > 0.1: # Confidence threshold
                        best_score = result['score']
                        best_answer = result['answer']
                except Exception:
                    continue
            if best_answer:
                return best_answer

        # Fallback to summarizing the most relevant chunks
        combined_text = " ".join(relevant_chunks)
        return self.summarizer.generate_contextual_summary(combined_text, max_length=250, min_length=50)

# --- Main Application Logic ---

def get_document_text(doc_files):
    processor = EnhancedDocumentProcessor()
    all_documents, doc_info = {}, []
    for doc_file in doc_files:
        file_text = ""
        file_name = doc_file.name
        if file_name.endswith('.pdf'):
            file_text = processor.extract_text_from_pdf(doc_file)
        elif file_name.endswith('.docx'):
            file_text = processor.extract_text_from_docx(doc_file)
        elif file_name.endswith('.txt'):
            file_text = processor.extract_text_from_txt(doc_file)
        
        if file_text.strip():
            all_documents[file_name] = file_text
            doc_info.append({'name': file_name, 'word_count': len(file_text.split()), 'status': '✅ Success'})
        else:
            doc_info.append({'name': file_name, 'word_count': 0, 'status': '❌ Failed'})
    return all_documents, doc_info

def create_text_chunks(documents_dict):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    all_chunks = []
    for doc_name, doc_text in documents_dict.items():
        # Using create_documents to handle metadata properly
        doc_chunks = text_splitter.create_documents([doc_text], metadatas=[{"source": doc_name}])
        for chunk in doc_chunks:
            # Prepend the document source to the content of each chunk
            chunk.page_content = f"[Document: {doc_name}]\n\n{chunk.page_content}"
            all_chunks.append(chunk.page_content)
    return all_chunks

def process_documents(doc_files):
    with st.spinner("Analyzing documents... This may take a moment."):
        documents_dict, doc_info = get_document_text(doc_files)
        
        # Display processing status
        st.sidebar.markdown("### Processing Status")
        for info in doc_info:
            st.sidebar.markdown(f"- **{info['name']}**: {info['status']}")

        if not documents_dict:
            st.error("No text could be extracted from the uploaded file(s). Please check the documents and try again.")
            return

        text_chunks = create_text_chunks(documents_dict)
        if not text_chunks:
            st.error("Could not create text segments from the documents.")
            return

        st.session_state.qa_system = SmartDocumentQA(text_chunks, doc_info)
        st.session_state.processed_files = list(documents_dict.keys())
        st.session_state.conversation_history = [] # Reset history on new processing
        st.success(f"Successfully processed {len(documents_dict)} document(s)!")

def handle_question(question):
    if "qa_system" in st.session_state and st.session_state.qa_system:
        with st.spinner("🤖 Thinking..."):
            answer = st.session_state.qa_system.generate_answer(question)
            st.session_state.conversation_history.append({
                "question": question,
                "answer": answer,
                "timestamp": datetime.now().strftime("%I:%M:%S %p")
            })
    else:
        st.error("Please upload and process documents first.")

def display_conversation():
    if st.session_state.conversation_history:
        st.markdown("---")
        for qa in reversed(st.session_state.conversation_history):
            st.markdown(f'<div class="question-box"><strong>You ({qa["timestamp"]}):</strong><br>{qa["question"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-box-container"><strong>Assistant:</strong><br>{qa["answer"]}</div>', unsafe_allow_html=True)

def create_sidebar():
    with st.sidebar:
        st.markdown("# 📁 Document Management")
        doc_files = st.file_uploader(
            "**Upload Your Documents**",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt"]
        )
        if st.button("Process Documents", type="primary", use_container_width=True):
            if doc_files:
                process_documents(doc_files)
                st.rerun() # Rerun to update the main page state
            else:
                st.warning("Please upload at least one document.")
        
        if st.session_state.processed_files:
            st.markdown("### ✅ Processed Documents")
            for name in st.session_state.processed_files:
                st.markdown(f'<div class="document-item"><span>{name}</span></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.conversation_history = []
            st.rerun()

def main():
    load_dotenv()
    st.set_page_config(page_title="Smart Document Assistant", page_icon="📚", layout="wide")
    st.markdown(enhanced_css, unsafe_allow_html=True)

    # Initialize session state variables
    if "qa_system" not in st.session_state:
        st.session_state.qa_system = None
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []

    st.markdown('<div class="main-header"><h1>Smart Document Assistant</h1><p>Ask questions and get intelligent answers from your documents.</p></div>', unsafe_allow_html=True)
    
    create_sidebar()

    st.markdown("## 💬 Ask a Question")
    if st.session_state.qa_system:
        # Chat input at the top for better UX
        user_question = st.chat_input("What would you like to know?")
        if user_question:
            handle_question(user_question)
        
        # Quick action buttons
        st.markdown("**Suggestions:**")
        cols = st.columns(3)
        quick_questions = ["Provide a comprehensive summary", "What are the key findings?", "List the main topics"]
        for i, q in enumerate(quick_questions):
            if cols[i].button(q, use_container_width=True):
                handle_question(q)
                st.rerun()
    else:
        st.markdown('<div class="alert-info"><h4>Welcome!</h4><p>Upload your documents in the sidebar and click \'Process Documents\' to begin.</p></div>', unsafe_allow_html=True)

    display_conversation()

if __name__ == "__main__":
    main()