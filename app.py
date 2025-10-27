import streamlit as st
import pandas as pd
import numpy as np
from utils.document_processor import DocumentProcessor
from utils.similarity_analyzer import SimilarityAnalyzer
from utils.visualization import create_similarity_heatmap
import os
import tempfile

# Configure page
st.set_page_config(
    page_title="BERT Semantic Plagiarism Checker",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'similarity_matrix' not in st.session_state:
    st.session_state.similarity_matrix = None
if 'doc_names' not in st.session_state:
    st.session_state.doc_names = []

# Initialize processors
@st.cache_resource
def load_similarity_analyzer():
    return SimilarityAnalyzer()

analyzer = load_similarity_analyzer()
processor = DocumentProcessor()

# Main title
st.title("🔍 BERT Semantic Plagiarism Checker")
st.markdown("Upload documents to detect semantic similarities using advanced BERT embeddings")

# Sidebar for controls
st.sidebar.header("Configuration")

# File upload
uploaded_files = st.sidebar.file_uploader(
    "Upload Documents",
    type=['txt', 'pdf', 'docx'],
    accept_multiple_files=True,
    help="Upload multiple documents to compare for plagiarism"
)

# Similarity threshold slider
similarity_threshold = st.sidebar.slider(
    "Similarity Threshold (%)",
    min_value=0,
    max_value=100,
    value=70,
    step=5,
    help="Documents above this threshold will be flagged as potentially plagiarized"
)

# Process uploaded files
if uploaded_files:
    # Clear previous data if new files uploaded
    if len(uploaded_files) != len(st.session_state.documents):
        st.session_state.documents = []
        st.session_state.embeddings = None
        st.session_state.similarity_matrix = None
        st.session_state.doc_names = []
    
    # Process documents if not already processed
    if not st.session_state.documents:
        with st.spinner("Processing documents..."):
            for uploaded_file in uploaded_files:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Extract text from document
                    text = processor.extract_text(tmp_path, uploaded_file.type)
                    if text.strip():
                        st.session_state.documents.append(text)
                        st.session_state.doc_names.append(uploaded_file.name)
                    else:
                        st.warning(f"Could not extract text from {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                finally:
                    # Clean up temporary file
                    os.unlink(tmp_path)
    
    # Generate embeddings and similarity matrix
    if st.session_state.documents and st.session_state.embeddings is None:
        with st.spinner("Generating BERT embeddings..."):
            try:
                st.session_state.embeddings = analyzer.generate_embeddings(st.session_state.documents)
                st.session_state.similarity_matrix = analyzer.compute_similarity_matrix(st.session_state.embeddings)
                st.success(f"Successfully processed {len(st.session_state.documents)} documents")
            except Exception as e:
                st.error(f"Error generating embeddings: {str(e)}")
                st.session_state.embeddings = None
                st.session_state.similarity_matrix = None

# Main content area
if st.session_state.similarity_matrix is not None:
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Similarity Matrix", "⚠️ Plagiarism Report", "📋 Document Details"])
    
    with tab1:
        st.subheader("Interactive Similarity Heatmap")
        st.markdown("Hover over cells to see exact similarity percentages")
        
        # Create and display heatmap
        fig = create_similarity_heatmap(
            st.session_state.similarity_matrix,
            st.session_state.doc_names
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display similarity matrix as table
        st.subheader("Similarity Matrix (Percentages)")
        similarity_df = pd.DataFrame(
            st.session_state.similarity_matrix * 100,
            index=st.session_state.doc_names,
            columns=st.session_state.doc_names
        )
        similarity_df = similarity_df.round(1)
        st.dataframe(similarity_df, use_container_width=True)
    
    with tab2:
        st.subheader("Plagiarism Detection Report")
        st.markdown(f"Documents with similarity ≥ {similarity_threshold}% are flagged")
        
        # Find potentially plagiarized pairs
        flagged_pairs = []
        n_docs = len(st.session_state.doc_names)
        
        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                similarity_percent = st.session_state.similarity_matrix[i][j] * 100
                if similarity_percent >= similarity_threshold:
                    flagged_pairs.append({
                        'Document 1': st.session_state.doc_names[i],
                        'Document 2': st.session_state.doc_names[j],
                        'Similarity (%)': round(similarity_percent, 1),
                        'Risk Level': 'High' if similarity_percent >= 85 else 'Medium' if similarity_percent >= 75 else 'Low'
                    })
        
        if flagged_pairs:
            st.error(f"⚠️ Found {len(flagged_pairs)} potentially plagiarized document pair(s)")
            
            flagged_df = pd.DataFrame(flagged_pairs)
            
            # Color code by risk level
            def highlight_risk(row):
                if row['Risk Level'] == 'High':
                    return ['background-color: #ffebee'] * len(row)
                elif row['Risk Level'] == 'Medium':
                    return ['background-color: #fff3e0'] * len(row)
                else:
                    return ['background-color: #f3e5f5'] * len(row)
            
            styled_df = flagged_df.style.apply(highlight_risk, axis=1)
            st.dataframe(styled_df, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("High Risk Pairs", len([p for p in flagged_pairs if p['Risk Level'] == 'High']))
            with col2:
                st.metric("Medium Risk Pairs", len([p for p in flagged_pairs if p['Risk Level'] == 'Medium']))
            with col3:
                st.metric("Low Risk Pairs", len([p for p in flagged_pairs if p['Risk Level'] == 'Low']))
        else:
            st.success("✅ No plagiarism detected above the threshold")
            st.info("All document pairs show acceptable levels of similarity")
    
    with tab3:
        st.subheader("Document Information")
        
        for i, (name, doc) in enumerate(zip(st.session_state.doc_names, st.session_state.documents)):
            with st.expander(f"📄 {name}"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric("Word Count", len(doc.split()))
                    st.metric("Character Count", len(doc))
                
                with col2:
                    st.text_area(
                        "Document Preview",
                        value=doc[:500] + "..." if len(doc) > 500 else doc,
                        height=100,
                        disabled=True,
                        key=f"preview_{i}"
                    )

else:
    # Welcome screen
    st.markdown("""
    ## How to use this plagiarism checker:
    
    1. **Upload Documents**: Use the sidebar to upload multiple text files (.txt), PDFs (.pdf), or Word documents (.docx)
    2. **Set Threshold**: Adjust the similarity threshold to control detection sensitivity
    3. **Review Results**: Examine the similarity matrix, plagiarism report, and document details
    
    ### Features:
    - 🧠 **BERT Embeddings**: Uses advanced semantic understanding to detect rephrased content
    - 📊 **Interactive Visualizations**: Hover over the heatmap to see exact similarity scores
    - ⚙️ **Adjustable Threshold**: Customize detection sensitivity based on your needs
    - 📋 **Detailed Reports**: Get comprehensive analysis of potential plagiarism
    
    ### Supported File Formats:
    - `.txt` - Plain text files
    - `.pdf` - PDF documents
    - `.docx` - Microsoft Word documents
    """)
    
    st.info("👆 Upload documents using the sidebar to get started!")

# Footer
st.markdown("---")
st.markdown(
    "Built with BERT embeddings (all-MiniLM-L6-v2) for semantic similarity detection | "
    "Powered by Streamlit"
)
