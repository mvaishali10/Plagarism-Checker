# Plagarism-Checker

# Overview

This is a BERT Semantic Plagiarism Checker built with Streamlit that analyzes documents for semantic similarities using advanced BERT embeddings. The application allows users to upload multiple documents (TXT, PDF, DOCX) and generates a similarity matrix to detect potential plagiarism based on semantic content rather than exact text matches. The system provides interactive visualizations through heatmaps to help users identify documents with high similarity scores.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Streamlit web application framework for rapid prototyping and deployment
- **Interface Design**: Single-page application with sidebar controls for file uploads and configuration
- **State Management**: Streamlit session state for maintaining uploaded documents, embeddings, and similarity matrices across user interactions
- **Caching Strategy**: Uses `@st.cache_resource` decorator for expensive operations like model loading

## Backend Architecture
- **Modular Design**: Separated into utility modules for specific functionalities:
  - `DocumentProcessor`: Handles text extraction from multiple file formats
  - `SimilarityAnalyzer`: Manages BERT embedding generation and similarity computations
  - `Visualization`: Creates interactive similarity heatmaps
- **Error Handling**: Comprehensive logging and fallback mechanisms throughout the system
- **File Processing**: Temporary file handling for uploaded documents with support for multiple formats

## Machine Learning Pipeline
- **Primary Model**: Sentence Transformers with 'all-MiniLM-L6-v2' BERT model for semantic embeddings
- **Fallback System**: TF-IDF based similarity calculation when BERT models are unavailable
- **Similarity Computation**: Cosine similarity between document embeddings to measure semantic relatedness

## Data Processing
- **Text Extraction**: Multi-format document processing (TXT, PDF, DOCX) with encoding fallbacks
- **Embedding Generation**: Converts text documents into high-dimensional vector representations
- **Similarity Matrix**: Pairwise similarity calculations stored in NumPy arrays for efficient computation

# External Dependencies

## Core Libraries
- **Streamlit**: Web application framework for the user interface
- **Sentence Transformers**: BERT-based model for generating semantic embeddings
- **NumPy**: Numerical computations for similarity matrices and embeddings
- **Pandas**: Data manipulation and analysis

## Visualization
- **Plotly**: Interactive plotting library for similarity heatmaps with custom color scales and hover effects

## Document Processing
- **PyPDF2/pdfplumber**: PDF text extraction (implied by PDF support)
- **python-docx**: Microsoft Word document processing (implied by DOCX support)

## Development and Deployment
- **Python Standard Library**: File handling, logging, and temporary file management
- **Hugging Face Model Hub**: Source for pre-trained BERT models through Sentence Transformers

Note: The application includes graceful degradation when BERT models are unavailable, falling back to traditional TF-IDF similarity measures to ensure functionality across different deployment environments.
