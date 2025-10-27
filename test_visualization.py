#!/usr/bin/env python3
"""Simple test to verify similarity computation works"""

import os
import sys
import numpy as np

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.similarity_analyzer import SimilarityAnalyzer

def test_similarity():
    """Test the similarity analyzer with sample documents"""
    
    # Initialize analyzer
    analyzer = SimilarityAnalyzer()
    
    # Sample documents with different similarity levels
    documents = [
        "The quick brown fox jumps over the lazy dog. This is a test document for plagiarism detection.",
        "A fast brown fox leaps above the sleepy dog. This document tests plagiarism identification systems.", 
        "The weather today is sunny and warm with a gentle breeze. It's perfect for outdoor activities."
    ]
    
    print("Testing similarity computation...")
    print(f"Model info: {analyzer.get_model_info()}")
    
    try:
        # Generate embeddings/features
        print("\nGenerating embeddings...")
        embeddings = analyzer.generate_embeddings(documents)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Compute similarity matrix
        print("\nComputing similarity matrix...")
        similarity_matrix = analyzer.compute_similarity_matrix(embeddings)
        print(f"Similarity matrix shape: {similarity_matrix.shape}")
        
        # Display results
        print("\nSimilarity Matrix (percentages):")
        for i in range(len(documents)):
            for j in range(len(documents)):
                print(f"Doc{i+1} vs Doc{j+1}: {similarity_matrix[i][j]*100:.1f}%")
        
        # Find similar pairs
        similar_pairs = analyzer.find_similar_pairs(similarity_matrix, threshold=0.3, document_names=[f"Doc{i+1}" for i in range(len(documents))])
        
        print(f"\nFound {len(similar_pairs)} similar pairs above 30% threshold:")
        for pair in similar_pairs:
            print(f"{pair['doc1_name']} vs {pair['doc2_name']}: {pair['similarity_percent']:.1f}%")
        
        # Verify expected behavior
        doc1_vs_doc2 = similarity_matrix[0][1] * 100  # Should be high (similar content)
        doc1_vs_doc3 = similarity_matrix[0][2] * 100  # Should be low (different content)
        
        print(f"\nValidation:")
        print(f"Doc1 vs Doc2 similarity: {doc1_vs_doc2:.1f}% (should be high)")
        print(f"Doc1 vs Doc3 similarity: {doc1_vs_doc3:.1f}% (should be low)")
        
        if doc1_vs_doc2 > doc1_vs_doc3:
            print("✓ Similarity computation working correctly!")
            return True
        else:
            print("✗ Similarity computation may have issues")
            return False
            
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_similarity()
    exit(0 if success else 1)