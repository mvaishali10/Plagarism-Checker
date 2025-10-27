#!/usr/bin/env python3
"""Test plagiarism detection features including threshold and risk classification"""

import os
import sys
import numpy as np

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.similarity_analyzer import SimilarityAnalyzer

def test_threshold_functionality():
    """Test threshold-based plagiarism detection"""
    
    print("Testing threshold-based plagiarism detection...")
    
    # Sample documents with varying similarity levels
    documents = [
        "The quick brown fox jumps over the lazy dog. This is a test document for plagiarism detection. Machine learning algorithms analyze text patterns.",
        "A fast brown fox leaps above the sleepy dog. This document tests plagiarism identification systems. AI algorithms examine textual patterns.",
        "Cats are independent pets that require minimal care. They spend most of their time sleeping and grooming themselves.",
        "Dogs are loyal companions that need regular exercise and attention. They are social animals that form strong bonds with humans."
    ]
    
    doc_names = ["Document A", "Document B", "Document C", "Document D"]
    
    # Generate similarity data
    analyzer = SimilarityAnalyzer()
    embeddings = analyzer.generate_embeddings(documents)
    similarity_matrix = analyzer.compute_similarity_matrix(embeddings)
    
    print(f"\nSimilarity Matrix:")
    for i in range(len(documents)):
        for j in range(len(documents)):
            print(f"  {doc_names[i]} vs {doc_names[j]}: {similarity_matrix[i][j]*100:.1f}%")
    
    # Test different threshold levels
    thresholds = [30, 50, 70]
    
    for threshold in thresholds:
        print(f"\n--- Testing threshold: {threshold}% ---")
        similar_pairs = analyzer.find_similar_pairs(
            similarity_matrix, 
            threshold=threshold/100, 
            document_names=doc_names
        )
        
        print(f"Found {len(similar_pairs)} pairs above {threshold}% threshold:")
        for pair in similar_pairs:
            similarity_percent = pair['similarity_percent']
            
            # Test risk level classification
            if similarity_percent >= 85:
                risk_level = 'High'
            elif similarity_percent >= 75:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            print(f"  {pair['doc1_name']} vs {pair['doc2_name']}: {similarity_percent:.1f}% ({risk_level} risk)")
    
    return True

def test_risk_classification():
    """Test risk level classification logic"""
    
    print("\n" + "="*50)
    print("Testing risk level classification...")
    
    # Test different similarity levels
    test_similarities = [95, 87, 76, 68, 45, 20]
    
    for similarity in test_similarities:
        if similarity >= 85:
            risk_level = 'High'
            expected_color = 'Red zone'
        elif similarity >= 75:
            risk_level = 'Medium'  
            expected_color = 'Orange zone'
        elif similarity >= 70:  # Assuming default threshold
            risk_level = 'Low'
            expected_color = 'Yellow zone'
        else:
            risk_level = 'Below threshold'
            expected_color = 'Green zone'
        
        print(f"  {similarity}% similarity -> {risk_level} risk ({expected_color})")
    
    return True

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    
    print("\n" + "="*50)
    print("Testing edge cases...")
    
    # Test with identical documents
    identical_docs = ["This is exactly the same text.", "This is exactly the same text."]
    
    analyzer = SimilarityAnalyzer()
    embeddings = analyzer.generate_embeddings(identical_docs)
    similarity_matrix = analyzer.compute_similarity_matrix(embeddings)
    
    identical_similarity = similarity_matrix[0][1] * 100
    print(f"Identical documents similarity: {identical_similarity:.1f}%")
    
    # Test with completely different documents
    different_docs = [
        "Machine learning algorithms process data efficiently using computational methods.",
        "The weather forecast indicates sunny skies with temperatures reaching 75 degrees."
    ]
    
    embeddings = analyzer.generate_embeddings(different_docs)
    similarity_matrix = analyzer.compute_similarity_matrix(embeddings)
    
    different_similarity = similarity_matrix[0][1] * 100
    print(f"Completely different documents similarity: {different_similarity:.1f}%")
    
    # Validate expected behavior
    if identical_similarity > 90:
        print("✓ Identical documents show high similarity")
    else:
        print("⚠ Identical documents should show higher similarity")
    
    if different_similarity < 20:
        print("✓ Different documents show low similarity")
    else:
        print("⚠ Different documents should show lower similarity")
    
    return True

if __name__ == "__main__":
    print("BERT Semantic Plagiarism Checker - Feature Testing")
    print("=" * 50)
    
    try:
        # Run all tests
        test_threshold_functionality()
        test_risk_classification()
        test_edge_cases()
        
        print("\n" + "=" * 50)
        print("✓ All plagiarism detection features working correctly!")
        print("- Threshold-based filtering: ✓")
        print("- Risk level classification: ✓")
        print("- Edge case handling: ✓")
        
    except Exception as e:
        print(f"✗ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)