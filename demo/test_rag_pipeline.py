#!/usr/bin/env python3
"""
Demo to test the RAG Pipeline for the APEC 2027 AI Chatbot.

Run this script to test the question-answering capabilities of the RAG Pipeline.
Ensure you have completed the following steps before testing:
1. python backend/data/crawler.py (to crawl data)
2. python backend/Embed/embed_data.py (to create embeddings)
3. python backend/VectorStore/build_faiss_index.py (to build the FAISS index)
4. Start the Ollama server with LLaMA3: ollama serve

Usage:
    cd demo
    python test_rag_pipeline.py
"""

import sys
from pathlib import Path

# Add the backend directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.rag_pipeline import RAGPipeline


def test_questions():
    """List of test questions."""
    questions = [
        "How many members does APEC have?",
        "Is Vietnam a member of APEC?",
        "Where will APEC 2025 take place?",
        "When does the Informal Senior Officials’ Meeting (ISOM) take place?",
        "Which cities in South Korea will host APEC 2025?",
        "What is APEC's mission?",
        "What is the APEC Vision 2040 about?",
        "What is APEC's mission?",
        "When will APEC 2025 take place?",
    ]
    return questions


def interactive_mode(rag_pipeline):
    """Interactive question-and-answer mode."""
    print("\n" + "="*80)
    print("INTERACTIVE Q&A MODE")
    print("="*80)
    print("Enter your question (or 'quit' to exit):")
    
    while True:
        try:
            user_query = input("\nQuestion: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not user_query:
                continue
                
            result = rag_pipeline.answer(user_query)
            
            print(f"\nAnswer: {result['answer']}")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def batch_test_mode(rag_pipeline):
    """Batch test mode with sample questions."""
    questions = test_questions()
    
    print("\n" + "="*80)
    print("BATCH TEST MODE")
    print("="*80)
    print(f"Testing {len(questions)} sample questions...")
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"QUESTION {i}/{len(questions)}: {question}")
        print("="*60)
        
        try:
            result = rag_pipeline.answer(question, k=3)  # Retrieve 3 context documents
            
            print(f"Answer:\n{result['answer']}")
            
            # Display context for debugging if desired
            show_context = input("\nDo you want to see the context? (y/n): ").strip().lower()
            if show_context == 'y':
                print(f"\nContext used:\n{result['context']}")
            
            # Pause between questions
            if i < len(questions):
                input("\nPress Enter to continue...")
                
        except Exception as e:
            print(f"Error processing question: {e}")
            continue


def main():
    """Main function for the demo."""
    print("RAG PIPELINE DEMO - APEC 2027 AI CHATBOT")
    print("=" * 80)
    
    try:
        # Initialize the RAG Pipeline
        print("Initializing RAG Pipeline...")
        rag = RAGPipeline()
        print("Initialization successful!")
        
        # Menu selection
        while True:
            print("\nSelect a test mode:")
            print("1. Batch test with sample questions")
            print("2. Interactive Q&A mode")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                batch_test_mode(rag)
            elif choice == "2":
                interactive_mode(rag)
            elif choice == "3":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
                
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure you have run the following steps:")
        print("1. python backend/data/crawler.py")
        print("2. python backend/Embed/embed_data.py") 
        print("3. python backend/VectorStore/build_faiss_index.py")
        
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("Please check your configuration and try again.")


if __name__ == "__main__":
    main()