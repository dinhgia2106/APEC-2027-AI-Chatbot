#!/usr/bin/env python3
"""
Demo test RAG Pipeline cho APEC 2027 AI Chatbot

Chạy script này để test khả năng trả lời câu hỏi của RAG Pipeline.
Đảm bảo đã chạy các bước sau trước khi test:
1. python backend/data/crawler.py (crawl dữ liệu)
2. python backend/Embed/embed_data.py (tạo embeddings)
3. python backend/VectorStore/build_faiss_index.py (xây dựng FAISS index)
4. Khởi động Ollama server với LLaMA3: ollama serve

Usage:
    cd demo
    python test_rag_pipeline.py
"""

import sys
from pathlib import Path

# Thêm backend vào Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.rag_pipeline import RAGPipeline


def test_questions():
    """Danh sách câu hỏi test"""
    questions = [
        "APEC có bao nhiêu thành viên?",
        "Việt Nam có phải là thành viên APEC không?",
        "APEC 2025 sẽ diễn ra ở đâu?",
        "Khi nào diễn ra cuộc họp Informal Senior Officials’ Meeting (ISOM)?",
        "Các thành phố nào ở Hàn Quốc sẽ tổ chức APEC 2025?",
        "Mission của APEC là gì?",
        "APEC Vision 2040 nói về điều gì?",
        "What is APEC's mission?",
        "When will APEC 2025 take place?",
    ]
    return questions


def interactive_mode(rag_pipeline):
    """Chế độ hỏi đáp tương tác"""
    print("\n" + "="*80)
    print("CHẾ ĐỘ HỎI ĐÁP TƯƠNG TÁC")
    print("="*80)
    print("Nhập câu hỏi của bạn (hoặc 'quit' để thoát):")
    
    while True:
        try:
            user_query = input("\nCâu hỏi: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q', 'thoát']:
                print("Tạm biệt!")
                break
                
            if not user_query:
                continue
                
            result = rag_pipeline.answer(user_query)
            
            print(f"\nCâu trả lời: {result['answer']}")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\nTạm biệt!")
            break
        except Exception as e:
            print(f"Lỗi: {e}")


def batch_test_mode(rag_pipeline):
    """Chế độ test hàng loạt với câu hỏi mẫu"""
    questions = test_questions()
    
    print("\n" + "="*80)
    print("CHẾ ĐỘ TEST HÀNG LOẠT")
    print("="*80)
    print(f"Sẽ test {len(questions)} câu hỏi mẫu...")
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"CÂU HỎI {i}/{len(questions)}: {question}")
        print("="*60)
        
        try:
            result = rag_pipeline.answer(question, k=3)  # Lấy 3 context documents
            
            print(f"Câu trả lời:\n{result['answer']}")
            
            # Hiển thị context nếu muốn debug
            show_context = input("\nBạn có muốn xem context? (y/n): ").strip().lower()
            if show_context == 'y':
                print(f"\nContext đã sử dụng:\n{result['context']}")
            
            # Pause giữa các câu hỏi
            if i < len(questions):
                input("\nNhấn Enter để tiếp tục...")
                
        except Exception as e:
            print(f"Lỗi khi xử lý câu hỏi: {e}")
            continue


def main():
    """Hàm chính của demo"""
    print("DEMO TEST RAG PIPELINE - APEC 2027 AI CHATBOT")
    print("=" * 80)
    
    try:
        # Khởi tạo RAG Pipeline
        print("Đang khởi tạo RAG Pipeline...")
        rag = RAGPipeline()
        print("Khởi tạo thành công!")
        
        # Menu lựa chọn
        while True:
            print("\nLựa chọn chế độ test:")
            print("1. Test hàng loạt với câu hỏi mẫu")
            print("2. Chế độ hỏi đáp tương tác")
            print("3. Thoát")
            
            choice = input("\nNhập lựa chọn (1-3): ").strip()
            
            if choice == "1":
                batch_test_mode(rag)
            elif choice == "2":
                interactive_mode(rag)
            elif choice == "3":
                print("Tạm biệt!")
                break
            else:
                print("Lựa chọn không hợp lệ. Vui lòng nhập 1, 2 hoặc 3.")
                
    except FileNotFoundError as e:
        print(f"\nLỗi: {e}")
        print("\nĐảm bảo đã chạy các bước sau:")
        print("1. python backend/data/crawler.py")
        print("2. python backend/Embed/embed_data.py") 
        print("3. python backend/VectorStore/build_faiss_index.py")
        
    except Exception as e:
        print(f"\nLỗi không mong muốn: {e}")
        print("Vui lòng kiểm tra lại cấu hình và thử lại.")


if __name__ == "__main__":
    main() 