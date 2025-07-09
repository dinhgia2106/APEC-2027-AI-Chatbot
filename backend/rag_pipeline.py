import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Import LLaMA3 function
sys.path.append(str(Path(__file__).resolve().parent.parent))
from llm import query_llm


class RAGPipeline:
    """
    Lớp RAG Pipeline tích hợp tất cả các thành phần:
    - Embedding model để encode query
    - FAISS index để tìm kiếm vector similarity
    - Mapping từ vector IDs về key paths  
    - Dữ liệu gốc để truy xuất text đầy đủ
    - LLM để sinh câu trả lời
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                 data_dir: str = "backend/data",
                 embed_dir: str = "backend/Embed", 
                 vector_dir: str = "backend/VectorStore"):
        """
        Khởi tạo RAG Pipeline
        
        Args:
            model_name: Tên model SentenceTransformer
            data_dir: Thư mục chứa data.json
            embed_dir: Thư mục chứa embeddings
            vector_dir: Thư mục chứa FAISS index
        """
        self.root = Path(__file__).resolve().parent.parent
        
        # 1. Load embedding model
        print("Đang load model embedding...")
        self.model = SentenceTransformer(model_name)
        
        # 2. Load FAISS index
        index_path = self.root / vector_dir / "apec.index"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index không tìm thấy tại: {index_path}")
        print("Đang load FAISS index...")
        self.index = faiss.read_index(str(index_path))
        
        # 3. Load id2key mapping
        id2key_path = self.root / vector_dir / "id2key.json"
        if not id2key_path.exists():
            raise FileNotFoundError(f"File id2key.json không tìm thấy tại: {id2key_path}")
        with open(id2key_path, "r", encoding="utf-8") as f:
            self.id2key = json.load(f)
        
        # 4. Load dữ liệu gốc
        data_path = self.root / data_dir / "data.json"
        if not data_path.exists():
            raise FileNotFoundError(f"File data.json không tìm thấy tại: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        print(f"RAG Pipeline đã sẵn sàng với {len(self.id2key)} vectors trong index.")
    
    def detect_language(self, text: str) -> str:
        """
        Phát hiện ngôn ngữ của text dựa trên ký tự đặc trưng
        
        Args:
            text: Chuỗi text cần phát hiện ngôn ngữ
            
        Returns:
            'vi' cho tiếng Việt, 'en' cho tiếng Anh
        """
        # Ký tự đặc trưng tiếng Việt với dấu (bao gồm cả chữ hoa và chữ thường)
        vietnamese_chars = r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđĐÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ]'
        
        # Kiểm tra ký tự đặc trưng tiếng Việt
        if re.search(vietnamese_chars, text):
            return 'vi'
        else:
            return 'en'

    def _get_value_from_keypath(self, keypath: str) -> str:
        """
        Truy xuất giá trị từ dữ liệu gốc theo key path
        
        Args:
            keypath: Đường dẫn key như "apec_general_info.title"
            
        Returns:
            Chuỗi text tương ứng
        """
        try:
            # Xử lý đặc biệt cho combined keypaths
            if keypath.endswith(".combined"):
                # Tái tạo combined text từ dữ liệu gốc
                base_keypath = keypath[:-9]  # Loại bỏ ".combined"
                parts = base_keypath.split(".")
                current = self.data
                
                for part in parts:
                    # Xử lý array index như [0], [1]
                    if "[" in part and "]" in part:
                        key = part.split("[")[0]
                        index = int(part.split("[")[1].split("]")[0])
                        current = current[key][index]
                    else:
                        current = current[part]
                
                # Tái tạo combined text giống như trong embed_data.py
                if isinstance(current, dict) and "event_title" in current and "date" in current:
                    combined_text = f"{current['event_title']}"
                    if "date" in current:
                        combined_text += f" diễn ra ngày {current['date']}"
                    if "venue" in current:
                        combined_text += f" tại {current['venue']}"
                    return combined_text
                else:
                    return f"[Không thể tái tạo combined text cho {keypath}]"
            
            # Xử lý bình thường cho các keypath khác
            parts = keypath.split(".")
            current = self.data
            
            for part in parts:
                # Xử lý array index như [0], [1]
                if "[" in part and "]" in part:
                    key = part.split("[")[0]
                    index = int(part.split("[")[1].split("]")[0])
                    current = current[key][index]
                else:
                    current = current[part]
            
            return str(current)
        except (KeyError, IndexError, ValueError) as e:
            return f"[Lỗi truy xuất keypath {keypath}: {e}]"
    
    def retrieve(self, query: str, k: int = 5) -> str:
        """
        Tìm kiếm context liên quan từ query
        
        Args:
            query: Câu hỏi của người dùng
            k: Số lượng kết quả trả về
            
        Returns:
            Chuỗi context được ghép từ các đoạn văn bản liên quan
        """
        # 1. Encode query thành vector
        query_vector = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        
        # 2. Tìm kiếm k vectors gần nhất
        scores, indices = self.index.search(query_vector.reshape(1, -1), k)
        
        # 3. Chuyển đổi indices thành key paths
        context_parts = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:  # FAISS trả về -1 nếu không tìm thấy
                continue
                
            keypath = self.id2key[idx]
            text = self._get_value_from_keypath(keypath)
            
            context_parts.append(f"{i+1}. {keypath}: {text}")
        
        return "\n".join(context_parts)
    
    def generate(self, context: str, query: str) -> str:
        """
        Sinh câu trả lời từ LLM với context và query
        
        Args:
            context: Ngữ cảnh từ hàm retrieve
            query: Câu hỏi gốc
            
        Returns:
            Câu trả lời từ LLM
        """
        # Phát hiện ngôn ngữ
        language = self.detect_language(query)
        
        # Template cho tiếng Việt
        VIETNAMESE_PROMPT_TEMPLATE = """Bạn là một trợ lý AI hữu ích và thân thiện của sự kiện APEC và về văn hóa Việt Nam.
Hãy sử dụng các thông tin được cung cấp dưới đây để trả lời câu hỏi của người dùng một cách chính xác và súc tích bằng tiếng Việt.

QUAN TRỌNG: 
- Chỉ trả lời trực tiếp, không sử dụng <think> hay bất kỳ tag nào
- Câu trả lời ngắn gọn, súc tích
- Nếu thông tin không có trong ngữ cảnh, hãy trả lời ngắn gọn rằng bạn không có thông tin về vấn đề đó

Ngữ cảnh:
{context}

Câu hỏi: {query}

Trả lời ngắn gọn bằng tiếng Việt:"""

        # Template cho tiếng Anh
        ENGLISH_PROMPT_TEMPLATE = """You are a helpful and friendly AI assistant for APEC events and Vietnamese culture.
Please use the information provided below to answer the user's question accurately and concisely in English.

IMPORTANT: 
- Answer directly without using <think> or any tags
- Keep answers concise and to the point
- If information is not available in the context, briefly respond that you don't have information about that topic

Context:
{context}

Question: {query}

Concise answer in English:"""
        
        # Chọn template phù hợp
        if language == 'en':
            template = ENGLISH_PROMPT_TEMPLATE
            print("Đã phát hiện ngôn ngữ: Tiếng Anh")
        else:
            template = VIETNAMESE_PROMPT_TEMPLATE
            print("Đã phát hiện ngôn ngữ: Tiếng Việt")
        
        # Tạo prompt hoàn chỉnh
        prompt = template.format(context=context, query=query)
        
        # Gọi LLM local qua Ollama
        try:
            response = query_llm(prompt)
            # Post-process để loại bỏ think tags và clean up
            cleaned_response = self._clean_response(response)
            return cleaned_response
        except Exception as e:
            return f"Lỗi khi gọi LLM: {e}"
    
    def _clean_response(self, response: str) -> str:
        """
        Làm sạch response từ LLM, loại bỏ think tags và metadata
        
        Args:
            response: Raw response từ LLM
            
        Returns:
            Cleaned response
        """
        import re
        
        # Loại bỏ <think>...</think> tags
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # Loại bỏ <language>...</language> tags  
        cleaned = re.sub(r'<language>.*?</language>', '', cleaned, flags=re.DOTALL)
        
        # Loại bỏ các dòng trống thừa
        cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
        
        # Trim whitespace
        cleaned = cleaned.strip()
        
        return cleaned
    
    def answer(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Trả lời câu hỏi hoàn chỉnh: retrieve context + generate answer
        
        Args:
            query: Câu hỏi của người dùng
            k: Số lượng context documents
            
        Returns:
            Dictionary chứa query, context, answer
        """
        print(f"Đang xử lý câu hỏi: {query}")
        
        # 1. Retrieve context
        print("Đang tìm kiếm context...")
        context = self.retrieve(query, k)
        
        # 2. Generate answer
        print("Đang sinh câu trả lời...")
        answer = self.generate(context, query)
        
        return {
            "query": query,
            "context": context,
            "answer": answer
        }


def main():
    """Demo function để test RAG Pipeline"""
    try:
        # Khởi tạo pipeline
        rag = RAGPipeline()
        
        # Test với một câu hỏi
        test_query = "APEC có bao nhiêu thành viên?"
        result = rag.answer(test_query)
        
        print("=" * 80)
        print("KẾT QUẢ TEST RAG PIPELINE")
        print("=" * 80)
        print(f"Câu hỏi: {result['query']}")
        print(f"\nContext tìm được:\n{result['context']}")
        print(f"\nCâu trả lời:\n{result['answer']}")
        
    except Exception as e:
        print(f"Lỗi: {e}")


if __name__ == "__main__":
    main() 