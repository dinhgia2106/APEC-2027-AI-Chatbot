import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Import LLaMA3 function
sys.path.append(str(Path(__file__).resolve().parent.parent))
from llm import query_llm


class RAGPipeline:
    """
    The RAG Pipeline class integrates all components:
    - An embedding model to encode the query.
    - A FAISS index for vector similarity search.
    - A mapping from vector IDs to key paths.
    - The original data source for retrieving full text.
    - An LLM to generate the final answer.
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                 data_dir: str = "backend/data",
                 embed_dir: str = "backend/Embed", 
                 vector_dir: str = "backend/VectorStore"):
        """
        Initializes the RAG Pipeline.
        
        Args:
            model_name: The name of the SentenceTransformer model.
            data_dir: The directory containing data.json.
            embed_dir: The directory containing embeddings.
            vector_dir: The directory containing the FAISS index.
        """
        self.root = Path(__file__).resolve().parent.parent
        
        # 1. Load embedding model
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        
        # 2. Load FAISS index
        index_path = self.root / vector_dir / "apec.index"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at: {index_path}")
        print("Loading FAISS index...")
        self.index = faiss.read_index(str(index_path))
        
        # 3. Load id2key mapping
        id2key_path = self.root / vector_dir / "id2key.json"
        if not id2key_path.exists():
            raise FileNotFoundError(f"File id2key.json not found at: {id2key_path}")
        with open(id2key_path, "r", encoding="utf-8") as f:
            self.id2key = json.load(f)
        
        # 4. Load original data
        data_path = self.root / data_dir / "data.json"
        if not data_path.exists():
            raise FileNotFoundError(f"File data.json not found at: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        print(f"RAG Pipeline is ready with {len(self.id2key)} vectors in the index.")
    
    def detect_language(self, text: str) -> str:
        """
        Detects the language of the text based on characteristic characters.
        
        Args:
            text: The text string to detect the language of.
            
        Returns:
            'vi' for Vietnamese, 'en' for English.
        """
        # Characteristic Vietnamese characters with diacritics (including uppercase and lowercase)
        vietnamese_chars = r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđĐÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ]'
        
        # Check for characteristic Vietnamese characters
        if re.search(vietnamese_chars, text):
            return 'vi'
        else:
            return 'en'
    
    def _is_list_all_query(self, query: str) -> bool:
        """
        Detects if the query is a request to list all items (e.g., "list all meetings").
        """
        list_patterns = [
            r'\b(list|all|tất cả|danh sách|liệt kê)\b.*\b(meeting|meetings|cuộc họp|sự kiện|event)\b',
            r'\b(toàn bộ|hết|đầy đủ)\b.*\b(lịch|schedule|meeting)\b',
            r'\b(show|display|hiển thị)\b.*\b(all|tất cả)\b.*\b(meeting|event)\b'
        ]
        
        for pattern in list_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def _is_today_schedule_query(self, query: str) -> bool:
        """
        Detects if the query is about today's schedule.
        """
        today_patterns = [
            r'\b(today|hôm nay|ngày hôm nay)\b.*\b(event|events|sự kiện|cuộc họp|meeting|lịch)\b',
            r'\b(lịch|schedule)\b.*\b(today|hôm nay|ngày hôm nay)\b',
            r'\b(sự kiện|event|meeting)\b.*\b(today|hôm nay|ngày hôm nay)\b'
        ]
        
        for pattern in today_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def _preprocess_query_with_date(self, query: str) -> tuple[str, str]:
        """
        Preprocesses the query, adding date information if necessary.
        
        Returns:
            A tuple (enhanced_query, date_info).
        """
        today = datetime.now()
        date_info = f"Current date: {today.strftime('%Y-%m-%d')} ({today.strftime('%A, %B %d, %Y')})"
        
        if self._is_today_schedule_query(query):
            # Add the current date to the query for better search results
            enhanced_query = f"{query} {today.strftime('%Y-%m-%d')} {today.strftime('%B %Y')}"
            return enhanced_query, date_info
        
        return query, ""
    
    def _get_all_meetings_context(self) -> str:
        """
        Gets all meetings from the data to answer "list all" queries.
        """
        try:
            meetings = self.data["apec_2025_korea"]["schedule"]["meetings"]
            context_parts = []
            
            for i, meeting in enumerate(meetings):
                combined_text = f"{meeting['event_title']}"
                if "date" in meeting and meeting["date"]:
                    combined_text += f" takes place on {meeting['date']}"
                if "venue" in meeting and meeting["venue"]:
                    combined_text += f" at {meeting['venue']}"
                
                context_parts.append(f"{i+1}. {combined_text}")
            
            return "\n".join(context_parts)
        except (KeyError, IndexError) as e:
            return f"[Error retrieving list of meetings: {e}]"

    def _get_value_from_keypath(self, keypath: str) -> str:
        """
        Retrieves a value from the original data using a key path.
        
        Args:
            keypath: The key path, e.g., "apec_general_info.title".
            
        Returns:
            The corresponding text string.
        """
        try:
            # Special handling for combined keypaths
            if keypath.endswith(".combined"):
                # Recreate the combined text from the original data
                base_keypath = keypath[:-9]  # Remove ".combined"
                parts = base_keypath.split(".")
                current = self.data
                
                for part in parts:
                    # Handle array indices like [0], [1]
                    if "[" in part and "]" in part:
                        key = part.split("[")[0]
                        index = int(part.split("[")[1].split("]")[0])
                        current = current[key][index]
                    else:
                        current = current[part]
                
                # Recreate the combined text as done in embed_data.py
                if isinstance(current, dict) and "event_title" in current and "date" in current:
                    combined_text = f"{current['event_title']}"
                    if "date" in current:
                        combined_text += f" takes place on {current['date']}"
                    if "venue" in current:
                        combined_text += f" at {current['venue']}"
                    return combined_text
                else:
                    return f"[Could not recreate combined text for {keypath}]"
            
            # Normal handling for other keypaths
            parts = keypath.split(".")
            current = self.data
            
            for part in parts:
                # Handle array indices like [0], [1]
                if "[" in part and "]" in part:
                    key = part.split("[")[0]
                    index = int(part.split("[")[1].split("]")[0])
                    current = current[key][index]
                else:
                    current = current[part]
            
            return str(current)
        except (KeyError, IndexError, ValueError) as e:
            return f"[Error retrieving keypath {keypath}: {e}]"
    
    def retrieve(self, query: str, k: int = 5) -> str:
        """
        Retrieves relevant context for a given query.
        
        Args:
            query: The user's query.
            k: The number of results to return.
            
        Returns:
            A string of context concatenated from relevant text snippets.
        """
        # 1. Encode the query into a vector
        query_vector = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        
        # 2. Search for the k nearest vectors
        scores, indices = self.index.search(query_vector.reshape(1, -1), k)
        
        # 3. Convert indices to key paths
        context_parts = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:  # FAISS returns -1 if not found
                continue
                
            keypath = self.id2key[idx]
            text = self._get_value_from_keypath(keypath)
            
            context_parts.append(f"{i+1}. {keypath}: {text}")
        
        return "\n".join(context_parts)
    
    def generate(self, context: str, query: str) -> str:
        """
        Generates an answer from the LLM using the context and query.
        
        Args:
            context: The context from the retrieve function.
            query: The original query.
            
        Returns:
            The answer from the LLM.
        """
        # Detect language
        language = self.detect_language(query)
        
        # Template for Vietnamese
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

        # Template for English
        ENGLISH_PROMPT_TEMPLATE = """You are a helpful and friendly AI assistant for APEC events and Vietnamese culture.
Please use the information provided below to answer the user's question accurately and concisely in English.

IMPORTANT: 
- Answer directly without using <think> or any tags.
- Keep answers concise and to the point.
- If information is not available in the context, briefly respond that you don't have information about that topic.

Context:
{context}

Question: {query}

Concise answer in English:"""
        
        # Select the appropriate template
        if language == 'en':
            template = ENGLISH_PROMPT_TEMPLATE
            print("Detected language: English")
        else:
            template = VIETNAMESE_PROMPT_TEMPLATE
            print("Detected language: Vietnamese")
        
        # Create the full prompt
        prompt = template.format(context=context, query=query)
        
        # Call the local LLM via Ollama
        try:
            response = query_llm(prompt)
            # Post-process to remove think tags and clean up
            cleaned_response = self._clean_response(response)
            return cleaned_response
        except Exception as e:
            return f"Error calling LLM: {e}"
    
    def _clean_response(self, response: str) -> str:
        """
        Cleans the response from the LLM, removing think tags and metadata.
        
        Args:
            response: The raw response from the LLM.
            
        Returns:
            The cleaned response.
        """
        # Remove <think>...</think> tags
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # Remove <language>...</language> tags  
        cleaned = re.sub(r'<language>.*?</language>', '', cleaned, flags=re.DOTALL)
        
        # Remove extra blank lines
        cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
        
        # Trim whitespace
        cleaned = cleaned.strip()
        
        return cleaned
    
    def answer(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Provides a complete answer to a query by retrieving context and generating a response.
        
        Args:
            query: The user's query.
            k: The number of context documents to retrieve.
            
        Returns:
            A dictionary containing the query, context, and answer.
        """
        print(f"Processing query: {query}")
        
        # Preprocess query and get date info if needed
        enhanced_query, date_info = self._preprocess_query_with_date(query)
        
        # 1. Retrieve context with special logic
        print("Retrieving context...")
        
        if self._is_list_all_query(query):
            # For "list all" queries, fetch all meetings
            print('Detected "list all" query - fetching all meetings...')
            context = self._get_all_meetings_context()
        else:
            # For "today schedule" queries, increase k to find more results
            if self._is_today_schedule_query(query):
                print('Detected "today schedule" query - increasing search count...')
                k = min(k * 3, 20)  # Increase by 3x but not more than 20
                
            # Retrieve context normally, using the enhanced query if available
            search_query = enhanced_query if enhanced_query != query else query
            context = self.retrieve(search_query, k)
            
            # Add date info if necessary
            if date_info:
                context = f"{date_info}\n\n{context}"
        
        # 2. Generate answer
        print("Generating answer...")
        answer = self.generate(context, query)
        
        return {
            "query": query,
            "enhanced_query": enhanced_query if enhanced_query != query else None,
            "context": context,
            "answer": answer,
            "date_info": date_info if date_info else None
        }


def main():
    """Demo function to test the RAG Pipeline."""
    try:
        # Initialize the pipeline
        rag = RAGPipeline()
        
        # Test with a sample query
        test_query = "How many members does APEC have?"
        result = rag.answer(test_query)
        
        print("=" * 80)
        print("RAG PIPELINE TEST RESULT")
        print("=" * 80)
        print(f"Query: {result['query']}")
        print(f"\nRetrieved Context:\n{result['context']}")
        print(f"\nAnswer:\n{result['answer']}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()