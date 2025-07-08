"""
Query Processor cho hệ thống RAG APEC 2025/2027
Xử lý truy vấn người dùng với Quick Replies, phân tích intent và routing
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class IntentType(Enum):
    """Các loại intent chính cho APEC"""
    OVERVIEW = "overview"           # Tổng quan APEC
    SCHEDULE = "schedule"           # Lịch trình
    LOCATION = "location"           # Địa điểm tổ chức  
    IMMIGRATION = "immigration"     # Thủ tục nhập cảnh, visa
    CULTURE = "culture"            # Văn hóa Việt Nam & Phú Quốc
    GENERAL = "general"            # Câu hỏi chung
    GREETING = "greeting"          # Chào hỏi


@dataclass
class QueryAnalysis:
    """Kết quả phân tích truy vấn"""
    intent: IntentType
    entities: List[str]
    language: str  # 'vi' hoặc 'en'
    confidence: float
    is_quick_reply: bool
    quick_response: Optional[str] = None


class QuickReplies:
    """Bộ câu trả lời nhanh cho các truy vấn phổ biến"""
    
    QUICK_RESPONSES = {
        # Tiếng Việt
        "apec là gì": {
            "vi": "APEC (Hiệp định Hợp tác Kinh tế Châu Á - Thái Bình Dương) là diễn đàn kinh tế khu vực được thành lập năm 1989, gồm 21 nền kinh tế thành viên nhằm thúc đẩy hợp tác kinh tế và thương mại tự do.",
            "en": "APEC (Asia-Pacific Economic Cooperation) is a regional economic forum established in 1989, comprising 21 member economies aimed at promoting economic cooperation and free trade."
        },
        "xin chào": {
            "vi": "Xin chào! Tôi là trợ lý AI cho APEC 2025/2027. Tôi có thể giúp bạn tìm hiểu về lịch trình, địa điểm, thủ tục, và văn hóa Việt Nam. Bạn cần hỗ trợ gì?",
            "en": "Hello! I'm the AI assistant for APEC 2025/2027. I can help you learn about schedules, venues, procedures, and Vietnamese culture. How can I assist you?"
        },
        "apec 2027 ở đâu": {
            "vi": "APEC 2027 sẽ được tổ chức tại Việt Nam, với địa điểm chính được đề xuất là tỉnh Kiên Giang (Phú Quốc) cho các sự kiện lớn.",
            "en": "APEC 2027 will be held in Vietnam, with Kien Giang province (Phu Quoc) proposed as the main venue for major events."
        },
        "phú quốc có gì": {
            "vi": "Phú Quốc nổi tiếng với các bãi biển đẹp như Bãi Sao, Bãi Dài, các điểm tham quan như Chùa Hộ Quốc, Vinpearl Safari, và ẩm thực đặc sắc như gỏi cá trích, bún kèn, cua Hàm Ninh.",
            "en": "Phu Quoc is famous for beautiful beaches like Starfish Beach, Long Beach, attractions like Ho Quoc Pagoda, Vinpearl Safari, and special cuisine like raw herring salad, bun ken, Ham Ninh crab."
        },
        
        # Tiếng Anh
        "what is apec": {
            "en": "APEC (Asia-Pacific Economic Cooperation) is a regional economic forum established in 1989, comprising 21 member economies aimed at promoting economic cooperation and free trade.",
            "vi": "APEC (Hiệp định Hợp tác Kinh tế Châu Á - Thái Bình Dương) là diễn đàn kinh tế khu vực được thành lập năm 1989, gồm 21 nền kinh tế thành viên nhằm thúc đẩy hợp tác kinh tế và thương mại tự do."
        },
        "hello": {
            "en": "Hello! I'm the AI assistant for APEC 2025/2027. I can help you learn about schedules, venues, procedures, and Vietnamese culture. How can I assist you?",
            "vi": "Xin chào! Tôi là trợ lý AI cho APEC 2025/2027. Tôi có thể giúp bạn tìm hiểu về lịch trình, địa điểm, thủ tục, và văn hóa Việt Nam. Bạn cần hỗ trợ gì?"
        },
        "where is apec 2027": {
            "en": "APEC 2027 will be held in Vietnam, with Kien Giang province (Phu Quoc) proposed as the main venue for major events.",
            "vi": "APEC 2027 sẽ được tổ chức tại Việt Nam, với địa điểm chính được đề xuất là tỉnh Kiên Giang (Phú Quốc) cho các sự kiện lớn."
        }
    }
    
    @classmethod
    def get_quick_response(cls, query: str) -> Optional[Dict[str, str]]:
        """Kiểm tra xem có câu trả lời nhanh không"""
        query_clean = query.lower().strip()
        
        # Kiểm tra exact match trước
        if query_clean in cls.QUICK_RESPONSES:
            return cls.QUICK_RESPONSES[query_clean]
        
        # Kiểm tra partial match
        for key, response in cls.QUICK_RESPONSES.items():
            if cls._is_similar_query(query_clean, key):
                return response
        
        return None
    
    @staticmethod
    def _is_similar_query(query: str, key: str) -> bool:
        """Kiểm tra độ tương tự giữa query và key"""
        # Các pattern chào hỏi
        greeting_patterns = [
            r'(xin\s+chào|hello|hi|chào|hey)',
            r'(bạn\s+có\s+thể|can\s+you|could\s+you)',
        ]
        
        for pattern in greeting_patterns:
            if re.search(pattern, query) and re.search(pattern, key):
                return True
        
        # Kiểm tra từ khóa chung
        query_words = set(query.split())
        key_words = set(key.split())
        common_words = query_words.intersection(key_words)
        
        return len(common_words) >= 2


class IntentAnalyzer:
    """Phân tích intent từ truy vấn người dùng"""
    
    INTENT_KEYWORDS = {
        IntentType.OVERVIEW: {
            'vi': ['tổng quan', 'giới thiệu', 'apec là gì', 'thông tin chung', 'khái niệm'],
            'en': ['overview', 'introduction', 'what is', 'general info', 'about']
        },
        IntentType.SCHEDULE: {
            'vi': ['lịch trình', 'thời gian', 'khi nào', 'chương trình', 'sự kiện', 'họp', 'hội nghị'],
            'en': ['schedule', 'timeline', 'when', 'program', 'event', 'meeting', 'conference']
        },
        IntentType.LOCATION: {
            'vi': ['địa điểm', 'ở đâu', 'nơi', 'tại', 'khách sạn', 'trung tâm hội nghị'],
            'en': ['location', 'where', 'venue', 'place', 'hotel', 'convention center']
        },
        IntentType.IMMIGRATION: {
            'vi': ['visa', 'nhập cảnh', 'thủ tục', 'giấy tờ', 'hộ chiếu', 'y tế', 'di chuyển'],
            'en': ['visa', 'immigration', 'entry', 'procedure', 'passport', 'health', 'travel']
        },
        IntentType.CULTURE: {
            'vi': ['văn hóa', 'ẩm thực', 'du lịch', 'phú quốc', 'việt nam', 'truyền thống', 'lễ hội'],
            'en': ['culture', 'cuisine', 'tourism', 'phu quoc', 'vietnam', 'tradition', 'festival']
        },
        IntentType.GREETING: {
            'vi': ['xin chào', 'chào', 'hello', 'hi', 'bạn là ai', 'giúp đỡ'],
            'en': ['hello', 'hi', 'hey', 'who are you', 'help']
        }
    }
    
    def analyze_intent(self, query: str) -> Tuple[IntentType, float]:
        """Phân tích intent chính từ query"""
        query_lower = query.lower()
        language = self._detect_language(query)
        
        intent_scores = {}
        
        for intent, keywords_dict in self.INTENT_KEYWORDS.items():
            keywords = keywords_dict.get(language, [])
            score = self._calculate_intent_score(query_lower, keywords)
            intent_scores[intent] = score
        
        # Tìm intent có điểm cao nhất
        best_intent = max(intent_scores, key=intent_scores.get)
        confidence = intent_scores[best_intent]
        
        # Nếu confidence thấp, trả về GENERAL
        if confidence < 0.3:
            return IntentType.GENERAL, confidence
        
        return best_intent, confidence
    
    def _detect_language(self, query: str) -> str:
        """Phát hiện ngôn ngữ của query"""
        # Đếm ký tự tiếng Việt
        vietnamese_chars = re.findall(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', query.lower())
        vietnamese_ratio = len(vietnamese_chars) / len(query) if query else 0
        
        return 'vi' if vietnamese_ratio > 0.1 else 'en'
    
    def _calculate_intent_score(self, query: str, keywords: List[str]) -> float:
        """Tính điểm cho một intent dựa trên keywords"""
        if not keywords:
            return 0.0
        
        matches = 0
        for keyword in keywords:
            if keyword in query:
                matches += 1
        
        return matches / len(keywords)


class EntityExtractor:
    """Trích xuất entities từ truy vấn"""
    
    ENTITY_PATTERNS = {
        'year': r'20(25|27)',
        'location': r'(phú quốc|phu quoc|kiên giang|kien giang|việt nam|vietnam|hàn quốc|korea)',
        'event_type': r'(summit|meeting|conference|họp|hội nghị|diễn đàn)',
        'date': r'(\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})',
    }
    
    def extract_entities(self, query: str) -> List[str]:
        """Trích xuất entities từ query"""
        entities = []
        query_lower = query.lower()
        
        for entity_type, pattern in self.ENTITY_PATTERNS.items():
            matches = re.findall(pattern, query_lower)
            for match in matches:
                if isinstance(match, tuple):
                    entities.extend([m for m in match if m])
                else:
                    entities.append(match)
        
        return list(set(entities))  # Loại bỏ duplicate


class QueryProcessor:
    """Bộ xử lý truy vấn chính"""
    
    def __init__(self):
        self.intent_analyzer = IntentAnalyzer()
        self.entity_extractor = EntityExtractor()
    
    def process_query(self, query: str) -> QueryAnalysis:
        """Xử lý truy vấn và trả về phân tích hoàn chỉnh"""
        
        # 1. Kiểm tra Quick Replies trước
        quick_response = QuickReplies.get_quick_response(query)
        if quick_response:
            return QueryAnalysis(
                intent=IntentType.GREETING,
                entities=[],
                language=self.intent_analyzer._detect_language(query),
                confidence=1.0,
                is_quick_reply=True,
                quick_response=quick_response
            )
        
        # 2. Phân tích intent
        intent, confidence = self.intent_analyzer.analyze_intent(query)
        
        # 3. Trích xuất entities
        entities = self.entity_extractor.extract_entities(query)
        
        # 4. Phát hiện ngôn ngữ
        language = self.intent_analyzer._detect_language(query)
        
        return QueryAnalysis(
            intent=intent,
            entities=entities,
            language=language,
            confidence=confidence,
            is_quick_reply=False
        )
    
    def get_quick_reply_suggestions(self) -> List[Dict[str, str]]:
        """Trả về danh sách quick reply suggestions"""
        return [
            {"text": "Tổng quan APEC 2025", "intent": "overview"},
            {"text": "Lịch trình chi tiết", "intent": "schedule"},
            {"text": "Địa điểm & Khách sạn", "intent": "location"},
            {"text": "Thủ tục nhập cảnh & Visa", "intent": "immigration"},
            {"text": "Văn hóa Việt Nam & Phú Quốc", "intent": "culture"},
        ]


# Test functions
def test_query_processor():
    """Test function cho Query Processor"""
    processor = QueryProcessor()
    
    test_queries = [
        "APEC là gì?",
        "Lịch trình APEC 2025 như thế nào?",
        "Phú Quốc có những điểm tham quan nào?",
        "Thủ tục visa cho APEC 2027",
        "What is APEC schedule?",
        "Hello, can you help me?"
    ]
    
    print("=== Test Query Processor ===")
    for query in test_queries:
        result = processor.process_query(query)
        print(f"Query: {query}")
        print(f"Intent: {result.intent.value}")
        print(f"Language: {result.language}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Entities: {result.entities}")
        print(f"Quick Reply: {result.is_quick_reply}")
        if result.quick_response:
            print(f"Quick Response: {result.quick_response}")
        print("-" * 50)


if __name__ == "__main__":
    test_query_processor() 