"""
Search Strategy cho hệ thống RAG APEC 2025/2027
Implement chiến lược tìm kiếm đa dạng với content types và boost factors
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import re
from .query_processor import IntentType, QueryAnalysis


class ContentType(Enum):
    """Các loại nội dung trong dữ liệu APEC"""
    OVERVIEW_TEXT = "overview_text"
    SCHEDULE_EVENTS = "schedule_events"
    VENUE_INFO = "venue_info"
    VISA_GUIDELINES = "visa_guidelines"
    CULTURE_TIPS = "culture_tips"
    GENERAL_INFO = "general_info"


class SearchMethod(Enum):
    """Các phương pháp tìm kiếm"""
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    DOMAIN_SPECIFIC = "domain_specific"
    HYBRID = "hybrid"


@dataclass
class SearchConfig:
    """Cấu hình tìm kiếm cho mỗi intent"""
    content_types: List[ContentType]
    boost_factors: Dict[ContentType, float]
    search_methods: List[SearchMethod]
    max_results: int = 20
    min_confidence: float = 0.3


@dataclass
class SearchResult:
    """Kết quả tìm kiếm"""
    content: str
    content_type: ContentType
    key_path: str
    score: float
    method: SearchMethod
    metadata: Dict[str, Any] = None


class ContentTypeMapper:
    """Mapping từ key paths trong data.json sang content types"""
    
    CONTENT_TYPE_PATTERNS = {
        ContentType.OVERVIEW_TEXT: [
            r'apec_general_info\.(introduction|mission|vision_2040)',
            r'apec_2025_korea\.overview',
            r'vietnam_context\.role_in_apec\.introduction'
        ],
        ContentType.SCHEDULE_EVENTS: [
            r'apec_2025_korea\.schedule\.meetings\[\d+\]',
            r'vietnam_context\.role_in_apec\.hosting_events',
            r'.*\.(date|schedule|timeline|meetings)'
        ],
        ContentType.VENUE_INFO: [
            r'apec_2025_korea\.venues',
            r'vietnam_context\.culture_and_tourism\.phu_quoc_focus\.attractions',
            r'.*\.(location|venue|place|hotel)'
        ],
        ContentType.VISA_GUIDELINES: [
            r'apec_2025_korea\.travel_guide\.(entry_and_visa|health_regulations)',
            r'.*\.(visa|immigration|entry|passport|health)'
        ],
        ContentType.CULTURE_TIPS: [
            r'vietnam_context\.culture_and_tourism',
            r'.*\.(culture|cuisine|festival|tradition|tourism)'
        ],
        ContentType.GENERAL_INFO: [
            r'apec_general_info\.member_economies',
            r'.*\.(title|description|general)'
        ]
    }
    
    @classmethod
    def get_content_type(cls, key_path: str) -> ContentType:
        """Xác định content type từ key path"""
        for content_type, patterns in cls.CONTENT_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, key_path):
                    return content_type
        return ContentType.GENERAL_INFO


class SearchStrategy:
    """Chiến lược tìm kiếm chính"""
    
    # Cấu hình cho từng intent
    INTENT_CONFIGS = {
        IntentType.OVERVIEW: SearchConfig(
            content_types=[ContentType.OVERVIEW_TEXT, ContentType.GENERAL_INFO],
            boost_factors={
                ContentType.OVERVIEW_TEXT: 25.0,
                ContentType.GENERAL_INFO: 15.0
            },
            search_methods=[SearchMethod.SEMANTIC, SearchMethod.KEYWORD],
            max_results=15
        ),
        IntentType.SCHEDULE: SearchConfig(
            content_types=[ContentType.SCHEDULE_EVENTS, ContentType.GENERAL_INFO],
            boost_factors={
                ContentType.SCHEDULE_EVENTS: 20.0,
                ContentType.GENERAL_INFO: 8.0
            },
            search_methods=[SearchMethod.KEYWORD, SearchMethod.SEMANTIC],
            max_results=20
        ),
        IntentType.LOCATION: SearchConfig(
            content_types=[ContentType.VENUE_INFO, ContentType.CULTURE_TIPS],
            boost_factors={
                ContentType.VENUE_INFO: 18.0,
                ContentType.CULTURE_TIPS: 12.0
            },
            search_methods=[SearchMethod.SEMANTIC, SearchMethod.DOMAIN_SPECIFIC],
            max_results=15
        ),
        IntentType.IMMIGRATION: SearchConfig(
            content_types=[ContentType.VISA_GUIDELINES, ContentType.GENERAL_INFO],
            boost_factors={
                ContentType.VISA_GUIDELINES: 22.0,
                ContentType.GENERAL_INFO: 10.0
            },
            search_methods=[SearchMethod.KEYWORD, SearchMethod.DOMAIN_SPECIFIC],
            max_results=12
        ),
        IntentType.CULTURE: SearchConfig(
            content_types=[ContentType.CULTURE_TIPS, ContentType.VENUE_INFO],
            boost_factors={
                ContentType.CULTURE_TIPS: 20.0,
                ContentType.VENUE_INFO: 14.0
            },
            search_methods=[SearchMethod.SEMANTIC, SearchMethod.KEYWORD],
            max_results=18
        ),
        IntentType.GENERAL: SearchConfig(
            content_types=[ct for ct in ContentType],
            boost_factors={
                ContentType.OVERVIEW_TEXT: 15.0,
                ContentType.SCHEDULE_EVENTS: 12.0,
                ContentType.VENUE_INFO: 12.0,
                ContentType.VISA_GUIDELINES: 12.0,
                ContentType.CULTURE_TIPS: 12.0,
                ContentType.GENERAL_INFO: 8.0
            },
            search_methods=[SearchMethod.HYBRID],
            max_results=25
        ),
        IntentType.GREETING: SearchConfig(
            content_types=[ContentType.OVERVIEW_TEXT],
            boost_factors={ContentType.OVERVIEW_TEXT: 10.0},
            search_methods=[SearchMethod.SEMANTIC],
            max_results=5
        )
    }
    
    # Keywords cho domain-specific search
    DOMAIN_KEYWORDS = {
        ContentType.SCHEDULE_EVENTS: [
            'meeting', 'conference', 'summit', 'forum', 'ministerial',
            'họp', 'hội nghị', 'diễn đàn', 'cấp bộ trưởng', 'thượng đỉnh'
        ],
        ContentType.VENUE_INFO: [
            'hotel', 'convention center', 'venue', 'location', 'city',
            'khách sạn', 'trung tâm hội nghị', 'địa điểm', 'thành phố'
        ],
        ContentType.VISA_GUIDELINES: [
            'visa', 'passport', 'immigration', 'entry', 'health', 'vaccination',
            'hộ chiếu', 'nhập cảnh', 'y tế', 'tiêm chủng'
        ],
        ContentType.CULTURE_TIPS: [
            'culture', 'tradition', 'cuisine', 'festival', 'tourism', 'attraction',
            'văn hóa', 'truyền thống', 'ẩm thực', 'lễ hội', 'du lịch', 'tham quan'
        ]
    }
    
    def __init__(self):
        self.content_mapper = ContentTypeMapper()
    
    def get_search_config(self, query_analysis: QueryAnalysis) -> SearchConfig:
        """Lấy cấu hình tìm kiếm cho intent"""
        return self.INTENT_CONFIGS.get(query_analysis.intent, self.INTENT_CONFIGS[IntentType.GENERAL])
    
    def filter_by_content_types(self, 
                               all_keys: List[str], 
                               target_types: List[ContentType]) -> List[str]:
        """Lọc keys theo content types"""
        filtered_keys = []
        for key in all_keys:
            content_type = self.content_mapper.get_content_type(key)
            if content_type in target_types:
                filtered_keys.append(key)
        return filtered_keys
    
    def keyword_search(self, 
                      query: str, 
                      keys: List[str], 
                      texts: List[str]) -> List[Tuple[int, float]]:
        """Tìm kiếm theo keyword"""
        query_words = set(query.lower().split())
        results = []
        
        for i, text in enumerate(texts):
            text_words = set(text.lower().split())
            
            # Exact matches
            exact_matches = len(query_words.intersection(text_words))
            
            # Partial matches (substring)
            partial_matches = 0
            for word in query_words:
                if any(word in text_word for text_word in text_words):
                    partial_matches += 1
            
            if exact_matches > 0 or partial_matches > 0:
                score = exact_matches * 2.0 + partial_matches * 1.0
                score = score / len(query_words)  # Normalize
                results.append((i, score))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def domain_specific_search(self, 
                             query: str,
                             keys: List[str], 
                             texts: List[str],
                             content_types: List[ContentType]) -> List[Tuple[int, float]]:
        """Tìm kiếm domain-specific với keywords được định nghĩa sẵn"""
        query_lower = query.lower()
        results = []
        
        # Collect domain keywords for target content types
        domain_keywords = set()
        for ct in content_types:
            if ct in self.DOMAIN_KEYWORDS:
                domain_keywords.update(self.DOMAIN_KEYWORDS[ct])
        
        for i, text in enumerate(texts):
            text_lower = text.lower()
            key = keys[i]
            content_type = self.content_mapper.get_content_type(key)
            
            if content_type not in content_types:
                continue
            
            # Score based on domain keywords
            score = 0.0
            for keyword in domain_keywords:
                if keyword in query_lower and keyword in text_lower:
                    score += 2.0
                elif keyword in text_lower:
                    score += 1.0
            
            if score > 0:
                results.append((i, score))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def apply_boost_factors(self, 
                          results: List[Tuple[int, float]], 
                          keys: List[str],
                          boost_factors: Dict[ContentType, float]) -> List[Tuple[int, float]]:
        """Áp dụng boost factors"""
        boosted_results = []
        
        for idx, score in results:
            key = keys[idx]
            content_type = self.content_mapper.get_content_type(key)
            boost = boost_factors.get(content_type, 1.0)
            boosted_score = score * boost
            boosted_results.append((idx, boosted_score))
        
        return sorted(boosted_results, key=lambda x: x[1], reverse=True)
    
    def combine_search_results(self, 
                             keyword_results: List[Tuple[int, float]],
                             semantic_results: List[Tuple[int, float]],
                             domain_results: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """Kết hợp kết quả từ nhiều phương pháp tìm kiếm"""
        combined_scores = {}
        
        # Combine scores with weights
        weights = {
            'keyword': 0.4,
            'semantic': 0.4,
            'domain': 0.2
        }
        
        for idx, score in keyword_results:
            combined_scores[idx] = combined_scores.get(idx, 0) + score * weights['keyword']
        
        for idx, score in semantic_results:
            combined_scores[idx] = combined_scores.get(idx, 0) + score * weights['semantic']
        
        for idx, score in domain_results:
            combined_scores[idx] = combined_scores.get(idx, 0) + score * weights['domain']
        
        return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    def prepare_search_context(self, 
                             query_analysis: QueryAnalysis,
                             all_keys: List[str]) -> Dict[str, Any]:
        """Chuẩn bị context cho tìm kiếm"""
        config = self.get_search_config(query_analysis)
        
        # Filter keys by content types
        filtered_keys = self.filter_by_content_types(all_keys, config.content_types)
        
        # Add entities to search context
        enhanced_query = query_analysis.intent.value
        if query_analysis.entities:
            enhanced_query += " " + " ".join(query_analysis.entities)
        
        return {
            'config': config,
            'filtered_keys': filtered_keys,
            'enhanced_query': enhanced_query,
            'original_query': enhanced_query,  # Will be replaced with actual query
            'language': query_analysis.language
        }


class SearchResultRanker:
    """Ranking và scoring kết quả tìm kiếm"""
    
    def __init__(self):
        pass
    
    def rank_results(self, 
                    results: List[SearchResult], 
                    query_analysis: QueryAnalysis) -> List[SearchResult]:
        """Ranking kết quả dựa trên nhiều yếu tố"""
        
        # Additional scoring factors
        for result in results:
            # Language preference scoring
            if query_analysis.language == 'vi':
                if any(vietnamese_char in result.content for vietnamese_char in 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'):
                    result.score *= 1.2
            
            # Content length preference (not too short, not too long)
            content_length = len(result.content)
            if 50 <= content_length <= 500:
                result.score *= 1.1
            elif content_length < 20:
                result.score *= 0.8
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def remove_duplicates(self, results: List[SearchResult]) -> List[SearchResult]:
        """Loại bỏ kết quả trùng lặp"""
        seen_content = set()
        unique_results = []
        
        for result in results:
            content_hash = hash(result.content[:100])  # Hash first 100 chars
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        return unique_results


# Test functions
def test_search_strategy():
    """Test Search Strategy components"""
    from .query_processor import QueryProcessor
    
    processor = QueryProcessor()
    strategy = SearchStrategy()
    ranker = SearchResultRanker()
    
    test_queries = [
        "Lịch trình APEC 2025",
        "Địa điểm tổ chức ở Phú Quốc",
        "Thủ tục visa cho APEC",
        "Văn hóa Việt Nam"
    ]
    
    print("=== Test Search Strategy ===")
    for query in test_queries:
        analysis = processor.process_query(query)
        config = strategy.get_search_config(analysis)
        
        print(f"Query: {query}")
        print(f"Intent: {analysis.intent.value}")
        print(f"Content Types: {[ct.value for ct in config.content_types]}")
        print(f"Boost Factors: {config.boost_factors}")
        print(f"Search Methods: {[sm.value for sm in config.search_methods]}")
        print("-" * 50)


if __name__ == "__main__":
    test_search_strategy() 