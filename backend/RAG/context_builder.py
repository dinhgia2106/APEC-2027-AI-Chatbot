"""
Context Builder cho hệ thống RAG APEC 2025/2027
Tổng hợp kết quả tìm kiếm và chuẩn bị context cho LLM
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

from .search_strategy import SearchResult, ContentType
from .query_processor import QueryAnalysis, IntentType


@dataclass
class ContextSection:
    """Một section trong context"""
    title: str
    content_type: ContentType
    items: List[Dict[str, Any]]
    priority: int = 1


@dataclass
class StructuredContext:
    """Context đã được cấu trúc cho LLM"""
    sections: List[ContextSection]
    metadata: Dict[str, Any]
    language: str
    total_items: int
    query_summary: str


class DataContentLoader:
    """Load nội dung thực từ data.json"""
    
    def __init__(self, data_path: Optional[Path] = None):
        if data_path is None:
            script_dir = Path(__file__).resolve().parent
            data_path = script_dir.parent / "data" / "data.json"
        
        self.data_path = data_path
        self.data = {}
        self._load_data()
    
    def _load_data(self):
        """Load data từ JSON file"""
        try:
            if self.data_path.exists():
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                print(f"Đã load data từ {self.data_path}")
            else:
                print(f"Không tìm thấy data file tại {self.data_path}")
        except Exception as e:
            print(f"Lỗi khi load data: {e}")
    
    def get_content_by_key_path(self, key_path: str) -> Optional[str]:
        """Lấy content từ key path"""
        try:
            # Parse key path như "apec_general_info.introduction"
            keys = key_path.split('.')
            current = self.data
            
            for key in keys:
                # Handle array indices like "meetings[0]"
                if '[' in key and ']' in key:
                    base_key = key.split('[')[0]
                    index_str = key.split('[')[1].split(']')[0]
                    index = int(index_str)
                    
                    if base_key in current and isinstance(current[base_key], list):
                        if index < len(current[base_key]):
                            current = current[base_key][index]
                        else:
                            return None
                    else:
                        return None
                else:
                    if key in current:
                        current = current[key]
                    else:
                        return None
            
            # Convert to string
            if isinstance(current, (str, int, float, bool)):
                return str(current)
            elif isinstance(current, (list, dict)):
                return json.dumps(current, ensure_ascii=False, indent=2)
            else:
                return str(current)
                
        except Exception as e:
            print(f"Lỗi khi lấy content cho key {key_path}: {e}")
            return None
    
    def get_related_metadata(self, key_path: str) -> Dict[str, Any]:
        """Lấy metadata liên quan từ key path"""
        metadata = {}
        
        # Extract section info
        if 'apec_general_info' in key_path:
            metadata['source'] = 'APEC General Information'
            metadata['category'] = 'Tổng quan'
        elif 'apec_2025_korea' in key_path:
            metadata['source'] = 'APEC 2025 Korea'
            metadata['category'] = 'APEC 2025'
        elif 'vietnam_context' in key_path:
            metadata['source'] = 'Vietnam Context'
            metadata['category'] = 'Việt Nam'
        
        # Extract specific info
        if 'schedule' in key_path or 'meetings' in key_path:
            metadata['type'] = 'Lịch trình'
        elif 'venue' in key_path or 'location' in key_path:
            metadata['type'] = 'Địa điểm'
        elif 'visa' in key_path or 'immigration' in key_path:
            metadata['type'] = 'Thủ tục'
        elif 'culture' in key_path or 'tourism' in key_path:
            metadata['type'] = 'Văn hóa'
        
        return metadata


class ContextBuilder:
    """Builder chính để tạo structured context"""
    
    # Mapping content types sang tiêu đề tiếng Việt
    CONTENT_TYPE_TITLES = {
        ContentType.OVERVIEW_TEXT: "Tổng quan APEC",
        ContentType.SCHEDULE_EVENTS: "Lịch trình và Sự kiện",
        ContentType.VENUE_INFO: "Địa điểm và Cơ sở vật chất",
        ContentType.VISA_GUIDELINES: "Thủ tục Nhập cảnh và Visa",
        ContentType.CULTURE_TIPS: "Văn hóa và Du lịch",
        ContentType.GENERAL_INFO: "Thông tin Chung"
    }
    
    # Mapping content types sang tiêu đề tiếng Anh
    CONTENT_TYPE_TITLES_EN = {
        ContentType.OVERVIEW_TEXT: "APEC Overview",
        ContentType.SCHEDULE_EVENTS: "Schedule and Events",
        ContentType.VENUE_INFO: "Venues and Facilities",
        ContentType.VISA_GUIDELINES: "Immigration and Visa Procedures",
        ContentType.CULTURE_TIPS: "Culture and Tourism",
        ContentType.GENERAL_INFO: "General Information"
    }
    
    # Priority cho các content types
    CONTENT_TYPE_PRIORITY = {
        ContentType.OVERVIEW_TEXT: 1,
        ContentType.SCHEDULE_EVENTS: 2,
        ContentType.VENUE_INFO: 3,
        ContentType.VISA_GUIDELINES: 4,
        ContentType.CULTURE_TIPS: 5,
        ContentType.GENERAL_INFO: 6
    }
    
    def __init__(self, data_loader: Optional[DataContentLoader] = None):
        self.data_loader = data_loader or DataContentLoader()
    
    def build_context(self, 
                     search_results: List[SearchResult],
                     query_analysis: QueryAnalysis,
                     max_items_per_section: int = 5) -> StructuredContext:
        """Build structured context từ search results"""
        
        # Group results by content type
        grouped_results = self._group_results_by_content_type(search_results)
        
        # Create sections
        sections = []
        total_items = 0
        
        for content_type, results in grouped_results.items():
            # Load actual content
            section_items = []
            for result in results[:max_items_per_section]:
                actual_content = self.data_loader.get_content_by_key_path(result.key_path)
                if actual_content:
                    metadata = self.data_loader.get_related_metadata(result.key_path)
                    section_items.append({
                        'content': actual_content,
                        'score': result.score,
                        'key_path': result.key_path,
                        'method': result.method.value,
                        'metadata': metadata
                    })
            
            if section_items:
                # Get title based on language
                title = self._get_section_title(content_type, query_analysis.language)
                priority = self.CONTENT_TYPE_PRIORITY.get(content_type, 10)
                
                sections.append(ContextSection(
                    title=title,
                    content_type=content_type,
                    items=section_items,
                    priority=priority
                ))
                total_items += len(section_items)
        
        # Sort sections by priority
        sections.sort(key=lambda x: x.priority)
        
        # Create query summary
        query_summary = self._create_query_summary(query_analysis)
        
        # Metadata
        metadata = {
            'query_intent': query_analysis.intent.value,
            'query_language': query_analysis.language,
            'query_entities': query_analysis.entities,
            'query_confidence': query_analysis.confidence,
            'sections_count': len(sections),
            'search_methods_used': list(set([item['method'] for section in sections for item in section.items]))
        }
        
        return StructuredContext(
            sections=sections,
            metadata=metadata,
            language=query_analysis.language,
            total_items=total_items,
            query_summary=query_summary
        )
    
    def _group_results_by_content_type(self, 
                                     results: List[SearchResult]) -> Dict[ContentType, List[SearchResult]]:
        """Group search results by content type"""
        grouped = defaultdict(list)
        for result in results:
            grouped[result.content_type].append(result)
        
        # Sort results within each group by score
        for content_type in grouped:
            grouped[content_type].sort(key=lambda x: x.score, reverse=True)
        
        return dict(grouped)
    
    def _get_section_title(self, content_type: ContentType, language: str) -> str:
        """Get section title based on content type and language"""
        if language == 'vi':
            return self.CONTENT_TYPE_TITLES.get(content_type, "Thông tin")
        else:
            return self.CONTENT_TYPE_TITLES_EN.get(content_type, "Information")
    
    def _create_query_summary(self, query_analysis: QueryAnalysis) -> str:
        """Create summary of the query for context"""
        if query_analysis.language == 'vi':
            summary = f"Truy vấn về {query_analysis.intent.value}"
            if query_analysis.entities:
                summary += f" liên quan đến: {', '.join(query_analysis.entities)}"
        else:
            summary = f"Query about {query_analysis.intent.value}"
            if query_analysis.entities:
                summary += f" related to: {', '.join(query_analysis.entities)}"
        
        return summary
    
    def format_context_for_llm(self, context: StructuredContext) -> str:
        """Format structured context cho LLM prompt"""
        
        if context.language == 'vi':
            prompt = f"""Dựa trên thông tin sau, hãy trả lời câu hỏi của người dùng một cách chi tiết và chính xác.

NGỮ CẢNH TRÀ LỜI:
{context.query_summary}

THÔNG TIN LIÊN QUAN:
"""
        else:
            prompt = f"""Based on the following information, please answer the user's question in detail and accurately.

RESPONSE CONTEXT:
{context.query_summary}

RELEVANT INFORMATION:
"""
        
        # Add sections
        for section in context.sections:
            prompt += f"\n### {section.title}\n"
            
            for i, item in enumerate(section.items, 1):
                prompt += f"\n{i}. {item['content']}\n"
                
                # Add metadata if available
                if item['metadata']:
                    metadata_str = " | ".join([f"{k}: {v}" for k, v in item['metadata'].items()])
                    prompt += f"   (Nguồn: {metadata_str})\n"
        
        # Add instructions
        if context.language == 'vi':
            prompt += """

HƯỚNG DẪN TRẢ LỜI:
- Trả lời bằng tiếng Việt và tiếng Anh (song ngữ)
- Sử dụng thông tin từ ngữ cảnh trên
- Cung cấp chi tiết cụ thể và chính xác
- Nếu thông tin không đủ, hãy nói rõ
- Thêm nguồn tham khảo nếu có
"""
        else:
            prompt += """

RESPONSE GUIDELINES:
- Answer in both Vietnamese and English (bilingual)
- Use information from the context above
- Provide specific and accurate details
- If information is insufficient, state clearly
- Include references if available
"""
        
        return prompt
    
    def get_context_statistics(self, context: StructuredContext) -> Dict[str, Any]:
        """Get statistics about the context"""
        stats = {
            'total_sections': len(context.sections),
            'total_items': context.total_items,
            'language': context.language,
            'query_intent': context.metadata['query_intent']
        }
        
        # Section breakdown
        section_stats = {}
        for section in context.sections:
            section_stats[section.content_type.value] = {
                'items_count': len(section.items),
                'avg_score': sum(item['score'] for item in section.items) / len(section.items) if section.items else 0
            }
        
        stats['sections_breakdown'] = section_stats
        stats['search_methods'] = context.metadata['search_methods_used']
        
        return stats


# Test functions
def test_context_builder():
    """Test Context Builder"""
    from .query_processor import QueryProcessor, QueryAnalysis, IntentType
    from .search_strategy import SearchResult, SearchMethod
    
    # Mock data
    mock_results = [
        SearchResult(
            content="APEC overview content",
            content_type=ContentType.OVERVIEW_TEXT,
            key_path="apec_general_info.introduction",
            score=0.95,
            method=SearchMethod.SEMANTIC
        ),
        SearchResult(
            content="Schedule content",
            content_type=ContentType.SCHEDULE_EVENTS,
            key_path="apec_2025_korea.schedule.meetings[0]",
            score=0.87,
            method=SearchMethod.KEYWORD
        ),
        SearchResult(
            content="Culture content",
            content_type=ContentType.CULTURE_TIPS,
            key_path="vietnam_context.culture_and_tourism.phu_quoc_focus",
            score=0.82,
            method=SearchMethod.DOMAIN_SPECIFIC
        )
    ]
    
    mock_analysis = QueryAnalysis(
        intent=IntentType.OVERVIEW,
        entities=["APEC", "2025"],
        language="vi",
        confidence=0.9,
        is_quick_reply=False
    )
    
    # Test context builder
    builder = ContextBuilder()
    context = builder.build_context(mock_results, mock_analysis)
    
    print("=== Test Context Builder ===")
    print(f"Sections: {len(context.sections)}")
    print(f"Total items: {context.total_items}")
    print(f"Language: {context.language}")
    
    # Print formatted context
    formatted = builder.format_context_for_llm(context)
    print("\n=== Formatted Context ===")
    print(formatted[:500] + "...")
    
    # Print statistics
    stats = builder.get_context_statistics(context)
    print("\n=== Context Statistics ===")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    test_context_builder() 