"""
Enhanced Vector Search cho hệ thống RAG APEC 2025/2027
Mở rộng FAISS search với filtering, boosting và multi-strategy search
"""

import json
import numpy as np
import faiss
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .search_strategy import (
    SearchStrategy, SearchResult, SearchMethod, ContentType, 
    ContentTypeMapper, SearchResultRanker, SearchConfig
)
from .query_processor import QueryAnalysis


@dataclass
class VectorSearchConfig:
    """Cấu hình cho vector search"""
    top_k: int = 20
    similarity_threshold: float = 0.3
    use_filtering: bool = True
    use_boosting: bool = True
    normalize_scores: bool = True


class EnhancedVectorSearch:
    """Vector search nâng cao với FAISS index"""
    
    def __init__(self, 
                 index_path: Optional[Path] = None,
                 id2key_path: Optional[Path] = None,
                 embeddings_path: Optional[Path] = None,
                 keys_path: Optional[Path] = None):
        
        # Đường dẫn mặc định
        script_dir = Path(__file__).resolve().parent
        vector_store_dir = script_dir.parent / "VectorStore"
        embed_dir = script_dir.parent / "Embed"
        
        self.index_path = index_path or vector_store_dir / "apec.index"
        self.id2key_path = id2key_path or vector_store_dir / "id2key.json"
        self.embeddings_path = embeddings_path or embed_dir / "embeddings.npy"
        self.keys_path = keys_path or embed_dir / "keys.txt"
        
        # Components
        self.search_strategy = SearchStrategy()
        self.content_mapper = ContentTypeMapper()
        self.ranker = SearchResultRanker()
        
        # Data
        self.index = None
        self.keys = []
        self.embeddings = None
        self.key_to_idx = {}
        
        # Load components
        self._load_components()
    
    def _load_components(self):
        """Load FAISS index, keys và embeddings"""
        try:
            # Load FAISS index
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                print(f"Đã load FAISS index từ {self.index_path}")
            else:
                print(f"Không tìm thấy FAISS index tại {self.index_path}")
                return
            
            # Load keys
            if self.id2key_path.exists():
                with open(self.id2key_path, 'r', encoding='utf-8') as f:
                    self.keys = json.load(f)
                print(f"Đã load {len(self.keys)} keys từ {self.id2key_path}")
            elif self.keys_path.exists():
                with open(self.keys_path, 'r', encoding='utf-8') as f:
                    self.keys = f.read().strip().split('\n')
                print(f"Đã load {len(self.keys)} keys từ {self.keys_path}")
            
            # Tạo mapping key -> index
            self.key_to_idx = {key: idx for idx, key in enumerate(self.keys)}
            
            # Load embeddings (cho filtering)
            if self.embeddings_path.exists():
                self.embeddings = np.load(self.embeddings_path)
                print(f"Đã load embeddings với shape {self.embeddings.shape}")
            
        except Exception as e:
            print(f"Lỗi khi load components: {e}")
    
    def _create_filtered_index(self, 
                              target_content_types: List[ContentType]) -> Tuple[faiss.Index, List[str], np.ndarray]:
        """Tạo index đã được filter theo content types"""
        if self.embeddings is None:
            # Fallback: sử dụng index gốc
            return self.index, self.keys, None
        
        # Filter indices theo content types
        filtered_indices = []
        filtered_keys = []
        
        for idx, key in enumerate(self.keys):
            content_type = self.content_mapper.get_content_type(key)
            if content_type in target_content_types:
                filtered_indices.append(idx)
                filtered_keys.append(key)
        
        if not filtered_indices:
            # Không có kết quả phù hợp, trả về index gốc
            return self.index, self.keys, None
        
        # Tạo embeddings đã filter
        filtered_embeddings = self.embeddings[filtered_indices]
        
        # Tạo index mới với embeddings đã filter
        filtered_index = faiss.IndexFlatIP(self.embeddings.shape[1])
        filtered_index.add(filtered_embeddings.astype('float32'))
        
        return filtered_index, filtered_keys, filtered_indices
    
    def semantic_search(self, 
                       query_embedding: np.ndarray,
                       search_config: SearchConfig,
                       vector_config: VectorSearchConfig) -> List[SearchResult]:
        """Semantic search sử dụng FAISS"""
        
        if self.index is None:
            return []
        
        try:
            # Lựa chọn index (filtered hoặc gốc)
            if vector_config.use_filtering and search_config.content_types:
                search_index, search_keys, original_indices = self._create_filtered_index(
                    search_config.content_types
                )
            else:
                search_index, search_keys = self.index, self.keys
                original_indices = None
            
            # Thực hiện search
            query_embedding = query_embedding.astype('float32').reshape(1, -1)
            similarities, indices = search_index.search(query_embedding, 
                                                       min(vector_config.top_k, search_index.ntotal))
            
            results = []
            for i in range(len(similarities[0])):
                similarity = float(similarities[0][i])
                idx = int(indices[0][i])
                
                if similarity < vector_config.similarity_threshold:
                    break
                
                # Lấy key và nội dung
                if idx < len(search_keys):
                    key = search_keys[idx]
                    
                    # Lấy content từ key (cần load từ data.json)
                    content = self._get_content_from_key(key)
                    if not content:
                        continue
                    
                    # Xác định content type
                    content_type = self.content_mapper.get_content_type(key)
                    
                    # Áp dụng boost factor
                    score = similarity
                    if vector_config.use_boosting and content_type in search_config.boost_factors:
                        score *= search_config.boost_factors[content_type]
                    
                    results.append(SearchResult(
                        content=content,
                        content_type=content_type,
                        key_path=key,
                        score=score,
                        method=SearchMethod.SEMANTIC,
                        metadata={
                            'similarity': similarity,
                            'original_idx': original_indices[idx] if original_indices else idx
                        }
                    ))
            
            return results
            
        except Exception as e:
            print(f"Lỗi trong semantic search: {e}")
            return []
    
    def keyword_search_with_embeddings(self,
                                     query: str,
                                     search_config: SearchConfig,
                                     vector_config: VectorSearchConfig) -> List[SearchResult]:
        """Keyword search kết hợp với embedding data"""
        
        # Lấy tất cả texts từ keys
        all_texts = []
        valid_keys = []
        
        for key in self.keys:
            content = self._get_content_from_key(key)
            if content:
                all_texts.append(content)
                valid_keys.append(key)
        
        if not all_texts:
            return []
        
        # Filter theo content types nếu cần
        if vector_config.use_filtering and search_config.content_types:
            filtered_texts = []
            filtered_keys = []
            
            for text, key in zip(all_texts, valid_keys):
                content_type = self.content_mapper.get_content_type(key)
                if content_type in search_config.content_types:
                    filtered_texts.append(text)
                    filtered_keys.append(key)
            
            all_texts, valid_keys = filtered_texts, filtered_keys
        
        # Thực hiện keyword search
        keyword_results = self.search_strategy.keyword_search(query, valid_keys, all_texts)
        
        results = []
        for idx, score in keyword_results[:vector_config.top_k]:
            if idx < len(valid_keys):
                key = valid_keys[idx]
                content = all_texts[idx]
                content_type = self.content_mapper.get_content_type(key)
                
                # Áp dụng boost factor
                if vector_config.use_boosting and content_type in search_config.boost_factors:
                    score *= search_config.boost_factors[content_type]
                
                results.append(SearchResult(
                    content=content,
                    content_type=content_type,
                    key_path=key,
                    score=score,
                    method=SearchMethod.KEYWORD,
                    metadata={'keyword_score': score}
                ))
        
        return results
    
    def domain_specific_search(self,
                             query: str,
                             search_config: SearchConfig,
                             vector_config: VectorSearchConfig) -> List[SearchResult]:
        """Domain-specific search"""
        
        # Lấy tất cả texts từ keys
        all_texts = []
        valid_keys = []
        
        for key in self.keys:
            content = self._get_content_from_key(key)
            if content:
                all_texts.append(content)
                valid_keys.append(key)
        
        if not all_texts:
            return []
        
        # Thực hiện domain-specific search
        domain_results = self.search_strategy.domain_specific_search(
            query, valid_keys, all_texts, search_config.content_types
        )
        
        results = []
        for idx, score in domain_results[:vector_config.top_k]:
            if idx < len(valid_keys):
                key = valid_keys[idx]
                content = all_texts[idx]
                content_type = self.content_mapper.get_content_type(key)
                
                # Áp dụng boost factor
                if vector_config.use_boosting and content_type in search_config.boost_factors:
                    score *= search_config.boost_factors[content_type]
                
                results.append(SearchResult(
                    content=content,
                    content_type=content_type,
                    key_path=key,
                    score=score,
                    method=SearchMethod.DOMAIN_SPECIFIC,
                    metadata={'domain_score': score}
                ))
        
        return results
    
    def hybrid_search(self,
                     query: str,
                     query_embedding: np.ndarray,
                     search_config: SearchConfig,
                     vector_config: VectorSearchConfig) -> List[SearchResult]:
        """Hybrid search kết hợp semantic, keyword và domain-specific"""
        
        # Thực hiện các loại search
        semantic_results = self.semantic_search(query_embedding, search_config, vector_config)
        keyword_results = self.keyword_search_with_embeddings(query, search_config, vector_config)
        domain_results = self.domain_specific_search(query, search_config, vector_config)
        
        # Kết hợp results
        all_results = semantic_results + keyword_results + domain_results
        
        # Remove duplicates và rank
        unique_results = self.ranker.remove_duplicates(all_results)
        
        # Limit results
        return unique_results[:search_config.max_results]
    
    def search(self,
              query: str,
              query_embedding: np.ndarray,
              query_analysis: QueryAnalysis,
              vector_config: Optional[VectorSearchConfig] = None) -> List[SearchResult]:
        """Main search method"""
        
        if vector_config is None:
            vector_config = VectorSearchConfig()
        
        # Lấy search config từ query analysis
        search_config = self.search_strategy.get_search_config(query_analysis)
        
        results = []
        
        # Thực hiện search theo methods được chỉ định
        for method in search_config.search_methods:
            if method == SearchMethod.SEMANTIC:
                results.extend(self.semantic_search(query_embedding, search_config, vector_config))
            elif method == SearchMethod.KEYWORD:
                results.extend(self.keyword_search_with_embeddings(query, search_config, vector_config))
            elif method == SearchMethod.DOMAIN_SPECIFIC:
                results.extend(self.domain_specific_search(query, search_config, vector_config))
            elif method == SearchMethod.HYBRID:
                results.extend(self.hybrid_search(query, query_embedding, search_config, vector_config))
        
        # Remove duplicates và rank final results
        unique_results = self.ranker.remove_duplicates(results)
        ranked_results = self.ranker.rank_results(unique_results, query_analysis)
        
        # Apply max results limit
        return ranked_results[:search_config.max_results]
    
    def _get_content_from_key(self, key_path: str) -> str:
        """Lấy content từ key path (cần implement load từ data.json)"""
        # TODO: Implement loading from data.json based on key path
        # Tạm thời return key path làm placeholder
        return f"Content for {key_path}"
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Lấy thống kê về index"""
        stats = {}
        
        if self.index:
            stats['total_vectors'] = self.index.ntotal
            stats['vector_dimension'] = self.index.d
        
        stats['total_keys'] = len(self.keys)
        
        if self.embeddings is not None:
            stats['embeddings_shape'] = self.embeddings.shape
        
        # Content type distribution
        content_type_counts = {}
        for key in self.keys:
            content_type = self.content_mapper.get_content_type(key)
            content_type_counts[content_type.value] = content_type_counts.get(content_type.value, 0) + 1
        
        stats['content_type_distribution'] = content_type_counts
        
        return stats


# Utility functions
def rebuild_enhanced_index(data_json_path: Path, 
                         output_dir: Path,
                         model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
    """Rebuild FAISS index với enhanced features"""
    
    # Import here để tránh circular import
    from sentence_transformers import SentenceTransformer
    import json
    
    # Load data
    with open(data_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Flatten data (sử dụng code từ embed_data.py)
    def flatten(data_obj, parent_key=""):
        items = []
        if isinstance(data_obj, dict):
            for k, v in data_obj.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                items.extend(flatten(v, new_key))
        elif isinstance(data_obj, list):
            for idx, v in enumerate(data_obj):
                new_key = f"{parent_key}[{idx}]"
                items.extend(flatten(v, new_key))
        else:
            text = str(data_obj).strip()
            if text:
                items.append((parent_key, text))
        return items
    
    entries = flatten(data)
    keys = [k for k, _ in entries]
    texts = [t for _, t in entries]
    
    # Generate embeddings
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, 
                            convert_to_numpy=True, normalize_embeddings=True)
    
    # Build FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    
    # Save files
    output_dir.mkdir(exist_ok=True)
    faiss.write_index(index, str(output_dir / "apec.index"))
    
    with open(output_dir / "id2key.json", 'w', encoding='utf-8') as f:
        json.dump(keys, f, ensure_ascii=False, indent=2)
    
    # Save embeddings và keys cho enhanced search
    np.save(output_dir.parent / "Embed" / "embeddings.npy", embeddings)
    with open(output_dir.parent / "Embed" / "keys.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(keys))
    
    print(f"Enhanced index rebuilt với {len(keys)} vectors tại {output_dir}")


# Test functions
def test_enhanced_vector_search():
    """Test Enhanced Vector Search"""
    from .query_processor import QueryProcessor
    
    processor = QueryProcessor()
    vector_search = EnhancedVectorSearch()
    
    print("=== Test Enhanced Vector Search ===")
    print("Index Stats:", vector_search.get_index_stats())
    
    # Test queries
    test_queries = [
        "Lịch trình APEC 2025",
        "Địa điểm Phú Quốc", 
        "Thủ tục visa"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        analysis = processor.process_query(query)
        
        # Mock query embedding (cần model thật để test)
        mock_embedding = np.random.rand(768).astype('float32')
        
        results = vector_search.search(query, mock_embedding, analysis)
        print(f"Found {len(results)} results")
        
        for i, result in enumerate(results[:3]):
            print(f"  {i+1}. {result.content_type.value}: {result.score:.3f}")
            print(f"     {result.content[:100]}...")


if __name__ == "__main__":
    test_enhanced_vector_search() 