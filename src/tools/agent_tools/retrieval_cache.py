"""
检索缓存工具 - 避免重复检索相同查询
"""
import hashlib
import time
from typing import Optional, Dict, Any


class RetrievalCache:
    """检索结果缓存"""
    
    def __init__(self, maxsize: int = 100, ttl: int = 1800):
        """
        Args:
            maxsize: 最大缓存条目数
            ttl: 缓存过期时间（秒），默认30分钟
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._maxsize = maxsize
        self._ttl = ttl
    
    def _get_cache_key(self, query: str, top_k: int) -> str:
        """生成缓存键"""
        key_str = f"{query}_{top_k}"
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()
    
    def get(self, query: str, top_k: int) -> Optional[str]:
        """获取缓存结果"""
        key = self._get_cache_key(query, top_k)
        
        if key in self._cache:
            cache_entry = self._cache[key]
            
            # 检查是否过期
            if time.time() - cache_entry['timestamp'] < self._ttl:
                cache_entry['hits'] += 1
                print(f"[Cache] ✅ 命中缓存: query_chars={len(query)}, hits={cache_entry['hits']}")
                return cache_entry['result']
            else:
                # 过期，删除
                del self._cache[key]
                print(f"[Cache] ⏰ 缓存过期: query_chars={len(query)}")
        
        return None
    
    def set(self, query: str, top_k: int, result: str):
        """设置缓存"""
        # 如果缓存满了，删除最旧的项
        if len(self._cache) >= self._maxsize:
            oldest_key = min(self._cache.keys(), 
                           key=lambda k: self._cache[k]['timestamp'])
            del self._cache[oldest_key]
            print(f"[Cache] 🗑️  缓存已满，删除最旧项")
        
        key = self._get_cache_key(query, top_k)
        self._cache[key] = {
            'result': result,
            'timestamp': time.time(),
            'hits': 0
        }
        print(f"[Cache] 💾 已缓存: query_chars={len(query)}")
    
    def clear(self):
        """清空缓存"""
        self._cache.clear()
        print("[Cache] 🧹 已清空所有缓存")
    
    def stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_hits = sum(entry['hits'] for entry in self._cache.values())
        return {
            'size': len(self._cache),
            'maxsize': self._maxsize,
            'total_hits': total_hits,
            'hit_rate': total_hits / max(len(self._cache), 1)
        }
