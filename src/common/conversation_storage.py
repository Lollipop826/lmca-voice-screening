"""
对话系统存储模块

用于保存完整的对话轮次，包括：
- 用户提问
- 生成的查询语句
- 检索的知识
- LLM的回答
"""

from typing import List, Dict, Any, Optional, TypedDict
from datetime import datetime
from pathlib import Path
import json
import os


class RetrievalResult(TypedDict, total=False):
    """检索结果"""
    rank: int
    text: str
    metadata: Dict[str, Any]
    score: Optional[float]
    highlighted_sentences: Optional[List[Dict[str, Any]]]


class DialogueTurn(TypedDict, total=False):
    """完整的对话轮次"""
    turn_id: int
    timestamp: str
    
    # 用户输入
    user_question: str
    user_emotion: Optional[str]
    
    # 系统内部处理
    dimension_id: Optional[str]  # 当前评估的维度
    dimension_name: Optional[str]
    generated_query: str  # 生成的查询语句
    query_keywords: List[str]
    
    # 检索结果
    retrieved_documents: List[RetrievalResult]
    retrieval_method: str  # "paragraph" | "paragraph_with_highlight"
    
    # LLM响应
    assistant_response: str
    response_metadata: Optional[Dict[str, Any]]  # 如token使用量等
    
    # 其他信息
    notes: Optional[str]


class ConversationSession(TypedDict, total=False):
    """完整的对话会话"""
    session_id: str
    user_id: Optional[str]
    start_time: str
    last_update: str
    
    # 用户画像
    profile: Optional[Dict[str, Any]]
    
    # 评估维度状态
    dimensions: List[Dict[str, Any]]
    
    # 对话历史
    dialogue_turns: List[DialogueTurn]
    
    # 会话元数据
    metadata: Optional[Dict[str, Any]]


class ConversationStorage:
    """对话存储管理器"""
    
    def __init__(self, storage_dir: str = "data/conversations"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def create_session(
        self, 
        session_id: str,
        user_id: Optional[str] = None,
        profile: Optional[Dict[str, Any]] = None,
        dimensions: Optional[List[Dict[str, Any]]] = None
    ) -> ConversationSession:
        """创建新的对话会话"""
        now = datetime.now().isoformat()
        
        session: ConversationSession = {
            "session_id": session_id,
            "user_id": user_id,
            "start_time": now,
            "last_update": now,
            "profile": profile,
            "dimensions": dimensions or [],
            "dialogue_turns": [],
            "metadata": {}
        }
        
        self._save_session(session)
        return session
    
    def add_turn(
        self,
        session_id: str,
        user_question: str,
        generated_query: str,
        retrieved_documents: List[RetrievalResult],
        assistant_response: str,
        dimension_id: Optional[str] = None,
        dimension_name: Optional[str] = None,
        query_keywords: Optional[List[str]] = None,
        user_emotion: Optional[str] = None,
        retrieval_method: str = "paragraph",
        response_metadata: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None
    ) -> ConversationSession:
        """添加一个对话轮次"""
        session = self.load_session(session_id)
        
        turn_id = len(session["dialogue_turns"]) + 1
        timestamp = datetime.now().isoformat()
        
        turn: DialogueTurn = {
            "turn_id": turn_id,
            "timestamp": timestamp,
            "user_question": user_question,
            "user_emotion": user_emotion,
            "dimension_id": dimension_id,
            "dimension_name": dimension_name,
            "generated_query": generated_query,
            "query_keywords": query_keywords or [],
            "retrieved_documents": retrieved_documents,
            "retrieval_method": retrieval_method,
            "assistant_response": assistant_response,
            "response_metadata": response_metadata,
            "notes": notes
        }
        
        session["dialogue_turns"].append(turn)
        session["last_update"] = timestamp
        
        self._save_session(session)
        return session
    
    def load_session(self, session_id: str) -> ConversationSession:
        """加载对话会话"""
        session_file = self.storage_dir / f"{session_id}.json"
        
        if not session_file.exists():
            raise FileNotFoundError(f"Session {session_id} not found")
        
        with open(session_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def update_dimensions(
        self,
        session_id: str,
        dimensions: List[Dict[str, Any]]
    ) -> ConversationSession:
        """更新维度状态"""
        session = self.load_session(session_id)
        session["dimensions"] = dimensions
        session["last_update"] = datetime.now().isoformat()
        self._save_session(session)
        return session
    
    def update_profile(
        self,
        session_id: str,
        profile: Dict[str, Any]
    ) -> ConversationSession:
        """更新用户画像"""
        session = self.load_session(session_id)
        session["profile"] = profile
        session["last_update"] = datetime.now().isoformat()
        self._save_session(session)
        return session
    
    def get_conversation_history(
        self,
        session_id: str,
        max_turns: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """获取对话历史（用于传给LLM）"""
        session = self.load_session(session_id)
        turns = session["dialogue_turns"]
        
        if max_turns:
            turns = turns[-max_turns:]
        
        history = []
        for turn in turns:
            history.append({"role": "user", "content": turn["user_question"]})
            history.append({"role": "assistant", "content": turn["assistant_response"]})
        
        return history
    
    def export_session_to_jsonl(self, session_id: str, output_file: str):
        """导出会话为JSONL格式（每个对话轮次一行）"""
        session = self.load_session(session_id)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # 写入会话元信息
            meta = {
                "type": "session_meta",
                "session_id": session["session_id"],
                "user_id": session.get("user_id"),
                "start_time": session["start_time"],
                "profile": session.get("profile"),
                "dimensions": session.get("dimensions")
            }
            f.write(json.dumps(meta, ensure_ascii=False) + '\n')
            
            # 写入每个对话轮次
            for turn in session["dialogue_turns"]:
                turn_record = {"type": "dialogue_turn", **turn}
                f.write(json.dumps(turn_record, ensure_ascii=False) + '\n')
    
    def list_sessions(self) -> List[str]:
        """列出所有会话ID"""
        return [f.stem for f in self.storage_dir.glob("*.json")]
    
    def _save_session(self, session: ConversationSession):
        """保存会话到文件"""
        session_file = self.storage_dir / f"{session['session_id']}.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session, f, ensure_ascii=False, indent=2)
    
    def delete_session(self, session_id: str):
        """删除会话"""
        session_file = self.storage_dir / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()


# 便捷函数
def create_storage(storage_dir: str = "data/conversations") -> ConversationStorage:
    """创建存储管理器"""
    return ConversationStorage(storage_dir)

