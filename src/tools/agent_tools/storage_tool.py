"""
对话存储工具 - 供Agent调用
"""

from typing import Type, Optional
import json

from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.tools import BaseTool

from src.common.conversation_storage import ConversationStorage


class ConversationStorageToolArgs(BaseModel):
    """对话存储工具参数"""
    session_id: str = Field(..., description="会话ID")
    action: str = Field(..., description="操作类型：'save_turn'（保存对话轮次）或'get_history'（获取历史）")
    turn_data: Optional[str] = Field(
        default=None,
        description="对话轮次数据（JSON格式），action='save_turn'时必需"
    )
    max_turns: Optional[int] = Field(
        default=5,
        description="获取历史时返回的最大轮次数"
    )


class ConversationStorageTool(BaseTool):
    """
    对话存储工具
    
    保存对话轮次或获取对话历史。
    """
    
    name: str = "conversation_storage"
    description: str = (
        "管理对话存储：保存对话轮次或获取对话历史。"
        "action='save_turn'：保存当前对话轮次（需要提供turn_data）。"
        "action='get_history'：获取对话历史（返回最近N轮对话）。"
    )
    
    args_schema: Type[BaseModel] = ConversationStorageToolArgs
    
    storage_dir: str = "data/conversations"
    
    _storage: ConversationStorage = PrivateAttr()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._storage = ConversationStorage(storage_dir=self.storage_dir)
    
    def _run(
        self,
        session_id: str,
        action: str = None,
        turn_data: Optional[str] = None,
        max_turns: int = 5,
        # 🔄 新增：支持CleanAgent的直接调用接口
        user_message: Optional[str] = None,
        agent_message: Optional[str] = None,
        message_type: Optional[str] = None,
    ) -> str:
        """
        执行存储操作
        
        Returns:
            JSON格式的操作结果
        """
        
        try:
            # 🔄 支持CleanAgent的直接调用方式
            if user_message is not None and agent_message is not None:
                # CleanAgent直接调用：自动构建turn_data
                turn_data_dict = {
                    "user_question": user_message,
                    "assistant_response": agent_message,
                    "message_type": message_type or "assessment",
                    "generated_query": "",
                    "query_keywords": [],
                    "retrieved_documents": [],
                    "dimension_id": None,
                    "dimension_name": None,
                    "user_emotion": "neutral",
                    "retrieval_method": "direct",
                    "response_metadata": {}
                }
                
                # 保存对话轮次
                session = self._storage.add_turn(
                    session_id=session_id,
                    **turn_data_dict
                )
                
                return json.dumps({
                    "success": True,
                    "action": "save_turn_direct",
                    "turn_id": len(session["dialogue_turns"]),
                    "message": "对话轮次已保存（CleanAgent直接调用）"
                }, ensure_ascii=False, indent=2)
            
            elif action == "save_turn":
                if not turn_data:
                    return json.dumps({
                        "success": False,
                        "error": "save_turn action requires turn_data"
                    }, ensure_ascii=False)
                
                # 解析turn_data
                data = json.loads(turn_data)
                
                # 保存对话轮次
                session = self._storage.add_turn(
                    session_id=session_id,
                    user_question=data.get("user_question", ""),
                    generated_query=data.get("generated_query", ""),
                    query_keywords=data.get("query_keywords", []),
                    retrieved_documents=data.get("retrieved_documents", []),
                    assistant_response=data.get("assistant_response", ""),
                    dimension_id=data.get("dimension_id"),
                    dimension_name=data.get("dimension_name"),
                    user_emotion=data.get("user_emotion"),
                    retrieval_method=data.get("retrieval_method", "paragraph"),
                    response_metadata=data.get("response_metadata")
                )
                
                return json.dumps({
                    "success": True,
                    "action": "save_turn",
                    "turn_id": len(session["dialogue_turns"]),
                    "message": "对话轮次已保存"
                }, ensure_ascii=False, indent=2)
            
            elif action == "get_history":
                history = self._storage.get_conversation_history(
                    session_id=session_id,
                    max_turns=max_turns
                )
                
                return json.dumps({
                    "success": True,
                    "action": "get_history",
                    "history": history,
                    "turn_count": len(history) // 2  # user+assistant为一轮
                }, ensure_ascii=False, indent=2)
            
            else:
                return json.dumps({
                    "success": False,
                    "error": f"Unknown action: {action}. Use 'save_turn' or 'get_history'"
                }, ensure_ascii=False)
        
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            }, ensure_ascii=False)
    
    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported")

