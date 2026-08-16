"""
认知表现记录工具 - 供Agent调用

记录每个维度的认知表现，跟踪评估历史，分析整体认知状态
注意：这是质量评估记录，不是固定分数制
"""

from __future__ import annotations

import os
import json
from typing import Optional, Type, Dict, Any, List
from pathlib import Path
from datetime import datetime

from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.tools import BaseTool


class ScoreRecordingToolArgs(BaseModel):
    """认知表现记录工具参数"""
    
    session_id: str = Field(..., description="会话ID")
    dimension_id: str = Field(default="", description="维度ID，如 orientation/registration 等")
    quality_level: str = Field(default="fair", description="回答质量等级：excellent/good/fair/poor")
    cognitive_performance: str = Field(default="正常", description="认知表现：正常/轻度异常/中度异常/重度异常")
    question: str = Field(default="", description="评估问题")
    answer: str = Field(default="", description="患者回答")
    evaluation_detail: str = Field(default="", description="评估详情")
    action: str = Field(default="save", description="操作类型：save（保存）/get（获取）/summary（汇总）")


class ScoreRecordingResult(BaseModel):
    """认知表现记录结果"""
    
    success: bool = Field(..., description="操作是否成功")
    message: str = Field(..., description="操作结果消息")
    current_performance: Optional[str] = Field(None, description="当前维度表现")
    overall_status: Optional[str] = Field(None, description="整体认知状态")
    completed_dimensions: Optional[List[str]] = Field(None, description="已完成的维度列表")
    performance_details: Optional[Dict[str, Any]] = Field(None, description="详细表现记录")


class ScoreRecordingTool(BaseTool):
    """
    认知表现记录工具
    
    记录每个维度的认知表现，支持保存、查询和汇总功能。
    """
    
    name: str = "score_recording_tool"
    description: str = (
        "记录和管理认知评估表现。"
        "输入参数：session_id（会话ID），dimension_id（维度ID），quality_level（质量等级），cognitive_performance（认知表现），action（操作类型）。"
        "支持操作：save（保存表现）、get（获取当前表现）、summary（获取整体汇总）。"
    )
    
    args_schema: Type[BaseModel] = ScoreRecordingToolArgs
    
    # 使用 PrivateAttr 避免 Pydantic 字段冲突
    _performance_dir: Path = PrivateAttr()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 创建认知表现存储目录
        self._performance_dir = Path("data/cognitive_performance")
        self._performance_dir.mkdir(parents=True, exist_ok=True)
    
    def _run(
        self,
        session_id: str,
        dimension_id: str = "",
        quality_level: str = "fair",
        cognitive_performance: str = "正常",
        question: str = "",
        answer: str = "",
        evaluation_detail: str = "",
        action: str = "save",
    ) -> str:
        """
        执行认知表现记录操作
        
        Returns:
            JSON格式的操作结果
        """
        # 容错处理：本地模型把整个JSON当作session_id参数传入
        if session_id and session_id.strip().startswith('{'):
            try:
                json_str = session_id.strip()
                brace_count = 0
                json_end = -1
                for i, char in enumerate(json_str):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                if json_end > 0:
                    json_str = json_str[:json_end]
                
                parsed = json.loads(json_str)
                session_id = parsed.get('session_id', '')
                dimension_id = parsed.get('dimension_id', '')
                quality_level = parsed.get('quality_level', 'fair')
                cognitive_performance = parsed.get('cognitive_performance', '正常')
                question = parsed.get('question', '')
                answer = parsed.get('answer', '')
                evaluation_detail = parsed.get('evaluation_detail', '')
                action = parsed.get('action', 'save')
                print(f"[ScoreTool] ✅ JSON解析成功")
            except Exception as e:
                print(f"[ScoreTool] ❌ JSON解析失败: {type(e).__name__}")
                # 解析失败时返回错误，避免使用错误的 session_id
                return json.dumps({
                    "success": False,
                    "message": f"参数解析失败: {e}，请确保传入正确的session_id"
                }, ensure_ascii=False)
        
        # 验证 session_id 不为空
        if not session_id or not session_id.strip():
            return json.dumps({
                "success": False,
                "message": "session_id 不能为空"
            }, ensure_ascii=False)
        
        performance_file = self._performance_dir / f"{session_id}_performance.json"
        
        # 读取现有记录
        performance_data = self._load_performance(performance_file)
        
        if action == "save":
            return self._save_performance(
                performance_data, performance_file, session_id, dimension_id, 
                quality_level, cognitive_performance, question, answer, evaluation_detail
            )
        elif action == "get":
            return self._get_performance(performance_data, dimension_id)
        elif action == "summary":
            return self._get_summary(performance_data)
        else:
            return json.dumps({
                "success": False,
                "message": f"不支持的操作类型: {action}"
            }, ensure_ascii=False)
    
    def _load_performance(self, performance_file: Path) -> Dict[str, Any]:
        """加载认知表现记录"""
        if performance_file.exists():
            try:
                with open(performance_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载认知表现文件失败: {type(e).__name__}")
                return self._init_performance_data()
        else:
            return self._init_performance_data()
    
    def _init_performance_data(self) -> Dict[str, Any]:
        """初始化认知表现数据结构"""
        return {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "dimensions": {},
            "overall_status": "未评估"
        }
    
    def _save_performance(
        self, 
        performance_data: Dict[str, Any], 
        performance_file: Path,
        session_id: str,
        dimension_id: str,
        quality_level: str,
        cognitive_performance: str,
        question: str,
        answer: str,
        evaluation_detail: str
    ) -> str:
        """保存认知表现"""
        try:
            # 更新维度表现记录
            if dimension_id not in performance_data["dimensions"]:
                performance_data["dimensions"][dimension_id] = {
                    "attempts": 0,
                    "quality_levels": [],
                    "cognitive_performances": [],
                    "records": [],
                    "latest_performance": "未评估",
                    "last_updated": datetime.now().isoformat()
                }
            
            dim_data = performance_data["dimensions"][dimension_id]
            
            # 记录单次表现
            dim_data["attempts"] += 1
            dim_data["quality_levels"].append(quality_level)
            dim_data["cognitive_performances"].append(cognitive_performance)
            dim_data["records"].append({
                "quality_level": quality_level,
                "cognitive_performance": cognitive_performance,
                "question": question,
                "answer": answer,
                "evaluation_detail": evaluation_detail,
                "timestamp": datetime.now().isoformat()
            })
            dim_data["latest_performance"] = cognitive_performance
            dim_data["last_updated"] = datetime.now().isoformat()
            
            # 计算该维度的整体表现（基于最近3次）
            recent_performances = dim_data["cognitive_performances"][-3:]
            if all(p == "正常" for p in recent_performances):
                dim_data["overall_status"] = "正常"
            elif any(p == "重度异常" for p in recent_performances):
                dim_data["overall_status"] = "重度异常"
            elif any(p == "中度异常" for p in recent_performances):
                dim_data["overall_status"] = "中度异常"
            elif any(p == "轻度异常" for p in recent_performances):
                dim_data["overall_status"] = "轻度异常"
            else:
                dim_data["overall_status"] = "正常"
            
            # 评估整体认知状态
            all_statuses = [d.get("overall_status", "未评估") for d in performance_data["dimensions"].values()]
            if len(all_statuses) > 0:
                if all(s == "正常" for s in all_statuses):
                    performance_data["overall_status"] = "整体正常"
                elif any(s == "重度异常" for s in all_statuses):
                    performance_data["overall_status"] = "存在重度认知障碍"
                elif any(s == "中度异常" for s in all_statuses):
                    performance_data["overall_status"] = "存在中度认知障碍"
                elif any(s == "轻度异常" for s in all_statuses):
                    performance_data["overall_status"] = "存在轻度认知障碍"
                else:
                    performance_data["overall_status"] = "整体正常"
            
            performance_data["updated_at"] = datetime.now().isoformat()
            
            # 保存到文件
            with open(performance_file, 'w', encoding='utf-8') as f:
                json.dump(performance_data, f, ensure_ascii=False, indent=2)
            
            result = ScoreRecordingResult(
                success=True,
                message=f"已记录 {dimension_id} 维度的认知表现",
                current_performance=cognitive_performance,
                overall_status=performance_data["overall_status"],
                completed_dimensions=list(performance_data["dimensions"].keys())
            )
            
            return json.dumps(result.model_dump(), ensure_ascii=False)
            
        except Exception as e:
            return json.dumps({
                "success": False,
                "message": f"保存认知表现失败: {str(e)}"
            }, ensure_ascii=False)
    
    def _get_performance(self, performance_data: Dict[str, Any], dimension_id: str) -> str:
        """获取指定维度的认知表现"""
        if dimension_id in performance_data["dimensions"]:
            dim_data = performance_data["dimensions"][dimension_id]
            result = ScoreRecordingResult(
                success=True,
                message=f"获取 {dimension_id} 维度表现成功",
                current_performance=dim_data.get("latest_performance", "未评估"),
                performance_details=dim_data
            )
        else:
            result = ScoreRecordingResult(
                success=False,
                message=f"未找到 {dimension_id} 维度的表现记录",
                current_performance="未评估"
            )
        
        return json.dumps(result.model_dump(), ensure_ascii=False)
    
    def _get_summary(self, performance_data: Dict[str, Any]) -> str:
        """获取整体认知表现汇总"""
        overall_status = performance_data.get("overall_status", "未评估")
        dimensions = performance_data.get("dimensions", {})
        
        # 统计各维度表现
        dimension_summary = {}
        for dim_id, dim_data in dimensions.items():
            records = dim_data.get("records") or []
            latest_quality = None
            latest_cognitive = dim_data.get("latest_performance", "未评估")
            if records:
                last_rec = records[-1]
                latest_quality = last_rec.get("quality_level")
                latest_cognitive = last_rec.get("cognitive_performance", latest_cognitive)
            dimension_summary[dim_id] = {
                "attempts": dim_data.get("attempts", 0),
                "latest_performance": dim_data.get("latest_performance", "未评估"),
                "overall_status": dim_data.get("overall_status", "未评估"),
                "latest_quality_level": latest_quality,
                "latest_cognitive_performance": latest_cognitive,
            }
        
        result = ScoreRecordingResult(
            success=True,
            message=f"整体认知状态: {overall_status}",
            overall_status=overall_status,
            completed_dimensions=list(dimensions.keys()),
            performance_details={
                "overall_status": overall_status,
                "dimension_summary": dimension_summary,
                "total_dimensions": 6,
                "completed_dimensions": len(dimensions),
                "completion_rate": len(dimensions) / 6 * 100
            }
        )
        
        return json.dumps(result.model_dump(), ensure_ascii=False)
    
    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported")
