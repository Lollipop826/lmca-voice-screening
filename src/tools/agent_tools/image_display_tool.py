"""
MMSE图片展示工具 - 用于命名任务和阅读任务
"""

import json
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


class ImageDisplayToolArgs(BaseModel):
    """图片展示工具参数"""
    image_id: str = Field(
        ..., 
        description="图片ID: watch(手表), pencil(铅笔), close_eyes(闭上眼睛), pentagons(五边形)"
    )
    title: str = Field(
        default="请看下面的图片", 
        description="图片标题提示文字"
    )
    action: str = Field(
        default="show",
        description="操作类型: show(显示), hide(隐藏)"
    )


class ImageDisplayTool(BaseTool):
    """
    MMSE图片展示工具
    
    用于在前端展示MMSE评估所需的图片：
    - watch: 手表（命名任务）
    - pencil: 铅笔（命名任务）
    - close_eyes: "闭上眼睛"文字卡（阅读任务）
    - pentagons: 两个相交的五边形（临摹任务）
    """
    
    name: str = "ImageDisplayTool"
    description: str = """
    用于在前端显示MMSE评估图片。
    
    使用场景：
    1. 语言维度-命名任务：显示手表或铅笔图片，让患者说出名称
    2. 语言维度-阅读任务：显示"闭上眼睛"文字卡，让患者照做
    3. 构图维度-临摹任务：显示五边形图形，让患者临摹
    
    参数：
    - image_id: 图片ID (watch/pencil/close_eyes/pentagons)
    - title: 图片标题 (默认"请看下面的图片")
    - action: show显示 或 hide隐藏
    
    返回JSON格式：
    {
        "success": true/false,
        "message": "操作结果",
        "image_id": "图片ID",
        "display_command": {...}  # 发送给前端的WebSocket消息
    }
    """
    args_schema: Type[BaseModel] = ImageDisplayToolArgs
    
    def _run(
        self,
        image_id: str,
        title: str = "请看下面的图片",
        action: str = "show"
    ) -> str:
        """
        执行图片展示
        
        注意：此工具只生成展示指令，实际展示需要voice_server.py通过WebSocket发送给前端
        """
        
        # 验证图片ID
        valid_images = ["watch", "pencil", "close_eyes", "pentagons", "writing_prompt"]
        if image_id not in valid_images:
            return json.dumps({
                "success": False,
                "message": f"无效的图片ID: {image_id}，支持: {', '.join(valid_images)}",
                "image_id": image_id
            }, ensure_ascii=False)
        
        # 生成前端展示指令
        if action == "show":
            display_command = {
                "type": "show_image",
                "image_id": image_id,
                "title": title
            }
            
            print(f"[ImageDisplay] 📋 准备展示图片: {image_id} - {title}")
            
            return json.dumps({
                "success": True,
                "message": f"已准备展示图片: {image_id}",
                "image_id": image_id,
                "title": title,
                "display_command": display_command
            }, ensure_ascii=False)
            
        elif action == "hide":
            display_command = {
                "type": "hide_image",
                "image_id": image_id
            }
            
            print(f"[ImageDisplay] 🚫 准备隐藏图片: {image_id}")
            
            return json.dumps({
                "success": True,
                "message": f"已准备隐藏图片: {image_id}",
                "image_id": image_id,
                "display_command": display_command
            }, ensure_ascii=False)
        
        else:
            return json.dumps({
                "success": False,
                "message": f"无效的操作类型: {action}，支持: show/hide",
                "image_id": image_id
            }, ensure_ascii=False)
    
    async def _arun(self, *args, **kwargs):
        """异步版本（暂不支持）"""
        raise NotImplementedError("ImageDisplayTool不支持异步调用")
