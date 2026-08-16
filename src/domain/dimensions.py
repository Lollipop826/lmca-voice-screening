from typing import List, Optional
import copy

from src.common.types import InfoDimension

MMSE_DIMENSIONS: List[InfoDimension] = [
    {"id": "orientation", "name": "定向力", "description": "时间/地点定向", "priority": 95, "status": "unknown"},
    {"id": "registration", "name": "即时记忆", "description": "三词登记/即时复述", "priority": 92, "status": "unknown"},
    {"id": "attention_calculation", "name": "注意力与计算", "description": "100-7连续减或倒拼", "priority": 90, "status": "unknown"},
    {"id": "recall", "name": "延迟回忆", "description": "延迟回忆三词", "priority": 88, "status": "unknown"},
    {"id": "language", "name": "语言", "description": "命名/复述/三步指令/阅读/口述句子", "priority": 85, "status": "unknown"},
    {"id": "copy", "name": "构图(临摹)", "description": "两五边形相交图形", "priority": 80, "status": "unknown"},
]


def new_session_dimensions() -> List[InfoDimension]:
    return copy.deepcopy(MMSE_DIMENSIONS)


def update_dimension_status(dimensions: List[InfoDimension], dim_id: str, status: str, value: Optional[str] = None) -> None:
    for d in dimensions:
        if d.get("id") == dim_id:
            d["status"] = status
            if value is not None:
                d["value"] = value
            return
