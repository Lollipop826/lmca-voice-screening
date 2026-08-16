from typing import Any, Dict, List, Optional, TypedDict


class Profile(TypedDict, total=False):
    user_id: str
    name: str
    age: int
    sex: str
    education_years: int
    notes: str


class InfoDimension(TypedDict, total=False):
    id: str
    name: str
    description: str
    priority: int
    status: str  # unknown | asking | known | inconsistent | skipped
    value: Optional[str]
    evidence: List[str]


class ConversationTurn(TypedDict, total=False):
    role: str
    content: str
    emotion: Optional[str]
    ts: Optional[str]


class SearchQueryResult(TypedDict, total=False):
    query: str
    keywords: List[str]
    target_dimensions: List[str]
    confidence: float
    rationale: str
    used_fallback: bool
