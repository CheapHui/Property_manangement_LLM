from __future__ import annotations
from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel, Field

Confidence = Literal["low", "medium", "high"]

class NextBestAction(BaseModel):
    priority: int = Field(..., description="行動優先級（1=最優先）")
    action: str = Field(..., description="建議下一步行動（一句）")
    why: str = Field(..., description="點解要做呢一步（一句）")
    required_inputs: List[str] = Field(default_factory=list, description="做呢一步需要嘅資料/文件")
    citations: List[str] = Field(
        default_factory=list,
        description="可選：如呢一步涉及法律/程序聲稱，引用對應 CIT payload（必須來自證據）"
    )


class DecisionBranch(BaseModel):
    if_condition: str = Field(..., description="分流條件（if）")
    then_action: str = Field(..., description="成立時建議（then）")
    else_action: Optional[str] = Field(None, description="不成立時建議（else，可選）")
    citations: List[str] = Field(
        default_factory=list,
        description="可選：如分流涉及法律/程序聲稱，引用對應 CIT payload（必須來自證據）"
    )

class RAGAnswer(BaseModel):
    answer_summary: str = Field(..., description="一段式重點答案（廣東話/繁中，保留英文專有名詞）")
    key_points: List[str] = Field(default_factory=list, description="要點（每點一句）")
    procedure_checklist: List[str] = Field(default_factory=list, description="可行步驟/Checklist")
    # --- Phase 2.2: Decision Guidance ---
    decision_frame: List[str] = Field(
        default_factory=list,
        description="決策框架：今次問題核心要決定咩（2-4點）"
    )
    required_facts: List[str] = Field(
        default_factory=list,
        description="作出決策前必需釐清/收集嘅事實資料（例如：源頭、責任界線、文件）"
    )
    clarifying_questions: List[str] = Field(
        default_factory=list,
        description="需要向用戶追問嘅關鍵澄清問題（建議最多3條）"
    )
    next_best_actions: List[NextBestAction] = Field(
        default_factory=list,
        description="下一步行動建議（按 priority 排序）"
    )
    decision_tree: List[DecisionBranch] = Field(
        default_factory=list,
        description="決策分流樹（if/then/else），用於引導用戶決定下一步"
    )
    citations_used: List[str] = Field(
        default_factory=list,
        description="只可填【證據】內出現過的 CIT payload（即 [CIT: ...] 內的 ...）"
    )
    evidence_gaps: List[str] = Field(default_factory=list, description="現有證據不足以支持的部分")
    route_info: Dict[str, Any] = Field(default_factory=dict, description="debug: route decision")
    confidence: Confidence = "medium"
    meeting_notice_template: Optional[List[str]] = None