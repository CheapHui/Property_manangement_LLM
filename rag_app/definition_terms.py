# rag_app/definition_terms.py
from typing import Dict, List, Optional

# -------------------------
# Core definition anchors
# -------------------------
# 用於「所有」法律定義題的泛化錨點
DEFINITION_ANCHORS = [
    # English
    "interpretation",
    "means",
    "definition",
    "for the purposes of this ordinance",
    "in this ordinance",
    "schedule",
    # Chinese
    "釋義",
    "定義",
    "指",
    "就本條例而言",
    "附表",
]

# -------------------------
# Term → keyword mapping
# -------------------------
# key = logical legal concept
# keywords = 出現喺 ordinance / case / guideline 入面嘅常見詞
DEFINITION_KEYWORD_MAP: Dict[str, List[str]] = {

    # 1️⃣ Common parts
    "common_parts": [
        "common parts",
        "common areas",
        "公用部分",
        "公共部分",
        "公用地方",
        "附表1",
        "schedule 1",
    ],

    # 2️⃣ Owner
    "owner": [
        "owner",
        "registered owner",
        "業主",
        "註冊業主",
        "section 2",
        "interpretation",
    ],

    # 3️⃣ Owners' Corporation
    "owners_corporation": [
        "owners' corporation",
        "incorporated owners",
        "法團",
        "業主立案法團",
        "corporation",
    ],

    # 4️⃣ Management Committee
    "management_committee": [
        "management committee",
        "管理委員會",
        "schedule 2",
        "composition and procedure",
    ],

    # 5️⃣ Deed of Mutual Covenant
    "dmc": [
        "deed of mutual covenant",
        "dmc",
        "大廈公契",
        "公契",
    ],

    # 6️⃣ Manager
    "manager": [
        "manager",
        "building management agent",
        "管理人",
        "物業管理人",
    ],

    # 7️⃣ General meeting
    "general_meeting": [
        "general meeting",
        "owners meeting",
        "業主大會",
        "schedule 3",
    ],

    # 8️⃣ Quorum
    "quorum": [
        "quorum",
        "法定人數",
        "minimum attendance",
    ],

    # 9️⃣ Resolution
    "resolution": [
        "resolution",
        "決議",
        "ordinary resolution",
        "special resolution",
    ],

    # 🔟 Maintenance / repair
    "maintenance": [
        "maintenance",
        "repair",
        "維修",
        "保養",
    ],

    # 11️⃣ Management fee
    "management_fee": [
        "management fee",
        "contribution",
        "管理費",
        "分擔費用",
    ],

    # 12️⃣ Unauthorized works
    "unauthorized_works": [
        "unauthorized building works",
        "unauthorized structure",
        "違建",
        "僭建",
    ],

    # 13️⃣ Procurement
    "procurement": [
        "procurement",
        "tender",
        "採購",
        "招標",
        "major procurement",
    ],

    # 14️⃣ Tribunal
    "tribunal": [
        "lands tribunal",
        "tribunal",
        "審裁處",
        "土地審裁處",
    ],

    # 15️⃣ Voting rights
    "voting_rights": [
        "voting rights",
        "voting power",
        "投票權",
        "表決權",
    ],
}

# -------------------------
# Helper functions
# -------------------------

def detect_definition_term(question: str) -> Optional[str]:
    """
    嘗試由問題內容判斷屬於邊一個 legal definition term
    """
    q = question.lower()
    for term, keywords in DEFINITION_KEYWORD_MAP.items():
        for kw in keywords:
            if kw.lower() in q:
                return term
    return None


def get_definition_keywords(term: Optional[str]) -> List[str]:
    """
    回傳某個 term 對應嘅 keyword list
    """
    if not term:
        return []
    return DEFINITION_KEYWORD_MAP.get(term, [])


def get_all_definition_anchors() -> List[str]:
    """
    所有定義題都適用嘅 anchor（Interpretation / 釋義 等）
    """
    return DEFINITION_ANCHORS