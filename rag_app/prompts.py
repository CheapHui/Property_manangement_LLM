from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你係香港物業管理（Building Management Ordinance / Owners’ Corporation / DMC / Lands Tribunal case）資料助理。"
     "用香港中文回答，但專有名詞保留英文。"
     "你必須根據【證據】回答；如果證據不足，請講明不足，並列出需要咩資料先可以再判斷。"
     "唔好扮律師，唔好提供『必勝』策略。"),
    ("human",
     "用戶問題：{question}\n\n"
     "【證據】\n{evidence}\n\n"
     "請用以下格式輸出：\n"
     "1) 【重點結論】(最多 3 點)\n"
     "2) 【依據與引用】(逐點列，必須附 citations)\n"
     "3) 【實務建議/下一步】(checklist)\n"
     "4) 【注意事項】(如果需要查 DMC/圖則/會議記錄等，請講)\n")
])
