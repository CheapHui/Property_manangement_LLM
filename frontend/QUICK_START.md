# 快速開始指南

## 1. 啟動前端

```bash
cd frontend
npm run dev
```

訪問：http://localhost:5174

## 2. 連接後端

### 選項 A：使用 `.env` 文件（推薦）

1. 創建 `.env` 文件：
```bash
cp .env.example .env
```

2. 編輯 `.env`：
```env
VITE_API_BASE_URL=http://localhost:8000
```

### 選項 B：直接修改代碼

編輯 `src/services/api.js`：
```javascript
const API_BASE_URL = 'http://localhost:8000';
```

## 3. 測試界面

### 預期行為

1. **發送消息**
   - 輸入問題，按 Enter 發送
   - 看到「正在輸入」效果（閃爍游標）

2. **串流回應**
   - Token 逐字顯示
   - 右側自動彈出證據面板（如有）

3. **證據面板**
   - 顯示引用文檔
   - 路由資訊（intent, confidence）
   - 警告（如有）

## 4. 調試

### 查看 SSE 事件

打開瀏覽器開發工具 → Network → 找到 `/run/` 請求 → EventStream

### 常見問題

**Q: 無法連接後端？**
- 檢查 `.env` 中的 `VITE_API_BASE_URL`
- 確認後端已啟動
- 檢查 CORS 配置

**Q: SSE 串流無效？**
- 檢查 Console 錯誤
- 確認後端返回正確的 SSE 格式
- 檢查 Content-Type: text/event-stream

**Q: 中文顯示異常？**
- 確認 HTML meta charset="UTF-8"
- 檢查後端 JSON 響應編碼

## 5. 生產部署

```bash
npm run build
```

構建產物在 `dist/` 目錄，可直接部署到 Nginx/Apache 等靜態服務器。

---

如需更多信息，請參閱 `README_ZH.md`
