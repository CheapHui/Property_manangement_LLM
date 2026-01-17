# 物業管理 AI 助手 - 前端

專業的香港物業管理法律及實務 AI Agent Web UI

## 功能特點

✅ **即時串流回應** - 使用 SSE (Server-Sent Events) 實現打字機效果
✅ **證據面板** - 顯示引用文檔、法例條文、案例資料
✅ **專業設計** - 簡約、專業的 UI/UX
✅ **響應式設計** - 支援桌面及移動設備
✅ **香港中文優化** - 針對繁體中文優化字體及排版

## 技術棧

- **React 19** - 前端框架
- **Vite** - 構建工具
- **Lucide React** - Icon 圖標庫
- **原生 CSS** - 無需額外 CSS 框架

## 目錄結構

```
frontend/
├── src/
│   ├── components/           # React 組件
│   │   ├── ChatMessage.jsx   # 聊天消息組件
│   │   ├── ChatInput.jsx     # 輸入框組件
│   │   └── EvidencePanel.jsx # 證據面板組件
│   ├── services/             # API 服務
│   │   └── api.js            # 後端 API 調用
│   ├── utils/                # 工具函數
│   │   └── sseStream.js      # SSE 串流處理
│   ├── App.jsx               # 主應用組件
│   ├── App.css               # 主應用樣式
│   ├── index.css             # 全局樣式
│   └── main.jsx              # 應用入口
├── .env.example              # 環境變量範本
└── package.json              # 依賴配置
```

## 安裝與啟動

### 1. 安裝依賴

```bash
cd frontend
npm install
```

### 2. 配置環境變量

複製 `.env.example` 為 `.env`：

```bash
cp .env.example .env
```

編輯 `.env` 文件，設置後端 API 地址：

```env
VITE_API_BASE_URL=http://localhost:8000
```

### 3. 啟動開發服務器

```bash
npm run dev
```

訪問 http://localhost:5173

### 4. 構建生產版本

```bash
npm run build
```

構建結果在 `dist/` 目錄

## API 集成

### SSE 事件格式

前端預期接收以下 SSE 事件：

#### 1. **start** - 開始回應
```
event: start
data: {"turn_id": "turn_12345"}
```

#### 2. **token** - 串流 token
```
event: token
data: {"text": "你"}
```

#### 3. **final** - 完整回應
```
event: final
data: {
  "assistant_answer": "完整答案...",
  "evidences": [...],
  "intent": "definition",
  "route": "ordinance_retrieval",
  "confidence": 0.95,
  "warnings": [],
  "property_intent": "common_parts"
}
```

#### 4. **error** - 錯誤
```
event: error
data: {"message": "錯誤訊息"}
```

### API Endpoint

```
POST /api/v1/conversations/{conversation_id}/run/
Content-Type: application/json

{
  "message": "用戶問題"
}
```

## 組件說明

### ChatMessage 組件
- 顯示用戶及助手消息
- 支援串流效果（閃爍游標）
- 區分用戶/助手/錯誤消息

### ChatInput 組件
- 自動調整高度的文本框
- 支援 Enter 發送、Shift+Enter 換行
- Loading 狀態禁用輸入

### EvidencePanel 組件
- 顯示法例、案例、指引等證據
- 展示路由資訊、信心度
- 可折疊的證據項目
- 警告提示

## 自定義配置

### 修改主題顏色

編輯 `src/App.css` 中的 header gradient：

```css
.app-header {
  background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
}
```

### 修改字體

編輯 `src/index.css`：

```css
:root {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', ...;
}
```

## 開發建議

1. **測試 SSE 連接**：使用瀏覽器開發工具的 Network 面板查看 SSE 事件
2. **錯誤處理**：檢查 Console 的錯誤日誌
3. **跨域問題**：確保後端配置正確的 CORS headers

## 部署

### Nginx 配置範例

```nginx
server {
  listen 80;
  server_name your-domain.com;
  root /path/to/frontend/dist;
  index index.html;

  location / {
    try_files $uri $uri/ /index.html;
  }

  # Proxy API requests to backend
  location /api/ {
    proxy_pass http://localhost:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection 'upgrade';
    proxy_set_header Host $host;
    proxy_cache_bypass $http_upgrade;
  }
}
```

## 授權

MIT License

---

如有問題或建議，請提交 Issue。
