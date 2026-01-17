// import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// 添加錯誤捕獲
console.log('🚀 main.jsx loading...');

try {
  const rootElement = document.getElementById('root');
  console.log('📦 Root element:', rootElement);

  if (!rootElement) {
    throw new Error('Root element not found');
  }

  console.log('🎨 Creating React root...');
  const root = createRoot(rootElement);

  console.log('🎭 Rendering App...');
  root.render(<App />);

  console.log('✅ React app rendered successfully');
} catch (error) {
  console.error('❌ Failed to render app:', error);
  document.body.innerHTML = `
    <div style="padding: 20px; font-family: sans-serif;">
      <h1 style="color: red;">應用啟動失敗</h1>
      <pre>${error.message}\n${error.stack}</pre>
    </div>
  `;
}
