import { useState } from 'react';

function AppTest() {
  const [count, setCount] = useState(0);
  
  return (
    <div style={{ padding: '40px', textAlign: 'center', fontFamily: 'sans-serif' }}>
      <h1 style={{ color: '#3b82f6' }}>React 測試頁面</h1>
      <p>如果你看到這個頁面，說明 React 正在運行</p>
      <button 
        onClick={() => setCount(count + 1)}
        style={{ 
          padding: '10px 20px', 
          fontSize: '16px',
          background: '#3b82f6',
          color: 'white',
          border: 'none',
          borderRadius: '8px',
          cursor: 'pointer'
        }}
      >
        點擊次數: {count}
      </button>
    </div>
  );
}

export default AppTest;
