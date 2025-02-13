import React from "react";
import ReactDOM from "react-dom/client"; // 使用 React 18 的正确导入方式
import "./index.css";
import App from "./App";

const rootElement = document.getElementById("root"); // 确保 HTML 里有 <div id="root"></div>
const root = ReactDOM.createRoot(rootElement); // 使用 createRoot

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
