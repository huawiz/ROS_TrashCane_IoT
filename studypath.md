# Docker、ESP32 和相關技術的循序漸進學習路線

## 階段 1: 基礎知識
1. Python 基礎
   - [ ] 變量、數據類型和基本語法
   - [ ] 函數和模組
   - [ ] 文件操作

   資源:
   
   - YouTube: Corey Schafer 的 Python 教程播放列表
   - Coursera: "Python for Everybody" 專項課程

2. 電子學基礎
   - [ ] 基本電路概念
   - [ ] 電阻、電容、電感
   - [ ] 數字邏輯基礎

   資源:
   
   - YouTube: EEVblog 頻道
   - Coursera: "Introduction to Electronics" 課程

3. Git 版本控制
   - [ ] 基本 Git 命令
   - [ ] 分支和合併
   - [ ] 遠程倉庫操作

   資源:
   
   - YouTube: The Coding Train 的 Git 和 GitHub 教程
   - Coursera: "Version Control with Git" 課程

## 階段 2: 嵌入式系統入門
4. Arduino 基礎（作為 ESP32 的前導）
   - [ ] Arduino IDE 使用
   - [ ] 基本 I/O 操作
   - [ ] 簡單感測器項目

   資源:
   
   - YouTube: Paul McWhorter 的 Arduino 教程
   - Coursera: "Introduction to the Internet of Things and Embedded Systems" 課程

5. ESP32 基礎
   - [ ] ESP32 硬體介紹
   - [ ] 在 Arduino IDE 中使用 ESP32
   - [ ] GPIO 操作
   - [ ] 串口通信基礎

   資源:
   
   - YouTube: DroneBot Workshop 的 ESP32 教程
   - YouTube: Murtaza's Workshop - Robotics and AI 的 ESP32 項目視頻

## 階段 3: 網絡和通信
6. 網絡基礎
   - [ ] TCP/IP 協議簡介
   - [ ] IP 地址和子網
   - [ ] 基本的網絡工具使用（ping, traceroute等）

   資源:
   
   - Coursera: "The Bits and Bytes of Computer Networking" 課程
   - YouTube: PowerCert Animated Videos 頻道

7. ESP32 網絡功能
   - [ ] Wi-Fi 連接
   - [ ] 簡單的 Web 服務器
   - [ ] MQTT 協議入門

   資源:
   
   - YouTube: Random Nerd Tutorials 的 ESP32 Wi-Fi 教程
   - YouTube: Andreas Spiess 的 IoT 視頻

8. 串口通信進階
   - [ ] 串口協議深入理解
   - [ ] Python 中使用 PySerial
   - [ ] 串口調試技巧

   資源:
   
   - YouTube: IMSAI Guy 的串口通信教程

## 階段 4: Docker 和容器化
9. Docker 基礎
   - [ ] Docker 核心概念
   - [ ] 基本 Docker 命令
   - [ ] 創建和運行容器

   資源:
   
   - YouTube: TechWorld with Nana 的 Docker 教程
   - Coursera: "Introduction to Cloud Computing" 課程（包含 Docker 部分）

10. Docker Compose
    - [ ] 多容器應用
    - [ ] 編寫 docker-compose.yml 文件
    - [ ] 管理開發環境

    資源:
    
    - YouTube: DevOps Directive 的 Docker Compose 教程

## 階段 5: 圖像處理和計算機視覺
11. 圖像處理基礎
    - [ ] 圖像格式和顏色模型
    - [ ] 基本圖像操作（裁剪、縮放等）
    - [ ] 圖像濾波和增強

    資源:
    
    - Coursera: "Image and Video Processing: From Mars to Hollywood with a Stop at the Hospital" 課程

12. OpenCV 與 Python
    - [ ] 圖像讀寫和顯示
    - [ ] 基本圖像處理操作
    - [ ] 簡單的計算機視覺應用

    資源:
    
    - YouTube: Murtaza's Workshop - Robotics and AI 的 OpenCV 課程
    - Coursera: "Computer Vision Basics" 課程

## 階段 6: ESP32-CAM 和高級應用
13. ESP32-CAM 項目
    - [ ] 設置 ESP32-CAM
    - [ ] 捕獲和傳輸圖像
    - [ ] 結合 Web 服務器顯示圖像

    資源:
    
    - YouTube: Random Nerd Tutorials 的 ESP32-CAM 項目視頻
    - YouTube: Andreas Spiess 的 ESP32-CAM 視頻

14. ROS 2 基礎（選修）
    - [ ] ROS 2 核心概念
    - [ ] 節點和話題
    - [ ] 簡單的 ROS 2 應用

    資源:
    
    - YouTube: The Construct 的 ROS 2 教程
    - Coursera: "Modern Robotics" 專項課程（涵蓋 ROS）

## 階段 7: 高級主題和系統集成
15. IoT 系統設計
    - [ ] IoT 架構設計原則
    - [ ] 安全性考慮
    - [ ] 雲端與邊緣計算

    資源:
    
    - Coursera: "A developer's guide to the Internet of Things (IoT)" 課程
    - YouTube: Internet of Things 頻道

16. CI/CD 基礎
    - [ ] CI/CD 概念和原則
    - [ ] 使用 GitHub Actions 或 GitLab CI
    - [ ] 自動化測試和部署 Docker 應用

    資源:
    
    - YouTube: DevOps Directive 的 CI/CD 教程
    - Coursera: "DevOps Culture and Mindset" 課程

17. 項目整合
    - [ ] 結合所學知識開發完整項目
    - [ ] 實現從 ESP32-CAM 捕獲圖像到雲端處理的完整流程
    - [ ] 應用 Docker 進行部署，使用 CI/CD 實現自動化

    資源:
    
    - YouTube: 搜索綜合項目教程和展示視頻
    - Coursera: "IoT Capstone: Create a System Design" 課程