# FJCU_SEDIA_23_IoT
輔大軟創系 專題


# 相關連結

VBOX(虛擬機依個人) - https://www.virtualbox.org/

```linux版本老師有指定 ubuntu Mate 22.04.3 LTS```

linux
https://ubuntu-mate.org/download/amd64/

Linux安裝步驟
https://ithelp.ithome.com.tw/articles/10317509

ROS 2
https://www.ros.org/

載-步驟
https://docs.ros.org/en/humble/Installation.html

Arduino
https://www.arduino.cc/

micro-ROS
https://micro.ros.org/docs/tutorials/core/first_application_linux/

ESP32S
https://ithelp.ithome.com.tw/articles/10260120

筆記：
https://zipkxlee.notion.site/8958f815905d4372986f521fa79e18f0?v=3386008d53304c25a9e0db9c8270e887


# AI整理學習清單(7/25)
# Docker、ESP32 和相關技術的循序漸進學習路線

## 階段 1: 基礎知識
1. Python 基礎
   - [ ] 變量、數據類型和基本語法
   - [ ] 函數和模組
   - [ ] 文件操作
  
   資源: Python 官方教程，Codecademy 的 Python 課程

2. 電子學基礎
   - [ ] 基本電路概念
   - [ ] 電阻、電容、電感
   - [ ] 數字邏輯基礎
   
   資源: All About Circuits 網站，《Make: Electronics》書籍

3. Git 版本控制
   - [ ] 基本 Git 命令
   - [ ] 分支和合併
   - [ ] 遠程倉庫操作
   
   資源: Git 官方文檔，GitHub Learning Lab

## 階段 2: 嵌入式系統入門
4. Arduino 基礎（作為 ESP32 的前導）
   - [ ] Arduino IDE 使用
   - [ ] 基本 I/O 操作
   - [ ] 簡單感測器項目
   
   資源: Arduino 官方教程，Arduino Project Hub

5. ESP32 基礎
   - [ ] ESP32 硬體介紹
   - [ ] 在 Arduino IDE 中使用 ESP32
   - [ ] GPIO 操作
   - [ ] 串口通信基礎
   
   資源: RandomNerdTutorials 網站，Espressif 官方文檔

## 階段 3: 網絡和通信
6. 網絡基礎
   - [ ] TCP/IP 協議簡介
   - [ ] IP 地址和子網
   - [ ] 基本的網絡工具使用（ping, traceroute等）
   
   資源: Coursera "The Bits and Bytes of Computer Networking" 課程

7. ESP32 網絡功能
   - [ ] Wi-Fi 連接
   - [ ] 簡單的 Web 服務器
   - [ ] MQTT 協議入門
   
   資源: ESP32 官方示例，MQTT.org

8. 串口通信進階
   - [ ] 串口協議深入理解
   - [ ] Python 中使用 PySerial
   - [ ] 串口調試技巧
   
   資源: PySerial 文檔，SparkFun 串口通信教程

## 階段 4: Docker 和容器化
9. Docker 基礎
   - [ ] Docker 核心概念
   - [ ] 基本 Docker 命令
   - [ ] 創建和運行容器
   
   資源: Docker 官方文檔，Docker 入門教程

10. Docker Compose
    - [ ] 多容器應用
    - [ ] 編寫 docker-compose.yml 文件
    - [ ] 管理開發環境
    
    資源: Docker Compose 官方文檔，實戰項目教程

## 階段 5: 圖像處理和計算機視覺
11. 圖像處理基礎
    - [ ] 圖像格式和顏色模型
    - [ ] 基本圖像操作（裁剪、縮放等）
    - [ ] 圖像濾波和增強
    
    資源: OpenCV-Python 教程，《Digital Image Processing》書籍

12. OpenCV 與 Python
    - [ ] 圖像讀寫和顯示
    - [ ] 基本圖像處理操作
    - [ ] 簡單的計算機視覺應用
    
    資源: OpenCV 官方教程，PyImageSearch 博客

## 階段 6: ESP32-CAM 和高級應用
13. ESP32-CAM 項目
    - [ ] 設置 ESP32-CAM
    - [ ] 捕獲和傳輸圖像
    - [ ] 結合 Web 服務器顯示圖像
    
    資源: RandomNerdTutorials ESP32-CAM 項目，GitHub 上的開源項目

14. ROS 2 基礎（選修）
    - [ ] ROS 2 核心概念
    - [ ] 節點和話題
    - [ ] 簡單的 ROS 2 應用
    
    資源: ROS 2 官方教程，《Programming Robots with ROS》書籍

## 階段 7: 高級主題和系統集成
15. IoT 系統設計
    - [ ] IoT 架構設計原則
    - [ ] 安全性考慮
    - [ ] 雲端與邊緣計算
    
    資源: AWS IoT 文檔，《IoT Fundamentals》書籍

16. CI/CD 基礎
    - [ ] CI/CD 概念和原則
    - [ ] 使用 GitHub Actions 或 GitLab CI
    - [ ] 自動化測試和部署 Docker 應用
    
    資源: GitHub Actions 文檔，GitLab CI/CD 文檔

17. 項目整合
    - [ ] 結合所學知識開發完整項目
    - [ ] 實現從 ESP32-CAM 捕獲圖像到雲端處理的完整流程
    - [ ] 應用 Docker 進行部署，使用 CI/CD 實現自動化
    
    資源: 個人項目實踐，開源社區參與