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
# Docker、ESP32、ESP32-CAM 和相關技術學習清單

## 1. Docker
- [ ] 基礎概念：容器、映像、Dockerfile
- [ ] 基本命令：run, build, exec, ps, logs 等
- [ ] Docker Compose
- [ ] 網絡配置和卷（Volumes）管理
- [ ] 在不同操作系統上的使用（Windows、Linux、macOS）
- [ ] 安全最佳實踐和性能優化
- [ ] 在 Docker 中運行 GUI 應用程序和 ROS 2

推薦學習資源：
- Docker 官方文檔：https://docs.docker.com/
- Udemy 課程："Docker Mastery" by Bret Fisher
- 書籍：《Docker Deep Dive》by Nigel Poulton
- YouTube 頻道：TechWorld with Nana

## 2. ESP32 和 ESP32-CAM
- [ ] 硬體架構
- [ ] Arduino IDE 中的編程
- [ ] GPIO 操作和串口通信
- [ ] Wi-Fi 和藍牙功能
- [ ] 低功耗模式
- [ ] FreeRTOS 多任務編程
- [ ] ESP-IDF 使用
- [ ] ESP32 相機庫使用

推薦學習資源：
- Espressif 官方文檔：https://docs.espressif.com/
- RandomNerdTutorials 網站：https://randomnerdtutorials.com/esp32-tutorials/
- YouTube 頻道：DroneBot Workshop
- 書籍：《Kolban's Book on ESP32》by Neil Kolban（免費電子書）

## 3. 串口通信
- [ ] 協議基礎和不同波特率的影響
- [ ] 使用 Python 進行串口通信（pyserial 庫）
- [ ] 串口調試工具（如 minicom, screen）
- [ ] 高速串口數據傳輸技術

推薦學習資源：
- PySerial 文檔：https://pythonhosted.org/pyserial/
- SparkFun 教程：https://learn.sparkfun.com/tutorials/serial-communication
- 書籍：《Serial Port Complete》by Jan Axelson

## 4. 開發環境和工具
- [ ] WSL 安裝、配置和 USB 設備使用
- [ ] usbipd-win 工具使用
- [ ] Visual Studio Code 遠程開發
- [ ] 邏輯分析儀進行硬體調試

推薦學習資源：
- Microsoft WSL 文檔：https://docs.microsoft.com/en-us/windows/wsl/
- Visual Studio Code 文檔：https://code.visualstudio.com/docs
- Saleae 邏輯分析儀教程：https://support.saleae.com/tutorials

## 5. ROS 2 (Robot Operating System 2)
- [ ] 基本概念：節點、話題、服務
- [ ] 在 Docker 中運行 ROS 2
- [ ] 與 ESP32 的集成
- [ ] 處理圖像數據

推薦學習資源：
- ROS 2 官方教程：https://docs.ros.org/en/foxy/Tutorials.html
- edX 課程："Hello (Real) World with ROS"
- 書籍：《Programming Robots with ROS》by Morgan Quigley et al.

## 6. 網絡和設備通信
- [ ] TCP/IP 基礎
- [ ] 網絡橋接和端口轉發
- [ ] USB 設備在不同操作系統中的識別和使用
- [ ] Socket 編程

推薦學習資源：
- Coursera 課程："The Bits and Bytes of Computer Networking" by Google
- 書籍：《TCP/IP Illustrated, Volume 1》by W. Richard Stevens
- Python Socket 編程教程：https://realpython.com/python-sockets/

## 7. 圖像處理和計算機視覺
- [ ] 圖像格式（JPEG, RGB 等）和壓縮技術
- [ ] OpenCV 基礎使用
- [ ] 基本圖像操作、顯示和保存
- [ ] 圖像增強技術

推薦學習資源：
- OpenCV 官方教程：https://docs.opencv.org/master/d9/df8/tutorial_root.html
- Udacity 課程："Introduction to Computer Vision"
- 書籍：《Learning OpenCV 3》by Adrian Kaehler & Gary Bradski

## 8. 編程語言和框架
- [ ] Python（PySerial, OpenCV）
- [ ] Processing（可選）
- [ ] Git 版本控制

推薦學習資源：
- Python 官方教程：https://docs.python.org/3/tutorial/
- Processing 官方教程：https://processing.org/tutorials/
- Git 官方文檔：https://git-scm.com/doc
- Coursera 課程："Version Control with Git" by Atlassian

## 9. IoT (Internet of Things) 概念
- [ ] 架構設計
- [ ] 設備安全
- [ ] 雲端與邊緣計算

推薦學習資源：
- edX 課程："Introduction to the Internet of Things (IoT)" by IEEE
- 書籍：《IoT Fundamentals》by David Hanes et al.
- AWS IoT 文檔：https://docs.aws.amazon.com/iot/

## 10. 持續集成和持續部署 (CI/CD)
- [ ] 為 Docker 項目設置 CI/CD 管道
- [ ] 自動化測試和部署

推薦學習資源：
- GitHub Actions 文檔：https://docs.github.com/en/actions
- GitLab CI/CD 文檔：https://docs.gitlab.com/ee/ci/
- 書籍：《Continuous Delivery》by Jez Humble & David Farley

## 11. 嵌入式系統和電子學基礎
- [ ] 微控制器基本概念
- [ ] 實時系統特性
- [ ] ESP32-CAM 硬件原理
- [ ] 基本電路知識

推薦學習資源：
- edX 課程："Embedded Systems - Shape The World" by University of Texas at Austin
- 書籍：《Making Embedded Systems》by Elecia White
- All About Circuits 教程：https://www.allaboutcircuits.com/education/

## 學習建議
- 根據個人興趣和需求選擇優先學習項目
- 結合理論學習和實際動手練習
- 循序漸進，從基礎開始逐步深入
- 參與開源項目或論壇討論，與他人交流學習經驗
