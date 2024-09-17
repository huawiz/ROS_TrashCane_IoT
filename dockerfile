# 使用官方 ROS 2 Jazzy 映像作為基礎
FROM osrf/ros:jazzy-desktop

# 切換到 root 用戶以進行安裝
USER root

# 安裝一些常用工具和依賴，包括 ESP32 所需的工具和 python3-venv
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-pytest-cov \
    ros-dev-tools \
    python3-pytest \
    python3-flake8 \
    python3-serial \
    usbutils \
    socat \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# 創建虛擬環境並在其中安裝 Python 包
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN /opt/venv/bin/pip3 install --no-cache-dir \
    pytest-repeat \
    pytest-rerunfailures \
    flake8-blind-except \
    flake8-builtins \
    flake8-class-newline \
    flake8-comprehensions \
    flake8-deprecated \
    flake8-docstrings \
    flake8-import-order \
    flake8-quotes \
    pytest-flake8 \
    pyserial

# 設置工作目錄
WORKDIR /root/ros2_ws

# 複製當前目錄下的檔案到容器中（如果有的話）
COPY . .

# 設置環境變量
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 創建一個腳本來設置 ESP32 連接
RUN echo '#!/bin/bash\n\
if [ -e /dev/ttyUSB0 ]; then\n\
    echo "ESP32 found at /dev/ttyUSB0"\n\
    socat pty,link=/dev/ttyESP32,raw,echo=0 /dev/ttyUSB0,raw,echo=0 &\n\
elif [ -e /dev/ttyACM0 ]; then\n\
    echo "ESP32 found at /dev/ttyACM0"\n\
    socat pty,link=/dev/ttyESP32,raw,echo=0 /dev/ttyACM0,raw,echo=0 &\n\
else\n\
    echo "ESP32 not found. Please check the connection."\n\
fi\n\
source /opt/ros/jazzy/setup.bash\n\
source /opt/venv/bin/activate\n\
exec bash' > /root/entrypoint.sh \
    && chmod +x /root/entrypoint.sh

# 設置 entrypoint
ENTRYPOINT ["/root/entrypoint.sh"]

# 開放 serial 端口
EXPOSE 5601