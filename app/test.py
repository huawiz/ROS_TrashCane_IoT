import serial
import serial.tools.list_ports
import cv2
import numpy as np
import time
import struct
import os
from datetime import datetime

def find_esp32_port():
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        try:
            ser = serial.Serial(port.device, 921600, timeout=1)
            start_time = time.time()
            while time.time() - start_time < 5:  # 等待最多5秒
                if ser.in_waiting:
                    data = ser.read(ser.in_waiting)
                    if b'\xFF\xD8\xFF' in data:  # 檢測JPEG開始標記
                        print(f"找到ESP32相機端口: {port.device}")
                        return ser
            ser.close()
        except (OSError, serial.SerialException):
            pass
    return None

def find_jpeg_start(ser):
    marker = b'\xFF\xD8\xFF'
    buffer = b''
    start_time = time.time()
    while time.time() - start_time < 5:  # 設置超時
        byte = ser.read(1)
        if not byte:
            continue
        buffer += byte
        if buffer.endswith(marker):
            return True
    return False

def read_image(ser):
    if not find_jpeg_start(ser):
        raise ValueError("無法找到JPEG開始標記")
    
    size_data = ser.read(4)
    if len(size_data) != 4:
        raise ValueError("無法讀取圖像大小")
    
    size = struct.unpack('<I', size_data)[0]
    if size > 1000000 or size < 1000:  # 假設圖像大小在1KB到1MB之間
        raise ValueError(f"無效的圖像大小: {size} 字節")
    
    img_data = ser.read(size)
    if len(img_data) != size:
        raise ValueError(f"圖像數據不完整: 預期 {size} 字節，實際接收 {len(img_data)} 字節")
    
    return img_data

def ensure_img_directory():
    """確保img資料夾存在"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, 'img')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
        print(f"建立影像儲存資料夾: {img_dir}")
    return img_dir

def main():
    # 確保img資料夾存在
    img_dir = ensure_img_directory()
    
    # 連接ESP32相機
    ser = find_esp32_port()
    if not ser:
        print("未找到ESP32相機端口")
        return

    try:
        print("串口成功打開")
        print("\n操作說明:")
        print("1. 按 's' 儲存當前畫面")
        print("2. 按 'q' 或 'ESC' 退出程式")
        
        cv2.namedWindow("ESP32-CAM Image", cv2.WINDOW_NORMAL)
        
        saved_count = 0  # 儲存的圖片計數

        while True:
            try:
                time.sleep(0.1)
                img_data = read_image(ser)

                # 讀取並顯示圖像
                frame = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                frame = cv2.flip(frame, -1)  # 上下左右顛倒
                display_frame = cv2.resize(frame, (800, 800))
                
                # 在畫面上顯示提示和計數
                cv2.putText(display_frame, f"Saved: {saved_count} | 's': save, 'q': quit", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("ESP32-CAM Image", display_frame)
                
                # 檢查按鍵
                key = cv2.waitKey(1) & 0xFF
                
                # 按's'儲存圖片
                if key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"esp32cam_{timestamp}.jpg"
                    save_path = os.path.join(img_dir, filename)
                    cv2.imwrite(save_path, frame)  # 儲存原始大小的圖片
                    saved_count += 1
                    print(f"已儲存圖片: {filename}")
                
                # 按'q'或ESC退出
                elif key in [27, ord('q')]:
                    print("退出程序")
                    break
            
            except (ValueError, serial.SerialException) as e:
                print(f"錯誤: {e}")
                print("嘗試重新同步...")
                ser.reset_input_buffer()
                time.sleep(1)

    except Exception as e:
        print(f"發生錯誤: {e}")
    
    finally:
        print(f"\n總共儲存了 {saved_count} 張圖片")
        if ser and ser.is_port:
            ser.close()
        cv2.destroyAllWindows()
        print("程序結束")

if __name__ == "__main__":
    main()