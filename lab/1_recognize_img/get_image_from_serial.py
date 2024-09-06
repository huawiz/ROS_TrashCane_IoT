import serial
import serial.tools.list_ports
import cv2
import numpy as np
import time
import struct

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
        print("無法找到JPEG開始標記")
        return None
    
    size_data = ser.read(4)
    if len(size_data) != 4:
        print("無法讀取圖像大小")
        return None
    
    size = struct.unpack('<I', size_data)[0]
    if size > 1000000 or size < 1000:  # 假設圖像大小在1KB到1MB之間
        print(f"無效的圖像大小: {size} 字節")
        return None
    
    img_data = ser.read(size)
    if len(img_data) != size:
        print(f"圖像數據不完整: 預期 {size} 字節，實際接收 {len(img_data)} 字節")
        return None
    
    return img_data

def main():
    ser = find_esp32_port()
    if not ser:
        print("未找到ESP32相機端口")
        return

    print("串口成功打開")
    print("按 'q' 退出程序，按 'r' 重置 ESP32")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("退出程序")
            break
        elif key == ord('r'):
            print("發送重置命令到 ESP32")
            ser.write(b'R')
            time.sleep(2)  # 給 ESP32 一些重置的時間

        if ser.in_waiting > 0:
            img_data = read_image(ser)
            if img_data is not None:
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    cv2.imshow('Image', img)
                else:
                    print("無法解碼圖像")
        else:
            time.sleep(0.01)

    if ser and ser.is_open:
        ser.close()
    cv2.destroyAllWindows()
    print("程序結束")

if __name__ == "__main__":
    main()