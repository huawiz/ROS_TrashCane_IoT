import serial
import serial.tools.list_ports
import cv2
import numpy as np
import time
import struct
import threading

def find_esp32_port():
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        try:
            ser = serial.Serial(port.device, 921600, timeout=1)
            start_time = time.time()
            while time.time() - start_time < 5:  # 等待最多5秒
                if ser.in_waiting:
                    data = ser.read(ser.in_waiting)
                    if b'\xFF\xD8\xFF' in data:  # 检测JPEG开始标记
                        print(f"找到ESP32相机端口: {port.device}")
                        return ser
            ser.close()
        except (OSError, serial.SerialException):
            pass
    return None

def find_jpeg_start(ser):
    marker = b'\xFF\xD8\xFF'
    buffer = b''
    start_time = time.time()
    while time.time() - start_time < 5:  # 设置超时
        byte = ser.read(1)
        if not byte:
            continue
        buffer += byte
        if buffer.endswith(marker):
            return True
    return False

def read_image(ser):
    if not find_jpeg_start(ser):
        raise ValueError("无法找到JPEG开始标记")
    
    size_data = ser.read(4)
    if len(size_data) != 4:
        raise ValueError("无法读取图像大小")
    
    size = struct.unpack('<I', size_data)[0]
    if size > 1000000 or size < 1000:  # 假设图像大小在1KB到1MB之间
        raise ValueError(f"无效的图像大小: {size} 字节")
    
    img_data = ser.read(size)
    if len(img_data) != size:
        raise ValueError(f"图像数据不完整: 预期 {size} 字节，实际接收 {len(img_data)} 字节")
    
    return img_data

def keyboard_input(stop_event):
    global should_reset
    while not stop_event.is_set():
        key = input()
        if key.lower() == 'q':
            print("用戶請求退出程序")
            stop_event.set()
        elif key.lower() == 'r':
            print("用戶請求重置 ESP32")
            should_reset = True

def main():
    global should_reset
    should_reset = False
    ser = find_esp32_port()
    if not ser:
        print("未找到ESP32相機端口")
        return

    stop_event = threading.Event()
    keyboard_thread = threading.Thread(target=keyboard_input, args=(stop_event,))
    keyboard_thread.start()

    try:
        print("串口成功打開")
        print("在控制台輸入 'q' 退出程序，輸入 'r' 重置 ESP32")
        
        while not stop_event.is_set():
            try:
                if should_reset:
                    print("發送重置命令到 ESP32")
                    ser.write(b'R')
                    time.sleep(2)  # 給 ESP32 一些重置的時間
                    should_reset = False

                if ser.in_waiting > 0:
                    img_data = read_image(ser)
                    
                    nparr = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if img is not None:
                        cv2.imshow('Image', img)
                        cv2.waitKey(1)  # 更新圖像顯示
                    else:
                        print("無法解碼圖像")
                else:
                    time.sleep(0.01)
            
            except (ValueError, serial.SerialException) as e:
                print(f"錯誤: {e}")
                print("嘗試重新同步...")
                ser.reset_input_buffer()
                time.sleep(1)

    except Exception as e:
        print(f"發生錯誤: {e}")
    
    finally:
        stop_event.set()
        keyboard_thread.join()
        if ser and ser.is_open:
            ser.close()
        cv2.destroyAllWindows()
        print("程序結束")

if __name__ == "__main__":
    main()