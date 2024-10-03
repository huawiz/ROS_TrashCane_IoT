import serial
import serial.tools.list_ports
import cv2
import numpy as np
import time
import struct
import os
from tensorflow.keras.models import load_model
import tensorflow as tf
import h5py
import json
import requests

def fix_keras_model(model_path):
    with h5py.File(model_path, mode="r+") as f:
        model_config = json.loads(f.attrs.get('model_config'))
        
        def remove_groups(config):
            if isinstance(config, dict):
                if 'groups' in config:
                    del config['groups']
                for key, value in config.items():
                    config[key] = remove_groups(value)
            elif isinstance(config, list):
                config = [remove_groups(item) for item in config]
            return config
        
        model_config = remove_groups(model_config)
        
        f.attrs.modify('model_config', json.dumps(model_config).encode())
        f.flush()

def load_model_with_fix(model_path):
    # 首先修復模型文件
    fix_keras_model(model_path)
    
    # 然後加載修復後的模型
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

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

def preprocess_image(img_data):
    nparr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1
    return image

def send_command(esp32_ip, command):
    url = f"http://{esp32_ip}/{command}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"成功發送指令: {command}")
        else:
            print(f"發送指令失敗: {command}, 狀態碼: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"發送指令時發生錯誤: {e}")

def main():
    print(tf.__version__)
    np.set_printoptions(suppress=True)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'keras_Model.h5')
    labels_path = os.path.join(current_dir, 'labels.txt')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")
    
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"找不到標籤文件: {labels_path}")

    print(f"模型文件路徑: {model_path}")
    print(f"標籤文件路徑: {labels_path}")

    model = load_model_with_fix(model_path)

    with open(labels_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    ser = find_esp32_port()
    if not ser:
        print("未找到ESP32相機端口")
        return

    try:
        print("串口成功打開")
        cv2.namedWindow("ESP32-CAM Image", cv2.WINDOW_NORMAL)
        
        # 初始化計數器和時間
        target_classes = ["1_pet_1", "2_other_7","3_object_x"]
        class_counts = {class_name: 0 for class_name in target_classes}
        last_reset_time = time.time()
        detection_interval = 3  # 檢測間隔，單位為秒
        threshold = 3  # 在檢測間隔內檢測到的閾值
        confidence_threshold = 50  # 信心閾值，百分比

        while True:
            try:
                img_data = read_image(ser)
                image = preprocess_image(img_data)
                cv2.imshow("ESP32-CAM Image", cv2.resize(cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR), (224, 224)))
                
                prediction = model.predict(image)
                index = np.argmax(prediction)
                class_name = class_names[index].strip()
                confidence_score = prediction[0][index]   # 轉換為百分比
                
                print("Class:", class_name[2:], end="")
                print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
                # 只有當檢測到目標類別且信心分數大於或等於閾值時才更新計數器
                if class_name[2:] in target_classes and np.round(confidence_score * 100) >= confidence_threshold:
                    class_counts[class_name[2:]] += 1
                
                ip = "192.168.0.11"
                # 檢查是否超過檢測間隔
                current_time = time.time()
                if current_time - last_reset_time >= detection_interval:
                    for name, count in class_counts.items():
                        if count >= threshold:
                            if name == "1_pet_1":
                                print(f"在過去 {detection_interval} 秒內，{name} 被檢測到 {count} 次 (信心分數 >= {confidence_threshold}%)")
                                send_command(ip,"R")
                            elif name == "2_other_7":
                                print(f"在過去 {detection_interval} 秒內，{name} 被檢測到 {count} 次 (信心分數 >= {confidence_threshold}%)")
                                send_command(ip,"L")
                            elif name == "3_object_x":
                                send_command(ip,"OFF")
                            
                    
                    # 重置計數器和時間
                    class_counts = {name: 0 for name in target_classes}
                    last_reset_time = current_time
                
                keyboard_input = cv2.waitKey(1) & 0xFF
                if keyboard_input == 27 or keyboard_input == ord('q'):
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
        if ser and ser.is_port:
            ser.close()
        cv2.destroyAllWindows()
        print("程序結束")

if __name__ == "__main__":
    main()