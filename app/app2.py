import cv2
import numpy as np
import time
import os
from tensorflow.keras.models import load_model
import tensorflow as tf
import h5py
import json
import requests
import easyocr

class VisionSystem:
    def __init__(self):
        print(f"TensorFlow version: {tf.__version__}")
        np.set_printoptions(suppress=True)
        
        self.model = None
        self.class_names = None
        self.cap = None
        # 初始化 EasyOCR
        print("初始化 EasyOCR...")
        self.reader = easyocr.Reader(['en'])  # 只使用英文辨識以提高速度
        
        # 系統參數
        self.target_classes = ["1_PET_ITEM", "2_HDPE_ITEM", "5_PP_ITEM", "6_PS_ITEM", "0_OTHER", "0_SIGN"]
        self.detection_interval = 7  # 檢測間隔（秒）
        self.threshold = 7  # 檢測閾值
        self.confidence_threshold = 80  # 信心閾值（%）
        self.esp32_ip = "192.168.0.39"
        
        # 初始化狀態
        self.class_counts = {class_name: 0 for class_name in self.target_classes}
        self.last_reset_time = time.time()

    def load_model_with_fix(self, model_path):
        """載入和修復模型"""
        try:
            # 修復模型
            with h5py.File(model_path, mode="r+") as f:
                model_config = json.loads(f.attrs.get('model_config'))
                model_config = self._remove_groups(model_config)
                f.attrs.modify('model_config', json.dumps(model_config).encode())
                f.flush()
            
            # 載入模型
            return tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            raise Exception(f"模型載入失敗: {str(e)}")

    def _remove_groups(self, config):
        """從模型配置中移除groups"""
        if isinstance(config, dict):
            if 'groups' in config:
                del config['groups']
            return {k: self._remove_groups(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._remove_groups(item) for item in config]
        return config

    def initialize_camera(self):
        """優化的攝影機初始化"""
        try:
            # 直接使用 DirectShow 後端（Windows）以加速初始化
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            
            if not self.cap.isOpened():
                raise Exception("無法開啟攝影機")

            # 關鍵參數設定
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)        # 最小緩衝區
            self.cap.set(cv2.CAP_PROP_FPS, 30)             # 設定 FPS
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)    # 設定解析度
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # 快速讀取一幀確認攝影機工作正常
            ret, _ = self.cap.read()
            if not ret:
                raise Exception("無法讀取影像")
                
            print("攝影機初始化成功")
            return True
                
        except Exception as e:
            print(f"攝影機初始化失敗: {str(e)}")
            if self.cap is not None:
                self.cap.release()
            return False

    def initialize_model(self):
        """初始化模型和標籤"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'keras_Model.h5')
        labels_path = os.path.join(current_dir, 'labels.txt')
        
        if not all(os.path.exists(p) for p in [model_path, labels_path]):
            raise FileNotFoundError("找不到必要的模型或標籤文件")
            
        print(f"載入模型: {model_path}")
        print(f"載入標籤: {labels_path}")
        
        self.model = self.load_model_with_fix(model_path)
        with open(labels_path, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]

    def preprocess_image(self, frame):
        """預處理圖像用於模型預測"""
        try:
            image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
            return (image / 127.5) - 1
        except Exception as e:
            raise ValueError(f"圖像處理失敗: {str(e)}")

    def recognize_number(self, frame):
        """使用 EasyOCR 辨識標誌中的數字"""
        try:
            # 影像預處理
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 增強對比度
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # EasyOCR 辨識
            results = self.reader.readtext(gray, 
                                         allowlist='1256',  # 限制只辨識1,2,5,6
                                         width_ths=0.7,     # 寬度閾值
                                         batch_size=1,      # 批次大小
                                         detail=0)          # 只返回文字結果
            
            # 處理結果
            if results:
                # 只取第一個辨識結果
                number = ''.join(filter(str.isdigit, results[0]))
                if number in ['1', '2', '5', '6']:
                    return number
            return None
            
        except Exception as e:
            print(f"OCR辨識錯誤: {str(e)}")
            return None

    def send_command(self, command):
        """發送指令到ESP32"""
        try:
            response = requests.get(f"http://{self.esp32_ip}/{command}", timeout=5)
            print(f"指令發送{'成功' if response.status_code == 200 else '失敗'}: {command}")
        except requests.exceptions.RequestException as e:
            print(f"指令發送錯誤: {e}")

    def process_detection(self, class_name, count):
        """處理檢測結果"""
        print(f"在過去 {self.detection_interval} 秒內，{class_name} 被檢測到 {count} 次")
        
        if class_name in ["1_PET_ITEM", "5_PP_ITEM"]:
            print("開右")
            # self.send_command("R")
        elif class_name in ["2_HDPE_ITEM", "6_PS_ITEM"]:
            print("開左")
            # self.send_command("L")
        elif class_name == "0_SIGN":
            number = self.recognize_number(self.current_frame)
            if number:
                print(f"辨識到回收標誌數字: {number}")
                if number in ["1", "5"]:
                    print("開右")
                    # self.send_command("R")
                elif number in ["2", "6"]:
                    print("開左")
                    # self.send_command("L")
            else:
                print("未能辨識到有效數字")

    def run(self):
        """主要運行循環"""
        try:
            
            print('初始化模型')
            self.initialize_model()
            print('初始化攝影機')
            self.initialize_camera()
            
            while True:
                time.sleep(0.5)
                ret, self.current_frame = self.cap.read()
                if not ret:
                    print("無法讀取影像")
                    break

                # 顯示影像
                display_frame = cv2.resize(self.current_frame, (224, 224))
                cv2.imshow(u"環智新境".encode('utf-8').decode('utf-8'), display_frame,)

                # 模型預測
                image = self.preprocess_image(self.current_frame)
                prediction = self.model.predict(image)
                index = np.argmax(prediction)
                class_name = self.class_names[index].strip()[2:]
                confidence_score = prediction[0][index]
                
                print(f"Class: {class_name}", end="  ")
                print(f"Confidence Score: {str(np.round(confidence_score * 100))[:-2]}%")

                # 更新計數器
                if (class_name in self.target_classes and 
                    np.round(confidence_score * 100) >= self.confidence_threshold):
                    self.class_counts[class_name] += 1

                # 檢查是否需要處理檢測結果
                current_time = time.time()
                if current_time - self.last_reset_time >= self.detection_interval:
                    for name, count in self.class_counts.items():
                        if count >= self.threshold:
                            self.process_detection(name, count)
                    
                    # 重置計數器
                    self.class_counts = {name: 0 for name in self.target_classes}
                    self.last_reset_time = current_time

                # 檢查退出條件
                if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                    print("程式結束")
                    break

        except Exception as e:
            print(f"執行錯誤: {e}")
        finally:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    vision_system = VisionSystem()
    vision_system.run()