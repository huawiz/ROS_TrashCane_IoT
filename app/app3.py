import os
import cv2
import torch
import numpy as np
import sys
import time
import pathlib
import requests
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# YOLOv5 目錄設置
YOLOV5_PATH = r"C:\yolo\yolov5"  
if YOLOV5_PATH not in sys.path:
    sys.path.append(YOLOV5_PATH)

from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox

class YOLODetector:
    def __init__(self, weights_path, camera_id=0, img_size=640, conf_thres=0.25, iou_thres=0.45, esp32_ip="192.168.1.100"):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size
        self.esp32_ip = esp32_ip
        
        # 修改檢測相關參數
        self.detection_interval = 5     # 檢測間隔
        self.required_count = 10         # 需要7次才觸發
        self.detection_counts = defaultdict(int)
        self.last_process_time = time.time()
        
        # 狀態顯示參數
        self.current_status = ""
        self.status_time = 0
        self.status_duration = 2        # 狀態顯示持續時間
        self.min_status_interval = 7    # 最小狀態改變間隔也改為7秒
        self.last_status_change = 0
        
        # 文字顯示設置
        self.text_settings = {
            'status': {
                'font_size': 80,       # 狀態文字大小
                'y_offset': -30,       # 垂直位置偏移
                'color': (0, 255, 0),  # 文字顏色 (BGR)
                'background_color': (0, 0, 0),  # 背景顏色
                'padding': 20,         # 背景padding
            },
            'fps': {
                'font_size': 30,       # FPS文字大小
                'position': (15, 25),  # 文字位置
                'color': (0, 255, 0),
                'background_color': (0, 0, 0),
                'padding': 8,
            },
            'size': {
                'font_size': 30,       # 尺寸文字大小
                'position': (15, 65),  # 文字位置
                'color': (0, 255, 0),
                'background_color': (0, 0, 0),
                'padding': 8,
            }
        }
        
        # 載入字體
        try:
            self.fonts = {
                'status': ImageFont.truetype("msyh.ttc", self.text_settings['status']['font_size']),
                'fps': ImageFont.truetype("msyh.ttc", self.text_settings['fps']['font_size']),
                'size': ImageFont.truetype("msyh.ttc", self.text_settings['size']['font_size'])
            }
        except:
            print("無法載入微軟正黑體，嘗試使用系統字體路徑")
            try:
                self.fonts = {
                    'status': ImageFont.truetype("C:\\Windows\\Fonts\\msyh.ttc", self.text_settings['status']['font_size']),
                    'fps': ImageFont.truetype("C:\\Windows\\Fonts\\msyh.ttc", self.text_settings['fps']['font_size']),
                    'size': ImageFont.truetype("C:\\Windows\\Fonts\\msyh.ttc", self.text_settings['size']['font_size'])
                }
            except:
                print("無法載入字體，使用默認字體")
                self.fonts = {key: ImageFont.load_default() for key in ['status', 'fps', 'size']}
        
        # 初始化模型和設備
        self.device = select_device('')
        print(f"正在載入模型: {weights_path}")
        self.model = DetectMultiBackend(weights_path, device=self.device)
        
        if hasattr(self.model, 'stride') and isinstance(self.model.stride, torch.Tensor):
            stride = int(self.model.stride.max())
        else:
            stride = 32
            
        self.img_size = check_img_size(self.img_size, s=stride)
        self.names = self.model.names
        print(f"檢測類別: {self.names}")
        
        # 初始化攝像頭
        print(f"正在初始化攝像頭 ID: {camera_id}")
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                raise ValueError(f"無法打開攝像頭 {camera_id}")
        
        self.optimize_camera()
        
        cv2.namedWindow('YOLOv5 Detection', cv2.WINDOW_NORMAL)
        self.model.warmup(imgsz=(1, 3, self.img_size, self.img_size))
    
    def optimize_camera(self):
        """優化相機設置"""
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # 清空緩衝區
        for _ in range(5):
            self.cap.read()

    def send_command(self, command):
        """發送指令到ESP32"""
        try:
            response = requests.get(f"http://{self.esp32_ip}/{command}", timeout=5)
            print(f"指令發送{'成功' if response.status_code == 200 else '失敗'}: {command}")
        except requests.exceptions.RequestException as e:
            print(f"指令發送錯誤: {e}")

    def process_detection(self, class_name, count):
        """處理檢測結果"""
        current_time = time.time()
        
        # 檢查是否達到最小狀態改變間隔
        if current_time - self.last_status_change < self.min_status_interval:
            return
        
        print(f"在過去 {self.detection_interval} 秒內，{class_name} 被檢測到 {count} 次")
        
        if class_name in ["1_PET_ITEM", "5_PP_ITEM","1_PET_SIGN", "5_PP_SIGN"]:
            print("開右")
            self.current_status = "開右"
            self.status_time = current_time
            self.last_status_change = current_time
            # self.send_command("R")
        elif class_name in ["2_HDPE_ITEM", "6_PS_ITEM","2_HDPE_SIGN", "6_PS_SIGN"]:
            print("開左")
            self.current_status = "開左"
            self.status_time = current_time
            self.last_status_change = current_time
            # self.send_command("L")

    def draw_chinese_text(self, img, text, font, position, color, bg_color, padding, show_background=True):
        """使用PIL繪製中文文字，可選擇是否顯示背景"""
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 獲取文字大小
        bbox = draw.textbbox(position, text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 只在需要時繪製背景
        if show_background:
            rect_x0 = position[0] - padding
            rect_y0 = position[1] - padding
            rect_x1 = position[0] + text_width + padding
            rect_y1 = position[1] + text_height + padding
            draw.rectangle([(rect_x0, rect_y0), (rect_x1, rect_y1)], fill=bg_color)
        
        # 繪製文字
        draw.text(position, text, font=font, fill=color)
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def draw_status(self, frame):
        """在畫面上顯示狀態"""
        current_time = time.time()
        if self.current_status and (current_time - self.status_time) < self.status_duration:
            settings = self.text_settings['status']
            h, w = frame.shape[:2]
            
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            bbox = draw.textbbox((0, 0), self.current_status, font=self.fonts['status'])
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (w - text_width) // 2
            y = (h - text_height) // 2 + settings['y_offset']
            
            # 繪製狀態文字，不顯示背景
            frame = self.draw_chinese_text(
                frame,
                self.current_status,
                self.fonts['status'],
                (x, y),
                settings['color'],
                settings['background_color'],
                settings['padding'],
                show_background=False  # 設定為False表示不顯示背景
            )
        
        return frame

    def preprocess_image(self, img):
        """預處理圖像"""
        img = letterbox(img, self.img_size, stride=32, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def detect(self):
        """執行檢測"""
        print(f"開始檢測，按 'q' 退出")
        print(f"設置：{self.detection_interval}秒內需要檢測到{self.required_count}次才觸發動作")
        fps_time = time.time()
        frame_count = 0
        fps_display = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("無法讀取攝像頭畫面")
                    continue
                
                orig_h, orig_w = frame.shape[:2]
                img = self.preprocess_image(frame)
                pred = self.model(img, augment=False, visualize=False)
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
                
                current_time = time.time()
                processed_frame = frame.copy()
                
                for i, det in enumerate(pred):
                    annotator = Annotator(processed_frame)
                    
                    if len(det):
                        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                        
                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)
                            class_name = self.names[c]
                            self.detection_counts[class_name] += 1
                            
                            # 更新檢測資訊顯示
                            print(f"類別: {class_name}, 當前計數: {self.detection_counts[class_name]}/{self.required_count}")
                            
                            label = f'{class_name} {conf:.2f}'
                            annotator.box_label(xyxy, label, color=colors(c, True))
                
                # 每隔指定時間處理一次檢測結果
                if current_time - self.last_process_time >= self.detection_interval:
                    print("\n=== 檢測間隔結束 ===")
                    for class_name, count in self.detection_counts.items():
                        if count > 0:
                            print(f"{class_name}: {count}/{self.required_count} 次")
                            self.process_detection(class_name, count)
                    
                    # 重置計數器
                    print("重置計數器")
                    self.detection_counts.clear()
                    self.last_process_time = current_time
                
                # 繪製狀態
                processed_frame = self.draw_status(processed_frame)
                
                # 更新FPS
                frame_count += 1
                if current_time - fps_time >= 1.0:
                    fps = frame_count / (current_time - fps_time)
                    fps_time = current_time
                    frame_count = 0
                    fps_display = fps
                
                # 繪製FPS和尺寸信息
                fps_settings = self.text_settings['fps']
                size_settings = self.text_settings['size']
                
                processed_frame = self.draw_chinese_text(
                    processed_frame,
                    f'FPS: {fps_display:.1f}',
                    self.fonts['fps'],
                    fps_settings['position'],
                    fps_settings['color'],
                    fps_settings['background_color'],
                    fps_settings['padding'],
                    show_background=True  # FPS顯示保留背景
                )
                
                processed_frame = self.draw_chinese_text(
                    processed_frame,
                    f'Size: {orig_w}x{orig_h}',
                    self.fonts['size'],
                    size_settings['position'],
                    size_settings['color'],
                    size_settings['background_color'],
                    size_settings['padding'],
                    show_background=True  # 尺寸信息保留背景
                )
                
                cv2.imshow('YOLOv5 Detection', processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except Exception as e:
            print(f"發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("檢測結束")

def test_cameras():
    """測試可用的攝像頭"""
    print("正在測試可用的攝像頭...")
    available_cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"攝像頭 ID {i} 可用")
                available_cameras.append(i)
            cap.release()
    return available_cameras

def main():
    try:
        available_cameras = test_cameras()
        if not available_cameras:
            print("找不到可用的攝像頭！")
            return
        
        weights_path = r"C:\yolo\yolov5\runs\train\yolo_custom_train11\weights\best.pt"
        
        if len(available_cameras) > 1:
            print("\n可用的攝像頭:")
            for i, cam_id in enumerate(available_cameras):
                print(f"{i}: Camera {cam_id}")
            choice = int(input("請選擇要使用的攝像頭 (輸入編號): "))
            camera_id = available_cameras[choice]
        else:
            camera_id = available_cameras[0]
        
        # 創建檢測器實例
        detector = YOLODetector(
            weights_path=weights_path,
            camera_id=camera_id,
            conf_thres=0.25,
            iou_thres=0.45,
            esp32_ip="192.168.1.100"  # 請修改為您的ESP32 IP地址
        )
        
        # 開始檢測
        detector.detect()
        
    except Exception as e:
        print(f"發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()