import os
import cv2
import torch
import numpy as np
import sys
import time
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# YOLOv5 目錄設置
YOLOV5_PATH = r"C:\yolo\yolov5"  # 修改為你的 YOLOv5 目錄路徑
if YOLOV5_PATH not in sys.path:
    sys.path.append(YOLOV5_PATH)

from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox

class YOLODetector:
    def __init__(self, weights_path, camera_id=0, img_size=640, conf_thres=0.25, iou_thres=0.45):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size
        
        # 選擇設備 (CPU/GPU)
        self.device = select_device('')
        
        # 載入模型
        print(f"正在載入模型: {weights_path}")
        self.model = DetectMultiBackend(weights_path, device=self.device)
        
        # 修改這裡的 stride 處理方式
        if hasattr(self.model, 'stride') and isinstance(self.model.stride, torch.Tensor):
            stride = int(self.model.stride.max())
        else:
            stride = 32  # 使用默認值
            
        self.img_size = check_img_size(self.img_size, s=stride)
        
        # 獲取類別名稱
        self.names = self.model.names
        print(f"檢測類別: {self.names}")
        
        # 設置攝像頭
        print(f"正在初始化攝像頭 ID: {camera_id}")
        self.cap = cv2.VideoCapture(camera_id,cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise ValueError(f"無法打開攝像頭 {camera_id}")
            
        # 設置視窗
        cv2.namedWindow('YOLOv5 Detection', cv2.WINDOW_NORMAL)
        
        # 模型預熱
        self.model.warmup(imgsz=(1, 3, self.img_size, self.img_size))
        
    def preprocess_image(self, img):
        """預處理圖像"""
        img = letterbox(img, self.img_size, stride=32, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img
    
    def detect(self):
        """執行檢測"""
        print("開始檢測，按 'q' 退出")
        fps_time = time.time()
        frame_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("無法讀取攝像頭畫面")
                    break
                
                # 顯示原始圖像尺寸
                orig_h, orig_w = frame.shape[:2]
                
                # 預處理
                img = self.preprocess_image(frame)
                
                # 推理
                pred = self.model(img, augment=False, visualize=False)
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
                
                # 處理結果
                for i, det in enumerate(pred):
                    annotator = Annotator(frame)
                    
                    if len(det):
                        # 縮放預測框到原始圖像大小
                        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                        
                        # 畫框和標籤
                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)
                            label = f'{self.names[c]} {conf:.2f}'
                            annotator.box_label(xyxy, label, color=colors(c, True))
                
                # 計算並顯示 FPS
                frame_count += 1
                if frame_count >= 30:
                    fps = frame_count / (time.time() - fps_time)
                    fps_time = time.time()
                    frame_count = 0
                    cv2.putText(frame, f'FPS: {fps:.1f}', (20, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 顯示圖像尺寸
                cv2.putText(frame, f'Size: {orig_w}x{orig_h}', (20, 80),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 顯示結果
                cv2.imshow('YOLOv5 Detection', frame)
                
                # 按 'q' 退出
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
        # 首先測試可用的攝像頭
        available_cameras = test_cameras()
        if not available_cameras:
            print("找不到可用的攝像頭！")
            return
        
        # 模型路徑
        weights_path = r"C:\yolo\yolov5\runs\train\yolo_custom_train11\weights\best.pt"  # 修改為你的模型路徑
        
        # 如果有多個攝像頭，讓用戶選擇
        if len(available_cameras) > 1:
            print("\n可用的攝像頭:")
            for i, cam_id in enumerate(available_cameras):
                print(f"{i}: Camera {cam_id}")
            choice = int(input("請選擇要使用的攝像頭 (輸入編號): "))
            camera_id = available_cameras[choice]
        else:
            camera_id = available_cameras[0]
        
        # 創建檢測器
        detector = YOLODetector(
            weights_path=weights_path,
            camera_id=camera_id,
            conf_thres=0.25,
            iou_thres=0.45
        )
        
        # 開始檢測
        detector.detect()
        
    except Exception as e:
        print(f"發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()