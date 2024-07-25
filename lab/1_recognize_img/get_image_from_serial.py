import serial
import cv2
import numpy as np

ser = serial.Serial('COM9', 921600)  # 根據實際情況修改串口

while True:
    # 讀取圖像大小
    size_data = ser.read(4)
    size = int.from_bytes(size_data, byteorder='little')
    
    # 讀取圖像數據
    img_data = ser.read(size)
    
    # 將數據轉換為圖像
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 顯示圖像
    cv2.imshow('Image', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
ser.close()