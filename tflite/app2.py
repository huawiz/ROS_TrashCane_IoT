import serial
import serial.tools.list_ports
import numpy as np
import time
import struct
import os
import tflite_runtime.interpreter as tflite

def find_esp32_port():
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        if "USB" in port.device:  # Most ESP32 boards appear as USB devices on Linux
            try:
                ser = serial.Serial(port.device, 921600, timeout=1)
                start_time = time.time()
                while time.time() - start_time < 5:  # Wait up to 5 seconds
                    if ser.in_waiting:
                        data = ser.read(ser.in_waiting)
                        if b'\xFF\xD8\xFF' in data:  # Check for JPEG start marker
                            print(f"Found ESP32 camera port: {port.device}")
                            return ser
                ser.close()
            except (OSError, serial.SerialException):
                pass
    return None

def find_jpeg_start(ser):
    marker = b'\xFF\xD8\xFF'
    buffer = b''
    start_time = time.time()
    while time.time() - start_time < 5:  # Set timeout
        byte = ser.read(1)
        if not byte:
            continue
        buffer += byte
        if buffer.endswith(marker):
            return True
    return False

def read_image(ser):
    if not find_jpeg_start(ser):
        raise ValueError("Unable to find JPEG start marker")
    
    size_data = ser.read(4)
    if len(size_data) != 4:
        raise ValueError("Unable to read image size")
    
    size = struct.unpack('<I', size_data)[0]
    if size > 1000000 or size < 1000:  # Assume image size is between 1KB and 1MB
        raise ValueError(f"Invalid image size: {size} bytes")
    
    img_data = ser.read(size)
    if len(img_data) != size:
        raise ValueError(f"Incomplete image data: expected {size} bytes, received {len(img_data)} bytes")
    
    return img_data

def preprocess_image(img_data):
    # We'll use PIL instead of OpenCV for image processing
    from PIL import Image
    import io
    image = Image.open(io.BytesIO(img_data))
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32)
    image = np.expand_dims(image, axis=0)
    image = (image / 127.5) - 1
    return image

def main():
    np.set_printoptions(suppress=True)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'model.tflite')
    labels_path = os.path.join(current_dir, 'labels.txt')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    print(f"Model file path: {model_path}")
    print(f"Labels file path: {labels_path}")

    # Load TFLite model
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    with open(labels_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    ser = find_esp32_port()
    if not ser:
        print("ESP32 camera port not found")
        return

    try:
        print("Serial port successfully opened")
        
        target_classes = ["1_pet_1", "2_other_7"]
        class_counts = {class_name: 0 for class_name in target_classes}
        last_reset_time = time.time()
        detection_interval = 3
        threshold = 3
        confidence_threshold = 50

        while True:
            try:
                img_data = read_image(ser)
                image = preprocess_image(img_data)
                
                # Set input tensor
                interpreter.set_tensor(input_details[0]['index'], image)

                # Run inference
                interpreter.invoke()

                # Get output tensor
                output_data = interpreter.get_tensor(output_details[0]['index'])
                
                index = np.argmax(output_data)
                class_name = class_names[index].strip()
                confidence_score = output_data[0][index]
                
                print("Class:", class_name[2:], end="")
                print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
                
                if class_name[2:] in target_classes and np.round(confidence_score * 100) >= confidence_threshold:
                    class_counts[class_name[2:]] += 1
                
                current_time = time.time()
                if current_time - last_reset_time >= detection_interval:
                    for name, count in class_counts.items():
                        if count >= threshold:
                            print(f"In the past {detection_interval} seconds, {name} was detected {count} times (confidence score >= {confidence_threshold}%)")
                    
                    class_counts = {name: 0 for name in target_classes}
                    last_reset_time = current_time
                
                # Check for user input to exit
                if input() in ['q', 'Q']:
                    print("Exiting program")
                    break
            
            except (ValueError, serial.SerialException) as e:
                print(f"Error: {e}")
                print("Attempting to resynchronize...")
                ser.reset_input_buffer()
                time.sleep(1)

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        if ser and ser.is_open:
            ser.close()
        print("Program ended")

if __name__ == "__main__":
    main()