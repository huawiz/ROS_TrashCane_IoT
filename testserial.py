import serial
import time

def test_serial():
    try:
        ser = serial.Serial('/dev/ttyUSB0', 921600, timeout=1)
        print("Serial port opened successfully")
        
        start_time = time.time()
        while time.time() - start_time < 10:  # Run for 10 seconds
            if ser.in_waiting:
                data = ser.read(ser.in_waiting)
                print(f"Received data: {data}")
            time.sleep(0.1)
        
        ser.close()
        print("Test completed")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_serial()