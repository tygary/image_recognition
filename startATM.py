import cv2
from bank_scanner import parse_form_image
import time

class ATM:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise Exception("Could not open video device")
        
    def start(self):
        print("ATM started. Press 'q' to quit.")
        try:
            while True:
                # Capture frame-by-frame
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Save the frame temporarily
                temp_image = "temp_capture.jpg"
                cv2.imwrite(temp_image, frame)
                
                # Try to parse the account number
                result = parse_form_image(temp_image)
                if result:
                    print(f"Detected account number: {result}")
                
                # Display the frame
                cv2.imshow('ATM Camera', frame)
                
                # Break the loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Add a small delay to prevent high CPU usage
                time.sleep(0.1)
                
        finally:
            self.camera.release()
            cv2.destroyAllWindows()

def main():
    atm = ATM()
    atm.start()

if __name__ == "__main__":
    main()
