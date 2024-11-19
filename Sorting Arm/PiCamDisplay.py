from picamera2 import Picamera2 #Library specifically for raspberry picam module
import cv2

# Function to initialize and test the camera
def initialize_camera():
    try:
        # Initialize Picamera2
        picam = Picamera2()
        picam.preview_configuration.main.size = (640, 480)  # Set resolution
        picam.preview_configuration.main.format = "RGB888"
        picam.preview_configuration.align()
        picam.configure("preview")
        picam.start()
        
        print("Camera initialized successfully!")
        return picam
    except Exception as e:
        print(f"Failed to initialize camera: {e}")
        return None

# Initialize the camera
camera = initialize_camera()

# Display the video feed from the camera
if camera:
    try:
        while True:
            # Capture a frame from the camera
            frame = camera.capture_array()

            # Print the shape of the array (height, width, channels)
            print(f"Frame shape: {frame.shape}")

            # Display the frame using OpenCV
            cv2.imshow('Camera Feed', frame)
            
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Video feed interrupted.")
    finally:
        # Stop the camera and cleanup
        camera.stop()
        cv2.destroyAllWindows()
else:
    print("Failed to initialize the camera.")
