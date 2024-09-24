# Put this code in your raspberry pi.
import socket
import cv2
import numpy as np
import struct
import pickle
from picamera2 import Picamera2

# Initialize the camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)  # Set desired resolution
picam2.preview_configuration.main.format = "RGB888"  # Set to RGB888
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Initialize socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 8080))  # Bind to any address on port 8080
server_socket.listen(1)

print("Waiting for a connection...")
conn, addr = server_socket.accept()
print(f"Connection from: {addr}")

try:
    while True:
        # Capture frame from the camera
        frame = picam2.capture_array()

        # Serialize the frame
        data = pickle.dumps(frame)
        # Pack the data length
        message_size = struct.pack("Q", len(data))  # Use Q for 64-bit

        # Send packed frame length followed by the frame data
        conn.sendall(message_size + data)
except (ConnectionResetError, BrokenPipeError):
    print("Client disconnected.")
finally:
    # Cleanup
    picam2.stop()
    conn.close()
    server_socket.close()
