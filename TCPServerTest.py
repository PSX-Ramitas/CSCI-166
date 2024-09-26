# Put this code on your Raspberry Pi
import socket
import cv2
import numpy as np
import struct
import pickle
from picamera2 import Picamera2
import threading

# Initialize the camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (320, 240)  # Set desired resolution
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

frame_count = 0

def listen_for_client_messages():
    while True:
        try:
            # Receive data from client
            message = conn.recv(1024)
            if message:
                print("Received message from client:", message.decode())
                if message == b"Hello":
                    print("Hello from client")
        except (ConnectionResetError, BrokenPipeError):
            print("Client disconnected.")
            break

# Start a thread to listen for client messages
listener_thread = threading.Thread(target=listen_for_client_messages)
listener_thread.start()

try:
    while True:
        # Capture frame from the camera, sending every 5th frame
        if frame_count % 5 == 0:
            frame = picam2.capture_array()

            # Compress the frame as JPEG
            _, frame_encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            # Serialize the encoded frame
            data = pickle.dumps(frame_encoded)
            # Pack the data length
            message_size = struct.pack("Q", len(data))

            # Send packed frame length followed by the frame data
            conn.sendall(message_size + data)

        frame_count += 1
except (ConnectionResetError, BrokenPipeError):
    print("Client disconnected.")
finally:
    # Cleanup
    picam2.stop()
    conn.close()
    server_socket.close()
