# This code serves to connect to a client PC in order to send its two cameras' input
# It encodes every fifth frame as a jpg file and packs them for sending to the client PC
# Not to be used in the final product but purely serves to test how the TCP socket handles two cameras being transferred

import socket
import cv2
import numpy as np
import struct
import pickle
from picamera2 import Picamera2

# Initialize both cameras
picam_left = Picamera2(camera_num=0)
picam_left.preview_configuration.main.size = (320, 240)
picam_left.preview_configuration.main.format = "RGB888"
picam_left.preview_configuration.align()
picam_left.configure("preview")
picam_left.start()

picam_right = Picamera2(camera_num=1)
picam_right.preview_configuration.main.size = (320, 240)
picam_right.preview_configuration.main.format = "RGB888"
picam_right.preview_configuration.align()
picam_right.configure("preview")
picam_right.start()

# Initialize socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 8080))
server_socket.listen(1)

print("Waiting for a connection...")
conn, addr = server_socket.accept()
print(f"Connection from: {addr}")

frame_count = 0

try:
    while True:
        # Capture frame from both cameras, sending every 5th frame
        if frame_count % 5 == 0:
            frame_left = picam_left.capture_array()
            frame_right = picam_right.capture_array()

            # Compress both frames as JPEG
            _, frame_encoded_left = cv2.imencode('.jpg', frame_left, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            _, frame_encoded_right = cv2.imencode('.jpg', frame_right, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            
            # Serialize the encoded frames
            data_left = pickle.dumps(frame_encoded_left)
            data_right = pickle.dumps(frame_encoded_right)

            # Pack the data lengths
            message_size_left = struct.pack("Q", len(data_left))
            message_size_right = struct.pack("Q", len(data_right))

            # Send packed frame lengths followed by the frame data
            conn.sendall(message_size_left + data_left + message_size_right + data_right)
        
        frame_count += 1
except (ConnectionResetError, BrokenPipeError):
    print("Client disconnected.")
finally:
    # Cleanup
    picam_left.stop()
    picam_right.stop()
    conn.close()
    server_socket.close()
