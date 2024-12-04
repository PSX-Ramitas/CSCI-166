# This code serves to connect to the Raspberry Pi in order to receive its two cameras' input
# From there, it displays the input onto the client PC and uses the haar cascade and draws a rectangle around detected faces
# Not to be used in the final product but purely serves to test how the TCP socket handles two cameras being transferred

import socket
import cv2
import pickle
import struct
import threading

# Initialize socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('129.8.237.172', 8080))  # Replace with your Pi's IP address

# Load the face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')  # Update the path

data = b""
payload_size = struct.calcsize("Q")

# Shared variables to store the latest frames
latest_frame_left = None
latest_frame_right = None
frame_lock = threading.Lock()

def receive_frames():
    global data, latest_frame_left, latest_frame_right
    while True:
        # Receive left frame size
        while len(data) < payload_size:
            data += client_socket.recv(4096)

        packed_msg_size_left = data[:payload_size]
        data = data[payload_size:]
        msg_size_left = struct.unpack("Q", packed_msg_size_left)[0]

        # Receive the left frame data
        while len(data) < msg_size_left:
            data += client_socket.recv(4096)

        frame_data_left = data[:msg_size_left]
        data = data[msg_size_left:]

        # Deserialize left frame data
        frame_encoded_left = pickle.loads(frame_data_left)

        # Decode the JPEG image for left camera
        frame_left = cv2.imdecode(frame_encoded_left, cv2.IMREAD_COLOR)

        # Receive right frame size
        while len(data) < payload_size:
            data += client_socket.recv(4096)

        packed_msg_size_right = data[:payload_size]
        data = data[payload_size:]
        msg_size_right = struct.unpack("Q", packed_msg_size_right)[0]

        # Receive the right frame data
        while len(data) < msg_size_right:
            data += client_socket.recv(4096)

        frame_data_right = data[:msg_size_right]
        data = data[msg_size_right:]

        # Deserialize right frame data
        frame_encoded_right = pickle.loads(frame_data_right)

        # Decode the JPEG image for right camera
        frame_right = cv2.imdecode(frame_encoded_right, cv2.IMREAD_COLOR)

        # Update the latest frames in a thread-safe manner
        with frame_lock:
            latest_frame_left = frame_left
            latest_frame_right = frame_right

def detect_faces_and_send():
    global client_socket, latest_frame_left, latest_frame_right
    while True:
        with frame_lock:
            if latest_frame_left is not None:
                # Convert to grayscale for face detection on left camera
                gray_left = cv2.cvtColor(latest_frame_left, cv2.COLOR_BGR2GRAY)
                faces_left = face_cascade.detectMultiScale(gray_left, 1.3, 5)

                # Draw rectangles around detected faces on left camera
                for (x, y, w, h) in faces_left:
                    cv2.rectangle(latest_frame_left, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if latest_frame_right is not None:
                # Convert to grayscale for face detection on right camera
                gray_right = cv2.cvtColor(latest_frame_right, cv2.COLOR_BGR2GRAY)
                faces_right = face_cascade.detectMultiScale(gray_right, 1.3, 5)

                # Draw rectangles around detected faces on right camera
                for (x, y, w, h) in faces_right:
                    cv2.rectangle(latest_frame_right, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frames
        if latest_frame_left is not None:
            cv2.imshow('Left Camera Feed', latest_frame_left)

        if latest_frame_right is not None:
            cv2.imshow('Right Camera Feed', latest_frame_right)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Create threads
receive_thread = threading.Thread(target=receive_frames)
detect_thread = threading.Thread(target=detect_faces_and_send)

receive_thread.start()
detect_thread.start()

receive_thread.join()
detect_thread.join()

# Cleanup
client_socket.close()
cv2.destroyAllWindows()
