# Put this code on your client PC
import socket
import cv2
import pickle
import struct
import threading

# Initialize socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('129.8.239.33', 8080))  # Replace with your Pi's IP address

# Load the face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')  # Update the path

data = b""
payload_size = struct.calcsize("Q")

# Shared variable to store the latest frame
latest_frame = None
frame_lock = threading.Lock()

def receive_frames():
    global data, latest_frame
    while True:
        # Receive frame size
        while len(data) < payload_size:
            data += client_socket.recv(4096)

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        # Receive the frame data
        while len(data) < msg_size:
            data += client_socket.recv(4096)

        frame_data = data[:msg_size]
        data = data[msg_size:]

        # Deserialize frame data
        frame_encoded = pickle.loads(frame_data)

        # Decode the JPEG image
        frame = cv2.imdecode(frame_encoded, cv2.IMREAD_COLOR)

        # Update the latest frame in a thread-safe manner
        with frame_lock:
            latest_frame = frame

def detect_faces_and_send():
    global client_socket, latest_frame
    while True:
        with frame_lock:
            if latest_frame is not None:
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                # Draw rectangles around detected faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(latest_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Send feedback to server if faces are detected
                if len(faces) > 0:
                    client_socket.sendall(b"Hello")

        # Display the frame
        if latest_frame is not None:
            cv2.imshow('Video Feed', latest_frame)

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
