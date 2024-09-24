# Put this code on client PC.
import socket
import cv2
import pickle
import struct

# Initialize socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('129.8.239.99', 8080))  # Replace with your Pi's IP address

data = b""
payload_size = struct.calcsize("Q")  # Use Q for 64-bit

while True:
    # Retrieve message size
    while len(data) < payload_size:
        data += client_socket.recv(4096)
    
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    
    msg_size = struct.unpack("Q", packed_msg_size)[0]  # Use Q for 64-bit

    # Retrieve frame data
    while len(data) < msg_size:
        data += client_socket.recv(4096)

    frame_data = data[:msg_size]
    data = data[msg_size:]

    # Deserialize frame data
    frame = pickle.loads(frame_data)

    # Display the frame directly without conversion
    cv2.imshow('Video Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
client_socket.close()
cv2.destroyAllWindows()