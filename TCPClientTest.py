import socket

# Define the server IP address and port
SERVER_IP = '192.168.137.176'  # Replace with your Raspberry Pi's IP
PORT = 12345                      # Same port as the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((SERVER_IP, PORT))
    message = 'Hello from the client!'
    s.sendall(message.encode())
    
    data = s.recv(1024)
    print(f"Received back: {data.decode()}")
