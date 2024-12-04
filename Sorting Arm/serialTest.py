#The following is designed to test data flow through the UART ports of the Pi given a jumper wire connected between the TXD and RXD ports
import serial

ser = serial.Serial("/dev/serial0", baudrate = 115200, timeout=1)

if ser.isOpen():
    print("Serial port is open!")
    while True:
        data = ser.read(9)
        print(f"Raw Data: {data}")
        if data[0] == 0x59 and data[1] == 0x59:
            distance = data[2] + (data[3] << 8)
            strength = data[4] + (data[5] << 8)
            temperature = data[6] + (data[7] << 8)
            temperature = (temperature / 8.0) - 256
            print(f"Distance: {distance/100.0} meters, Strength: {strength}, Temp: {temperature} C")
        else:
            print("Invalid data packet")
else:
    print("Serial port failed to open")
