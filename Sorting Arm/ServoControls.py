import socket
from gpiozero import Servo
from time import sleep

# Initialize each servo on a specific GPIO pin
servos = [
    Servo(2, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000),  # Base
    Servo(3, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000),  # Shoulder
    Servo(4, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000),  # Elbow
    Servo(17, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000), # Wrist Pitch
    Servo(27, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000), # Wrist Roll
    Servo(22, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)  # Claw
]

def set_servo_position(index, position):
    # Check for valid index
    if index < 0 or index >= len(servos):
        print("Invalid servo index.")
        return
    
    # Apply limited range for claw
    if index == 5:
        # Clamp position to range -0.5 to 0
        position = max(-0.5, min(0, position))
    else:
        # Clamp position to range -1 to 1
        position = max(-1, min(1, position))
    
    # Set servo position
    servos[index].value = position
    print(f"Servo {index} moved to degree {position * 90}")

# Example usage
set_servo_position(0, -1)   # Full range servo
sleep(1)
set_servo_position(5, -0.5) # Limited range servo
sleep(1)
set_servo_position(5, 0)    # Limited range servo
sleep(1)
