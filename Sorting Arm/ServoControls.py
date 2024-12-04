# THis code uses GPIO zero to adjust six servos based off of recived positions
# Designed for eventual use in the final robot

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

def move_servos(positions):
    """
    Adjust all servos at once.
    :param positions: List of positions for each servo, values between -1 and 1.
                      For the claw servo (index 5), the range is -0.5 to 0.
    """
    if len(positions) != len(servos):
        print("Error: Provide a position for each servo.")
        return

    for index, position in enumerate(positions):
        # Clamp position based on servo type
        if index == 5:  # Claw servo
            position = max(-0.5, min(0, position))
        else:
            position = max(-1, min(1, position))

        # Set servo position
        servos[index].value = position
        print(f"Servo {index} moved to degree {position * 90}")

# Example usage
positions = [-1, 0.5, -0.5, 1, -1, -0.25]  # Desired positions for all servos
move_servos(positions)  # Adjust all servos simultaneously
sleep(1)
