# Utilizing Deep Q-Learning for Efficient Object Sorting with a Robotic Arm
Authors: Takara Busby, Tomas Garcia Gallardo, Marcus Ramirez
CSU Fresno - CSCI 166 - Fall Semester 2024

Robotics is a growing field in today's market and the growing capabilities of machine learning has been a major boon for it.  In this project we attempt to create a neural network that can control a robotic arm comprised of six servos to sort colored cubes.  To do this we used Deep Q Learning and a simulation for the training of the model.  The model used CNN's and fully connected layers in order to map features from input of a camera and servo angles.  The simulation was done in PyBullet and the network made with PyTorch.  Testing showed that the model was very susceptible to changes in learning rate and we actually had problems getting it to converge with it, oscillating between good rewards and bad rewards while remaining in the initial phase.  This implies the model requires further fine-tuning and testing in order to get it to a state where it can be used in the real world.

## Objectives
- Design & implement a 6 Degrees of Freedom (DOF) robotic arm to grab and sort colored blocks in a simulated environment
- Leverage Deep Q-Learning to optimize behavior w/ iterative interactions
- Ensure precise object detection and manipulation based on visual input
- Design a system adaptable for real-world deployment beyond simulation

    
## Software Dependencies
 - Python
 - PyTorch
 - TorchVision
 - PyBullet
 - matplotlib
 - OpenCV
 - GPIOZero
 - PiCamera2

## Hardware Utilized
 - [Raspberry Pi 5](https://www.raspberrypi.com/products/raspberry-pi-5/)
 - [Raspberry Pi Camera Module V2](https://www.raspberrypi.com/products/camera-module-v2/)
 - [6 DOF Mechanical Arm w/ 6 MG996R digital servo motors](https://www.amazon.com/Mechanical-Programmable-Manipulator-Parameter-Industrial/dp/B0B1368WN1?_encoding=UTF8&pd_rd_i=B0B1368WN1&pd_rd_w=zN26W&content-id=amzn1.sym.5334573a-029b-477d-a49c-2bbdfe16b2cb&pf_rd_p=5334573a-029b-477d-a49c-2bbdfe16b2cb&pf_rd_r=9ZQX3MFBXTJ13WV4JQ1A&pd_rd_wg=6NhHz&pd_rd_r=87a46c1a-eb4d-4331-aba9-f906f2583bd0)
 - [TF Luna Lidar Sensor](https://www.amazon.com/Benewake-TF-Luna-Single-Point-Ranging-Interface/dp/B086MJQSLR)
 - [Power Supply](https://www.amazon.com/Zeee-Battery-3600mAh-Connector-Associated/dp/B0B8YXSY35?dib=eyJ2IjoiMSJ9.wg0HEr1b7FZK-74krDku93LF9nW6C4_ZEmNfs_D2nXO154CX5RTnNxNyByEGle9HDYRZpg9BVhjIJCqiJJateSzBgzWMbCzKvYuG2kmoJJWYBI8dkWbYbRNN1XakCmVYGJ2SSX6oB08fcJBt9dg2gp6mCEoc3pUO0Ys36YEWTFPjbcGPKEpxS0Z8IPeFfI5dHK9O_p1e4v7WcOGLDSidGZpIiGFdu74gjUzRitGPxOtzWV0Dv6BCcf7bGBP7tRZ4S1PHdC18Zd-XTEMd3TT5WoWlkAydkuEXKdbGqpl_FSQ.ionJckw7DwCykVPUFtur77HxghyUZR8od9i4w5W3coU&dib_tag=se&keywords=7.2v+rc+battery&qid=1732057157&sr=8-8)

## About `SortingArm` Folder
The folder mentioned contains several test scripts that are for testing between the client PC and the Raspberry Pi.  Many of it is sample stuff that could be easily ignored, but the intention of including it is to show all the algorithms taken into consideration for implementation in real world testing.

## About `throwaway_code` Folder
The folder mentioned above also contains test scripts, but for loading a sample URDF file of the robot arm in a pybullet simulation.  It attempts to train the robot with a PPO algorithm, albeit faulty.
