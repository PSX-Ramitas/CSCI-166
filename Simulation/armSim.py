import pybullet as p
import time
import pybullet_data

class ArmEnv:
    def __init__(self, gui=True):
        # initialize variables
        self.gui = gui
        self.physics_client = None

    def connect(self):
        mode = p.GUI if self.gui else p.DIRECT
        self.physics_client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        print("connected to pybullet simulation")

    def disconnect(self):
        # diconnect form the simulation
        if self.physics_client is not None:
            p.disconnect()
            self.physics_client = None
            print("diconnected from pybullet simulation")

    def reset(self):
        if self.physics_client is not None:
            p.resetSimulation()
            print("Simulation Reset")
