from rocket import Rocket
import numpy as np

class Environment():
    def __init__(self) -> None:
        # Rocket in the environment
        self.rocket: Rocket = Rocket()

        # Size of the environment in pixels
        self.width_px: int = 1000
        self.height_px: int = 800
        
        # Scale factor (must match visualization)
        self.M_TO_PX: float = 40.0
        
        # Size of the environment in meters
        self.x_max: float = self.width_px / self.M_TO_PX
        self.y_max: float = self.height_px / self.M_TO_PX

    def iterate(self, delta_time: float, e1: bool, e2: bool, e3: bool):
        # Iterate the rocket
        self.rocket.iteration(e1, e2, e3, delta_time)

        # Get the position and orientation of the rocket
        position: np.array
        orientation: np.array
        position, orientation = self.rocket.get_position_and_orientation()

        # If the rocket goes out of bounds
        # Note: position is in meters.
        if position[0] > self.x_max or position[0] < 0 or position[1] > self.y_max or position[1] < 0:
            self.rocket.reset_position()
        
        return position, orientation