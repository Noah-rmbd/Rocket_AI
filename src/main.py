import sys
import os
import time
import numpy as np
import pygame
from environment import Environment
from visualization import Visualization
#import keyboardMac

def main():
    pygame.init()
    # State of the program
    running: bool = True

    # State of the visualization
    visualize: bool = True
    visualization: Visualization = Visualization()

    # Environment
    environment: Environment = Environment()

    # Refresh rate
    frame_time: float = 1/60

    # Engine states
    engine1: bool = False
    engine2: bool = False
    engine3: bool = False

    while running:
        start_time: float = time.time()

        for event in pygame.event.get():
            # Check for the 'X' button on the window
            if event.type == pygame.QUIT:
                running = False
            
            # Check for key *presses* (fires only once per press)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                
                if event.key == pygame.K_v:
                    visualize = not visualize
                    print(f"Visualization: {visualize}")
        
        # Check for held keys
        keys = pygame.key.get_pressed()
        engine1 = keys[pygame.K_i]
        engine2 = keys[pygame.K_o]
        engine3 = keys[pygame.K_p]
        
        # Update the environment
        position: np.array
        rotation: np.array
        position, rotation = environment.iterate(frame_time, engine1, engine2, engine3)

        # Draw the rocket
        if visualize:
            visualization.draw_rocket(position, rotation, environment.rocket.y_cm)
        
        # Framerate control
        end_time: float = time.time()
        delta_time: float = end_time - start_time

        if visualize and delta_time < frame_time:
            time.sleep(frame_time - delta_time)

    # End the visualization
    visualization.end_visualization()
    sys.exit()

if __name__ == "__main__":
    main()