import sys
import os
import time
import numpy as np
import pygame
import argparse
import torch
from environment import Environment
from visualization import Visualization
from agent import DQNAgent
from trainer import train

def run_agent(model_path="final_model.pth"):
    pygame.init()
    running = True
    visualize = True
    visualization = Visualization()
    environment = Environment()
    frame_time = 1/60
    
    # Load Agent
    state_size = 3
    action_size = 8
    agent = DQNAgent(state_size, action_size, seed=0)
    try:
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    except:
        print(f"Could not load model from {model_path}, starting with random weights")

    # Action map
    action_map = {
        0: (False, False, False),
        1: (False, False, True),
        2: (False, True, False),
        3: (False, True, True),
        4: (True, False, False),
        5: (True, False, True),
        6: (True, True, False),
        7: (True, True, True)
    }

    environment.rocket.reset_position()
    # Randomize initial state for testing
    environment.rocket.orientation = np.random.uniform(-0.5, 0.5)
    environment.rocket.angular_velocity = np.random.uniform(-1.0, 1.0)

    while running:
        start_time = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_v:
                    visualize = not visualize
                if event.key == pygame.K_r:
                    environment.rocket.reset_position()
                    environment.rocket.orientation = np.random.uniform(-0.5, 0.5)
                    environment.rocket.angular_velocity = np.random.uniform(-1.0, 1.0)
                    environment.rocket.velocity = np.zeros(2)

        # Get state
        theta = environment.rocket.orientation
        av = environment.rocket.angular_velocity
        state = np.array([np.cos(theta), np.sin(theta), av], dtype=np.float32)

        # Agent action
        action_idx = agent.act(state, eps=0.0) # Greedy action
        engine1, engine2, engine3 = action_map[action_idx]
        
        # Update environment
        position, rotation = environment.iterate(frame_time, engine1, engine2, engine3)

        # Draw
        if visualize:
            visualization.draw_rocket(position, rotation, environment.rocket.y_cm)
        
        # Framerate
        end_time = time.time()
        delta_time = end_time - start_time
        if visualize and delta_time < frame_time:
            time.sleep(frame_time - delta_time)

    visualization.end_visualization()
    sys.exit()

def main():
    parser = argparse.ArgumentParser(description='Rocket AI Control')
    parser.add_argument('--mode', type=str, default='manual', choices=['manual', 'train', 'test'], help='Mode: manual, train, or test')
    parser.add_argument('--model', type=str, default='final_model.pth', help='Path to model file for test mode')
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        run_agent(args.model)
    else:
        # Manual Mode
        pygame.init()
        running = True
        visualize = True
        visualization = Visualization()
        environment = Environment()
        frame_time = 1/60
        engine1 = False
        engine2 = False
        engine3 = False

        while running:
            start_time = time.time()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    
                    if event.key == pygame.K_v:
                        visualize = not visualize
                        print(f"Visualization: {visualize}")
            
            keys = pygame.key.get_pressed()
            engine1 = keys[pygame.K_i]
            engine2 = keys[pygame.K_o]
            engine3 = keys[pygame.K_p]
            
            position, rotation = environment.iterate(frame_time, engine1, engine2, engine3)

            if visualize:
                visualization.draw_rocket(position, rotation, environment.rocket.y_cm)
            
            end_time = time.time()
            delta_time = end_time - start_time

            if visualize and delta_time < frame_time:
                time.sleep(frame_time - delta_time)

        visualization.end_visualization()
        sys.exit()

if __name__ == "__main__":
    main()