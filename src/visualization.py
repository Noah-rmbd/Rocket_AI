import numpy as np
import pygame

class Visualization():
    def __init__(self) -> None:
        #pygame.init()

        # Window dimensions
        width: int = 1000
        height: int = 800

        # Window element
        self.window = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Rocket Visualization")

        # Colors of the interface
        self.colors = {"WHITE": (255, 255, 255), "BLACK": (0, 0, 0), "RED": (255, 0, 0), "GREEN": (0, 255, 0), "BLUE": (0, 0, 255)}
        
        # Rocket dimensions (scaled for visualization)
        self.rectangle_width = 20
        self.rectangle_length = 80
        self.triangle_height = 30

    def draw_rocket(self, position, orientation, y_cm):
        # Clear screen
        self.window.fill(self.colors["WHITE"])

        # Scale factor
        M_TO_PX = 40.0

        # Convert position to pixels
        pos_px = (int(position[0] * M_TO_PX), int(position[1] * M_TO_PX))
        
        # Dimensions in Pixels
        w = 0.5 * M_TO_PX
        h_rect = 2.0 * M_TO_PX
        h_tri = 0.433 * M_TO_PX # approx sqrt(0.5^2 - 0.25^2) = sqrt(0.1875) = 0.433
        y_cm_px = y_cm * M_TO_PX
        
        # Rectangle
        # Top-Left, Top-Right, Bot-Right, Bot-Left
        rect_points = [
            (-w/2, y_cm_px - h_rect),
            (w/2, y_cm_px - h_rect),
            (w/2, y_cm_px),
            (-w/2, y_cm_px),
        ]
        
        # Triangle
        # Base-Left, Base-Right, Tip
        triangle_points = [
            (-w/2, y_cm_px - h_rect),
            (w/2, y_cm_px - h_rect),
            (0, y_cm_px - h_rect - h_tri),
        ]

        # Rotate points
        rotation_matrix = np.array([
            [np.cos(orientation), -np.sin(orientation)],
            [np.sin(orientation), np.cos(orientation)],
        ])
        rect_points = [np.dot(rotation_matrix, point) for point in rect_points]
        triangle_points = [np.dot(rotation_matrix, point) for point in triangle_points]

        # Translate points to rocket position
        rect_points = [(point[0] + pos_px[0], point[1] + pos_px[1]) for point in rect_points]
        triangle_points = [(point[0] + pos_px[0], point[1] + pos_px[1]) for point in triangle_points]

        # Draw rectangle
        pygame.draw.polygon(self.window, self.colors["RED"], rect_points)
        # Draw triangle
        pygame.draw.polygon(self.window, self.colors["BLUE"], triangle_points)

        # Present frame
        pygame.display.flip()

    def end_visualization(self):

        pygame.quit()

