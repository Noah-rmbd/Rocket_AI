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
        # Physics: (0,0) is top-left? No, usually standard cartesian.
        # Pygame: (0,0) is top-left. +Y is down.
        # Physics: Gravity is +Y. So Physics +Y is Down.
        # So Position conversion is direct scaling.
        pos_px = (int(position[0] * M_TO_PX), int(position[1] * M_TO_PX))

        # Rocket dimensions in meters (from rocket.py)
        # We should probably pass these or hardcode them to match rocket.py
        # rocket.py: width=0.5, rect_height=2.0, tri_height=~0.43
        # Let's use the passed y_cm to offset correctly.
        
        # We need to define the shape relative to the COM.
        # Physics Base is at y=0. COM is at y=y_cm.
        # So Base is at y = -y_cm relative to COM.
        # Wait, if +Y is Down in physics (Gravity +), and Base is 0, Top is +Height?
        # rocket.py: y_triangle = rect_height + ...
        # So Triangle is at +Y.
        # So Physics: Base=0, Top=Positive.
        # Thrust is -200 (Negative Y). So Thrust pushes towards Base (0).
        # This implies the rocket points towards +Y (Down)?
        # If Thrust is -Y, and Rocket points +Y, then Thrust is a "Retro-rocket" or it flies backwards?
        # Usually Rocket points UP (-Y). Thrust is UP (-Y).
        # If Rocket points UP, then Head should be at -Y. Base at +Y.
        # In rocket.py: Triangle is at rect_height (2.0). Base at 0.
        # So Triangle is at +Y.
        # If Thrust is -Y, it pushes the Rocket towards 0 (Base).
        # So the rocket moves in the direction of its Base? That's backwards.
        # Unless the "Head" is the exhaust? No, "Triangle head".
        
        # OK, let's fix the Visual Orientation to match "Standard Rocket".
        # Standard: Head points UP (-Y). Thrust pushes UP (-Y).
        # Physics: Gravity +Y (Down).
        # If we want Head to point UP (-Y):
        #   Head should be at Negative Y relative to COM.
        #   Base should be at Positive Y relative to COM.
        #   Thrust should be Negative Y (pushing up).
        
        # Currently in rocket.py:
        #   Base = 0. Head = +2.0.
        #   So Head is DOWN.
        #   Thrust = -200. Pushes UP.
        #   So Thrust pushes the Head towards the Base.
        #   This is correct if the rocket is falling tail-first?
        #   No, usually Thrust is opposite to orientation.
        
        # Let's just flip the Visual so 0 degrees = Pointing UP.
        # And we assume the Physics model meant "Head is at +Y" means "Head is Down".
        # But we want to see it pointing Up.
        # So we will draw the Head at -Y relative to COM.
        # And Base at +Y relative to COM.
        
        # Dimensions in Pixels
        w = 0.5 * M_TO_PX
        h_rect = 2.0 * M_TO_PX
        h_tri = 0.433 * M_TO_PX # approx sqrt(0.5^2 - 0.25^2) = sqrt(0.1875) = 0.433
        
        # COM offset in pixels
        # y_cm is distance from Base to COM.
        # If we want Base at +Y (Bottom), then Base y = +y_cm_px.
        # Head (Triangle) is at Base - Total Height?
        # No, let's stick to the Physics coordinates but rotated 180?
        # Or just define points such that Head is Up.
        
        y_cm_px = y_cm * M_TO_PX
        
        # Define points relative to COM, assuming 0 rotation = Pointing UP (-Y)
        # Base is at +y_cm_px (Bottom)
        # Rect Top is at +y_cm_px - h_rect
        # Triangle Top is at +y_cm_px - h_rect - h_tri
        
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
        # Physics orientation: 0.
        # If we draw it pointing UP at 0, and physics simulates it pointing DOWN at 0...
        # We need to be careful.
        # Physics: Thrust is -Y.
        # If Orientation is 0, Thrust is [0, -200].
        # So Thrust is UP.
        # If we draw Rocket pointing UP at 0, then Thrust aligns with Rocket.
        # So this visual mapping is correct.
        
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

