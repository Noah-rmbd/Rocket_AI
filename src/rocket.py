import sys
from math import sqrt
import numpy as np

class Rocket():
    def __init__(self) -> None:
        # Masses (kg)
        m_triangle: int = 10 # Mass of the triangular head
        m_rectangle: int = 20 # Mass of the rectangular base

        # Dimensions (m)
        rectangle_width: float = 0.5
        rectangle_height: float = 2.0
        triangle_height: float = sqrt(pow(rectangle_width, 2) - pow(rectangle_width / 2, 2))

        # Position of the rocket (2D) - Center of Mass (cm)
        # Initialized in meters. Screen is 1000x1000 px, so 25x25 m with 40px/m scale.
        self.position = np.array([12.5, 12.5])

        # Positions of centers of mass (relative to the base of the rocket)
        # Assuming the base of the rectangle is at y=0
        y_triangle: float = rectangle_height + triangle_height/3
        y_rectangle: float = rectangle_height/2

        # Moment of inertia of the triangle about its centroid
        I_triangle: float = (m_triangle * triangle_height**2) / 18
        
        # Moment of inertia of the rectangle about its centroid
        I_rectangle: float = (m_rectangle * (rectangle_width**2 + rectangle_height**2)) / 12

        # Total mass
        self.total_mass: int = m_triangle + m_rectangle

        # Center of mass of the entire rocket
        self.y_cm = (m_triangle * y_triangle + m_rectangle * y_rectangle) / self.total_mass

        # Distance from the rocket's center of mass to each component's center of mass
        d_triangle = abs(y_triangle - self.y_cm)
        d_rectangle = abs(y_rectangle - self.y_cm)

        # Parallel axis theorem: I_total = I_triangle + m_triangle d_triangle^2 + I_rectangle + m_rectangle * d_rectangle^2
        self.I_total: float = I_triangle + m_triangle * d_triangle**2 + I_rectangle + m_rectangle * d_rectangle**2

        # Forces (N) and their points of application (m from the base) - 2D vectors
        self.F1_local: np.array = np.array([0.0, -400.0])  # Force from reactor 1
        self.F2_local: np.array = np.array([0.0, -680.0])  # Force from reactor 2
        self.F3_local: np.array = np.array([0.0, -400.0])   # Force from reactor 3

        # Positions of the forces (relative to the base)
        self.r1_local: np.array = np.array([-1.0, -self.y_cm])  # Position of reactor 1
        self.r2_local: np.array = np.array([0.0, -self.y_cm]) # Position of reactor 2
        self.r3_local: np.array = np.array([1.0, -self.y_cm])  # Position of reactor 3

        # Gravitational force
        g: float = 9.81  # m/s^2
        self.F_gravity: float = np.array([0.0, self.total_mass * g])

        # Air Resistance Constants
        self.Cd: float = 0.5 # Drag coefficient
        self.rho: float = 1.225 # Air density (kg/m^3)
        self.A: float = rectangle_width # Reference area (m^2) - using width for 2D profile

        # Net force
        self.F_net: np.array = np.array([0.0, 0.0])

        # Linear acceleration
        self.a_linear: np.array = self.F_net / self.total_mass

        # Net torque
        self.tau_net: np.array = np.array([0.0, 0.0])

        # Angular acceleration
        self.alpha: float = self.tau_net / self.I_total

        self.angular_velocity: float = 0.0
        self.velocity: np.array = np.array([0.0, 0.0])
        self.orientation: float = 0.0

    def iteration(self, engine1: bool, engine2: bool, engine3: bool, delta_time: float):
        # Create a 2D rotation matrix from the rocket's current orientation
        c, s = np.cos(self.orientation), np.sin(self.orientation)
        rotation_matrix = np.array([[c, -s], 
                                    [s,  c]])

        # Indicate for each engine if it is on (1) or off (0)
        e1 = 1 if engine1 else 0
        e2 = 1 if engine2 else 0
        e3 = 1 if engine3 else 0

        # Rotate the LOCAL engine forces to get the WORLD forces
        F1_world = rotation_matrix @ self.F1_local * e1
        F2_world = rotation_matrix @ self.F2_local * e2
        F3_world = rotation_matrix @ self.F3_local * e3
        
        # Calculate Drag Force
        # F_drag = -0.5 * rho * |v| * v * Cd * A
        velocity_magnitude = np.linalg.norm(self.velocity)
        if velocity_magnitude > 0:
            F_drag = -0.5 * self.rho * velocity_magnitude * self.velocity * self.Cd * self.A
        else:
            F_drag = np.array([0.0, 0.0])

        # Calculate the net force
        self.F_net = (F1_world + F2_world + F3_world + self.F_gravity + F_drag)
        
        self.a_linear = self.F_net / self.total_mass
        
        tau1 = np.cross(self.r1_local, self.F1_local * e1)
        tau2 = np.cross(self.r2_local, self.F2_local * e2)
        tau3 = np.cross(self.r3_local, self.F3_local * e3)

        # Calculate the net torque
        damping_coefficient = 5.0
        tau_damping = -damping_coefficient * self.angular_velocity * abs(self.angular_velocity)
        
        self.tau_net = tau1 + tau2 + tau3 + tau_damping
        self.alpha = self.tau_net / self.I_total
        
        # Update velocity and position
        self.velocity += self.a_linear * delta_time
        self.position += self.velocity * delta_time

        # Update angular velocity and orientation
        self.angular_velocity += self.alpha * delta_time
        self.orientation += self.angular_velocity * delta_time

    def get_position_and_orientation(self):
        return self.position, self.orientation
    
    def reset_position(self):
        self.position = np.array([12.5, 12.5])