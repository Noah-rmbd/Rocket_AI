import numpy as np
from rocket import Rocket

def test_physics():
    rocket = Rocket()
    print(f"Initial Position: {rocket.position}")
    
    # Test 1: Gravity + Drag (Terminal Velocity)
    print("\n--- Test 1: Gravity + Drag (Terminal Velocity) ---")
    expected_v_term = np.sqrt(2 * rocket.total_mass * 9.81 / (rocket.rho * rocket.Cd * rocket.A))
    print(f"Expected Terminal Velocity: {expected_v_term:.2f} m/s")
    
    for _ in range(1000): # 1000 steps of 0.1s = 100s
        rocket.iteration(False, False, False, 0.1)
        
    print(f"Velocity after 100s: {rocket.velocity}")
    
    # Test 2: Angular Damping
    print("\n--- Test 2: Angular Damping ---")
    rocket = Rocket()
    # Apply a kick to start spinning
    rocket.angular_velocity = 10.0 # rad/s
    print(f"Initial Angular Velocity: {rocket.angular_velocity}")
    
    # Simulate for 10 seconds
    for _ in range(100): # 100 steps of 0.1s = 10s
        rocket.iteration(False, False, False, 0.1)
        
    print(f"Angular Velocity after 10s: {rocket.angular_velocity}")
    if abs(rocket.angular_velocity) < 10.0:
        print("SUCCESS: Angular velocity decreased.")
    else:
        print("FAILURE: Angular velocity did not decrease.")

if __name__ == "__main__":
    test_physics()
