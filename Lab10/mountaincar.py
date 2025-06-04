import gym
import numpy as np
import time

#CIĄGŁY I CIĄGŁY


# Add compatibility fix for numpy
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

def run_mountain_car_random():
    """Run MountainCar with random actions"""
    env = gym.make('MountainCarContinuous-v0', render_mode="human")
    
    observation, info = env.reset(seed=42)
    
    for step in range(1000):
        # Random action between -1 and 1 (continuous action space)
        action = np.random.uniform(-1, 1, size=(1,))
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Print info every 100 steps
        if step % 100 == 0:
            position, velocity = observation
            print(f"Step {step}: Position={position:.3f}, Velocity={velocity:.3f}, Action={action[0]:.3f}")
        
        time.sleep(0.01)  # Small delay to see the movement
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break
    
    env.close()

def run_mountain_car_smart():
    """Run MountainCar with a smart strategy"""
    env = gym.make('MountainCarContinuous-v0', render_mode="human")
    
    observation, info = env.reset(seed=42)
    
    for step in range(1000):
        position, velocity = observation
        
        # Smart strategy: push in direction of velocity to build momentum
        if velocity > 0:
            action = np.array([1.0])  # Push right when moving right
        elif velocity < 0:
            action = np.array([-1.0])  # Push left when moving left
        else:
            action = np.array([1.0])   # Default push right
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Print progress
        if step % 50 == 0:
            print(f"Step {step}: Position={position:.3f}, Velocity={velocity:.3f}, Action={action[0]:.3f}")
        
        time.sleep(0.05)  # Slower to see the strategy
        
        if terminated or truncated:
            if position >= 0.45:  # Goal position
                print(f"🎉 SUCCESS! Reached the goal at step {step}!")
            else:
                print(f"Episode ended at step {step}")
            break
    
    env.close()

def run_mountain_car_oscillation():
    """Run MountainCar with oscillation strategy"""
    env = gym.make('MountainCarContinuous-v0', render_mode="human")
    
    observation, info = env.reset(seed=42)
    
    for step in range(1000):
        position, velocity = observation
        
        # Oscillation strategy: build momentum by pushing opposite to position
        if position < -0.25:  # If in left valley
            action = np.array([1.0])   # Push right
        elif position > -0.25:  # If climbing right
            action = np.array([-1.0])  # Push left to build momentum
        else:
            action = np.array([0.0])   # No action at center
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        if step % 25 == 0:
            print(f"Step {step}: Pos={position:.3f}, Vel={velocity:.3f}, Action={action[0]:.3f}")
        
        time.sleep(0.02)
        
        if terminated or truncated:
            if position >= 0.45:
                print(f"🎉 SUCCESS! Car reached the goal at step {step}!")
            else:
                print(f"Episode ended at step {step}")
            break
    
    env.close()

def understand_mountain_car():
    """Understand the MountainCar environment"""
    env = gym.make('MountainCarContinuous-v0')
    
    print("=== MountainCar Environment Info ===")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Range: {env.action_space.low} to {env.action_space.high}")
    print(f"Position Range: {env.observation_space.low[0]} to {env.observation_space.high[0]}")
    print(f"Velocity Range: {env.observation_space.low[1]} to {env.observation_space.high[1]}")
    
    print("\n=== Game Rules ===")
    print("- Goal: Get the car to position >= 0.45 (top of right hill)")
    print("- Start: Car starts at bottom of valley (position ≈ -0.5)")
    print("- Engine: Not powerful enough to drive straight up")
    print("- Strategy: Build momentum by rocking back and forth")
    print("- Actions: Continuous values from -1 (full left) to +1 (full right)")
    
    env.close()

if __name__ == "__main__":
    print("MountainCar Game Runner")
    print("=" * 40)
    
    print("\nChoose option:")
    print("1. Random actions")
    print("2. Smart momentum strategy")
    print("3. Oscillation strategy")
    print("4. Understand the environment")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        print("Running with random actions...")
        run_mountain_car_random()
    elif choice == "2":
        print("Running with smart momentum strategy...")
        run_mountain_car_smart()
    elif choice == "3":
        print("Running with oscillation strategy...")
        run_mountain_car_oscillation()
    elif choice == "4":
        understand_mountain_car()
    else:
        print("Invalid choice, running smart strategy...")
        run_mountain_car_smart()