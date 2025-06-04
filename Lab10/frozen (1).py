import gym
import numpy as np
import time

# Add compatibility fix for numpy
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

def simple_right_then_down_strategy():
    """Simple strategy: go max right, then max down"""
    
    # Create environment to get map size - DISABLE SLIPPERY!
    env = gym.make('FrozenLake8x8-v1', is_slippery=False)
    desc = env.unwrapped.desc
    nrow, ncol = desc.shape
    env.close()
    
    print(f"Map size: {nrow}x{ncol}")
    
    # Generate action sequence: RIGHT (ncol-1) times, then DOWN (nrow-1) times
    actions = []
    
    # Go right to the rightmost column
    for _ in range(ncol - 1):
        actions.append(2)  # RIGHT
    
    # Go down to the bottom row
    for _ in range(nrow - 1):
        actions.append(1)  # DOWN
    
    print(f"Action sequence: {actions}")
    print(f"Total moves: {len(actions)}")
    print("Strategy: Go RIGHT 7 times, then DOWN 7 times")
    
    return actions

def play_simple_strategy():
    """Play the game with simple right-then-down strategy"""
    
    # Get the action sequence
    actions = simple_right_then_down_strategy()
    
    # Now play with visualization - DISABLE SLIPPERY!
    env = gym.make('FrozenLake8x8-v1', render_mode="human", is_slippery=False)
    
    observation, info = env.reset(seed=42)
    print(f"\nStarting game with {len(actions)} planned moves...")
    print("🧊 Slippery mode DISABLED - actions will be deterministic!")
    
    total_reward = 0
    
    for step, action in enumerate(actions):
        action_name = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}[action]
        print(f"\nStep {step + 1}: Taking action {action} ({action_name})")
        
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        time.sleep(1)  # Pause to see the movement
        
        if terminated or truncated:
            if reward > 0:
                print(f"🎉 SUCCESS! Reached the goal in {step + 1} steps!")
                print(f"Total reward: {total_reward}")
            else:
                print(f"💀 Failed! Game ended at step {step + 1}")
                print(f"Total reward: {total_reward}")
            break
    else:
        # If we completed all actions without terminating
        print("Completed all planned actions!")
    
    env.close()


def play_both_modes():
    """Compare slippery vs non-slippery modes"""
    actions = simple_right_then_down_strategy()
    
    print("\n" + "="*60)
    print("COMPARING SLIPPERY vs NON-SLIPPERY MODES")
    print("="*60)
    
    # Test with slippery mode (default)
    print("\n🧊 SLIPPERY MODE (default behavior):")
    env_slippery = gym.make('FrozenLake8x8-v1', render_mode="human")
    observation, info = env_slippery.reset(seed=42)
    
    for step, action in enumerate(actions):  # Just first 5 moves
        action_name = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}[action]
        print(f"Step {step + 1}: Want {action_name}")
        
        observation, reward, terminated, truncated, info = env_slippery.step(action)
        time.sleep(1)
        
        if terminated or truncated:
            break
    
    env_slippery.close()
    
    # Test without slippery mode
    print("\n🎯 NON-SLIPPERY MODE (deterministic):")
    env_deterministic = gym.make('FrozenLake8x8-v1', render_mode="human", is_slippery=False)
    observation, info = env_deterministic.reset(seed=42)
    
    for step, action in enumerate(actions):  # Just first 5 moves
        action_name = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}[action]
        print(f"Step {step + 1}: Want {action_name}")
        
        observation, reward, terminated, truncated, info = env_deterministic.step(action)
        time.sleep(1)
        
        if terminated or truncated:
            break
    
    env_deterministic.close()

if __name__ == "__main__":
    print("Simple FrozenLake Strategy: Right then Down")
    print("="*50)
    
    print("\nChoose option:")
    print("1. Play once with visualization (NON-SLIPPERY)")
    print("2. Compare slippery vs non-slippery modes")
    
    choice = input("Enter choice (1-2): ").strip()
    
    if choice == "1":
        play_simple_strategy()
    
    elif choice == "2":
        play_both_modes()
    else:
        print("Invalid choice, playing once...")
        play_simple_strategy()