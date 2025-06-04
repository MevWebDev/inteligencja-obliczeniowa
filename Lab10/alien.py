import gym
import numpy as np
import time

#dyskretny i ciągły

# Add compatibility fix for numpy
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

def get_alien_env():
    """Try different ways to create the Alien environment"""
    env_names = [
        "ALE/Alien-v5",      # Latest format
        "Alien-v5",          # Alternative format
        "Alien-v4",          # Older version
        "AlienNoFrameskip-v4"  # No frameskip version
    ]
    
    for env_name in env_names:
        try:
            env = gym.make(env_name, render_mode="human")
            print(f"Successfully created environment: {env_name}")
            return env
        except Exception as e:
            print(f"Failed to create {env_name}: {e}")
            continue
    
    print("Could not create any Alien environment. Trying Breakout instead...")
    try:
        return gym.make("ALE/Breakout-v5", render_mode="human")
    except:
        print("No Atari environments available. Using CartPole instead...")
        return gym.make("CartPole-v1", render_mode="human")

def run_alien_random():
    """Run Alien with random actions"""
    env = get_alien_env()
    
    observation, info = env.reset(seed=42)
    total_score = 0
    
    print("Starting game with random actions...")
    print(f"Action space: {env.action_space}")
    print(f"Available actions: {env.action_space.n}")
    
    for step in range(5000):  # Play for 5000 steps
        action = env.action_space.sample()  # Random action
        
        observation, reward, terminated, truncated, info = env.step(action)
        total_score += reward
        
        # Print score updates
        if reward > 0:
            print(f"Step {step}: Got {reward} points! Total score: {total_score}")
        
        time.sleep(0.05)  # Slow down to see the game
        
        if terminated or truncated:
            print(f"Game Over! Final score: {total_score}")
            break
    
    env.close()

def run_alien_smart():
    """Run Alien with a basic strategy"""
    env = get_alien_env()
    
    observation, info = env.reset(seed=42)
    total_score = 0
    step_count = 0
    
    print("Starting game with smart strategy...")
    
    # Get available actions
    n_actions = env.action_space.n
    
    # Basic strategy: prefer action 1 (usually FIRE) and some movement
    preferred_actions = [1, 2, 3, 4, 5] if n_actions > 5 else list(range(n_actions))
    
    for step in range(5000):
        # Use smart action selection
        if step % 10 < 6:  # Fire/action more frequently
            action = 1 if n_actions > 1 else 0
        else:
            action = np.random.choice(preferred_actions)
        
        observation, reward, terminated, truncated, info = env.step(action)
        total_score += reward
        step_count += 1
        
        if reward > 0:
            print(f"Step {step}: Scored {reward} points! Total: {total_score}")
        
        # Show progress every 500 steps
        if step % 500 == 0:
            print(f"Step {step}: Score = {total_score}")
        
        time.sleep(0.03)  # Slightly faster than random
        
        if terminated or truncated:
            print(f"Game Over! Final score: {total_score} in {step_count} steps")
            break
    
    env.close()

def list_available_envs():
    """List available Atari environments"""
    print("Checking available environments...")
    
    atari_envs = [
        "ALE/Alien-v5", "ALE/Breakout-v5", "ALE/Pong-v5", "ALE/SpaceInvaders-v5",
        "Alien-v5", "Breakout-v5", "Pong-v5", "SpaceInvaders-v5",
        "Alien-v4", "Breakout-v4", "Pong-v4", "SpaceInvaders-v4"
    ]
    
    available = []
    for env_name in atari_envs:
        try:
            env = gym.make(env_name)
            available.append(env_name)
            env.close()
            print(f"✅ {env_name}")
        except Exception as e:
            print(f"❌ {env_name}: {str(e)[:50]}...")
    
    if available:
        print(f"\nFound {len(available)} available Atari environments!")
        return available[0]  # Return first available
    else:
        print("\nNo Atari environments found. You may need to install ALE:")
        print("pip install 'gymnasium[atari,accept-rom-license]'")
        return None

def run_any_available_game():
    """Run any available Atari game"""
    available_env = list_available_envs()
    
    if available_env:
        print(f"\nRunning {available_env}...")
        env = gym.make(available_env, render_mode="human")
        
        observation, info = env.reset(seed=42)
        total_score = 0
        
        for step in range(2000):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total_score += reward
            
            if reward > 0:
                print(f"Step {step}: Scored {reward}! Total: {total_score}")
            
            time.sleep(0.05)
            
            if terminated or truncated:
                print(f"Game Over! Final score: {total_score}")
                break
        
        env.close()
    else:
        print("No Atari games available. Install with:")
        print("pip install 'gymnasium[atari,accept-rom-license]'")

if __name__ == "__main__":
    print("Atari Game Runner")
    print("=" * 40)
    
    print("\nChoose option:")
    print("1. Try to run Alien (random actions)")
    print("2. Try to run Alien (smart strategy)")
    print("3. List available environments")
    print("4. Run any available Atari game")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        print("Trying to run Alien with random actions...")
        run_alien_random()
    elif choice == "2":
        print("Trying to run Alien with smart strategy...")
        run_alien_smart()
    elif choice == "3":
        list_available_envs()
    elif choice == "4":
        run_any_available_game()
    else:
        print("Invalid choice, trying to run any available game...")
        run_any_available_game()