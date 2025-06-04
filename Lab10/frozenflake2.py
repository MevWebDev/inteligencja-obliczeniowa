import pygad
import numpy as np
import gym
import matplotlib.pyplot as plt

# Add compatibility fix for numpy
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

# Create FrozenLake environment to get the map
env = gym.make('FrozenLake8x8-v1', is_slippery=False)
desc = env.unwrapped.desc
nrow, ncol = desc.shape

print("FrozenLake 8x8 Map:")
for i, row in enumerate(desc):
    row_str = ""
    for j, cell in enumerate(row):
        if cell == b'S':
            row_str += "S "  # Start
        elif cell == b'F':
            row_str += ". "  # Frozen (safe)
        elif cell == b'H':
            row_str += "X "  # Hole (danger)
        elif cell == b'G':
            row_str += "G "  # Goal
    print(f"{i}: {row_str}")

# Find start and goal positions
start_pos = None
goal_pos = None
holes = []

for i in range(nrow):
    for j in range(ncol):
        if desc[i][j] == b'S':
            start_pos = (i, j)
        elif desc[i][j] == b'G':
            goal_pos = (i, j)
        elif desc[i][j] == b'H':
            holes.append((i, j))

print(f"\nStart position: {start_pos}")
print(f"Goal position: {goal_pos}")
print(f"Number of holes: {len(holes)}")

env.close()

# IMPROVED Genetic Algorithm Parameters
gene_space = [0, 1, 2, 3]  # 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP
num_genes = 30  # Increased for longer paths if needed
sol_per_pop = 200  # Larger population
num_parents_mating = 50  # More parents
num_generations = 500  # More generations
parent_selection_type = "tournament"
keep_parents = 20
crossover_type = "two_points"  # Better crossover
mutation_type = "random"
mutation_percent_genes = 15  # Reduced mutation rate

def simulate_path(actions):
    """Simulate a path through FrozenLake and return fitness metrics"""
    current_pos = start_pos
    path = [current_pos]
    visited = set([current_pos])
    
    steps_taken = 0
    fell_in_hole = False
    reached_goal = False
    wall_hits = 0
    revisits = 0
    
    for action in actions:
        row, col = current_pos
        
        # Calculate new position based on action
        if action == 0:    # LEFT
            new_col = col - 1
            new_row = row
        elif action == 1:  # DOWN
            new_row = row + 1
            new_col = col
        elif action == 2:  # RIGHT
            new_col = col + 1
            new_row = row
        elif action == 3:  # UP
            new_row = row - 1
            new_col = col
        
        steps_taken += 1
        
        # Check bounds
        if new_row < 0 or new_row >= nrow or new_col < 0 or new_col >= ncol:
            wall_hits += 1
            continue  # Stay in same position
        
        new_pos = (new_row, new_col)
        
        # Check what's at the new position
        cell_type = desc[new_row][new_col]
        
        if cell_type == b'H':  # Hole
            fell_in_hole = True
            current_pos = new_pos
            path.append(current_pos)
            break
        
        # Valid move
        current_pos = new_pos
        path.append(current_pos)
        
        # Check for revisits
        if new_pos in visited:
            revisits += 1
        else:
            visited.add(new_pos)
        
        # Check if reached goal
        if cell_type == b'G':
            reached_goal = True
            break
    
    return {
        'path': path,
        'reached_goal': reached_goal,
        'fell_in_hole': fell_in_hole,
        'steps_taken': steps_taken,
        'wall_hits': wall_hits,
        'revisits': revisits,
        'final_pos': current_pos
    }

def improved_fitness_func(ga_instance, solution, solution_idx):
    """IMPROVED fitness function with better guidance"""
    result = simulate_path(solution)
    
    # Base fitness starts higher
    fitness = 50.0
    
    # Distance-based fitness (stronger guidance toward goal)
    distance_to_goal = abs(result['final_pos'][0] - goal_pos[0]) + abs(result['final_pos'][1] - goal_pos[1])
    fitness += 200.0 / (1.0 + distance_to_goal)  # Increased weight
    
    # MASSIVE bonus for reaching the goal
    if result['reached_goal']:
        fitness += 2000  # Increased bonus
        # Extra bonus for shorter successful paths
        fitness += 200 * (num_genes - result['steps_taken'])
        print(f"🎉 SOLUTION FOUND! Steps: {result['steps_taken']}, Fitness: {fitness:.2f}")
    
    # Severe penalty for falling in hole
    if result['fell_in_hole']:
        fitness -= 1000  # Increased penalty
    
    # Progress rewards
    progress_x = abs(start_pos[1] - result['final_pos'][1])  # Horizontal progress
    progress_y = abs(start_pos[0] - result['final_pos'][0])  # Vertical progress
    fitness += 50 * (progress_x + progress_y)  # Reward any progress
    
    # Penalties for inefficient behavior
    fitness -= 20 * result['wall_hits']      # Increased penalty for hitting walls
    fitness -= 10 * result['revisits']       # Increased penalty for revisiting positions
    
    # Bonus for exploration
    unique_positions = len(set(result['path']))
    fitness += 10 * unique_positions  # Increased exploration bonus
    
    # Bonus for getting closer to goal
    if distance_to_goal <= 2:  # Very close to goal
        fitness += 300
    elif distance_to_goal <= 4:  # Moderately close
        fitness += 150
    elif distance_to_goal <= 6:  # Getting closer
        fitness += 75
    
    return max(fitness, 0)  # Ensure non-negative fitness

def find_simple_path():
    """Find a simple hardcoded path for comparison"""
    print("\n🔍 Searching for simple paths...")
    
    # Try some basic strategies
    strategies = [
        # Strategy 1: Right then down
        [2] * 7 + [1] * 7,
        # Strategy 2: Down then right  
        [1] * 7 + [2] * 7,
        # Strategy 3: Mixed approach
        [2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1],
        # Strategy 4: Zigzag
        [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
    ]
    
    for i, strategy in enumerate(strategies, 1):
        result = simulate_path(strategy)
        print(f"Strategy {i}: Goal={result['reached_goal']}, Hole={result['fell_in_hole']}, Steps={result['steps_taken']}")
        
        if result['reached_goal']:
            print(f"✅ Simple strategy {i} works!")
            return strategy, result
    
    return None, None

def display_path(path, title="FrozenLake Path"):
    """Display the FrozenLake map with the path highlighted"""
    plt.figure(figsize=(12, 12))
    
    # Create visualization matrix
    viz = np.zeros((nrow, ncol))
    
    # Set cell values: 0=safe, 1=hole, 2=start, 3=goal, 4=path
    for i in range(nrow):
        for j in range(ncol):
            if desc[i][j] == b'F':
                viz[i][j] = 0  # Safe (white)
            elif desc[i][j] == b'H':
                viz[i][j] = 1  # Hole (black)
            elif desc[i][j] == b'S':
                viz[i][j] = 2  # Start (green)
            elif desc[i][j] == b'G':
                viz[i][j] = 3  # Goal (red)
    
    # Mark the path
    for i, pos in enumerate(path[1:-1], 1):  # Don't overwrite start and goal
        if len(path) > 2 and desc[pos[0]][pos[1]] != b'S' and desc[pos[0]][pos[1]] != b'G':
            viz[pos] = 4  # Path (blue)
    
    # Create color map
    colors = ['white', 'black', 'green', 'red', 'blue']
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)
    
    plt.imshow(viz, cmap=cmap, vmin=0, vmax=4)
    
    # Add grid
    plt.grid(True, color='gray', linewidth=0.5)
    plt.xticks(range(ncol))
    plt.yticks(range(nrow))
    
    # Add labels
    for i in range(nrow):
        for j in range(ncol):
            if desc[i][j] == b'S':
                plt.text(j, i, 'S', ha='center', va='center', fontsize=14, fontweight='bold')
            elif desc[i][j] == b'G':
                plt.text(j, i, 'G', ha='center', va='center', fontsize=14, fontweight='bold')
            elif desc[i][j] == b'H':
                plt.text(j, i, 'H', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # Add path numbers
    for i, pos in enumerate(path):
        if i > 0 and i < len(path) - 1:  # Skip start and goal
            plt.text(pos[1], pos[0], str(i), ha='center', va='center', 
                    fontsize=8, color='white', fontweight='bold')
    
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def run_improved_genetic_algorithm():
    """Run the improved genetic algorithm"""
    print("\n🧬 Starting IMPROVED Genetic Algorithm...")
    print(f"Population size: {sol_per_pop}")
    print(f"Generations: {num_generations}")
    print(f"Chromosome length: {num_genes} moves")
    print(f"Mutation rate: {mutation_percent_genes}%")
    
    # Track progress
    best_fitness_per_generation = []
    
    def on_generation(ga_instance):
        """Callback function to track progress"""
        generation = ga_instance.generations_completed
        best_fitness = ga_instance.best_solution()[1]
        best_fitness_per_generation.append(best_fitness)
        
        if generation % 50 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness:.2f}")
    
    # Create GA instance with callback
    ga_instance = pygad.GA(
        gene_space=gene_space,
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=improved_fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        parent_selection_type=parent_selection_type,
        keep_parents=keep_parents,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        mutation_percent_genes=mutation_percent_genes,
        on_generation=on_generation,
        random_seed=None  # Remove fixed seed for variety
    )
    
    # Run the algorithm
    ga_instance.run()
    
    # Get the best solution
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    
    print(f"\n🎯 Best solution fitness: {solution_fitness:.2f}")
    
    # Analyze the best solution
    result = simulate_path(solution)
    
    print(f"\n📊 Path Analysis:")
    print(f"Reached goal: {'✅ YES' if result['reached_goal'] else '❌ NO'}")
    print(f"Fell in hole: {'💀 YES' if result['fell_in_hole'] else '✅ NO'}")
    print(f"Steps taken: {result['steps_taken']}")
    print(f"Wall hits: {result['wall_hits']}")
    print(f"Revisits: {result['revisits']}")
    print(f"Final position: {result['final_pos']}")
    print(f"Distance to goal: {abs(result['final_pos'][0] - goal_pos[0]) + abs(result['final_pos'][1] - goal_pos[1])}")
    
    # Show the path
    action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
    print(f"\n🗺️  Action sequence (first {min(result['steps_taken'], 20)} moves):")
    for i, action in enumerate(solution[:min(result['steps_taken'], 20)]):
        print(f"Step {i+1}: {action_names[action]}")
    
    return result, ga_instance, best_fitness_per_generation

def quick_test():
    """Quick test with fewer generations"""
    print("\n⚡ Quick Test (100 generations)...")
    
    global num_generations
    original_generations = num_generations
    num_generations = 100
    
    result, ga_instance, fitness_progress = run_improved_genetic_algorithm()
    
    # Restore original value
    num_generations = original_generations
    
    return result



import gym
import time

def play_frozen_lake_with_moves(moves, show_steps=True, delay=0.5):
    """Play FrozenLake with given moves and show the game"""
    # Create environment with rendering
    env = gym.make('FrozenLake8x8-v1', is_slippery=False, render_mode="human")
    
    print(f"\n🎮 Playing FrozenLake with {len(moves)} moves...")
    print("Actions: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP")
    
    # Reset environment
    observation, info = env.reset(seed=42)
    total_reward = 0
    
    action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
    
    print(f"Starting position: {observation}")
    time.sleep(delay)
    
    for step, action in enumerate(moves):
        if show_steps:
            print(f"\nStep {step + 1}: Taking action {action} ({action_names[action]})")
        
        # Take the action
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if show_steps:
            print(f"  Reward: {reward}, Total reward: {total_reward}")
            print(f"  New position: {observation}")
        
        time.sleep(delay)  # Pause to see the movement
        
        # Check if game ended
        if terminated or truncated:
            if reward > 0:
                print(f"\n🎉 SUCCESS! Reached the goal in {step + 1} steps!")
                print(f"Total reward: {total_reward}")
            else:
                print(f"\n💀 FAILED! Game ended at step {step + 1}")
                print(f"Total reward: {total_reward}")
            break
    else:
        # If we used all moves without terminating
        print(f"\nUsed all {len(moves)} moves. Final reward: {total_reward}")
    
    env.close()
    return total_reward > 0  # Return True if successful


def play_with_genetic_solution():
    """Play with the best solution found by genetic algorithm"""
    print("\n🧬 First, let's run the genetic algorithm to find a solution...")
    
    # Run a quick genetic algorithm
    result = quick_test()
    
    if result['reached_goal']:
        print(f"\n✅ Genetic algorithm found a solution!")
        print(f"Path length: {result['steps_taken']} steps")
        
        # Extract the successful moves
        successful_moves = []
        for i, pos in enumerate(result['path'][1:], 0):
            if i < len(result['path']) - 1:
                current_pos = result['path'][i]
                next_pos = result['path'][i + 1]
                
                # Calculate the action taken
                row_diff = next_pos[0] - current_pos[0]
                col_diff = next_pos[1] - current_pos[1]
                
                if row_diff == -1:  # UP
                    successful_moves.append(3)
                elif row_diff == 1:   # DOWN
                    successful_moves.append(1)
                elif col_diff == -1:  # LEFT
                    successful_moves.append(0)
                elif col_diff == 1:   # RIGHT
                    successful_moves.append(2)
        
        print(f"Extracted moves: {successful_moves}")
        
        # Now play the game with these moves
        print(f"\n🎮 Playing FrozenLake with genetic algorithm solution...")
        success = play_frozen_lake_with_moves(successful_moves, show_steps=True, delay=1.0)
        
        return success
    else:
        
        return False

if __name__ == "__main__":
    print("🧊 IMPROVED FrozenLake 8x8 Genetic Algorithm Solver")
    print("=" * 60)
    
    # First, try simple hardcoded strategies
    simple_strategy, simple_result = find_simple_path()
    
    if simple_result and simple_result['reached_goal']:
        print("✅ Found working simple strategy!")
        display_path(simple_result['path'], "Simple Strategy Solution")
    
    # Display the initial map
    display_path([start_pos], "FrozenLake 8x8 Map")
    
    # Choose running mode
    print("\nChoose mode:")
    print("1. Quick test (100 generations)")
    print("2. Full run (500 generations)")
    print("3. Multiple attempts (5 runs)")
    print("4. 🎮 Play with genetic algorithm solution")
    
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        result = quick_test()
        if result['reached_goal']:
            display_path(result['path'], f"GA Solution Found ({result['steps_taken']} steps)")
    elif choice == "2":
        result, ga_instance, fitness_progress = run_improved_genetic_algorithm()
        if result['reached_goal']:
            display_path(result['path'], f"GA Solution Found ({result['steps_taken']} steps)")
    elif choice == "3":
        successful_runs = 0
        best_result = None
        
        for run in range(5):
            print(f"\n--- RUN {run + 1}/5 ---")
            result = quick_test()
            
            if result['reached_goal']:
                successful_runs += 1
                if best_result is None or result['steps_taken'] < best_result['steps_taken']:
                    best_result = result
                print(f"✅ Run {run + 1}: SUCCESS in {result['steps_taken']} steps!")
            else:
                print(f"❌ Run {run + 1}: Failed")
        
        print(f"\n🏆 FINAL RESULTS: {successful_runs}/5 successful runs")
        if best_result:
            display_path(best_result['path'], f"Best GA Solution ({best_result['steps_taken']} steps)")
    elif choice == "4":
        play_with_genetic_solution()
    
    else:
        print("Invalid choice, running quick test...")
        result = quick_test()