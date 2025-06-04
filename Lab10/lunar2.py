import pygad
import numpy as np
import gym
import matplotlib.pyplot as plt
import time

# Add compatibility fix for numpy
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

# Genetic Algorithm Parameters
num_genes = 80  # Number of actions in sequence
sol_per_pop = 300  # Population size
num_parents_mating = 10
num_generations = 100
parent_selection_type = "rank"
keep_parents = 5
crossover_type = "uniform"
mutation_type = "random"
mutation_percent_genes = 25
gene_space = gene_space = {"low": 0, "high": 3}  # 0=nothing, 1=left engine, 2=main engine, 3=right engine

def simulate_lunar_landing(actions):
    """Simulate a lunar landing attempt with given actions"""
    env = gym.make("LunarLander-v2")
    
    observation, info = env.reset(seed=42)
    total_reward = 0
    steps = 0
    
    # Extract initial state
    x_pos, y_pos, x_vel, y_vel, angle, angular_vel, leg1, leg2 = observation
    
    results = {
        'total_reward': 0,
        'steps': 0,
        'final_position': (0, 0),
        'final_velocity': (0, 0),
        'final_angle': 0,
        'legs_contact': False,
        'crashed': False,
        'landed_successfully': False,
        'fuel_efficiency': 0,
        'trajectory': []
    }
    
    for action in actions:
        action=int(action)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # Extract state information
        x_pos, y_pos, x_vel, y_vel, angle, angular_vel, leg1, leg2 = observation
        
        # Store trajectory point
        results['trajectory'].append({
            'position': (x_pos, y_pos),
            'velocity': (x_vel, y_vel),
            'angle': angle,
            'legs_contact': (leg1, leg2)
        })
        
        # Check if episode ended
        if terminated or truncated:
            break
    
    # Calculate final metrics
    results['total_reward'] = total_reward
    results['steps'] = steps
    results['final_position'] = (x_pos, y_pos)
    results['final_velocity'] = (x_vel, y_vel)
    results['final_angle'] = angle
    results['legs_contact'] = leg1 and leg2
    results['crashed'] = total_reward < -100
    results['landed_successfully'] = total_reward > 200
    
    # Calculate fuel efficiency (penalize excessive engine use)
    fuel_used = sum(1 for a in actions[:steps] if a != 0)
    results['fuel_efficiency'] = max(0, steps - fuel_used)
    
    env.close()
    return results

def advanced_fitness_function(ga_instance, solution, solution_idx):
    """Advanced fitness function for lunar landing"""
    solution = [int(x) for x in solution]
    result = simulate_lunar_landing(solution)
    
    # Base fitness
    fitness = 100.0
    
    # Primary reward: game score
    fitness += result['total_reward']
    
    # Landing success bonus
    if result['landed_successfully']:
        fitness += 500
        print(f"🎉 SUCCESSFUL LANDING! Reward: {result['total_reward']:.1f}, Fitness: {fitness:.1f}")
    
    # Crash penalty
    if result['crashed']:
        fitness -= 300
    
    # Position-based rewards
    final_x, final_y = result['final_position']
    
    # Reward for being close to landing pad (x=0, y=0)
    distance_to_pad = abs(final_x) + abs(final_y)
    fitness += 50 / (1 + distance_to_pad)
    
    # Reward for low altitude (close to ground)
    if final_y > -0.5:  # Close to ground
        fitness += 100
    elif final_y > -1.0:  # Moderately close
        fitness += 50
    
    # Velocity control rewards
    final_x_vel, final_y_vel = result['final_velocity']
    
    # Reward for low velocities (controlled landing)
    velocity_magnitude = abs(final_x_vel) + abs(final_y_vel)
    fitness += 50 / (1 + velocity_magnitude * 10)
    
    # Angle control rewards
    angle_penalty = abs(result['final_angle']) * 100
    fitness -= angle_penalty
    
    # Legs contact bonus
    if result['legs_contact']:
        fitness += 200
    
    # Fuel efficiency bonus
    fitness += result['fuel_efficiency'] * 2
    
    # Trajectory analysis bonus
    if len(result['trajectory']) > 10:
        # Reward for stable descent (consistent downward movement)
        y_positions = [point['position'][1] for point in result['trajectory']]
        if len(y_positions) > 1:
            descent_consistency = sum(1 for i in range(1, len(y_positions)) 
                                    if y_positions[i] <= y_positions[i-1])
            fitness += descent_consistency * 2
        
        # Reward for angle stability
        angles = [abs(point['angle']) for point in result['trajectory']]
        avg_angle = np.mean(angles)
        fitness += 20 / (1 + avg_angle)
    
    return max(fitness, 0)  # Ensure non-negative fitness

def smart_fitness_function(ga_instance, solution, solution_idx):
    """Simpler but effective fitness function"""
    solution = [int(x) for x in solution]
    result = simulate_lunar_landing(solution)
    
    fitness = 200.0  # Base fitness
    
    # Game reward (most important)
    fitness += result['total_reward'] * 2
    
    # Success bonuses
    if result['landed_successfully']:
        fitness += 1000
        print(f"🚀 PERFECT LANDING! Score: {result['total_reward']:.1f}")
    elif result['legs_contact']:
        fitness += 300
        print(f"👍 Legs touched ground! Score: {result['total_reward']:.1f}")
    
    # Position bonus (close to center)
    final_x, final_y = result['final_position']
    if abs(final_x) < 0.1:  # Very close to center
        fitness += 100
    elif abs(final_x) < 0.3:  # Reasonably close
        fitness += 50
    
    # Altitude bonus (close to ground)
    if final_y > -0.2:
        fitness += 150
    elif final_y > -0.5:
        fitness += 75
    
    # Velocity control
    final_x_vel, final_y_vel = result['final_velocity']
    if abs(final_x_vel) < 0.2 and abs(final_y_vel) < 0.2:
        fitness += 200  # Very controlled
    elif abs(final_x_vel) < 0.5 and abs(final_y_vel) < 0.5:
        fitness += 100  # Somewhat controlled
    
    # Angle control
    if abs(result['final_angle']) < 0.1:
        fitness += 100  # Very upright
    elif abs(result['final_angle']) < 0.3:
        fitness += 50   # Reasonably upright
    
    return fitness

def faster_fitness_function(ga_instance, solution, solution_idx):
    """Optimized fitness function for faster learning"""
    solution = [int(x) for x in solution]
    result = simulate_lunar_landing(solution)
    
    fitness = 0.0
    
    # Main reward with higher weight
    fitness += result['total_reward'] * 5  # Increased weight
    
    # Massive success bonus
    if result['landed_successfully']:
        fitness += 3000  # Much higher bonus
        print(f"🚀 SUCCESS! Score: {result['total_reward']:.1f}, Fitness: {fitness:.1f}")
        return fitness
    
    # Strong guidance rewards
    final_x, final_y = result['final_position']
    final_x_vel, final_y_vel = result['final_velocity']
    
    # Distance to landing pad (stronger guidance)
    distance = abs(final_x) + abs(final_y) 
    fitness += 800 / (1 + distance * 3)  # Stronger guidance
    
    # Velocity control (critical for landing)
    velocity = abs(final_x_vel) + abs(final_y_vel)
    fitness += 500 / (1 + velocity * 15)
    
    # Angle control (very important)
    fitness += 400 / (1 + abs(result['final_angle']) * 8)
    
    # Legs contact bonus (huge bonus)
    if result['legs_contact']:
        fitness += 1200  # Much higher
    
    # Altitude bonus (get close to ground)
    if final_y > -0.1:  # Very close
        fitness += 600
    elif final_y > -0.3:  # Close
        fitness += 300
    elif final_y > -0.5:  # Moderately close
        fitness += 150
    
    # Crash penalty
    if result['crashed']:
        fitness -= 200
    
    return max(fitness, 1)

def visualize_landing(actions, title="Lunar Landing Simulation"):
    """Visualize a lunar landing attempt"""
    env = gym.make("LunarLander-v2", render_mode="human")
    
    print(f"\n🚀 {title}")
    print("Actions: 0=NOTHING, 1=LEFT, 2=MAIN, 3=RIGHT")
    
    observation, info = env.reset(seed=42)
    total_reward = 0
    
    action_names = {0: "NOTHING", 1: "LEFT", 2: "MAIN", 3: "RIGHT"}
    
    for step, action in enumerate(actions):
        print(f"Step {step + 1}: {action_names[action]}")
        
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        time.sleep(0.1)  # Slow down to see the action
        
        if terminated or truncated:
            x_pos, y_pos, x_vel, y_vel, angle, angular_vel, leg1, leg2 = observation
            
            print(f"\nLanding complete after {step + 1} steps!")
            print(f"Final score: {total_reward:.1f}")
            print(f"Position: ({x_pos:.3f}, {y_pos:.3f})")
            print(f"Velocity: ({x_vel:.3f}, {y_vel:.3f})")
            print(f"Angle: {angle:.3f}")
            print(f"Legs contact: {leg1 and leg2}")
            
            if total_reward > 200:
                print("🎉 SUCCESSFUL LANDING!")
            elif total_reward > 0:
                print("👍 Decent attempt!")
            else:
                print("💥 CRASHED!")
            
            break
    
    env.close()
    return total_reward

def run_genetic_algorithm(fitness_func=smart_fitness_function):
    """Run the genetic algorithm to find optimal landing sequence"""
    print("\n🧬 Starting Genetic Algorithm for Lunar Landing...")
    print(f"Population size: {sol_per_pop}")
    print(f"Generations: {num_generations}")
    print(f"Action sequence length: {num_genes}")
    
    # Track progress
    best_fitness_history = []
    successful_solutions = []
    
    def on_generation(ga_instance):
        generation = ga_instance.generations_completed
        best_fitness = ga_instance.best_solution()[1]
        best_fitness_history.append(best_fitness)
        
        if generation % 20 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness:.2f}")
            
            # Check if we found a good solution
            if best_fitness > 1500:  # Likely a successful landing
                solution = ga_instance.best_solution()[0]
                result = simulate_lunar_landing(solution)
                if result['landed_successfully']:
                    successful_solutions.append({
                        'generation': generation,
                        'solution': solution.copy(),
                        'fitness': best_fitness,
                        'reward': result['total_reward']
                    })
    
    # Create and run GA
    ga_instance = pygad.GA(
        gene_space=gene_space,
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        parent_selection_type=parent_selection_type,
        keep_parents=keep_parents,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        mutation_percent_genes=mutation_percent_genes,
        on_generation=on_generation,
        random_seed=None
    )
    
    ga_instance.run()
    
    # Get best solution
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    
    print(f"\n🎯 Evolution Complete!")
    print(f"Best fitness: {solution_fitness:.2f}")
    print(f"Successful solutions found: {len(successful_solutions)}")
    
    # Analyze best solution
    result = simulate_lunar_landing(solution)
    
    print(f"\n📊 Best Solution Analysis:")
    print(f"Total reward: {result['total_reward']:.2f}")
    print(f"Successfully landed: {'✅ YES' if result['landed_successfully'] else '❌ NO'}")
    print(f"Legs contact: {'✅ YES' if result['legs_contact'] else '❌ NO'}")
    print(f"Final position: ({result['final_position'][0]:.3f}, {result['final_position'][1]:.3f})")
    print(f"Final velocity: ({result['final_velocity'][0]:.3f}, {result['final_velocity'][1]:.3f})")
    print(f"Final angle: {result['final_angle']:.3f}")
    print(f"Steps taken: {result['steps']}")
    
    return solution, result, successful_solutions, best_fitness_history

def test_manual_strategies():
    """Test some manual landing strategies"""
    print("\n🧪 Testing Manual Strategies...")
    
    strategies = [
        # Strategy 1: Mostly main engine with some corrections
        [2] * 20 + [1, 2, 3, 2] * 5 + [2] * 10,
        
        # Strategy 2: Conservative approach
        [0] * 10 + [2] * 15 + [1, 2, 3, 2] * 6 + [2] * 5,
        
        # Strategy 3: Active control
        [2, 1, 2, 3, 2] * 10,
        
        # Strategy 4: Late intervention
        [0] * 25 + [2] * 20 + [1, 3] * 2 + [2],
    ]
    
    for i, strategy in enumerate(strategies, 1):
        print(f"\n--- Testing Strategy {i} ---")
        reward = visualize_landing(strategy, f"Manual Strategy {i}")
        print(f"Strategy {i} score: {reward:.1f}")
        time.sleep(2)

def run_multiple_attempts(num_attempts=5):
    """Run multiple GA attempts to find the best solution"""
    print(f"\n🔄 Running {num_attempts} GA attempts...")
    
    best_overall_solution = None
    best_overall_reward = -float('inf')
    successful_attempts = 0
    
    for attempt in range(num_attempts):
        print(f"\n{'='*50}")
        print(f"ATTEMPT {attempt + 1}/{num_attempts}")
        print(f"{'='*50}")
        
        solution, result, successful_solutions, fitness_history = run_genetic_algorithm()
        
        if result['landed_successfully']:
            successful_attempts += 1
            if result['total_reward'] > best_overall_reward:
                best_overall_reward = result['total_reward']
                best_overall_solution = solution
                print(f"🌟 NEW BEST SOLUTION! Reward: {result['total_reward']:.1f}")
    
    print(f"\n🏆 FINAL RESULTS:")
    print(f"Successful attempts: {successful_attempts}/{num_attempts}")
    
    if best_overall_solution is not None:
        print(f"Best solution reward: {best_overall_reward:.1f}")
        print(f"\n🚀 Demonstrating best solution:")
        visualize_landing(best_overall_solution, "Best GA Solution")
        return best_overall_solution
    else:
        print("❌ No successful solutions found.")
        return None

if __name__ == "__main__":
    print("🚀 Lunar Lander Genetic Algorithm Solver")
    print("=" * 50)
    
    print("\nChoose mode:")
    print("1. Single GA run (smart fitness)")
    print("2. Single GA run (advanced fitness)")
    print("3. Multiple GA attempts (5 runs)")
    print("4. Test manual strategies")
    print("5. Quick demo")
    
    choice = input("Enter choice (1-5): ").strip()
    
    if choice == "1":
        solution, result, successful_solutions, fitness_history = run_genetic_algorithm(smart_fitness_function)
        if result['landed_successfully']:
            print(f"\n🎮 Demonstrating solution:")
            visualize_landing(solution, "GA Solution (Smart Fitness)")
    
    elif choice == "2":
        solution, result, successful_solutions, fitness_history = run_genetic_algorithm(advanced_fitness_function)
        if result['landed_successfully']:
            print(f"\n🎮 Demonstrating solution:")
            visualize_landing(solution, "GA Solution (Advanced Fitness)")
    
    elif choice == "3":
        run_multiple_attempts(5)
    
    elif choice == "4":
        test_manual_strategies()

    elif choice == "5":
        print("\n⚡ Quick demo with optimized settings...")
        
        
        # Temporary faster settings
        num_generations = 100
        sol_per_pop = 200
        mutation_percent_genes = 35
        
        solution, result, successful_solutions, fitness_history = run_genetic_algorithm(faster_fitness_function)
        if result['landed_successfully']:
            print(f"\n🎮 Demonstrating solution:")
            visualize_landing(solution, "Quick GA Demo")
        else:
            print("No success in quick demo. Try option 3 for multiple attempts.")

    
    
    
    else:
        print("Invalid choice, running single GA attempt...")
        solution, result, successful_solutions, fitness_history = run_genetic_algorithm()
        if result['landed_successfully']:
            visualize_landing(solution, "Default GA Solution")