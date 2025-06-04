import numpy as np
import gym
import pyswarms as ps
import matplotlib.pyplot as plt
import time

# Add compatibility fix for numpy
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

class MountainCarPSO:
    def __init__(self, max_steps=200, env_type='continuous'):
        """
        Initialize MountainCar PSO solver
        
        Args:
            max_steps: Maximum steps per episode
            env_type: 'continuous' or 'discrete'
        """
        self.max_steps = max_steps
        self.env_type = env_type
        
        if env_type == 'continuous':
            self.env_name = 'MountainCarContinuous-v0'
            self.action_dim = 1  # Continuous action between -1 and 1
        else:
            self.env_name = 'MountainCar-v0'
            self.action_dim = 1  # Discrete actions: 0, 1, 2
            
        print(f"Using {env_type} MountainCar environment")
        
    def create_policy_network(self, weights, input_size=2, hidden_size=16):
        """
        Create a simple neural network policy from PSO weights
        
        Args:
            weights: Flattened weights from PSO particle
            input_size: Input size (position, velocity)
            hidden_size: Hidden layer size
            
        Returns:
            Dictionary containing network weights
        """
        # Calculate weight matrix sizes
        w1_size = input_size * hidden_size
        b1_size = hidden_size
        w2_size = hidden_size * self.action_dim
        b2_size = self.action_dim
        
        # Split weights
        idx = 0
        w1 = weights[idx:idx + w1_size].reshape(input_size, hidden_size)
        idx += w1_size
        
        b1 = weights[idx:idx + b1_size]
        idx += b1_size
        
        w2 = weights[idx:idx + w2_size].reshape(hidden_size, self.action_dim)
        idx += w2_size
        
        b2 = weights[idx:idx + b2_size]
        
        return {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
    
    def neural_network_policy(self, state, network_weights):
        """
        Forward pass through neural network policy
        
        Args:
            state: Current state [position, velocity]
            network_weights: Network weights dictionary
            
        Returns:
            Action for the environment
        """
        # Forward pass
        hidden = np.tanh(np.dot(state, network_weights['w1']) + network_weights['b1'])
        output = np.dot(hidden, network_weights['w2']) + network_weights['b2']
        
        if self.env_type == 'continuous':
            # Continuous action: use tanh to bound between -1 and 1
            action = np.tanh(output[0])
            return np.array([action])
        else:
            # Discrete action: use softmax for action probabilities
            exp_output = np.exp(output - np.max(output))  # Numerical stability
            probs = exp_output / np.sum(exp_output)
            action = np.random.choice(len(probs), p=probs)
            return action
    
    def simulate_episode(self, weights):
        """
        Simulate one episode with given policy weights
        
        Args:
            weights: PSO particle weights
            
        Returns:
            Dictionary with episode results
        """
        env = gym.make(self.env_name)
        
        # Create policy network
        network = self.create_policy_network(weights)
        
        # Reset environment
        observation, info = env.reset(seed=42)
        total_reward = 0
        steps = 0
        
        positions = []
        velocities = []
        actions_taken = []
        
        for step in range(self.max_steps):
            position, velocity = observation
            positions.append(position)
            velocities.append(velocity)
            
            # Get action from policy
            action = self.neural_network_policy(observation, network)
            actions_taken.append(action)
            
            # Take step
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Check if reached goal
            if terminated or truncated:
                break
        
        env.close()
        
        final_position, final_velocity = observation
        
        return {
            'total_reward': total_reward,
            'steps': steps,
            'final_position': final_position,
            'final_velocity': final_velocity,
            'reached_goal': final_position >= 0.45,  # Goal position
            'positions': positions,
            'velocities': velocities,
            'actions': actions_taken
        }
    
    def fitness_function(self, particles):
        """
        Fitness function for PSO optimization
        
        Args:
            particles: Array of particle weights (n_particles, n_dimensions)
            
        Returns:
            Array of fitness values (higher is better)
        """
        n_particles = particles.shape[0]
        fitness_values = np.zeros(n_particles)
        
        for i, particle in enumerate(particles):
            result = self.simulate_episode(particle)
            
            # Base fitness
            fitness = 0.0
            
            # Primary reward: total episode reward
            fitness += result['total_reward']
            
            # Major bonus for reaching goal
            if result['reached_goal']:
                fitness += 1000
                # Bonus for reaching goal quickly
                fitness += (self.max_steps - result['steps']) * 5
                print(f"🎉 Particle {i}: REACHED GOAL in {result['steps']} steps! Fitness: {fitness:.1f}")
            
            # Progress reward: how far right the car got
            progress = (result['final_position'] + 1.2) / 1.7  # Normalize to 0-1
            fitness += progress * 200
            
            # Velocity reward: higher rightward velocity is good
            if result['final_velocity'] > 0:
                fitness += result['final_velocity'] * 100
            
            # Exploration bonus: reward for reaching high positions
            max_position = max(result['positions']) if result['positions'] else -1.2
            height_bonus = max(0, (max_position + 1.2) / 1.7) * 150
            fitness += height_bonus
            
            # Energy efficiency: fewer steps is better (if goal reached)
            if result['reached_goal']:
                efficiency_bonus = max(0, (self.max_steps - result['steps']) * 2)
                fitness += efficiency_bonus
            
            fitness_values[i] = fitness
        
        return fitness_values
    
    def optimize(self, n_particles=50, n_iterations=100, hidden_size=16):
        """
        Run PSO optimization to find best policy
        
        Args:
            n_particles: Number of particles in swarm
            n_iterations: Number of PSO iterations
            hidden_size: Size of neural network hidden layer
            
        Returns:
            Best weights and optimization results
        """
        # Calculate total number of weights needed
        input_size = 2  # position, velocity
        total_weights = (input_size * hidden_size +  # w1
                        hidden_size +                # b1
                        hidden_size * self.action_dim +  # w2
                        self.action_dim)             # b2
        
        print(f"\n🧠 Neural Network Architecture:")
        print(f"Input size: {input_size}")
        print(f"Hidden size: {hidden_size}")
        print(f"Output size: {self.action_dim}")
        print(f"Total weights: {total_weights}")
        
        # PSO parameters
        options = {
            'c1': 2.0,  # Cognitive parameter
            'c2': 2.0,  # Social parameter
            'w': 0.9    # Inertia weight
        }
        
        # Weight bounds
        bounds = (np.full(total_weights, -2.0), np.full(total_weights, 2.0))
        
        print(f"\n🔄 Starting PSO optimization...")
        print(f"Particles: {n_particles}")
        print(f"Iterations: {n_iterations}")
        print(f"Weight bounds: [-2.0, 2.0]")
        
        # Initialize PSO
        optimizer = ps.single.GlobalBestPSO(
            n_particles=n_particles,
            dimensions=total_weights,
            options=options,
            bounds=bounds
        )
        
        # Track best fitness over iterations
        best_fitness_history = []
        successful_solutions = []
        
        def progress_callback(optimizer):
            """Callback to track progress"""
            iteration = len(best_fitness_history)
            best_fitness = optimizer.swarm.best_cost
            best_fitness_history.append(-best_fitness)  # Convert to positive
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Best fitness = {-best_fitness:.2f}")
            
            # Check for successful solutions
            if -best_fitness > 1000:  # Likely reached goal
                result = self.simulate_episode(optimizer.swarm.best_pos)
                if result['reached_goal']:
                    successful_solutions.append({
                        'iteration': iteration,
                        'weights': optimizer.swarm.best_pos.copy(),
                        'fitness': -best_fitness,
                        'steps': result['steps']
                    })
        
        # Run optimization (minimize negative fitness)
        best_cost, best_pos = optimizer.optimize(
            self.fitness_function,
            iters=n_iterations,
            n_processes=1,  # Single process for gym compatibility
            verbose=False
        )
        
        print(f"\n🎯 Optimization completed!")
        print(f"Best fitness: {-best_cost:.2f}")
        print(f"Successful solutions found: {len(successful_solutions)}")
        
        # Test best solution
        best_result = self.simulate_episode(best_pos)
        
        print(f"\n📊 Best Solution Analysis:")
        print(f"Reached goal: {'✅ YES' if best_result['reached_goal'] else '❌ NO'}")
        print(f"Final position: {best_result['final_position']:.4f}")
        print(f"Final velocity: {best_result['final_velocity']:.4f}")
        print(f"Steps taken: {best_result['steps']}")
        print(f"Total reward: {best_result['total_reward']:.2f}")
        
        return {
            'best_weights': best_pos,
            'best_fitness': -best_cost,
            'best_result': best_result,
            'fitness_history': best_fitness_history,
            'successful_solutions': successful_solutions
        }
    
    def visualize_solution(self, weights, title="PSO MountainCar Solution"):
        """
        Visualize the solution by rendering the environment
        
        Args:
            weights: Policy weights to test
            title: Title for the visualization
        """
        env = gym.make(self.env_name, render_mode="human")
        
        print(f"\n🎮 {title}")
        print("Watch the car attempt to reach the goal!")
        
        network = self.create_policy_network(weights)
        observation, info = env.reset(seed=42)
        total_reward = 0
        
        action_names = {0: "LEFT", 1: "NOTHING", 2: "RIGHT"} if self.env_type == 'discrete' else None
        
        for step in range(self.max_steps):
            position, velocity = observation
            
            # Get action from policy
            action = self.neural_network_policy(observation, network)
            
            if self.env_type == 'discrete':
                print(f"Step {step + 1}: Pos={position:.3f}, Vel={velocity:.3f}, Action={action_names[action]}")
            else:
                print(f"Step {step + 1}: Pos={position:.3f}, Vel={velocity:.3f}, Action={action[0]:.3f}")
            
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            time.sleep(0.05)  # Slow down to see the movement
            
            if terminated or truncated:
                position, velocity = observation
                if position >= 0.45:
                    print(f"\n🎉 SUCCESS! Reached goal in {step + 1} steps!")
                    print(f"Final position: {position:.4f}")
                    print(f"Final velocity: {velocity:.4f}")
                    print(f"Total reward: {total_reward:.2f}")
                else:
                    print(f"\n⏰ Time limit reached after {step + 1} steps")
                    print(f"Final position: {position:.4f}")
                    print(f"Best position reached: {position:.4f}")
                break
        
        env.close()
        return total_reward

def plot_learning_progress(fitness_history, title="PSO Learning Progress"):
    """Plot the learning progress over iterations"""
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(fitness_history)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    # Plot moving average
    window_size = min(10, len(fitness_history) // 4)
    if len(fitness_history) >= window_size:
        moving_avg = np.convolve(fitness_history, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(fitness_history)), moving_avg, 'r-', label=f'Moving Average ({window_size})')
        plt.plot(fitness_history, 'b-', alpha=0.3, label='Raw Fitness')
        plt.legend()
    else:
        plt.plot(fitness_history, 'b-')
    
    plt.title('Fitness with Moving Average')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def run_multiple_experiments(env_type='continuous', num_experiments=3):
    """Run multiple PSO experiments and find the best solution"""
    print(f"\n🔬 Running {num_experiments} PSO experiments...")
    
    best_overall_result = None
    best_overall_fitness = -float('inf')
    all_results = []
    
    for experiment in range(num_experiments):
        print(f"\n{'='*50}")
        print(f"EXPERIMENT {experiment + 1}/{num_experiments}")
        print(f"{'='*50}")
        
        solver = MountainCarPSO(max_steps=200, env_type=env_type)
        result = solver.optimize(n_particles=50, n_iterations=100, hidden_size=16)
        
        all_results.append(result)
        
        if result['best_result']['reached_goal']:
            print(f"✅ Experiment {experiment + 1}: SUCCESS!")
            if result['best_fitness'] > best_overall_fitness:
                best_overall_fitness = result['best_fitness']
                best_overall_result = result
                print(f"🌟 NEW BEST SOLUTION!")
        else:
            print(f"❌ Experiment {experiment + 1}: No success")
    
    # Summary
    successful_experiments = sum(1 for r in all_results if r['best_result']['reached_goal'])
    print(f"\n🏆 FINAL RESULTS:")
    print(f"Successful experiments: {successful_experiments}/{num_experiments}")
    
    if best_overall_result:
        print(f"Best solution fitness: {best_overall_fitness:.2f}")
        print(f"Best solution steps: {best_overall_result['best_result']['steps']}")
        
        # Visualize best solution
        solver = MountainCarPSO(max_steps=200, env_type=env_type)
        solver.visualize_solution(best_overall_result['best_weights'], "Best PSO Solution")
        
        # Plot learning curve
        plot_learning_progress(best_overall_result['fitness_history'], "Best Experiment Learning Progress")
        
        return best_overall_result
    else:
        print("❌ No successful solutions found in any experiment.")
        return None

if __name__ == "__main__":
    print("🏔️ MountainCar PSO Solver")
    print("=" * 40)
    
    print("\nChoose environment:")
    print("1. Continuous MountainCar (recommended)")
    print("2. Discrete MountainCar")
    print("3. Multiple experiments (continuous)")
    print("4. Multiple experiments (discrete)")
    print("5. Quick test (continuous, small)")
    
    choice = input("Enter choice (1-5): ").strip()
    
    if choice == "1":
        solver = MountainCarPSO(max_steps=200, env_type='continuous')
        result = solver.optimize(n_particles=50, n_iterations=100, hidden_size=16)
        if result['best_result']['reached_goal']:
            solver.visualize_solution(result['best_weights'])
            plot_learning_progress(result['fitness_history'])
    
    elif choice == "2":
        solver = MountainCarPSO(max_steps=200, env_type='discrete')
        result = solver.optimize(n_particles=50, n_iterations=100, hidden_size=16)
        if result['best_result']['reached_goal']:
            solver.visualize_solution(result['best_weights'])
            plot_learning_progress(result['fitness_history'])
    
    elif choice == "3":
        run_multiple_experiments('continuous', 3)
    
    elif choice == "4":
        run_multiple_experiments('discrete', 3)
    
    elif choice == "5":
        print("\n⚡ Quick test with reduced parameters...")
        solver = MountainCarPSO(max_steps=200, env_type='continuous')
        result = solver.optimize(n_particles=30, n_iterations=50, hidden_size=12)
        if result['best_result']['reached_goal']:
            solver.visualize_solution(result['best_weights'], "Quick PSO Test")
    
    else:
        print("Invalid choice, running default continuous experiment...")
        solver = MountainCarPSO(max_steps=200, env_type='continuous')
        result = solver.optimize(n_particles=50, n_iterations=100, hidden_size=16)
        if result['best_result']['reached_goal']:
            solver.visualize_solution(result['best_weights'])