import gym
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# Add compatibility fix for numpy
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

TRAINING = False

LEARNING_RATE = [0.01, 0.001, 0.0001]
DISCOUNT_FACTOR = [0.9, 0.99, 0.999]
EPSILON_DECAY = [0.99910, 0.99941, 0.99954, 0.99973, 0.99987]

LEARNING_EPISODES = 5000
TESTING_EPISODES = 100
REPLAY_BUFFER_SIZE = 250000
REPLAY_BUFFER_BATCH_SIZE = 32
MINIMUM_REWARD = -250
STATE_SIZE = 8
NUMBER_OF_ACTIONS = 4
WEIGHTS_FILENAME = './weights/weights.h5'

class Agent:
    def __init__(self, training, learning_rate, discount_factor, epsilon_decay):
        self.training = training
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_decay = epsilon_decay
        self.epsilon = 1.0 if self.training else 0.0
        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

        self._create_networks()
        
        if not training:
            self._load_weights()

    def _create_networks(self):
        # Main Q-network
        self.q_network = keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(STATE_SIZE,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(NUMBER_OF_ACTIONS, activation=None)
        ])
        
        # Target Q-network
        self.q_target_network = keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(STATE_SIZE,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(NUMBER_OF_ACTIONS, activation=None)
        ])
        
        self.q_network.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        # Initialize target network with same weights
        self.update_target_network()

    def choose_action(self, s):
        if not self.training or np.random.rand() > self.epsilon:
            s_reshaped = np.reshape(s, [1, STATE_SIZE])
            q_values = self.q_network.predict(s_reshaped, verbose=0)
            return np.argmax(q_values[0])
        
        return np.random.choice(NUMBER_OF_ACTIONS)

    def store(self, s, a, r, s_, is_terminal):
        if self.training:
            self.replay_buffer.append((s, a, r, s_, is_terminal))

    def optimize(self, s, a, r, s_, is_terminal):
        if self.training and len(self.replay_buffer) > REPLAY_BUFFER_BATCH_SIZE:
            batch = random.sample(list(self.replay_buffer), REPLAY_BUFFER_BATCH_SIZE)
            
            states = np.array([e[0] for e in batch])
            actions = np.array([e[1] for e in batch])
            rewards = np.array([e[2] for e in batch])
            next_states = np.array([e[3] for e in batch])
            terminals = np.array([e[4] for e in batch])
            
            # Current Q values
            current_q_values = self.q_network.predict(states, verbose=0)
            
            # Next Q values from target network
            next_q_values = self.q_target_network.predict(next_states, verbose=0)
            
            # Calculate target Q values
            target_q_values = current_q_values.copy()
            
            for i in range(REPLAY_BUFFER_BATCH_SIZE):
                if terminals[i]:
                    target_q_values[i][actions[i]] = rewards[i]
                else:
                    target_q_values[i][actions[i]] = rewards[i] + self.discount_factor * np.max(next_q_values[i])
            
            # Train the network
            self.q_network.fit(states, target_q_values, epochs=1, verbose=0)

    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.q_target_network.set_weights(self.q_network.get_weights())

    def update(self):
        if self.training:
            # Update target network every episode
            self.update_target_network()
        
        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay

    def close(self):
        if self.training:
            print("Saving agent weights to disk...")
            # Create weights directory if it doesn't exist
            os.makedirs(os.path.dirname(WEIGHTS_FILENAME), exist_ok=True)
            self.q_network.save_weights(WEIGHTS_FILENAME)

    def _load_weights(self):
        print("Loading agent weights from disk...")
        try:
            if os.path.exists(WEIGHTS_FILENAME):
                self.q_network.load_weights(WEIGHTS_FILENAME)
                self.update_target_network()
                print("Weights loaded successfully!")
            else:
                print(f"Weights file {WEIGHTS_FILENAME} not found. Using random weights.")
        except Exception as e:
            print("Error loading agent weights from disk:", e)

if __name__ == "__main__":
    np.set_printoptions(precision=2)

    # Create environment
    env = gym.make("LunarLander-v2", render_mode="human" if not TRAINING else None)
    average_reward = deque(maxlen=100)

    agent = Agent(TRAINING, LEARNING_RATE[2], DISCOUNT_FACTOR[1], EPSILON_DECAY[1])

    print("Alpha: %.4f Gamma: %.3f Epsilon %.5f" % (agent.learning_rate, agent.discount_factor, agent.epsilon_decay))
    
    episodes = LEARNING_EPISODES if TRAINING else TESTING_EPISODES
    
    for episode in range(episodes):
        current_reward = 0

        # Handle new gym API
        observation = env.reset()
        if isinstance(observation, tuple):
            s = observation[0]  # Extract state from (state, info) tuple
        else:
            s = observation

        for t in range(1000):
            a = agent.choose_action(s)
            
            # Handle new gym API
            step_result = env.step(a)
            if len(step_result) == 5:  # New API: (obs, reward, terminated, truncated, info)
                s_, r, terminated, truncated, info = step_result
                is_terminal = terminated or truncated
            else:  # Old API: (obs, reward, done, info)
                s_, r, is_terminal, info = step_result

            current_reward += r

            agent.store(s, a, r, s_, is_terminal)
            agent.optimize(s, a, r, s_, is_terminal)

            s = s_

            if is_terminal or current_reward < MINIMUM_REWARD:
                break

        agent.update()
        average_reward.append(current_reward)

        print("%i, %.2f, %.2f, %.2f" % (episode, current_reward, np.average(average_reward), agent.epsilon))

        # Save weights periodically during training
        if TRAINING and episode % 500 == 0 and episode > 0:
            agent.close()

    env.close()
    agent.close()
    
    print(f"\nTraining completed! Final average reward: {np.average(average_reward):.2f}")