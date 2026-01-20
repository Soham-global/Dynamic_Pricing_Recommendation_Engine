import numpy as np
class QLearningAgent:
    """
    Q-Learning Agent for optimal pricing
    """
    def __init__(self, n_states=30, n_actions=5):
        self.n_states = n_states
        self.n_actions = n_actions
        
        # Hyperparameters
        self.learning_rate = 0.1      # How fast to learn (alpha)
        self.discount_factor = 0.9    # Importance of future rewards (gamma)
        self.epsilon = 0.3            # Exploration rate (30% random)
        self.epsilon_decay = 0.995    # Decay epsilon over time
        self.epsilon_min = 0.05       # Minimum exploration
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((n_states, n_actions)) 
    
    def get_action(self, state, training=True):
        """
        Choose action using epsilon-greedy strategy
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, self.n_actions)
        else:
            # Exploit: best known action
            return np.argmax(self.q_table[state])
    
    def update_q_table(self, state, action, reward, next_state):
        """
        Update Q-value using Q-learning formula:
        Q(s,a) = Q(s,a) + α * [R + γ * max(Q(s',a')) - Q(s,a)]
        """
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state, action] = new_q
    
    def decay_epsilon(self):
        """
        Gradually reduce exploration over time
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)