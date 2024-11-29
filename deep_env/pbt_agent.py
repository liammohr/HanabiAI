import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class HanabiAgent:
    def __init__(self, env, learning_rate=1e-3, entropy_reg=0.01):
        self.env = env
        self.learning_rate = learning_rate
        self.entropy_reg = entropy_reg
        self.action_space = 20  # Total number of discrete actions
        self.policy_network, self.value_network = self._build_model()
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate, rho=0.99, momentum=0)

    def _build_model(self):
        """
        Builds the policy and value networks.

        Returns:
            Tuple of policy network and value network.
        """
        # Shared MLP feature extractor
        inputs = layers.Input(shape=(322,))
        x = layers.Dense(256, activation="relu")(inputs)

        # Policy Network
        lstm_input = layers.Reshape((1, 256))(x)  # Reshape for LSTM input
        lstm_output = layers.LSTM(256, return_sequences=True)(lstm_input)
        lstm_output = layers.LSTM(256)(lstm_output)
        policy_logits = layers.Dense(self.action_space)(lstm_output)

        # Value Network
        value_hidden = layers.Dense(256, activation="relu")(x)
        value_output = layers.Dense(1)(value_hidden)

        # Compile models
        policy_model = Model(inputs, policy_logits, name="PolicyNetwork")
        value_model = Model(inputs, value_output, name="ValueNetwork")

        return policy_model, value_model


    def select_action(self, observation: np.ndarray) -> int:
        """
        Select an action based on the policy, with masking of illegal actions.

        Args:
            observation (np.ndarray): Current state observation.

        Returns:
            int: Selected action index.
        """
        # Get legal actions mask
        legal_actions = self.env.get_legal_actions()  # Binary mask (1 for legal, 0 for illegal)
        if legal_actions is None or len(legal_actions) != self.action_space:
            # Fallback: Assume all actions are legal if mask is invalid
            legal_actions = np.ones(self.action_space, dtype=int)

        # Get policy logits
        logits = self.policy_network.predict(observation[np.newaxis, :],verbose=0)[0]

        # Replace invalid logits with fallback
        if np.any(np.isnan(logits)):
            logits = np.zeros_like(logits)

        # Mask illegal actions
        masked_logits = logits * legal_actions + (1 - legal_actions) * -1e9

        # Numerical stability in softmax
        max_logit = np.max(masked_logits) if np.isfinite(np.max(masked_logits)) else 0
        exp_logits = np.exp(masked_logits - max_logit)
        probabilities = exp_logits / np.sum(exp_logits)

        # Ensure no NaN probabilities
        if np.any(np.isnan(probabilities)) or not np.isfinite(probabilities).all():
            # Fallback: Uniform distribution over legal actions
            probabilities = legal_actions / np.sum(legal_actions)

        # Sample an action
        return np.random.choice(len(probabilities), p=probabilities)


    def train_step(self, observations, actions, rewards, gamma=0.999):
        """
        Performs a single training step using policy gradients with a value baseline.

        Args:
            observations: List of observations from the episode.
            actions: List of actions taken in the episode.
            rewards: List of rewards obtained in the episode.
            gamma: Discount factor.
        """
        returns = self._compute_returns(rewards, gamma)

        observations = np.array(observations)
        actions = np.array(actions)
        returns = np.array(returns)

        with tf.GradientTape() as tape:
            logits = self.policy_network(observations)
            values = self.value_network(observations)

            # Compute policy loss
            action_masks = tf.one_hot(actions, self.action_space)
            log_probs = tf.reduce_sum(action_masks * tf.math.log(tf.nn.softmax(logits) + 1e-10), axis=1)
            advantages = returns - tf.squeeze(values, axis=-1)
            entropy_loss = -tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(logits) * tf.math.log(tf.nn.softmax(logits) + 1e-10), axis=1))
            policy_loss = -tf.reduce_mean(log_probs * advantages) - self.entropy_reg * entropy_loss

            # Compute value loss
            value_loss = tf.reduce_mean(tf.square(returns - tf.squeeze(values, axis=-1)))

            # Total loss
            loss = policy_loss + 0.25 * value_loss

        gradients = tape.gradient(loss, self.policy_network.trainable_variables + self.value_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables + self.value_network.trainable_variables))

    def evaluate(self):
        """
        Evaluate the agent on 10 episodes to compute the average reward.

        Returns:
            float: Average reward over 10 episodes.
        """
        rewards = []
        for _ in range(10):
            obs = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.select_action(obs)
                obs, reward, done, _ = self.env.step(action)
                total_reward += reward
            rewards.append(np.sum(self.env.game_state.board))
        return np.mean(rewards)

    def _compute_returns(self, rewards, gamma):
        """
        Computes discounted returns for a trajectory.

        Args:
            rewards: List of rewards from the episode.
            gamma: Discount factor.

        Returns:
            Discounted returns.
        """
        returns = []
        g = 0
        for reward in reversed(rewards):
            g = reward + gamma * g
            returns.insert(0, g)
        return returns
