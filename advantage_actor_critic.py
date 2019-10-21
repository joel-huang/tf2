import gym
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import Model
from tensorflow.keras import optimizers

# Seed rng for reproducibility.
tf.random.set_seed(17)

class CategoricalSampler(Model):
    def call(self, logits):
        # Draw a sample from a particular class.
        # logits is a 2D tensor, with each row representing event
        # probabilities of a different categorical distribution.
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class PolicyValueEstimator(Model):
    def __init__(self, num_actions):
        super().__init__()
        # Value regression layers
        self.fc1 = layers.Dense(128, activation='relu')
        self.value = layers.Dense(1)

        # Action choice layers
        self.fc2 = layers.Dense(128, activation='relu')
        self.logits = layers.Dense(num_actions)
        self.sampler = CategoricalSampler()

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        # Perform value regression
        fc1_out = self.fc1(x)
        # Perform action categorical choice
        fc2_out = self.fc2(x)
        return self.logits(fc2_out), self.value(fc1_out)

    def action_value(self, obs):
        logits, value = self.predict(obs, batch_size=1)
        action = self.sampler.predict(logits)
        return tf.squeeze(action, axis=-1), tf.squeeze(value, axis=-1)

class A2CAgent:
    def __init__(self, model, batch_size=32, value=0.5, entropy=0.0001, gamma=0.99):
        self.batch_size = batch_size
        self.value = value
        self.entropy = entropy
        self.gamma = gamma
        self.model = model
        self.model.compile(
            optimizer=optimizers.Adam(lr=0.005),
            loss=[self._logits_loss, self._value_loss]
        )
    
    def test(self, env, render=True):
        obs, ep_reward, done = env.reset(), 0, False
        while not done:
            print(obs[None, :])
            action, _ = self.model.action_value(obs[None, :])
            action_index = action.numpy()
            obs, reward, done, _ = env.step(action_index)
            ep_reward += reward
            if render:
                env.render()
        return ep_reward

    def _value_loss(self, returns, value):
        # Value loss is the MSE between the value estimate and returns.
        return self.value * losses.mean_squared_error(returns, value)

    def _logits_loss(self, actions_and_advantages, logits):
        actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)

        # from_logits ensures transformation into normalized probabilities.
        sparse_ce = losses.SparseCategoricalCrossentropy(from_logits=True)

        # Policy loss is defined by policy gradients and weighted by advantages.
        # Advantages are the difference between returns and some baseline.
        policy_loss = sparse_ce(actions, logits, sample_weight=advantages)

        # Entropy loss ensures sufficient exploration, and is the CE over itself.
        # Entropy measures how random the distribution is (uniform distribution maximizes entropy).
        entropy_loss = losses.categorical_crossentropy(logits, logits, from_logits=True)

        return policy_loss - self.entropy * entropy_loss

    def _returns_advantages(self, rewards, dones, values, next_value):
        # next_value is the bootstrap estimate of a future state.
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # returns is the discounted sum of future rewards.
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t+1] * (1 - dones[t])
        returns = returns[:-1]
        # Calculate advantages by the difference between the returns and bootstrap estimate.
        advantages = returns - values
        return returns, advantages

    def train(self, env, updates=10, render=True, verbose=True):
        # Storage arrays for one batch of data.
        actions = np.empty((self.batch_size,), dtype=np.int32)
        rewards, dones, values = np.empty((3, self.batch_size))
        observations = np.empty((self.batch_size,) + env.observation_space.shape)

        # Training loop.
        obs, ep_rewards = env.reset(), [0]
        for update in range(updates):
            for step in range(self.batch_size):
                observations[step] = obs.copy()
                action, value = self.model.action_value(obs[None, :])
                actions[step], values[step] = action.numpy(), value.numpy()
                obs, rewards[step], dones[step], _ = env.step(action.numpy())
                ep_rewards[-1] += rewards[step]
                if render:
                    env.render()
                if dones[step]:
                    ep_rewards.append(0)
                    obs = env.reset()
            _, next_value = self.model.action_value(obs[None, :])
            returns, advantages = self._returns_advantages(rewards, dones, values, next_value)
            actions_and_advantages = np.concatenate([actions[:, None], advantages[:, None]], axis=-1)
            update_loss = self.model.train_on_batch(observations, [actions_and_advantages, returns])
            if verbose:
                print(update_loss)
        return ep_rewards

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    model = PolicyValueEstimator(num_actions=env.action_space.n)
    agent = A2CAgent(model)
    total_reward = agent.train(env)
    print(total_reward)
    env.close()