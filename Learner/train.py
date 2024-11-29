import numpy as np
from ray.air import session
import os
import pickle

from ray import tune
from ray.train import Checkpoint
from ray.tune.schedulers import PopulationBasedTraining

from deep_env.env import HanabiEnv
from deep_env.pbt_agent import HanabiAgent

from ray.air import session
import os
import pickle


from ray.air import session
from ray.train import Checkpoint
import os
import pickle

from ray.air import session
from ray.train import Checkpoint
import os
import pickle

def train_hanabi_agent(config):
    # Initialize environment and agent
    env = HanabiEnv(num_players=2)
    agent = HanabiAgent(env, learning_rate=config["learning_rate"], entropy_reg=config["entropy_reg"])

    # Load checkpoint if available
    checkpoint = session.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pkl")
            with open(checkpoint_path, "rb") as f:
                state = pickle.load(f)
            agent.policy_network.set_weights(state["policy_weights"])
            agent.value_network.set_weights(state["value_weights"])
            agent.optimizer.learning_rate.assign(state["learning_rate"])
            agent.entropy_reg = state["entropy_reg"]

    for step in range(config["training_steps"]):
        observations, actions, rewards = [], [], []
        obs = env.reset()
        done = False

        # Play an episode
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            obs = next_obs

        # Train the agent
        agent.train_step(observations, actions, rewards, gamma=0.999)

        # Evaluate the agent
        avg_reward = agent.evaluate()

        # Save checkpoint periodically
        if step % 10000 == 0:
            checkpoint_dir = f"checkpoint_step_{step}"  # Use a step-specific checkpoint directory
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pkl")
            state = {
                "policy_weights": agent.policy_network.get_weights(),
                "value_weights": agent.value_network.get_weights(),
                "learning_rate": agent.optimizer.learning_rate.numpy(),
                "entropy_reg": agent.entropy_reg,
            }
            with open(checkpoint_path, "wb") as f:
                pickle.dump(state, f)
            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            session.report({"reward": avg_reward}, checkpoint=checkpoint)

        # Report metrics periodically
        session.report({"reward": avg_reward})

    # Final checkpoint and metrics
    checkpoint_dir = "final_checkpoint"  # Use a specific directory for the final checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pkl")
    state = {
        "policy_weights": agent.policy_network.get_weights(),
        "value_weights": agent.value_network.get_weights(),
        "learning_rate": agent.optimizer.learning_rate.numpy(),
        "entropy_reg": agent.entropy_reg,
    }
    with open(checkpoint_path, "wb") as f:
        pickle.dump(state, f)
    checkpoint = Checkpoint.from_directory(checkpoint_dir)
    session.report({"reward": avg_reward}, checkpoint=checkpoint)

pbt_scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="reward",
    mode="max",
    perturbation_interval=5,
    hyperparam_mutations={
        "learning_rate": lambda: np.random.uniform(1e-4, 4e-4),
        "entropy_reg": lambda: np.random.uniform(1e-2, 5e-2),
    },
)
local_dir = os.path.abspath("./pbt_checkpoints")  # Convert to absolute path

analysis = tune.run(
    train_hanabi_agent,
    config={
        "learning_rate": tune.uniform(1e-4, 4e-4),
        "entropy_reg": tune.uniform(1e-2, 5e-2),
        "training_steps": 1000,
    },
    num_samples=30,
    scheduler=pbt_scheduler,
    stop={"training_iteration": 50},
    verbose=1,
    local_dir=local_dir,  # Use absolute path
)