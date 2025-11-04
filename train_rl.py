import os
from sb3_contrib import MaskablePPO as PPO 
from stable_baselines3.common.env_checker import check_env
# L'import qui contient SubprocVecEnv (celui qui manque probablement)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from pokemon_env import PokemonEnv

def make_env(rank: int, seed: int = 0, team1_file="myteam.txt", team2_file="oppteam.txt"):
    """
    Utility function for multiprocessed envs.
    :param rank: index of the subprocess
    :param seed: the initial seed for RNG
    :param team1_file: path to player 1 team
    :param team2_file: path to player 2 team
    """
    def _init():

        monitor_dir = os.path.join(log_dir, "monitor_logs")
        os.makedirs(monitor_dir, exist_ok=True)

        env = PokemonEnv(team1_file=team1_file, team2_file=team2_file)
        env.reset(seed=seed + rank)
        env = Monitor(env, filename=os.path.join(monitor_dir, f"monitor_{rank}"))
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    # --- Configuration ---
    log_dir = "./rl_logs/" # Directory to save training logs and model
    os.makedirs(log_dir, exist_ok=True)
    
    team1 = "myteam.txt" # Your RL agent's team
    team2 = "oppteam.txt" # Opponent team
    
    num_cpu = 6 # Number of parallel environments (adjust based on your CPU cores)
    total_timesteps = 10_000_000 # Total training steps (start small, e.g., 50k, then increase)
    model_save_path = os.path.join(log_dir, "ppo_pokemon_agent")
    tensorboard_log_path = os.path.join(log_dir, "ppo_tensorboard/")

    # --- Create the Environment ---
    print("Creating environment(s)...")
    if num_cpu > 1:
        env = SubprocVecEnv([make_env(i, team1_file=team1, team2_file=team2) for i in range(num_cpu)])
    else:
        env = DummyVecEnv([make_env(0, team1_file=team1, team2_file=team2)])

    # --- Define and Train the Agent ---
    print("Defining the PPO agent...")
    # MlpPolicy is a standard feedforward neural network policy
    # Adjust hyperparameters as needed (learning_rate, n_steps, batch_size, etc.)
    model = PPO("MlpPolicy",
                env,
                verbose=1, # Print training progress
                tensorboard_log=tensorboard_log_path,
                n_steps=2048, # Steps per update per env
                batch_size=64, # Minibatch size
                n_epochs=10, # Epochs per update
                gamma=0.99, # Discount factor
                gae_lambda=0.9,
                clip_range=0.2,
                ent_coef=0.02,
                vf_coef=0.5,
                max_grad_norm=0.5,
                learning_rate=3e-4) # Common starting learning rate

    print(f"Starting training for {total_timesteps} timesteps...")
    # The 'learn' method handles the training loop
    model.learn(total_timesteps=total_timesteps,
                log_interval=1) # Log stats every update
                # callback=YourCustomCallback() # Optional: for more complex logging/saving

    # --- Save the Trained Model ---
    print(f"Training finished. Saving model to {model_save_path}.zip")
    model.save(model_save_path)

    # --- Clean up ---
    print("Closing environment.")
    env.close()

    print("\nTraining complete. You can now run evaluate_rl.py to see the agent play.")
    print(f"To visualize training (optional), run: tensorboard --logdir {tensorboard_log_path}")