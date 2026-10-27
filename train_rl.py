"""Train a MaskablePPO agent against the Run & Bun AI.

Usage: python train_rl.py [--timesteps N] [--envs N] [--opponent teams/opponent_team_easy.txt]
"""
import argparse
import os

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from pokemon_env import PokemonEnv

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT, "rl_logs")
MODEL_PATH = os.path.join(LOG_DIR, "ppo_pokemon_agent")
DEFAULT_TEAM = os.path.join(ROOT, "teams", "my_team.txt")
DEFAULT_OPPONENT = os.path.join(ROOT, "teams", "opponent_team.txt")


def make_env(rank: int, team1_file: str, team2_file: str, seed: int = 0):
    """Factory for (possibly multiprocessed) environments."""
    def _init():
        monitor_dir = os.path.join(LOG_DIR, "monitor_logs")
        os.makedirs(monitor_dir, exist_ok=True)
        env = PokemonEnv(team1_file=team1_file, team2_file=team2_file)
        env.reset(seed=seed + rank)
        return Monitor(env, filename=os.path.join(monitor_dir, f"monitor_{rank}"))
    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--team", default=DEFAULT_TEAM, help="Agent team (Showdown format)")
    parser.add_argument("--opponent", default=DEFAULT_OPPONENT, help="Opponent team (Showdown format)")
    parser.add_argument("--timesteps", type=int, default=10_000_000, help="Total training steps")
    parser.add_argument("--envs", type=int, default=6, help="Parallel environments")
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    env_fns = [make_env(i, args.team, args.opponent) for i in range(args.envs)]
    env = SubprocVecEnv(env_fns) if args.envs > 1 else DummyVecEnv(env_fns)

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=os.path.join(LOG_DIR, "ppo_tensorboard"),
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.9,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=3e-4,
    )

    print(f"Training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps, log_interval=1)

    model.save(MODEL_PATH)
    env.close()
    print(f"Model saved to {MODEL_PATH}.zip")
    print("Watch it play with: python evaluate_rl.py")
