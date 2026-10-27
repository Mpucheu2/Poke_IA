"""Watch a trained agent fight, turn by turn, and report its win rate.

Usage: python evaluate_rl.py [--episodes N] [--fast]
"""
import argparse
import os
import time

from sb3_contrib import MaskablePPO

from pokemon_env import PokemonEnv
from train_rl import DEFAULT_OPPONENT, DEFAULT_TEAM, MODEL_PATH

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=MODEL_PATH, help="Path to the trained model")
    parser.add_argument("--team", default=DEFAULT_TEAM)
    parser.add_argument("--opponent", default=DEFAULT_OPPONENT)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--fast", action="store_true", help="No battle log, no pauses: just the win rate")
    args = parser.parse_args()

    env = PokemonEnv(args.team, args.opponent, render_mode=None if args.fast else "human")
    model = MaskablePPO.load(args.model, env=env)

    wins = 0
    for episode in range(1, args.episodes + 1):
        obs, info = env.reset()
        done, total_reward = False, 0.0
        while not done:
            # deterministic=True: always pick the action the agent rates best
            action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            if not args.fast:
                print(f"\n--- Battle {episode}, turn {info['turn']} ---")
                env.render()
                time.sleep(0.5)

        won = info.get("winner") == 1
        wins += won
        result = "win" if won else ("timeout" if truncated else "loss")
        print(f"Battle {episode}: {result}, total reward {total_reward:.2f}")

    print(f"\nAgent win rate: {wins}/{args.episodes} ({100 * wins / args.episodes:.0f}%)")
    env.close()
