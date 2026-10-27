from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

from pokemon_env import PokemonEnv

TEAMS = Path(__file__).resolve().parent.parent / "teams"


def make_env():
    return PokemonEnv(TEAMS / "my_team.txt", TEAMS / "opponent_team.txt")


def test_gymnasium_api_compliance():
    check_env(make_env(), skip_render_check=True)


def test_observation_shape_and_bounds():
    env = make_env()
    obs, info = env.reset(seed=0)
    assert obs.shape == (106,)
    assert env.observation_space.contains(obs)
    assert len(info["action_masks"]) == env.action_space.n


def test_random_masked_policy_finishes_battles():
    env = make_env()
    rng = np.random.default_rng(0)
    for seed in range(3):
        env.reset(seed=seed)
        done = False
        while not done:
            valid = np.flatnonzero(env.action_masks())
            _, _, terminated, truncated, info = env.step(int(rng.choice(valid)))
            done = terminated or truncated
        assert info["turn"] <= 201
        assert terminated == ("winner" in info)


def test_invalid_action_is_penalised():
    env = make_env()
    env.reset(seed=0)
    # Keep a single move so that move slot 3 becomes invalid
    env.battle.active1.moves = env.battle.active1.moves[:1]
    _, reward, terminated, _, _ = env.step(3)
    assert reward < 0 and not terminated
