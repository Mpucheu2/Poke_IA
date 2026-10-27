"""Gymnasium environment: the RL agent controls team 1 against the Run & Bun AI (team 2)."""
import random
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pokemon_simulator import Battle, showdown_loader

POKEMON_TYPES = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
    "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug",
    "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy", "None",  # "None" = no second type
]
NUM_TYPES = len(POKEMON_TYPES)

# Order matters: index in the one-hot vector
STATUS_CONDITIONS = ["ok", "burn", "paralysis", "poison", "sleep", "freeze"]
NUM_STATUSES = len(STATUS_CONDITIONS)

MAX_TEAM_SIZE = 6
BENCH_SIZE = MAX_TEAM_SIZE - 1
NUM_MOVES = 4

MAX_TURNS = 200
MAX_STAT_APPROX = 500  # Used to scale stats to roughly [0, 2]

# Rewards
KO_REWARD = 1.0
SWITCH_PENALTY = 0.01      # Discourages endless switching
INVALID_ACTION_PENALTY = 10.0
TIMEOUT_PENALTY = 200.0    # Must be worse than anything else, to stop stalling
CRASH_PENALTY = 20.0


class PokemonEnv(gym.Env):
    """6v6 singles battle. One step = one turn.

    Actions (9): use move 0-3, or switch to bench Pokemon 0-4.
    Invalid actions are exposed through `action_masks()` for MaskablePPO.

    Observation (106 floats):
      - HP ratio of both active Pokemon                   (2)
      - types of both active Pokemon, one-hot             (19 x 2 x 2)
      - status of both active Pokemon, one-hot            (6 x 2)
      - HP ratio of each bench Pokemon, -1 when empty     (5 x 2)
      - Atk / Def / Spe of both active Pokemon            (3 x 2)
    """
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(self, team1_file: str, team2_file: str, render_mode: Optional[str] = None):
        super().__init__()

        self.team1_base = showdown_loader(team1_file)
        self.team2_base = showdown_loader(team2_file)
        if not self.team1_base or not self.team2_base:
            raise ValueError("Could not load one of the teams.")

        self.num_moves = NUM_MOVES
        self.num_switches = BENCH_SIZE
        self.action_space = spaces.Discrete(self.num_moves + self.num_switches)

        low = np.array(
            [0.0] * 2
            + [0.0] * (NUM_TYPES * 4)
            + [0.0] * (NUM_STATUSES * 2)
            + [-1.0] * (BENCH_SIZE * 2)
            + [0.0] * 6,
            dtype=np.float32,
        )
        high = np.array(
            [1.0] * 2
            + [1.0] * (NUM_TYPES * 4)
            + [1.0] * (NUM_STATUSES * 2)
            + [1.0] * (BENCH_SIZE * 2)
            + [2.0] * 6,
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self._default_obs = low.copy()

        self.battle: Optional[Battle] = None
        self.render_mode = render_mode
        self._action_log: List[str] = []

    # --- Observation -----------------------------------------------------

    @staticmethod
    def _types_one_hot(pokemon) -> List[float]:
        type1 = np.zeros(NUM_TYPES, dtype=np.float32)
        type2 = np.zeros(NUM_TYPES, dtype=np.float32)
        types = list(pokemon.types or []) + ["None", "None"]
        for vector, type_name in ((type1, types[0]), (type2, types[1])):
            if type_name in POKEMON_TYPES:
                vector[POKEMON_TYPES.index(type_name)] = 1.0
        return list(type1) + list(type2)

    @staticmethod
    def _status_one_hot(pokemon) -> List[float]:
        vector = np.zeros(NUM_STATUSES, dtype=np.float32)
        status = pokemon.status if pokemon.status in STATUS_CONDITIONS else "ok"  # e.g. flinch -> ok
        vector[STATUS_CONDITIONS.index(status)] = 1.0
        return list(vector)

    @staticmethod
    def _bench_hp(team, active) -> List[float]:
        bench = [p for p in team if p is not active and not p.is_fainted()]
        values = [p.current_hp / p.max_hp if p.max_hp > 0 else 0.0 for p in bench[:BENCH_SIZE]]
        return values + [-1.0] * (BENCH_SIZE - len(values))

    def _get_obs(self) -> np.ndarray:
        if not self.battle or not self.battle.active1 or not self.battle.active2:
            return self._default_obs.copy()

        p1, p2 = self.battle.active1, self.battle.active2
        obs: List[float] = []

        for p in (p1, p2):
            obs.append(p.current_hp / p.max_hp if p.max_hp > 0 else 0.0)
        for p in (p1, p2):
            obs.extend(self._types_one_hot(p))
        for p in (p1, p2):
            obs.extend(self._status_one_hot(p))
        obs.extend(self._bench_hp(self.battle.team1, p1))
        obs.extend(self._bench_hp(self.battle.team2, p2))
        for p in (p1, p2):
            for stat in ("atk", "def", "spe"):
                obs.append(p.stats.get(stat, 0) / MAX_STAT_APPROX)

        return np.clip(np.array(obs, dtype=np.float32), self.observation_space.low, self.observation_space.high)

    def _get_info(self) -> Dict:
        if not self.battle:
            return {"turn": 0, "p1_active": None, "p2_active": None}
        info = {
            "turn": self.battle.turn,
            "p1_active": self.battle.active1.name if self.battle.active1 else None,
            "p2_active": self.battle.active2.name if self.battle.active2 else None,
        }
        if self.battle.battle_over():
            info["winner"] = 1 if all(p.is_fainted() for p in self.battle.team2) else 2
        return info

    # --- Gymnasium API ---------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        battle_rng = random.Random(int(self.np_random.integers(0, 1e9)))
        self.battle = Battle(deepcopy(self.team1_base), deepcopy(self.team2_base), rng=battle_rng)
        self._action_log = ["--- New battle ---"]

        info = self._get_info()
        info["action_masks"] = self.action_masks()
        return self._get_obs(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if not self.battle:
            return self._default_obs.copy(), 0.0, True, False, {}

        if self.battle.battle_over() or self.battle.turn > MAX_TURNS:
            info = self._get_info()
            info["action_masks"] = [False] * self.action_space.n
            return self._get_obs(), 0.0, self.battle.battle_over(), not self.battle.battle_over(), info

        reward = 0.0
        valid_actions = self.action_masks()
        action_valid = 0 <= action < self.action_space.n and valid_actions[action]

        # 1. Translate the agent's action
        active_p1 = self.battle.active1
        bench = [p for p in self.battle.team1 if p != active_p1 and not p.is_fainted()]
        player1_action = None
        if action_valid and action < self.num_moves:
            move = active_p1.moves[action]
            player1_action = ("move", move)
            self._action_log.append(f"Agent uses {move.name}")
        elif action_valid:
            target = bench[action - self.num_moves]
            player1_action = ("switch", target)
            self._action_log.append(f"Agent switches to {target.name}")
            reward -= SWITCH_PENALTY

        # 2. Invalid action: penalise and skip the turn (should not happen with MaskablePPO)
        if not action_valid:
            self._action_log.append(f"Invalid action {action}, turn skipped.")
            info = self._get_info()
            info["action_masks"] = self.action_masks()
            return self._get_obs(), reward - INVALID_ACTION_PENALTY, False, False, info

        # 3. Opponent action (Run & Bun AI)
        player2_action = self.battle.ai_team2.choose_action(
            self.battle, self.battle.active2, self.battle.active1,
            is_perfect_luck=self.battle.perfect_luck_opponent,
        )
        self._action_log.append(f"Opponent: {player2_action[0]} {player2_action[1].name}")

        # 4. Play the turn
        hp_before_p1 = self.battle.active1.current_hp
        hp_before_p2 = self.battle.active2.current_hp
        log_start = len(self.battle.log_entries)
        try:
            self.battle.do_turn(player1_action, player2_action)
        except Exception as e:
            print(f"ERROR during battle.do_turn: {e}")
            info = self._get_info()
            info["error"] = str(e)
            return self._get_obs(), -CRASH_PENALTY, True, False, info
        self._action_log.extend(self.battle.log_entries[log_start:])
        self.battle.turn += 1

        # 5. Rewards: +1 per KO dealt, -1 per KO taken (Nuzlocke mindset)
        if hp_before_p2 > 0 and self.battle.active2.current_hp <= 0:
            reward += KO_REWARD
        if hp_before_p1 > 0 and self.battle.active1.current_hp <= 0:
            reward -= KO_REWARD

        # 6. Replace fainted Pokemon (the agent's replacement is simply the next one alive)
        if self.battle.active1.is_fainted() and not self.battle.battle_over():
            self.battle.active1 = self.battle.get_next_alive(self.battle.team1)
            self._action_log.append(f"Player 1 sends out {self.battle.active1.name}")
        if self.battle.active2.is_fainted() and not self.battle.battle_over():
            self.battle.active2 = self.battle.ai_team2.choose_switch_in(self.battle, self.battle.team2, self.battle.active1)
            self._action_log.append(f"Opponent sends out {self.battle.active2.name}")

        terminated = self.battle.battle_over()
        truncated = not terminated and self.battle.turn > MAX_TURNS
        if truncated:
            reward -= TIMEOUT_PENALTY

        info = self._get_info()
        info["action_masks"] = self.action_masks()
        return self._get_obs(), reward, terminated, truncated, info

    def action_masks(self) -> List[bool]:
        """Valid actions: existing moves, and switches to bench Pokemon that are still alive."""
        if not self.battle or not self.battle.active1 or self.battle.active1.is_fainted():
            return [False] * self.action_space.n

        active = self.battle.active1
        bench = [p for p in self.battle.team1 if p != active and not p.is_fainted()]
        move_mask = [i < len(active.moves) for i in range(self.num_moves)]
        switch_mask = [i < len(bench) for i in range(self.num_switches)]
        # PP are not simulated, so a Pokemon with moves always has a valid action
        return move_mask + switch_mask

    def render(self) -> Optional[str]:
        if not self.battle:
            return None
        p1, p2 = self.battle.active1, self.battle.active2

        if self.render_mode == "human":
            print("\n".join(self._action_log[-15:]))
            print(f"  Player 1: {p1.name} ({p1.current_hp}/{p1.max_hp} HP)")
            print(f"  Player 2: {p2.name} ({p2.current_hp}/{p2.max_hp} HP)")
            if self.battle.battle_over():
                winner = "Player 1" if all(p.is_fainted() for p in self.battle.team2) else "Player 2"
                print(f"--- Battle over! Winner: {winner} ---")
            return None

        if self.render_mode == "ansi":
            state = f"T{self.battle.turn}: P1({p1.name} {p1.current_hp}/{p1.max_hp}) vs P2({p2.name} {p2.current_hp}/{p2.max_hp})"
            return state + "\n" + "\n".join(self._action_log[-5:])
        return None

    def close(self):
        self.battle = None
