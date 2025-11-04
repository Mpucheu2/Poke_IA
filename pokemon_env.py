import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from copy import deepcopy
from typing import List, Dict, Optional, Any, Tuple

try:
    from pokemon_simulator import (
        Pokemon, Battle, Move, Item, Ability, RunAndBunAI,
        showdown_loader, compute_damage, apply_move, get_type_multiplier, 
        normalize_stat_key, calculate_stat
    )
except ImportError as e:
    print(f"Erreur d'importation dans pokemon_env.py: {e}")
    print("Assurez-vous que pokemon_env.py et pokemon_simulator.py sont dans le même dossier.")
    raise
# --- Fin import ---

POKEMON_TYPES = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
    "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug",
    "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy", "None" # 'None' pour le second type si absent
]
NUM_TYPES = len(POKEMON_TYPES)

# Liste des statuts possibles (ordre important)
STATUS_CONDITIONS = ["ok", "brn", "par", "psn", "slp", "frz"] # ok = pas de statut
NUM_STATUSES = len(STATUS_CONDITIONS)

# Taille maximale de l'équipe (pour le banc)
MAX_TEAM_SIZE = 6
BENCH_SIZE = MAX_TEAM_SIZE - 1

class PokemonEnv(gym.Env):
    """
    Environnement Gymnasium pour un combat Pokémon 1v1 simplifié,
    où l'agent RL contrôle l'équipe 1 contre l'IA RunAndBunAI.
    """
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(self, team1_file: str, team2_file: str, render_mode: Optional[str] = None):
        """
        Initialise l'environnement.

        Args:
            team1_file: Chemin vers le fichier de l'équipe de l'agent RL.
            team2_file: Chemin vers le fichier de l'équipe adverse (contrôlée par RunAndBunAI).
            render_mode: Mode d'affichage ('human' pour print, 'ansi' pour string, None pour pas d'affichage).
        """
        super().__init__()

        # Charger les équipes de base
        try:
            self.team1_base = showdown_loader(team1_file)
            self.team2_base = showdown_loader(team2_file)
            if not self.team1_base or not self.team2_base:
                 raise ValueError("Échec du chargement d'une ou des deux équipes.")
        except Exception as e:
            print(f"ERREUR lors du chargement des équipes dans PokemonEnv: {e}")
            # Gérer l'erreur comme nécessaire, peut-être sortir ou lever une exception plus spécifique
            raise

        self.max_team_size = 6
        self.num_moves = 4
        self.num_switches = self.max_team_size - 1 # On peut switcher vers 5 autres max
        self.action_space = spaces.Discrete(self.num_moves + self.num_switches)

        obs_dim = 2 + (NUM_TYPES * 4) + (NUM_STATUSES * 2) + (BENCH_SIZE * 2) + 6 # Doit inclure BENCH_SIZE * 2

        # Low: HP=0, Types=0, Status=0, Bench HP=-1, Stats=0
        low = np.array(
            ([0.0] * 2) + # HPs actifs
            ([0.0] * (NUM_TYPES * 4)) + # Types actifs
            ([0.0] * (NUM_STATUSES * 2)) + # Statuts actifs
            ([-1.0] * BENCH_SIZE * 2) + # HPs banc J1 & J2 (-1 pour padding)
            ([0.0] * 6), # Stats actives (Atk, Def, Spe pour J1 & J2)
            dtype=np.float32
        )
        # High: HP=1, Types=1, Status=1, Bench HP=1, Stats=~2.0 (après normalisation)
        high = np.array(
            ([1.0] * 2) + # HPs actifs
            ([1.0] * (NUM_TYPES * 4)) + # Types actifs
            ([1.0] * (NUM_STATUSES * 2)) + # Statuts actifs
            ([1.0] * BENCH_SIZE * 2) + # HPs banc J1 & J2
            ([2.0] * 6), # Stats actives (valeur max approx après normalisation)
            dtype=np.float32
        )

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # --- Initialisation interne ---
        self.battle: Optional[Battle] = None # L'instance de combat en cours
        self.render_mode = render_mode
        self._action_log = [] # Pour stocker les actions lors du rendu


    # REMPLACEZ la fonction _get_obs

    # REMPLACEZ la fonction _get_obs par celle-ci

    def _get_obs(self) -> np.ndarray:
        """Construit l'observation numérique très améliorée."""
        obs_list = []
        MAX_STAT_APPROX = 500 # Valeur approx pour normaliser les stats

        # --- Vecteur d'observation par défaut ---
        default_obs = np.array(
            ([0.0] * 2) + # HPs actifs
            ([0.0] * (NUM_TYPES * 4)) + # Types actifs
            ([0.0] * (NUM_STATUSES * 2)) + # Statuts actifs
            ([-1.0] * BENCH_SIZE * 2) + # Banc J1 & J2
            ([0.0] * 6), # Stats actives
            dtype=np.float32
        )
        expected_dim = default_obs.shape[0] # Taille attendue (106)


        if not self.battle or not self.battle.active1 or not self.battle.active2:
            # Retourne le vecteur par défaut si le combat n'est pas prêt
             # Vérifier si la taille correspond (debug)
            if default_obs.shape[0] != self.observation_space.shape[0]:
                 print(f"ERREUR INIT: Taille obs par défaut ({default_obs.shape[0]}) != Espace obs ({self.observation_space.shape[0]})")
            return default_obs

        p1 = self.battle.active1
        p2 = self.battle.active2

        # 1. HPs actifs (Normalisés 0-1)
        obs_list.append(p1.current_hp / p1.max_hp if p1.max_hp > 0 else 0.0)
        obs_list.append(p2.current_hp / p2.max_hp if p2.max_hp > 0 else 0.0)

        # 2. Types actifs (One-Hot Encoded)
        for pokemon in [p1, p2]:
            type1_onehot = np.zeros(NUM_TYPES, dtype=np.float32)
            type2_onehot = np.zeros(NUM_TYPES, dtype=np.float32)
            types = pokemon.types if pokemon and hasattr(pokemon, 'types') else []
            if types:
                try:
                    type1_idx = POKEMON_TYPES.index(types[0])
                    type1_onehot[type1_idx] = 1.0
                except (ValueError, IndexError): pass # Type inconnu ou liste vide
                if len(types) > 1:
                    try:
                        type2_idx = POKEMON_TYPES.index(types[1])
                        type2_onehot[type2_idx] = 1.0
                    except (ValueError, IndexError): # Type inconnu ou juste 1 type
                         none_idx = POKEMON_TYPES.index("None")
                         type2_onehot[none_idx] = 1.0
                else:
                     none_idx = POKEMON_TYPES.index("None")
                     type2_onehot[none_idx] = 1.0
            else:
                 none_idx = POKEMON_TYPES.index("None")
                 type1_onehot[none_idx] = 1.0
                 type2_onehot[none_idx] = 1.0
            obs_list.extend(type1_onehot)
            obs_list.extend(type2_onehot)


        # 3. Statuts actifs (One-Hot Encoded)
        for pokemon in [p1, p2]:
            status_onehot = np.zeros(NUM_STATUSES, dtype=np.float32)
            status_str = pokemon.status if pokemon and hasattr(pokemon, 'status') else "ok"
            status_str = status_str if status_str else "ok" # Assure "ok" si status est None
            try:
                # Utilise les 3 premières lettres, en minuscule
                status_idx = STATUS_CONDITIONS.index(status_str[:3].lower())
                status_onehot[status_idx] = 1.0
            except ValueError:
                status_onehot[0] = 1.0 # Défaut à "ok" si statut inconnu
            obs_list.extend(status_onehot)

        # 4. HP% du Banc J1 & J2 (avec padding -1.0)
        for team in [self.battle.team1, self.battle.team2]:
            active_pokemon = self.battle.active1 if team == self.battle.team1 else self.battle.active2
            bench_hp_list = []
            # Assurer que 'team' est bien une liste de Pokemon
            available_bench = []
            if isinstance(team, list):
                 available_bench = [p for p in team if p != active_pokemon and p and hasattr(p,'is_fainted') and not p.is_fainted()]

            for i in range(BENCH_SIZE):
                if i < len(available_bench):
                    p_bench = available_bench[i]
                    bench_hp_list.append(p_bench.current_hp / p_bench.max_hp if p_bench.max_hp > 0 else 0.0)
                else:
                    bench_hp_list.append(-1.0) # Padding
            obs_list.extend(bench_hp_list)

        # 5. Stats Actives (Atk, Def, Spe - Normalisées approx. 0-2)
        for pokemon in [p1, p2]:
             # Vérifier que pokemon et stats existent
            stats_dict = pokemon.stats if pokemon and hasattr(pokemon, 'stats') else {}
            obs_list.append(stats_dict.get('atk', 0) / MAX_STAT_APPROX)
            obs_list.append(stats_dict.get('def', 0) / MAX_STAT_APPROX)
            obs_list.append(stats_dict.get('sp', 0) / MAX_STAT_APPROX)

        # --- Vérification finale de la taille ---
        final_obs_array = np.array(obs_list, dtype=np.float32)

        if final_obs_array.shape[0] != expected_dim:
             print(f"--- ERREUR CRITIQUE DANS _get_obs ---")
             print(f"Taille d'observation incorrecte! Attendu={expected_dim}, Obtenu={final_obs_array.shape[0]}")
             print(f"Contenu partiel obs_list (len={len(obs_list)}): {obs_list[:10]}...{obs_list[-10:]}")
             # Essayer de renvoyer le tableau par défaut pour éviter un crash immédiat
             # mais l'entraînement sera probablement instable.
             return default_obs

        return final_obs_array

    def _get_info(self) -> Dict:
        """Retourne des informations additionnelles (non utilisées par l'IA, mais utiles pour le débogage)."""
        if not self.battle:
            return {"turn": 0, "p1_active": None, "p2_active": None}
        return {
            "turn": self.battle.turn,
            "p1_active": self.battle.active1.name if self.battle.active1 else None,
            "p2_active": self.battle.active2.name if self.battle.active2 else None,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Réinitialise l'environnement pour un nouveau combat (épisode)."""
        super().reset(seed=seed) # Important pour la reproductibilité

        seed_value = int(self.np_random.integers(0, 1e9))
        battle_rng = random.Random(seed_value)
        self.battle = Battle(deepcopy(self.team1_base), deepcopy(self.team2_base), rng=battle_rng)

        self._action_log = ["--- Nouveau Combat ---"] # Réinitialise le log pour le rendu

        observation = self._get_obs()
        info = self._get_info()

        info["action_masks"] = self.action_masks()

        if self.render_mode == "human":
            pass

        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Exécute une étape. Logique Nuzlocke (KO) + Anti-boucle de switch.
        """
        # --- 0. Vérifications initiales ---
        if not self.battle:
            return np.zeros_like(self.observation_space.low), 0, True, False, {}

        terminated = self.battle.battle_over()
        truncated = self.battle.turn > 200 

        if terminated or truncated:
            obs = self._get_obs()
            reward = 0
            info = self._get_info()
            info["action_masks"] = [False] * self.action_space.n 
            is_done = terminated or truncated
            return obs, reward, is_done, truncated, info 

        # --- 1. Initialiser la récompense et le masque ---
        reward = 0.0 # Récompense de base pour ce tour
        valid_actions = self.action_masks()
        info = {"action_masks": valid_actions}

        # --- 2. Traduire l'action de l'agent RL ---
        player1_action: Optional[Tuple] = None
        action_valid = valid_actions[action] 
        action_description = f"Action Agent: {action}"

        active_p1 = self.battle.active1
        available_moves = active_p1.moves
        available_bench = [p for p in self.battle.team1 if p != active_p1 and not p.is_fainted()]

        if 0 <= action < self.num_moves: # Attaque
            move_index = action
            if action_valid: 
                move_chosen = available_moves[move_index]
                player1_action = ('move', move_chosen)
                action_description += f" -> Utilise {move_chosen.name}"
            else: 
                action_description += " -> Attaque Invalide (Masquée)"

        elif self.num_moves <= action < (self.num_moves + self.num_switches): # Switch
            switch_index = action - self.num_moves
            if action_valid:
                player1_action = ('switch', available_bench[switch_index])
                action_description += f" -> Switch vers {available_bench[switch_index].name}"
                reward -= 0.01 # Petite pénalité pour décourager le switch infini

            else:
                action_description += " -> Switch Invalide (Masqué)"
        else: 
            action_valid = False 
            action_description += " -> Action Inconnue"

        self._action_log.append(action_description)

                # --- 3. Gérer l'action invalide ---
        if not action_valid:
            self._action_log.append(f"Action invalide ({action}) choisie ! Tour sauté.")
            reward -= 10.0 # Grosse pénalité

            # On ne joue pas le tour. On renvoie l'état actuel.
            observation = self._get_obs()
            info = self._get_info()
            info["action_masks"] = self.action_masks()

            # On simule l'action de l'adversaire (qui joue seul)
            player2_action = self.battle.ai_team2.choose_action(
                self.battle, self.battle.active2, self.battle.active1,
                is_perfect_luck=self.battle.perfect_luck_opponent
            )
            # On force l'action de l'agent à "ne rien faire" (Struggle-like, mais sans dégâts)
            # C'est complexe. Le plus simple est de juste renvoyer :

            return observation, reward, terminated, truncated, info
        
        # --- 4. Obtenir l'action de l'IA adverse ---
        player2_action = self.battle.ai_team2.choose_action(
            self.battle, 
            self.battle.active2, 
            self.battle.active1,
            is_perfect_luck=self.battle.perfect_luck_opponent 
        )
        self._action_log.append(f"Action Adversaire: {player2_action[0]} {player2_action[1].name if player2_action[0]=='move' else player2_action[1].name}")

        # --- 5. Exécuter le tour de combat (Garder les HP d'avant) ---
        hp_before_p1 = self.battle.active1.current_hp if self.battle.active1 else 0
        hp_before_p2 = self.battle.active2.current_hp if self.battle.active2 else 0

        try:
            self.battle.do_turn(player1_action, player2_action)
            self._action_log.extend(self.battle.log_entries[-self.battle.turn:]) 
        except Exception as e:
            print(f"ERREUR pendant battle.do_turn: {e}")
            observation = self._get_obs()
            reward = -20.0 # Pénalité pour crash
            terminated = True
            info = self._get_info()
            info["error"] = str(e)
            return observation, reward, terminated, False, info

        # --- 6. Gérer les KOs et les Switchs Post-KO ---
        if self.battle.active1 and self.battle.active1.is_fainted() and not self.battle.battle_over():
            next_p1 = self.battle.get_next_alive(self.battle.team1)
            if next_p1:
                self.battle.active1 = next_p1
                self._action_log.append(f"Joueur 1 envoie {next_p1.name}")

        if self.battle.active2 and self.battle.active2.is_fainted() and not self.battle.battle_over():
            next_p2 = self.battle.ai_team2.choose_switch_in(self.battle, self.battle.team2, self.battle.active1)
            if next_p2:
                self.battle.active2 = next_p2
                self._action_log.append(f"Adversaire envoie {next_p2.name}")

        # --- 7. LOGIQUE DE RÉCOMPENSE "NUZLOCKE KO" ---
        hp_after_p2 = self.battle.active2.current_hp if self.battle.active2 else 0
        if hp_before_p2 > 0 and hp_after_p2 <= 0: 
            reward += 1.0  # Récompense simple pour un KO infligé

        hp_after_p1 = self.battle.active1.current_hp if self.battle.active1 else 0
        if hp_before_p1 > 0 and hp_after_p1 <= 0: 
            reward -= 1.0 # Pénalité simple pour un KO subi

        terminated = self.battle.battle_over()
        truncated = self.battle.turn > 200 

        if terminated:
            pass # Pas de bonus/malus de victoire/défaite
        elif truncated:
            # La pénalité pour combat trop long doit être PIRE que de switcher
            reward -= 200.0 

        # --- 8. Obtenir la nouvelle observation et info ---
        observation = self._get_obs()
        info = self._get_info()
        info["action_masks"] = self.action_masks() 

        if self.render_mode == "human":
            pass 

        is_done = terminated or truncated

        return observation, reward, is_done, truncated, info

    def action_masks(self) -> List[bool]:
        """Retourne une liste de booléens indiquant les actions valides."""

        # Initialise toutes les actions comme potentiellement valides
        mask = [True] * self.action_space.n 

        if not self.battle or not self.battle.active1 or self.battle.active1.is_fainted():
            # Si pas de Pokémon actif pour nous, aucune action n'est valide
            # (sauf peut-être pendant le choix post-KO, mais on gère ça simplement)
            return [False] * self.action_space.n

        active_p1 = self.battle.active1
        available_moves = active_p1.moves
        available_bench = [p for p in self.battle.team1 if p != active_p1 and not p.is_fainted()]

        # Masquer les attaques invalides
        for i in range(self.num_moves):
            if i >= len(available_moves):
                mask[i] = False # Pas d'attaque à cet index
            # TODO: Ajouter une vérification des PP si vous les simulez
            # elif available_moves[i].pp <= 0:
            #     mask[i] = False 

        # Masquer les switches invalides
        for i in range(self.num_switches):
            if i >= len(available_bench):
                mask[self.num_moves + i] = False # Pas de Pokémon sur le banc à cet index

        # Cas spécial: On ne peut pas switcher s'il n'y a personne sur le banc
        if not available_bench:
             for i in range(self.num_switches):
                 mask[self.num_moves + i] = False

        # Assurer qu'au moins une action est possible (sinon, c'est Struggle - non implémenté ici)
        if not any(mask):
             # Si aucune action n'est valide (ex: toutes attaques à 0 PP et pas de switch),
             # on pourrait forcer une action (ex: Struggle) ou laisser le masque vide.
             # Pour Stable Baselines, laisser au moins une action valide est mieux.
             # On pourrait exceptionnellement réactiver la première attaque comme "Struggle proxy".
             if len(available_moves) > 0: mask[0] = True # Fallback très basique

        return mask
    
    def render(self) -> Optional[str]:
            """Affiche l'état actuel du combat."""
            if self.render_mode == "human":
                # Affiche les 15 dernières actions/logs du combat
                print("\n".join(self._action_log[-15:])) 
                if self.battle:
                    p1 = self.battle.active1
                    p2 = self.battle.active2
                    if p1: print(f"  Joueur 1: {p1.name} ({p1.current_hp}/{p1.max_hp} HP)")
                    if p2: print(f"  Joueur 2: {p2.name} ({p2.current_hp}/{p2.max_hp} HP)")
                    if self.battle.battle_over():
                        winner = "Joueur 1" if all(p.is_fainted() for p in self.battle.team2) else "Joueur 2 (ou Nul)"
                        print(f"--- Combat Terminé! Vainqueur: {winner} ---")
                return None # Ne retourne rien pour le mode human
            
            elif self.render_mode == "ansi":
                # Retourne une représentation textuelle simple
                if not self.battle: return "Combat non initialisé"
                p1 = self.battle.active1
                p2 = self.battle.active2
                log_str = "\n".join(self._action_log[-5:]) # 5 dernières actions
                state_str = f"T{self.battle.turn}: P1({p1.name} {p1.current_hp}/{p1.max_hp}) vs P2({p2.name} {p2.current_hp}/{p2.max_hp})"
                return state_str + "\n" + log_str
            
            else:
                # Mode de rendu non géré
                return None

    def close(self):
        """Nettoie les ressources si nécessaire."""
        self.battle = None # Libère l'instance de combat