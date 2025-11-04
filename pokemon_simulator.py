from __future__ import annotations
import argparse
import json
import math
import random
from copy import deepcopy
from typing import List, Dict, Optional, Any

# --- Imports Corrigés ---
# Importe depuis tes fichiers de modules séparés
from data_loader import (
    load_json, get_species, get_move, get_item, get_ability, get_nature_data
)
from stat_calculator import calculate_stat # Importe depuis la racine

# --- Classes de base (Pokémon, Move, etc.) ---

class Move:
    def __init__(self, name: str, data: Dict[str, Any]):
        self.name = name
        # Lit "bp", "basePower", ou "power" depuis le JSON
        self.power = data.get("bp", data.get("basePower", data.get("power", 0)))
        # Lit "category" ou "cat"
        self.category = data.get("category", data.get("cat", "Status"))
        self.type = data.get("type", "Normal")
        self.accuracy = data.get("accuracy", 100)
        if self.accuracy is True: self.accuracy = 101 # Gère "true" pour 100%
        self.pp = data.get("pp", 10)
        self.priority = data.get("priority", 0)
        self.effect = data.get("effect") # Pour les effets secondaires

class Item:
    def __init__(self, name: str, data: Optional[Dict[str, Any]] = None):
        self.name = name
        self.data = data or {}

class Ability:
    def __init__(self, name: str, data: Optional[Dict[str, Any]] = None):
        self.name = name
        self.data = data or {}

class Pokemon:
    def __init__(self, name: str, level: int, moves: List[Move], item: Optional[Item], ability: Ability, stats: Dict[str, int], types: List[str]):
        self.name = name
        self.level = level
        self.moves = moves
        self.item = item
        self.ability = ability
        self.stats = stats
        self.types = types
        self.max_hp = stats.get("hp", 1)
        self.current_hp = self.max_hp
        self.is_seeded = False
        self.status = None # 'burn', 'paralysis', 'sleep', 'poison', 'freeze', 'flinch'

    def is_fainted(self) -> bool:
        return self.current_hp <= 0

# --- Logique de Combat & Calculs ---

def get_type_multiplier(attack_type: str, target_types: List[str]) -> float:
    """Calcule le multiplicateur de type en utilisant une table d'efficacité."""
    type_effectiveness = {
        'Normal': {'Rock': 0.5, 'Ghost': 0, 'Steel': 0.5},
        'Fire': {'Fire': 0.5, 'Water': 0.5, 'Grass': 2, 'Ice': 2, 'Bug': 2, 'Rock': 0.5, 'Dragon': 0.5, 'Steel': 2},
        'Water': {'Fire': 2, 'Water': 0.5, 'Grass': 0.5, 'Ground': 2, 'Rock': 2, 'Dragon': 0.5},
        'Electric': {'Water': 2, 'Electric': 0.5, 'Grass': 0.5, 'Ground': 0, 'Flying': 2, 'Dragon': 0.5},
        'Grass': {'Fire': 0.5, 'Water': 2, 'Grass': 0.5, 'Poison': 0.5, 'Ground': 2, 'Flying': 0.5, 'Bug': 0.5, 'Rock': 2, 'Dragon': 0.5, 'Steel': 0.5},
        'Ice': {'Fire': 0.5, 'Water': 0.5, 'Grass': 2, 'Ice': 0.5, 'Ground': 2, 'Flying': 2, 'Dragon': 2, 'Steel': 0.5},
        'Fighting': {'Normal': 2, 'Ice': 2, 'Poison': 0.5, 'Flying': 0.5, 'Psychic': 0.5, 'Bug': 0.5, 'Rock': 2, 'Ghost': 0, 'Dark': 2, 'Steel': 2, 'Fairy': 0.5},
        'Poison': {'Grass': 2, 'Poison': 0.5, 'Ground': 0.5, 'Rock': 0.5, 'Ghost': 0.5, 'Steel': 0, 'Fairy': 2},
        'Ground': {'Fire': 2, 'Electric': 2, 'Grass': 0.5, 'Poison': 2, 'Flying': 0, 'Bug': 0.5, 'Rock': 2, 'Steel': 2},
        'Flying': {'Electric': 0.5, 'Grass': 2, 'Fighting': 2, 'Bug': 2, 'Rock': 0.5, 'Steel': 0.5},
        'Psychic': {'Fighting': 2, 'Poison': 2, 'Psychic': 0.5, 'Dark': 0, 'Steel': 0.5},
        'Bug': {'Fire': 0.5, 'Grass': 2, 'Fighting': 0.5, 'Poison': 0.5, 'Flying': 0.5, 'Psychic': 2, 'Ghost': 0.5, 'Dark': 2, 'Steel': 0.5, 'Fairy': 0.5},
        'Rock': {'Fire': 2, 'Ice': 2, 'Fighting': 0.5, 'Ground': 0.5, 'Flying': 2, 'Bug': 2, 'Steel': 0.5},
        'Ghost': {'Normal': 0, 'Psychic': 2, 'Ghost': 2, 'Dark': 0.5},
        'Dragon': {'Dragon': 2, 'Steel': 0.5, 'Fairy': 0},
        'Dark': {'Fighting': 0.5, 'Psychic': 2, 'Ghost': 2, 'Dark': 0.5, 'Fairy': 0.5},
        'Steel': {'Fire': 0.5, 'Water': 0.5, 'Electric': 0.5, 'Ice': 2, 'Rock': 2, 'Steel': 0.5, 'Fairy': 2},
        'Fairy': {'Fire': 0.5, 'Fighting': 2, 'Poison': 0.5, 'Dragon': 2, 'Dark': 2, 'Steel': 0.5},
    }

    if attack_type not in type_effectiveness:
        return 1.0

    multiplier = 1.0
    attack_chart = type_effectiveness[attack_type]
    for target_type in target_types:
        multiplier *= attack_chart.get(target_type, 1.0) 

    return multiplier

def compute_damage(battle: Battle, attacker: Pokemon, defender: Pokemon, move: Move, critical: bool=False, rng_factor: Optional[float]=None, force_perfect: bool = False) -> int:
    """Calcule les dégâts d'une attaque."""
    
    category_lower = move.category.lower()
    
    if category_lower == 'status':
        return 0
    
    if force_perfect:
        rng_factor = 1.0
        critical = True
    
    if rng_factor is None:
        rng_factor = random.uniform(0.85, 1.0)

    if move.power is None or move.power <= 0:
        return 0

    A_stat = 'atk'
    D_stat = 'def'
    if category_lower == 'physical':
        A_stat = 'atk'
        D_stat = 'def'
    elif category_lower == 'special':
        A_stat = 'spa'
        D_stat = 'spd'
    else:
        return 0
        
    A = attacker.stats.get(A_stat, 0)
    D = defender.stats.get(D_stat, 1)
    
    # Objets modifiant A ou D AVANT le calcul
    if attacker.item:
        item_name_lower = attacker.item.name.lower()
        if item_name_lower == "thick club" and attacker.name in ["Cubone", "Marowak"] and A_stat == 'atk':
            A *= 2
        elif item_name_lower == "light ball" and attacker.name == "Pikachu":
             A *= 2
             
    if defender.item:
         if defender.item.name.lower() == "metal powder" and defender.name == "Ditto" and D_stat == 'def':
              D *= 1.5 

    if D <= 0: D = 1

    stab = 1.5 if move.type in attacker.types else 1.0
    type_mult = get_type_multiplier(move.type, defender.types)
    burn_mul = 0.5 if attacker.status == 'burn' and category_lower == 'physical' else 1.0
    
    # Multiplicateur d'objet
    item_mul = 1.0
    if attacker.item:
        item_name_lower = attacker.item.name.lower()
        if item_name_lower == "life orb":
            item_mul *= 1.3
        elif item_name_lower == "choice specs" and category_lower == "special":
            item_mul *= 1.5
        
        TYPE_BOOST_ITEMS = {
            "charcoal": "Fire", "mystic water": "Water", "magnet": "Electric",
            "black belt": "Fighting", "black glasses": "Dark", "hard stone": "Rock",
            "miracle seed": "Grass", "never-melt ice": "Ice", "poison barb": "Poison",
            "sharp beak": "Flying", "silver powder": "Bug", "soft sand": "Ground",
            "spell tag": "Ghost", "twisted spoon": "Psychic", "dragon fang": "Dragon",
            "metal coat": "Steel", "pink bow": "Normal", "polkadot bow": "Normal" 
        }
        if item_name_lower in TYPE_BOOST_ITEMS and TYPE_BOOST_ITEMS[item_name_lower] == move.type:
            item_mul *= 1.1 
    
    # Multiplicateur Climat
    weather_mul = 1.0
    if battle.weather == "Sun":
        if move.type == "Fire": weather_mul = 1.5
        elif move.type == "Water": weather_mul = 0.5
    elif battle.weather == "Rain":
        if move.type == "Water": weather_mul = 1.5
        elif move.type == "Fire": weather_mul = 0.5

    # Formule
    damage = (((((2 * attacker.level) / 5) + 2) * move.power * (A / D)) / 50) + 2
    damage *= stab
    damage *= type_mult
    damage *= burn_mul
    damage *= item_mul
    damage *= weather_mul
    damage *= rng_factor
    
    final_damage = max(1, math.floor(damage)) if type_mult > 0 else 0
    
    return final_damage

def apply_move(battle: Battle, attacker: Pokemon, defender: Pokemon, move: Move, rng: random.Random) -> Dict[str, Any]:
    """Applique une attaque et retourne le résultat."""
    result = {'hit': True, 'damage': 0, 'failed': False, 'flinch': False} 
    move_name_lower = move.name.lower()

    # Gestion de Solar Beam
    if move_name_lower == "solar beam":
        if battle.weather != "Sun":
             battle.log(f"{attacker.name} a besoin de soleil pour {move.name} !")
             result['failed'] = True
             return result 
    
    # Gestion Précision (Thunder, Hurricane, Bright Powder)
    accuracy = move.accuracy
    
    if accuracy < 101: # 101+ = ne peut pas échouer
        if battle.weather == "Rain":
             if move_name_lower == "thunder" or move_name_lower == "hurricane":
                  accuracy = 101 
        elif battle.weather == "Sun":
             if move_name_lower == "thunder" or move_name_lower == "hurricane":
                  accuracy = 50 

        if defender.item and defender.item.name.lower() == "bright powder":
            accuracy *= 0.9 

        if rng.uniform(0, 100) > accuracy:
            result['hit'] = False
            return result

    type_mult = get_type_multiplier(move.type, defender.types)
    if type_mult == 0:
        result['failed'] = True
        return result

    # Gestion Weather Ball
    original_move_type = move.type
    original_move_power = move.power
    if move_name_lower == "weather ball" and battle.weather is not None:
        move.power *= 2 
        if battle.weather == "Sun": move.type = "Fire"
        elif battle.weather == "Rain": move.type = "Water"
        type_mult = get_type_multiplier(move.type, defender.types) # Recalculer
        if type_mult == 0:
            result['failed'] = True
            move.type = original_move_type 
            move.power = original_move_power
            return result
    
    # Gestion Taux de Critique (Scope Lens, etc.)
    crit_chance = 1/16 
    high_crit_moves = ["razorleaf", "crabhammer", "slash", "aeroblast", "crosschop", "karatechop"]
    if move_name_lower in high_crit_moves:
         crit_chance *= 8
    if attacker.item:
        item_name_lower = attacker.item.name.lower()
        if item_name_lower == "scope lens": crit_chance *= 2
        elif item_name_lower == "lucky punch" and attacker.name == "Chansey": crit_chance = 1/2
        elif item_name_lower == "stick" and attacker.name == "Farfetch'd": crit_chance = 1/2
              
    is_crit = rng.random() < crit_chance
    crit_mul = 1.5 if is_crit else 1.0 

    # Calcul Dégâts
    damage = compute_damage(battle, attacker, defender, move, critical=is_crit, rng_factor=rng.uniform(0.85, 1.0))
    damage = math.floor(damage * crit_mul) # Appliquer le critique *après* le calcul de base

    # Rétablir Weather Ball
    if move_name_lower == "weather ball":
         move.type = original_move_type
         move.power = original_move_power
    
    # Gestion Focus Band / Focus Sash
    focus_item_activated = False
    if defender.item:
        item_name_lower = defender.item.name.lower()
        if item_name_lower == "focus band" and damage >= defender.current_hp:
             if rng.random() < (1/10): 
                  damage = defender.current_hp - 1
                  focus_item_activated = True
                  result['effect'] = 'focus_band'
        elif item_name_lower == "focus sash" and defender.current_hp == defender.max_hp and damage >= defender.current_hp:
            damage = defender.current_hp - 1 
            focus_item_activated = True
            result['effect'] = 'focus_sash'

    # Appliquer les dégâts
    if damage > 0:
        defender.current_hp = max(0, defender.current_hp - damage)
        result['damage'] = damage
    
    # Effets secondaires (simplifié)
    if move.effect == 'burn10' and defender.status is None and rng.random() < 0.1:
        defender.status = 'burn'
        result['effect'] = 'burn'
    
    if focus_item_activated and defender.item.name.lower() == "focus sash":
        defender.item = None # Consommé

    # Gestion King's Rock
    if attacker.item and attacker.item.name.lower() == "king's rock" and move.category.lower() != 'status':
         if rng.random() < 0.1: 
             result['potential_flinch'] = True
    
    # Gestion Recul (Volt Tackle, Head Smash)
    is_recoil = 'recoil' in (move.effect or '') or move_name_lower in ["volt tackle", "head smash", "flare blitz"]
    if is_recoil and (not attacker.ability or attacker.ability.name.lower() != 'rock head'):
        recoil_damage = max(1, math.floor(damage / 3))
        attacker.current_hp = max(0, attacker.current_hp - recoil_damage)
        result['recoil_damage'] = recoil_damage

    return result

# --- IA (SmartAI) - Votre IA ---

class SmartAI:
    """IA avancée qui évalue les switchs, talents, vitesse, et statut."""
    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()

    def choose_action(self, battle: Battle, active_pokemon: Pokemon, team: List[Pokemon], opponent: Pokemon, is_perfect_luck: bool = False) -> tuple:
        """L'IA décide s'il faut attaquer ou changer de Pokémon."""
        
        best_move, best_move_score = None, -float('inf')
        for move in active_pokemon.moves:
            score = self.score_move(battle, active_pokemon, opponent, move, is_perfect_luck=is_perfect_luck)
            if score > best_move_score:
                best_move_score = score
                best_move = move

        best_switch, best_switch_score = None, -float('inf')
        benched_pokemon = [p for p in team if p != active_pokemon and not p.is_fainted()]
        if benched_pokemon:
            for bench_p in benched_pokemon:
                score = self.score_switch(battle, bench_p, opponent)
                if score > best_switch_score:
                    best_switch_score = score
                    best_switch = bench_p
        
        SWITCH_THRESHOLD = 1.25
        if best_switch and best_switch_score > best_move_score * SWITCH_THRESHOLD:
            # print(f"  [IA de {active_pokemon.name}] Décision : Switcher sur {best_switch.name} (Score Switch: {best_switch_score:.1f} vs Score Attaque: {best_move_score:.1f})")
            return ('switch', best_switch)
        else:
            # print(f"  [IA de {active_pokemon.name}] Décision : Attaquer avec {best_move.name if best_move else 'Struggle'} (Score Attaque: {best_move_score:.1f} vs Score Switch: {best_switch_score:.1f})")
            default_move = active_pokemon.moves[0] if active_pokemon.moves else Move("Struggle", {"power": 50, "type": "Normal", "category": "physical", "accuracy": 100}) 
            return ('move', best_move if best_move else default_move)

    def score_move(self, battle: Battle, attacker: Pokemon, defender: Pokemon, move: Move, is_perfect_luck: bool = False) -> float:
        """Attribue un score à une attaque (logique SmartAI)."""
        if get_type_multiplier(move.type, defender.types) == 0:
            return 0
        
        move_category_lower = move.category.lower()

        if move_category_lower == 'status':
            if defender.status is not None: return 0 
            score = 0
            effect_name = (move.effect or move.name).lower()
            if 'paralysis' in effect_name: score = 60 if defender.stats['spe'] > attacker.stats['spe'] else 30
            elif 'burn' in effect_name: score = 60 if defender.stats['atk'] > defender.stats['spa'] else 30
            elif 'howl' in effect_name: score = 35
            elif 'shell smash' in effect_name: score = 75 if attacker.current_hp / attacker.max_hp > 0.7 else 10
            return score

        damage = compute_damage(battle, attacker, defender, move, rng_factor=0.925, force_perfect=is_perfect_luck)
        score = (damage / defender.max_hp) * 100

        if 'spa_boost_70' in (move.effect or ''): score += 20 
        if 'burn10' in (move.effect or '') and defender.status is None: score += 8 

        # Calcul de la vitesse effective
        speed_attacker = attacker.stats['spe']
        speed_defender = defender.stats['spe']
        if battle.weather == "Sun":
            if attacker.ability and attacker.ability.name.lower() == "chlorophyll": speed_attacker *= 2
            if defender.ability and defender.ability.name.lower() == "chlorophyll": speed_defender *= 2
        elif battle.weather == "Rain":
            if attacker.ability and attacker.ability.name.lower() == "swift swim": speed_attacker *= 2
            if defender.ability and defender.ability.name.lower() == "swift swim": speed_defender *= 2
        
        is_faster = speed_attacker > speed_defender
        
        if is_faster and damage >= defender.current_hp:
            score *= 1.5 

        if not is_faster:
            opponent_offensive_moves = [m for m in defender.moves if m.category.lower() != 'status']
            if opponent_offensive_moves:
                damage_taken = max(compute_damage(battle, defender, attacker, opp_move, rng_factor=1.0) for opp_move in opponent_offensive_moves)
                if damage_taken >= attacker.current_hp:
                    score *= 0.6 

        is_recoil = 'recoil' in (move.effect or '') or move.name in ["Volt Tackle", "Head Smash", "Flare Blitz"]
        if is_recoil and (not attacker.ability or attacker.ability.name.lower() != 'rock head'):
            recoil_dmg = (damage / 3) / attacker.max_hp * 100
            score -= recoil_dmg * 0.75

        return score

    def score_switch(self, battle: Battle, incoming: Pokemon, opponent: Pokemon) -> float:
        """Évalue l'avantage d'un switch (logique SmartAI)."""
        
        offensive_score = 0
        offensive_moves = [m for m in incoming.moves if m.category.lower() != 'status']
        if offensive_moves:
            offensive_score = max(self.score_move(battle, incoming, opponent, move) for move in offensive_moves)

        worst_damage = 0
        worst_move_type = "Normal"
        opponent_offensive_moves = [m for m in opponent.moves if m.category.lower() != 'status']
        
        if not opponent_offensive_moves:
             survival_score = 100
        else:
            for opp_move in opponent_offensive_moves:
                dmg = compute_damage(battle, opponent, incoming, opp_move, rng_factor=1.0)
                if dmg > worst_damage:
                    worst_damage = dmg
                    worst_move_type = opp_move.type
            survival_score = 100 - ((worst_damage / incoming.max_hp) * 100)

        ability_name = incoming.ability.name.lower() if incoming.ability else ""

        if (ability_name == 'levitate' and worst_move_type == 'Ground') or \
           (ability_name == 'volt absorb' and worst_move_type == 'Electric') or \
           (ability_name == 'water absorb' and worst_move_type == 'Water') or \
           (ability_name == 'flash fire' and worst_move_type == 'Fire'):
            
            worst_damage_after_immunity = 0
            opponent_offensive_moves_filtered = [m for m in opponent_offensive_moves if m.type != worst_move_type]
            
            if opponent_offensive_moves_filtered:
                 for opp_move in opponent_offensive_moves_filtered:
                    dmg = compute_damage(battle, opponent, incoming, opp_move, rng_factor=1.0)
                    if dmg > worst_damage_after_immunity:
                        worst_damage_after_immunity = dmg
            
            survival_score = 100 - ((worst_damage_after_immunity / incoming.max_hp) * 100)
            score = (offensive_score * 0.5) + (survival_score * 0.5) + 100 # +100 pour l'immunité
        else:
            score = (offensive_score * 0.6) + (survival_score * 0.4)

        if ability_name == 'intimidate' and opponent.stats['atk'] > opponent.stats['spa']:
            score *= 1.3
            
        # Calcul Vitesse
        speed_incoming = incoming.stats['spe']
        speed_opponent = opponent.stats['spe']
        if battle.weather == "Sun":
            if incoming.ability and incoming.ability.name.lower() == "chlorophyll": speed_incoming *= 2
            if opponent.ability and opponent.ability.name.lower() == "chlorophyll": speed_opponent *= 2
        elif battle.weather == "Rain":
            if incoming.ability and incoming.ability.name.lower() == "swift swim": speed_incoming *= 2
            if opponent.ability and opponent.ability.name.lower() == "swift swim": speed_opponent *= 2

        if speed_incoming > speed_opponent:
            score *= 1.2

        return score
            
    def choose_switch_in(self, battle: Battle, team: List[Pokemon], opponent: Pokemon) -> Pokemon:
        """IA simple pour le switch-in de l'agent RL (prend le premier dispo)."""
        # (L'IA RL n'utilise pas cette fonction, c'est l'env qui gère son switch post-KO)
        return next((p for p in team if not p.is_fainted()), team[0])

# --- IA (RunAndBunAI) - IA Ennemie ---

class RunAndBunAI:
    """Implémente la logique de scoring de l'IA de Run and Bun (basée sur les PDF)."""
    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()

    def choose_action(self, battle: Battle, attacker: Pokemon, opponent: Pokemon, is_perfect_luck: bool = False) -> tuple:
        """Choisit la meilleure ATTAQUE basée sur le scoring du document RnB."""
        best_move = None
        best_score = -float('inf')

        move_damages = {}
        highest_damage = 0
        for move in attacker.moves:
            if move.category.lower() != 'status':
                dmg = compute_damage(battle, attacker, opponent, move, rng_factor=1.0, force_perfect=is_perfect_luck)
                move_damages[move.name] = dmg
                if dmg > highest_damage:
                    highest_damage = dmg
            else:
                move_damages[move.name] = 0

        for move in attacker.moves:
            is_highest_damage = (move_damages[move.name] == highest_damage and highest_damage > 0)
            score = self.get_move_score(battle, attacker, opponent, move, move_damages[move.name], is_highest_damage)
            
            if score > best_score:
                best_score = score
                best_move = move
            elif score == best_score and self.rng.random() < 0.5:
                best_move = move
        
        return ('move', best_move if best_move else attacker.moves[0])

    def get_move_score(self, battle: Battle, attacker: Pokemon, opponent: Pokemon, move: Move, damage: int, is_highest_damage: bool) -> float:
        """Calcule le score d'un mouvement basé sur le doc RnB 1.07 (plus complet)."""
        
        move_name_lower = move.name.lower()
        move_effect_lower = (move.effect or '').lower()
        move_category_lower = move.category.lower()
        status_conditions = ['paralysis', 'burn', 'sleep', 'poison', 'freeze']

        if move_category_lower == 'status' and opponent.status is not None:
             if any(s in move_name_lower or s in move_effect_lower for s in status_conditions):
                 return -20 

        if get_type_multiplier(move.type, opponent.types) == 0 and move_category_lower != 'status':
             return -20

        # Vitesse effective (pour les bonus de statut, etc.)
        speed_attacker = attacker.stats['spe']
        speed_opponent = opponent.stats['spe']
        if battle.weather == "Sun":
            if attacker.ability and attacker.ability.name.lower() == "chlorophyll": speed_attacker *= 2
            if opponent.ability and opponent.ability.name.lower() == "chlorophyll": speed_opponent *= 2
        elif battle.weather == "Rain":
            if attacker.ability and attacker.ability.name.lower() == "swift swim": speed_attacker *= 2
            if opponent.ability and opponent.ability.name.lower() == "swift swim": speed_opponent *= 2
        
        is_faster = speed_attacker > speed_opponent
        can_kill = damage >= opponent.current_hp

        if move_category_lower == 'status':
            score = 6.0 
            if 'stealth rock' in move_name_lower: score = 8.75
            elif 'spikes' in move_name_lower: score = 7.5
            elif 'toxic spikes' in move_name_lower: score = 7.5
            elif 'thunder wave' in move_name_lower:
                score = 8.0 if not is_faster else 7.0 
            elif 'will-o-wisp' in move_name_lower:
                score = 7.0 if opponent.stats['atk'] > opponent.stats['spa'] else 6.0
            elif 'howl' in move_name_lower: score = 6.2 # Gardé bas pour éviter la boucle
            
            setup_moves = ['swords dance', 'dragon dance', 'calm mind', 'shell smash', 'growth'] 
            if any(s in move_name_lower for s in setup_moves):
                 opponent_offensive_moves = [m for m in opponent.moves if m.category.lower() != 'status']
                 if not opponent_offensive_moves:
                     damage_taken = 0
                 else:
                     damage_taken = max(compute_damage(battle, opponent, attacker, opp_move, rng_factor=1.0) for opp_move in opponent_offensive_moves)
                 
                 if attacker.current_hp > damage_taken:
                     score += 1.5
                 else: 
                     score -= 2.0
            return score

        score = 0.0
        if is_highest_damage:
            # --- LOGIQUE 80/20 DU PDF ---
            if self.rng.random() < 0.20: # 20% du temps
                score = 8.0
            else: # 80% du temps
                score = 6.0
            # --- FIN LOGIQUE 80/20 ---

            if get_type_multiplier(move.type, opponent.types) > 1:
                score += 1.5


        if can_kill:
            score += 6.0 if is_faster else 3.0
        
        if not is_highest_damage:
             score = (damage / opponent.max_hp) * 5.0
             if get_type_multiplier(move.type, opponent.types) > 1:
                 score += 1.0

        if 'u-turn' in move_name_lower or 'volt switch' in move_name_lower:
             score += 0.5 
        elif 'knock off' in move_name_lower and opponent.item:
             score += 2.0 

        is_recoil = 'recoil' in move_effect_lower or move_name_lower in ["volt tackle", "head smash", "flare blitz"]
        if is_recoil and (not attacker.ability or attacker.ability.name.lower() != 'rock head'):
            recoil_dmg_percent = (damage / 3) / attacker.max_hp * 100
            if recoil_dmg_percent > 30: 
                score -= 3.0

        return score

    def choose_switch_in(self, battle: Battle, team: List[Pokemon], opponent: Pokemon) -> Pokemon:
        """Choisit le meilleur Pokémon à envoyer après un KO, basé sur le doc Switch Scores."""
        available_pokemon = [p for p in team if not p.is_fainted()]
        if not available_pokemon:
            return team[0] # Ne devrait pas arriver

        best_pokemon = None
        best_score = -float('inf')

        for pokemon in available_pokemon:
            score = self._calculate_switch_score(battle, pokemon, opponent)
            
            if score > best_score:
                best_score = score
                best_pokemon = pokemon
            # En cas d'égalité, le doc dit "premier dans l'ordre de l'équipe", donc on ne met pas à jour
        
        return best_pokemon if best_pokemon else available_pokemon[0]

    def _calculate_switch_score(self, battle: Battle, incoming: Pokemon, opponent: Pokemon) -> float:
        """Calcule le score de switch basé sur le PDF 'Post-KO Switch-in AI'."""
        
        # Calculer vitesse effective
        speed_incoming = incoming.stats['spe']
        speed_opponent = opponent.stats['spe']
        if battle.weather == "Sun":
            if incoming.ability and incoming.ability.name.lower() == "chlorophyll": speed_incoming *= 2
            if opponent.ability and opponent.ability.name.lower() == "chlorophyll": speed_opponent *= 2
        elif battle.weather == "Rain":
            if incoming.ability and incoming.ability.name.lower() == "swift swim": speed_incoming *= 2
            if opponent.ability and opponent.ability.name.lower() == "swift swim": speed_opponent *= 2
        
        is_faster = speed_incoming > speed_opponent

        # Calculer dégâts max infligés et reçus
        max_damage_dealt = 0
        for move in incoming.moves:
            if move.category.lower() == 'status': continue
            dmg = compute_damage(battle, incoming, opponent, move, rng_factor=1.0)
            if dmg > max_damage_dealt:
                max_damage_dealt = dmg
        
        max_damage_taken = 0
        opponent_offensive_moves = [m for m in opponent.moves if m.category.lower() != 'status']
        if opponent_offensive_moves:
            for opp_move in opponent_offensive_moves:
                 dmg = compute_damage(battle, opponent, incoming, opp_move, rng_factor=1.0)
                 if dmg > max_damage_taken:
                     max_damage_taken = dmg
        
        can_ohko_opponent = max_damage_dealt >= opponent.current_hp
        gets_ohkoed = max_damage_taken >= incoming.current_hp
        deals_more_percent_damage = (max_damage_dealt / opponent.max_hp) > (max_damage_taken / incoming.max_hp)

        # Appliquer les règles du PDF
        if is_faster and can_ohko_opponent:
            return 5 # "AI's Pokémon is faster than the player's and OHKO's it."
        
        if not is_faster and can_ohko_opponent and not gets_ohkoed:
            return 4 # "AI's Pokémon is slower... but OHKO's it and is not OHKO'd."
        
        if is_faster and deals_more_percent_damage:
            return 3 # "AI's Pokémon is faster... and deals more damage than it takes (%)"
        
        if not is_faster and deals_more_percent_damage:
            return 2 # "AI's Pokémon is slower... and deals more damage than it takes (%)"
            
        if is_faster:
            return 1 # "AI's Pokémon is faster than the player's."
            
        if not is_faster and gets_ohkoed:
            return -1 # "AI's Pokémon is slower... and is OHKO'd."

        return 0 # "Default."

# --- Classe de Bataille ---

class Battle:
    def __init__(self, team1: List[Pokemon], team2: List[Pokemon], rng: Optional[random.Random] = None, perfect_luck_opponent: bool = False):
        self.team1 = deepcopy(team1)
        self.team2 = deepcopy(team2)
        self.active1 = self.team1[0]
        self.active2 = self.team2[0]
        self.turn = 1
        self.rng = rng or random.Random()
        self.log_entries = []
        self.perfect_luck_opponent = perfect_luck_opponent

        # --- Variables de Climat ---
        self.weather = None 
        self.weather_turns = 0 

        self.ai_team1 = SmartAI(rng=self.rng)
        self.ai_team2 = RunAndBunAI(rng=self.rng)
    
    def log(self, msg: str):
        self.log_entries.append(msg)

    def set_weather(self, weather_type: str, setter: Pokemon):
        """Définit le climat et sa durée, en vérifiant les objets."""
        if self.weather == weather_type:
            return 
            
        base_duration = 5
        item_message = ""
        
        if setter.item:
            item_name_lower = setter.item.name.lower()
            if weather_type == "Sun" and item_name_lower == "heat rock":
                base_duration = 8
                item_message = f" (tenu par {setter.name})!"
            elif weather_type == "Rain" and item_name_lower == "damp rock":
                base_duration = 8
                item_message = f" (tenu par {setter.name})!"
        
        self.weather = weather_type
        self.weather_turns = base_duration
        
        if weather_type == "Sun":
            self.log(f"☀️ Le soleil brille intensément{item_message}")
        elif weather_type == "Rain":
            self.log(f"🌧️ Il commence à pleuvoir{item_message}")

    def run_battle(self):
        """Simule un combat et retourne un rapport détaillé."""
        self.log(f"Début du combat : {self.active1.name} VS {self.active2.name}")
        
        MAX_TURNS = 200 
        
        while not self.battle_over():
            if self.turn > MAX_TURNS:
                self.log(f"Limite de {MAX_TURNS} tours atteinte ! Match nul.")
                winner = 0
                break 
            
            action1 = self.ai_team1.choose_action(
                self, # Passe l'instance 'battle'
                self.active1, self.team1, self.active2, 
                is_perfect_luck=self.perfect_luck_opponent 
            )
            
            action2 = self.ai_team2.choose_action(
                self, # Passe l'instance 'battle'
                self.active2, self.active1,
                is_perfect_luck=self.perfect_luck_opponent 
            )
            
            self.do_turn(action1, action2)

            # --- GESTION DES SWITCHS POST-KO ---
            if self.active1.is_fainted() and not self.battle_over(): 
                # L'IA 1 choisit son remplaçant
                next_pokemon = self.ai_team1.choose_switch_in(self, self.team1, self.active2)
                if next_pokemon:
                     self.active1 = next_pokemon
                     self.log(f"➡️ {self.active1.name} entre en jeu pour l'équipe 1 !")

            if self.active2.is_fainted() and not self.battle_over():
                # L'IA 2 (RunAndBunAI) choisit son remplaçant
                next_pokemon = self.ai_team2.choose_switch_in(self, self.team2, self.active1)
                if next_pokemon:
                    self.active2 = next_pokemon
                    self.log(f"➡️ {self.active2.name} entre en jeu pour l'équipe 2 !")
            # --- FIN GESTION SWITCHS ---
            
            if self.battle_over(): 
                break
            self.turn += 1

        winner = 0
        if self.turn <= MAX_TURNS: # Si ce n'est pas un match nul par limite de tours
            if all(p.is_fainted() for p in self.team2):
                self.log("🎉 Équipe 1 gagne le combat !")
                winner = 1
            elif all(p.is_fainted() for p in self.team1):
                self.log("💀 Équipe 2 gagne le combat !")
                winner = 2

        return {
            "winner": winner,
            "team1_survivors": sum(1 for p in self.team1 if not p.is_fainted()),
            "log": self.log_entries
        }

    def get_next_alive(self, team: List[Pokemon]) -> Optional[Pokemon]:
        for p in team:
            if not p.is_fainted():
                return p
        return None

    def battle_over(self) -> bool:
        return all(p.is_fainted() for p in self.team1) or all(p.is_fainted() for p in self.team2)

    def do_turn(self, action1: tuple, action2: tuple):
        """Exécute un tour complet en gérant la priorité, Quick Claw, et la vitesse météo."""
        p1, p2 = self.active1, self.active2
        act1 , act2  = action1 , action2
        
        p1_moves_first = None # None = use speed, True = p1 via Claw, False = p2 via Claw
        if p1.item and p1.item.name.lower() == "quick claw" and self.rng.random() < 0.2: 
             p1_moves_first = True
             self.log(f"Le Vive Griffe de {p1.name} s'active !")
        elif p2.item and p2.item.name.lower() == "quick claw" and self.rng.random() < 0.2:
             p1_moves_first = False
             self.log(f"Le Vive Griffe de {p2.name} s'active !")
        
        p1.status = None if p1.status == 'flinch' else p1.status
        p2.status = None if p2.status == 'flinch' else p2.status
        
        p1_prio = 1 if action1[0] == 'switch' else (action1[1].priority if action1[0] == 'move' else 0)
        p2_prio = 1 if action2[0] == 'switch' else (action2[1].priority if action2[0] == 'move' else 0)
        
        if p1_prio > p2_prio:
            first_p, second_p = p1, p2
            first_act, second_act = act1, act2
        elif p2_prio > p1_prio:
            first_p, second_p = p2, p1
            first_act, second_act = act2, act1
        elif p1_moves_first is True: # Quick Claw (p1)
             first_p, second_p = p1, p2
             first_act, second_act = act1, act2
        elif p1_moves_first is False: # Quick Claw (p2)
             first_p, second_p = p2, p1
             first_act, second_act = act2, act1
        
        # --- BLOC DE VITESSE (GÈRE MAINTENANT LE CLIMAT) ---
        else:
            speed_p1 = p1.stats['spe']
            speed_p2 = p2.stats['spe']
            
            if self.weather == "Sun":
                if p1.ability and p1.ability.name.lower() == "chlorophyll": speed_p1 *= 2
                if p2.ability and p2.ability.name.lower() == "chlorophyll": speed_p2 *= 2
            elif self.weather == "Rain":
                if p1.ability and p1.ability.name.lower() == "swift swim": speed_p1 *= 2
                if p2.ability and p2.ability.name.lower() == "swift swim": speed_p2 *= 2
            
            if speed_p1 > speed_p2:
                first_p, second_p = p1, p2
                first_act, second_act = act1, act2
            elif speed_p2 > speed_p1:
                first_p, second_p = p2, p1
                first_act, second_act = act2, act1
            else: # Égalité de vitesse
                if self.rng.random() < 0.5:
                    first_p, second_p = p1, p2
                    first_act, second_act = act1, act2
                else:
                    first_p, second_p = p2, p1
                    first_act, second_act = act2, act1
        # --- FIN BLOC VITESSE ---

        res1 = {}
        if first_p.status == 'flinch':
             self.log(f"{first_p.name} est apeuré et ne peut pas bouger !")
             first_p.status = None 
        else:
             res1 = self.execute_action(first_p, second_p, first_act) 
             if res1.get('potential_flinch'):
                   second_p.status = 'flinch'

        if second_p.is_fainted():
            self.handle_end_of_turn()
            return 

        actor_for_second_action = second_p
        defender_for_second = self.active2 if actor_for_second_action == self.active1 else self.active1
        is_still_active = (actor_for_second_action == self.active1) or (actor_for_second_action == self.active2)

        res2 = {}
        if actor_for_second_action.is_fainted():
             pass 
        elif actor_for_second_action.status == 'flinch':
             self.log(f"{actor_for_second_action.name} est apeuré et ne peut pas bouger !")
             actor_for_second_action.status = None 
        elif not is_still_active:
             pass 
        else:
             res2 = self.execute_action(actor_for_second_action, defender_for_second, second_act)
             if res2.get('potential_flinch'):
                   defender_for_second.status = 'flinch'
                   
        self.handle_end_of_turn()

    def execute_action(self, user: Pokemon, target: Pokemon, action: tuple) -> Dict:
        """Exécute une action (attaque ou switch) et affiche le résultat."""
        res = {}
        action_type, obj = action
        if action_type == 'switch':
            
            incoming_pokemon = obj 
            
            if user == self.active1: self.active1 = incoming_pokemon
            else: self.active2 = incoming_pokemon
            self.log(f"{user.name} est rappelé ! {incoming_pokemon.name} entre sur le terrain !")

            # Déclencher les talents météo
            if incoming_pokemon.ability and hasattr(incoming_pokemon.ability, 'name'):
                 ability_name_lower = incoming_pokemon.ability.name.lower()
                 if ability_name_lower == "drought":
                      self.set_weather("Sun", incoming_pokemon)
                 elif ability_name_lower == "drizzle":
                      self.set_weather("Rain", incoming_pokemon)

        elif action_type == 'move':
            self.log(f"{user.name} utilise {obj.name} !")
            # Passe 'self' (l'instance Battle) à apply_move
            res = apply_move(self, user, target, obj, self.rng) 
            
            if res.get('failed'): self.log("Mais cela a échoué !")
            elif not res['hit']: self.log("L'attaque a manqué !")
            else:
                # Log dégâts
                hp_left_target = max(0, target.current_hp)
                self.log(f"➡️ {target.name} perd {res['damage']} PV. (Il lui reste {hp_left_target}/{target.max_hp} PV)")

                # Log Effets
                if res.get('effect') == 'burn': self.log(f"🔥 {target.name} a été brûlé !")
                if res.get('effect') == 'focus_sash': self.log(f"La Ceinture Force de {target.name} lui permet de tenir !")
                if res.get('effect') == 'focus_band': self.log(f"Le Bandeau de {target.name} lui permet de tenir !")

                # Log Recul
                if res.get('recoil_damage'):
                     hp_left_user = max(0, user.current_hp)
                     self.log(f"💥 {user.name} subit {res['recoil_damage']} PV de recul. (Il lui reste {hp_left_user}/{user.max_hp} PV)")
                     if user.is_fainted():
                         self.log(f"💀 {user.name} est KO à cause du recul !")

                # Log KO (vérifier après le recul)
                if not user.is_fainted() and target.is_fainted():
                    self.log(f"💀 {target.name} est KO !")

        return res
        
    def handle_end_of_turn(self):
        """Gère les effets de fin de tour comme Leftovers, statut, Baies ET CLIMAT."""
        for p in [self.active1, self.active2]:
            if p.is_fainted():
                continue
                
            berry_consumed = False
            if p.item:
                 item_name = p.item.name
                 # Baies de Guérison de Statut
                 STATUS_HEAL_BERRIES = {
                      "Burnt Berry": "freeze", "Ice Berry": "burn",
                      "Mint Berry": "sleep", "PRZ Cure Berry": "paralysis",
                      "PSN Cure Berry": "poison", "Miracle Berry": "any" 
                 }
                 if item_name in STATUS_HEAL_BERRIES and p.status is not None:
                     heal_condition = STATUS_HEAL_BERRIES[item_name]
                     if heal_condition == "any" or p.status == heal_condition:
                          self.log(f"La {item_name} de {p.name} guérit son statut {p.status} !")
                          p.status = None
                          berry_consumed = True

                 # Baies de Restauration PV (Gen 2)
                 elif item_name == "Berry Juice" and p.current_hp <= p.max_hp / 2:
                      heal_amount = 20
                      p.current_hp = min(p.max_hp, p.current_hp + heal_amount)
                      self.log(f"Le Jus de Baie de {p.name} lui rend {heal_amount} PV.")
                      berry_consumed = True
                 elif item_name == "Gold Berry" and p.current_hp <= p.max_hp / 2:
                      heal_amount = 30
                      p.current_hp = min(p.max_hp, p.current_hp + heal_amount)
                      self.log(f"La Baie Dorée de {p.name} lui rend {heal_amount} PV.")
                      berry_consumed = True

            if berry_consumed:
                 p.item = None 
            
            # Effet Leftovers
            elif p.item and p.item.name.lower() == "leftovers":
                if p.current_hp < p.max_hp:
                    heal_amount = max(1, p.max_hp // 16)
                    p.current_hp = min(p.max_hp, p.current_hp + heal_amount)
                    self.log(f"Restes de {p.name} lui rend {heal_amount} PV.")
            
            # Effet de statut (Brûlure, Poison)
            if p.status == 'burn':
                burn_dmg = max(1, p.max_hp // 16) 
                p.current_hp = max(0, p.current_hp - burn_dmg)
                self.log(f"La brûlure inflige {burn_dmg} PV à {p.name}.")
                if p.is_fainted():
                    self.log(f"💀 {p.name} est KO à cause de la brûlure !")
            elif p.status == 'poison':
                 poison_dmg = max(1, p.max_hp // 16) 
                 p.current_hp = max(0, p.current_hp - poison_dmg)
                 self.log(f"Le poison inflige {poison_dmg} PV à {p.name}.")
                 if p.is_fainted():
                     self.log(f"💀 {p.name} est KO à cause du poison !")
        
        # Gestion du Climat
        if self.weather is not None:
            self.weather_turns -= 1
            if self.weather_turns <= 0:
                if self.weather == "Sun": self.log("Le soleil s'est couché.")
                elif self.weather == "Rain": self.log("La pluie s'est arrêtée.")
                self.weather = None
                self.weather_turns = 0

# --- Chargement des Équipes (Showdown Parser) ---

def normalize_stat_key(key: str) -> str:
    """Normalise les abréviations de stats (ex: 'at' -> 'atk')."""
    key = key.lower()
    mapping = {
        'hp': 'hp', 'at': 'atk', 'atk': 'atk', 'df': 'def', 'def': 'def',
        'sa': 'spa', 'sl': 'spa', 'spa': 'spa', 'sd': 'spd', 'spd': 'spd',
        'sp': 'spe', 'spe': 'spe',
    }
    return mapping.get(key, key)

def showdown_loader(filepath: str) -> List[Pokemon]:
    """Charge une équipe depuis un fichier texte au format Showdown."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pokemon_blocks = content.strip().split('\n\n')
    team = []
    
    for block in pokemon_blocks:
        lines = block.strip().split('\n')
        if not lines:
            continue
        
        parsed = {"moves": [], "ivs": {}, "evs": {}}
        
        name_line = lines.pop(0)
        if '@' in name_line:
            name, item_name = name_line.split(' @ ')
            parsed["name"] = name.strip()
            parsed["item"] = item_name.strip()
        else:
            parsed["name"] = name_line.strip()
            parsed["item"] = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('- '):
                parsed["moves"].append(line[2:].strip().replace('[', '').replace(']', ''))
            elif line.startswith('Ability: '):
                parsed["ability"] = line.replace('Ability: ', '').strip()
            elif line.startswith('Level: '):
                parsed["level"] = int(line.replace('Level: ', ''))
            elif line.startswith('IVs: '):
                for iv_stat in line.replace('IVs: ', '').split(' / '):
                    val, key = iv_stat.split(' ')
                    parsed["ivs"][key] = int(val)
            elif line.startswith('EVs: '):
                 for ev_stat in line.replace('EVs: ', '').split(' / '):
                    val, key = ev_stat.split(' ')
                    parsed["evs"][key] = int(val)
            elif ' Nature' in line:
                parsed["nature"] = line.replace(' Nature', '').strip()

        species = get_species(parsed["name"])
        if not species:
            raise ValueError(f"Espèce Pokémon introuvable : {parsed['name']}")
        
        moves_list = []
        for move_name in parsed["moves"]:
            move_data = get_move(move_name)
            if not move_data:
                print(f"AVERTISSEMENT: Attaque introuvable : {move_name}. Ignorée.")
                continue
            moves_list.append(Move(name=move_name, data=move_data))
        
        item = None
        if parsed.get("item"):
            item_data = get_item(parsed["item"])
            item = Item(name=parsed["item"], data=item_data)
        
        ability_name = parsed.get("ability", species.get("abilities", {}).get("0"))
        ability_data = get_ability(ability_name)
        ability = Ability(name=ability_name, data=ability_data)

        # Statistiques
        base_bs = species.get("bs", {})
        STAT_KEYS = ["hp", "atk", "def", "spa", "spd", "spe"]
        
        normalized_base = {key: 0 for key in STAT_KEYS}
        if base_bs and isinstance(base_bs, dict):
            for raw_key, value in base_bs.items():
                clean_key = normalize_stat_key(raw_key) 
                if clean_key in STAT_KEYS and isinstance(value, (int, float)):
                    normalized_base[clean_key] = max(normalized_base.get(clean_key, 0), int(value))

        ivs = {key: 31 for key in STAT_KEYS}
        parsed_ivs = parsed.get("ivs", {}) or {}
        for k, v in parsed_ivs.items():
            key = normalize_stat_key(k)
            if key in ivs: ivs[key] = int(v)

        evs = {key: 0 for key in STAT_KEYS} # Par défaut à 0
        parsed_evs = parsed.get("evs", {}) or {}
        for k, v in parsed_evs.items():
            key = normalize_stat_key(k)
            if key in evs: evs[key] = int(v)
        
        level = int(parsed.get("level", 50))
        nature_name = parsed.get("nature", None)
        nature_mults = {key: 1.0 for key in STAT_KEYS}

        if nature_name:
            nature_data = get_nature_data(nature_name) 
            if nature_data and isinstance(nature_data, list) and len(nature_data) == 2:
                plus_stat_raw, minus_stat_raw = nature_data
                if plus_stat_raw != minus_stat_raw:
                    plus_stat = normalize_stat_key(plus_stat_raw)
                    minus_stat = normalize_stat_key(minus_stat_raw)
                    if plus_stat in nature_mults: nature_mults[plus_stat] = 1.1
                    if minus_stat in nature_mults: nature_mults[minus_stat] = 0.9

        final_stats = {}
        for stat, base in normalized_base.items():
            final_stats[stat] = calculate_stat(
                base=base,
                iv=ivs.get(stat, 31), # Utiliser .get avec défaut
                ev=evs.get(stat, 0), # Utiliser .get avec défaut
                level=level,
                nature_multiplier=nature_mults[stat],
                is_hp=(stat == 'hp')
            )
        
        team.append(Pokemon(
            name=parsed["name"],
            level=level,
            moves=moves_list,
            item=item,
            ability=ability,
            stats=final_stats,
            types=species.get("types", ["Normal"])
        ))
    
    return team

# --- Planificateur de Scénarios (pour l'ancienne simulation) ---

class ScenarioPlanner:
    def __init__(self, team1_base, team2_base, num_sims):
        self.team1_base = team1_base
        self.team2_base = team2_base
        self.num_sims = num_sims
        self.normal_results = []
        self.pessimistic_results = [] # On le garde, mais on ne l'utilise plus

    def run_simulations(self):
        print(f"\n⏳ Lancement de {self.num_sims} simulations en arrière-plan...")
        for i in range(self.num_sims):
            battle_norm = Battle(self.team1_base, self.team2_base, perfect_luck_opponent=False)
            self.normal_results.append(battle_norm.run_battle())
            
        print("✅ Simulations terminées.")

    def analyze_and_display(self):
        print("\n" + "="*50)
        print("📊 ANALYSE DES SCÉNARIOS DE COMBAT 📊")
        print("="*50)

        self._display_scenario(
            title="🏆 Scénario 1 : Victoire à tout prix",
            results=self.normal_results,
            condition=lambda r: r["winner"] == 1
        )

        self._display_scenario(
            title="✨ Scénario 2 : Victoire Parfaite (0 KO)",
            results=self.normal_results,
            condition=lambda r: r["winner"] == 1 and r["team1_survivors"] == len(self.team1_base)
        )

    def _display_scenario(self, title, results, condition):
        print("\n" + "-"*50)
        print(title)
        print("-"*50)

        successful_sims = [r for r in results if condition(r)]
        win_rate = (len(successful_sims) / self.num_sims * 100) if self.num_sims > 0 else 0

        print(f"📈 Probabilité de succès : {win_rate:.1f}%")
        
        if successful_sims:
            print("\n📜 Exemple de déroulement du combat :")
            log_to_show = successful_sims[0]["log"]
            for line in log_to_show:
                print(f"   {line}")
        else:
            print("\n❌ Ce scénario ne s'est produit dans aucune simulation.")

# --- Analyseur d'Objets (pour l'ancienne simulation) ---

ITEMS_TO_TEST = [
    "Life Orb", "Focus Sash", "Choice Scarf", 
    "Choice Specs", "Leftovers", "Expert Belt"
]

def suggest_item_changes(team1_base: List[Pokemon], team2_base: List[Pokemon], sims_per_item: int = 30):
    """Suggère la MEILLEURE optimisation d'objet."""
    print("\n" + "="*50)
    print("💡 ANALYSE ET SUGGESTIONS D'OBJETS 💡")
    print("="*50)
    print(f"Analyse en cours avec {sims_per_item} simulations par objet...\n")

    has_suggestion = False
    for pokemon in team1_base:
        if "Mega" in pokemon.name:
            print(f"ⓘ  Analyse d'objet ignorée pour {pokemon.name} (Méga-Évolution).")
            continue

        original_item_name = pokemon.item.name if pokemon.item else "Aucun"
        
        baseline_wins = 0
        for _ in range(sims_per_item):
            battle = Battle(deepcopy(team1_base), deepcopy(team2_base))
            result = battle.run_battle()
            if result["winner"] == 1:
                baseline_wins += 1
        baseline_win_rate = (baseline_wins / sims_per_item) * 100

        best_item_name = ""
        best_win_rate = baseline_win_rate
        
        for item_name in ITEMS_TO_TEST:
            if item_name.lower() == original_item_name.lower():
                continue

            test_team = deepcopy(team1_base)
            pokemon_to_change = next(p for p in test_team if p.name == pokemon.name)
            
            item_data = get_item(item_name)
            pokemon_to_change.item = Item(name=item_name, data=item_data)
            
            new_wins = 0
            for _ in range(sims_per_item):
                battle = Battle(deepcopy(test_team), deepcopy(team2_base))
                result = battle.run_battle()
                if result["winner"] == 1:
                    new_wins += 1
            new_win_rate = (new_wins / sims_per_item) * 100

            if new_win_rate > best_win_rate:
                best_win_rate = new_win_rate
                best_item_name = item_name

        improvement = best_win_rate - baseline_win_rate
        if improvement > 10: 
            print(f"➡️ Pour {pokemon.name}:")
            print(f"   En remplaçant '{original_item_name}' par '{best_item_name}',")
            print(f"   le taux de victoire estimé passe de {baseline_win_rate:.0f}% à {best_win_rate:.0f}% (+{improvement:.0f}%).\n")
            has_suggestion = True

    if not has_suggestion:
        print("Aucune suggestion d'objet n'a montré d'amélioration significative. Votre équipement semble déjà optimal !")

# --- Point d'Entrée Principal (Non utilisé par l'IA RL) ---

def main_cli():
    parser = argparse.ArgumentParser(description="Analyseur de scénarios Pokémon")
    parser.add_argument('--team1', type=str, default='myteam.txt', help="Fichier de votre équipe")
    parser.add_argument('--team2', type=str, default='oppteam.txt', help="Fichier de l'équipe adverse") 
    parser.add_argument('--sims', type=int, default=200, help="Nombre de simulations par scénario")
    args = parser.parse_args()

    try:
        team1 = showdown_loader(args.team1)
        team2 = showdown_loader(args.team2)
    except Exception as e:
        print(f"❌ Erreur lors du chargement des équipes : {e}")
        return

    # suggest_item_changes(team1, team2) # Désactivé pour la vitesse

    planner = ScenarioPlanner(team1, team2, args.sims)
    planner.run_simulations()
    planner.analyze_and_display()

if __name__ == "__main__":
    # N'est exécuté que si on lance 'python pokemon_simulator.py'
    main_cli()