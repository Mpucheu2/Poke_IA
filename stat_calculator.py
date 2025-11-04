import math

def calculate_stat(base: int, iv: int, ev: int, level: int, nature_multiplier: float, is_hp: bool) -> int:
    """
    Calcule la statistique finale d'un Pokémon en utilisant la formule officielle.
    """
    if is_hp:
        # La formule pour les PV est différente et n'utilise pas de nature.
        if base == 1:  # Cas spécial pour Shedinja
            return 1
        return math.floor(((2 * base + iv + (ev // 4)) * level) / 100) + level + 10
    else:
        # Formule pour Atk, Def, SpA, SpD, Spe
        stat = math.floor(((2 * base + iv + (ev // 4)) * level) / 100) + 5
        return math.floor(stat * nature_multiplier)