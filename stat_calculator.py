import math


def calculate_stat(base: int, iv: int, ev: int, level: int, nature_multiplier: float, is_hp: bool) -> int:
    """Compute a final stat with the official formula (Gen 3 onwards)."""
    if is_hp:
        # HP uses its own formula and is not affected by nature
        if base == 1:  # Shedinja
            return 1
        return math.floor(((2 * base + iv + (ev // 4)) * level) / 100) + level + 10

    stat = math.floor(((2 * base + iv + (ev // 4)) * level) / 100) + 5
    return math.floor(stat * nature_multiplier)
