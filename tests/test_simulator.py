import random
from pathlib import Path

from pokemon_simulator import Battle, get_type_multiplier, showdown_loader
from stat_calculator import calculate_stat

TEAMS = Path(__file__).resolve().parent.parent / "teams"


def test_stat_formula():
    # Garchomp, base 108 HP / 130 Atk, 31 IVs, 252 EVs, level 100, Adamant
    assert calculate_stat(108, 31, 252, 100, 1.0, is_hp=True) == 420
    assert calculate_stat(130, 31, 252, 100, 1.1, is_hp=False) == 394
    assert calculate_stat(1, 31, 252, 100, 1.0, is_hp=True) == 1  # Shedinja


def test_type_chart():
    assert get_type_multiplier("Water", ["Fire"]) == 2
    assert get_type_multiplier("Ground", ["Flying"]) == 0
    assert get_type_multiplier("Ice", ["Dragon", "Flying"]) == 4
    assert get_type_multiplier("Fire", ["Water", "Rock"]) == 0.25


def test_showdown_loader():
    team = showdown_loader(TEAMS / "my_team.txt")
    assert [p.name for p in team] == ["Charizard", "Magnezone", "Swampert", "Gengar", "Landorus-Therian", "Snorlax"]
    charizard = team[0]
    assert charizard.level == 50
    assert charizard.item.name == "Heavy-Duty Boots"
    assert charizard.types == ["Fire", "Flying"]
    assert len(charizard.moves) == 4


def test_battle_always_finishes():
    team1 = showdown_loader(TEAMS / "my_team.txt")
    team2 = showdown_loader(TEAMS / "opponent_team.txt")
    for seed in range(5):
        result = Battle(team1, team2, rng=random.Random(seed)).run_battle()
        assert result["winner"] in (0, 1, 2)
        assert result["log"]


def test_battle_is_reproducible_with_a_seed():
    team1 = showdown_loader(TEAMS / "my_team.txt")
    team2 = showdown_loader(TEAMS / "opponent_team.txt")
    first = Battle(team1, team2, rng=random.Random(42)).run_battle()
    second = Battle(team1, team2, rng=random.Random(42)).run_battle()
    assert first["log"] == second["log"]
