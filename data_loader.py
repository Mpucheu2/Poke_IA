"""Load Pokemon data (species, moves, items, abilities, natures) from data/.

The JSON files were exported from the Run & Bun damage calculator (see the
`calc` submodule). Each file is loaded once, on first use, then cached.
"""
import json
import os

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

_caches = {}


def load_json(filename):
    """Load a JSON file from the data folder, or return None on error."""
    filepath = os.path.join(DATA_PATH, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: JSON file not found: {filepath}")
    except json.JSONDecodeError:
        print(f"ERROR: malformed JSON file: {filepath}")
    return None


def _clean_key(name):
    """Normalise a name so that "King's Rock", "kings-rock" and "Kings Rock" match."""
    for char in "- %:'":
        name = name.replace(char, "")
    return name.lower()


def _merge(raw_data):
    """The calc exports a list of dicts (one per generation); merge them, later ones win."""
    merged = {}
    if isinstance(raw_data, list):
        for item in raw_data:
            if isinstance(item, dict):
                merged.update(item)
    elif isinstance(raw_data, dict):
        merged = raw_data
    return merged


def _table(filename):
    """Return the cached {clean_name: data} table for a JSON file."""
    if filename not in _caches:
        raw = load_json(filename)
        if raw is None:
            return None
        _caches[filename] = {_clean_key(key): value for key, value in _merge(raw).items()}
    return _caches[filename]


def _lookup(filename, name):
    table = _table(filename)
    if table is None or not name:
        return None
    return table.get(_clean_key(name))


def get_species(name: str):
    return _lookup("species.json", name)


def get_move(name: str):
    return _lookup("moves.json", name)


def get_item(name: str):
    return _lookup("items.json", name)


def get_ability(name: str):
    return _lookup("abilities.json", name)


def get_nature_data(name: str):
    """Return [boosted_stat, lowered_stat] for a nature, e.g. ["atk", "spa"] for Adamant."""
    return _lookup("natures.json", name)
