import json
import os

# --- Définition du chemin absolu ---
# Cela garantit que "data_from_calc/" est trouvé, peu importe d'où vous lancez le script
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _SCRIPT_DIR = os.path.abspath(os.getcwd())
DATA_PATH = os.path.join(_SCRIPT_DIR, "data_from_calc/")


def load_json(filename):
    """Charge un fichier JSON depuis le dossier data."""
    filepath = os.path.join(DATA_PATH, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERREUR CRITIQUE: Fichier JSON introuvable : {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"ERREUR CRITIQUE: Fichier JSON mal formé : {filepath}")
        return None

# --- Initialisation des Caches ---
_species_cache = None
_moves_cache = None
_items_cache = None
_abilities_cache = None
_natures_cache = None

# --- Helper pour convertir les listes JSON en Dictionnaires ---
def _normalize_json_data(raw_data):
    """Convertit les listes de dicts JSON en un seul dict pour une recherche rapide."""
    data_dict = {}
    if isinstance(raw_data, list):
        for item in raw_data:
            if isinstance(item, dict):
                data_dict.update(item)
    elif isinstance(raw_data, dict):
        data_dict = raw_data
    
    # Nettoie les clés
    normalized_dict = {}
    for key, value in data_dict.items():
        clean_key = key.lower().replace('-', '').replace(' ', '').replace('%', '').replace(':', '').replace("'", "")
        normalized_dict[clean_key] = value
    return normalized_dict

# --- Fonctions de chargement corrigées ---

def get_species(name: str):
    """Charge les espèces depuis species.json et les met en cache."""
    global _species_cache
    if _species_cache is None:
        raw = load_json("species.json")
        if raw is None: return None
        _species_cache = _normalize_json_data(raw)

    clean_name = name.lower().replace('-', '').replace(' ', '').replace('%', '').replace(':', '').replace("'", "")
    
    if clean_name in _species_cache:
        return _species_cache[clean_name]
    
    for key, value in _species_cache.items():
        if value.get('name', '').lower() == name.lower():
            return value
    return None

def get_move(name: str):
    """Charge une attaque depuis moves.json et la met en cache."""
    global _moves_cache
    if _moves_cache is None:
        raw = load_json("moves.json")
        if raw is None: return None
        _moves_cache = _normalize_json_data(raw)

    clean_name = name.lower().replace('-', '').replace(' ', '')
    return _moves_cache.get(clean_name)

def get_item(name: str):
    """Charge un objet depuis items.json et la met en cache."""
    global _items_cache
    if _items_cache is None:
        raw = load_json("items.json")
        if raw is None: return None
        _items_cache = _normalize_json_data(raw)

    clean_name = name.lower().replace('-', '').replace(' ', '')
    return _items_cache.get(clean_name)

def get_ability(name: str):
    """Charge un talent depuis abilities.json et la met en cache."""
    global _abilities_cache
    if _abilities_cache is None:
        raw = load_json("abilities.json")
        if raw is None: return None
        _abilities_cache = _normalize_json_data(raw)

    clean_name = name.lower().replace('-', '').replace(' ', '')
    return _abilities_cache.get(clean_name)

def get_nature_data(name: str):
    """Charge les natures depuis natures.json et la met en cache."""
    global _natures_cache
    if _natures_cache is None:
        raw = load_json("natures.json")
        if raw is None: return None
        
        _natures_cache = {}
        data_dict = {}
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    data_dict.update(item)
        elif isinstance(raw, dict):
            data_dict = raw
        
        for key, value in data_dict.items():
            _natures_cache[key.lower()] = value

    return _natures_cache.get(name.lower())