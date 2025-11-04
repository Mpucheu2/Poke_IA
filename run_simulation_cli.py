import argparse
from copy import deepcopy
from typing import List

# Importe les composants nécessaires depuis le moteur de simulation
try:
    from pokemon_simulator import (
        Pokemon, Battle, showdown_loader, 
        ScenarioPlanner, suggest_item_changes, ITEMS_TO_TEST
    )
except ImportError as e:
    print(f"Erreur d'importation: {e}")
    print("Assurez-vous que pokemon_simulator.py est dans le même dossier.")
    exit()


def main_cli():
    """Point d'entrée pour la simulation basée sur les règles (l'ancienne simulation)."""
    
    parser = argparse.ArgumentParser(description="Analyseur de scénarios Pokémon (Basé sur les règles)")
    parser.add_argument('--team1', type=str, default='myteam.txt', help="Fichier de votre équipe")
    parser.add_argument('--team2', type=str, default='oppteam.txt', help="Fichier de l'équipe adverse")
    parser.add_argument('--sims', type=int, default=200, help="Nombre de simulations par scénario")
    args = parser.parse_args()

    print("--- Lancement du Simulateur Basé sur les Règles ---")

    try:
        print(f"Chargement de l'équipe 1: {args.team1}")
        team1 = showdown_loader(args.team1)
        print(f"Chargement de l'équipe 2: {args.team2}")
        team2 = showdown_loader(args.team2)
        print("Équipes chargées avec succès.")
    except Exception as e:
        print(f"❌ Erreur lors du chargement des équipes : {e}")
        return

    # Exécuter l'analyse des objets
    suggest_item_changes(team1, team2)

    # Exécuter le planificateur de scénarios
    planner = ScenarioPlanner(team1, team2, args.sims)
    planner.run_simulations()
    planner.analyze_and_display()

if __name__ == "__main__":
    main_cli()