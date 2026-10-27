"""Rule-based analysis: simulate many AI-vs-AI battles between two teams.

Team 1 is played by SmartAI (a heuristic AI), team 2 by the Run & Bun AI.
Prints the win rate, an example battle, and optionally held item suggestions.

Usage: python run_simulation_cli.py [--team1 FILE] [--team2 FILE] [--sims N] [--suggest-items]
"""
import argparse
import os

from pokemon_simulator import ScenarioPlanner, showdown_loader, suggest_item_changes

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="Pokemon battle scenario analyser (rule-based)")
    parser.add_argument("--team1", default=os.path.join(ROOT, "teams", "my_team.txt"), help="Your team (Showdown format)")
    parser.add_argument("--team2", default=os.path.join(ROOT, "teams", "opponent_team.txt"), help="Opponent team (Showdown format)")
    parser.add_argument("--sims", type=int, default=200, help="Number of simulated battles")
    parser.add_argument("--suggest-items", action="store_true", help="Also test alternative held items (slower)")
    args = parser.parse_args()

    try:
        team1 = showdown_loader(args.team1)
        team2 = showdown_loader(args.team2)
    except Exception as e:
        print(f"Could not load the teams: {e}")
        return

    if args.suggest_items:
        suggest_item_changes(team1, team2)

    planner = ScenarioPlanner(team1, team2, args.sims)
    planner.run_simulations()
    planner.analyze_and_display()


if __name__ == "__main__":
    main()
