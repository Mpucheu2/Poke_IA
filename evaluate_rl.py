import time
from sb3_contrib import MaskablePPO as PPO
from pokemon_env import *

# 1. Charger l'environnement
# Mettre render_mode="human" pour voir le déroulement dans la console
env = PokemonEnv(team1_file="myteam.txt", team2_file="oppteam.txt", render_mode="human")

# 2. Charger le modèle entraîné
# Assure-toi que le chemin correspond à celui où le modèle a été sauvegardé
model_path = "./rl_logs/ppo_pokemon_agent"
model = PPO.load(model_path, env=env)

# 3. Lancer des combats d'évaluation
num_episodes = 5 # Nombre de combats à regarder
for episode in range(num_episodes):
    print(f"\n--- Combat d'évaluation #{episode+1} ---")
    obs, info = env.reset()
    terminated, truncated = False, False
    total_reward = 0
    turn_count = 0
    while not terminated and not truncated:
        turn_count += 1
        print(f"\n--- Tour {turn_count} ---")
        # deterministic=True : l'IA choisit la meilleure action qu'elle connaît
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render() # Affiche l'état après l'action
        total_reward += reward
        time.sleep(0.5) # Pause pour pouvoir lire les logs
        # input("Appuyez sur Entrée pour le tour suivant...") # Décommente pour avancer pas à pas

    winner = "Agent RL (Équipe 1)" if total_reward > 0 else "RunAndBunAI (Équipe 2) ou Nul" # Approximation basée sur la récompense
    print(f"Fin du combat #{episode+1}. Récompense totale: {total_reward:.2f}. Vainqueur (estimé): {winner}")
    time.sleep(2) # Pause avant le prochain combat

env.close()