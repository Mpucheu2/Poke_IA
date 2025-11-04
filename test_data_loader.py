from data_loader import get_species, get_move, get_item, get_ability

names = ["Charizard", "Pikachu", "Mewtwo"]
print("=== Test species ===")
for n in names:
    s = get_species(n)
    print(n, "->", "OK" if s else "NOT FOUND")

print("\n=== Test moves ===")
for m in ["Flamethrower", "Dragon Claw", "Thunderbolt", "Solar Beam"]:
    mv = get_move(m)
    print(f"{m} ->", mv and f"Power={mv['power']} Type={mv['type']} Cat={mv['category']}" or "NOT FOUND")
