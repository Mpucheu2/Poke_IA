from data_loader import get_item, get_move, get_nature_data, get_species


def test_species_are_found():
    for name in ["Charizard", "Pikachu", "Mewtwo", "Landorus-Therian"]:
        assert get_species(name) is not None, name


def test_move_data():
    move = get_move("Flamethrower")
    assert move["type"] == "Fire"
    assert move["bp"] == 90


def test_lookup_ignores_case_spaces_and_punctuation():
    assert get_move("dragon claw") == get_move("Dragon-Claw")
    assert get_item("kings rock") == get_item("King's Rock")


def test_nature():
    assert get_nature_data("Adamant") == ["atk", "spa"]


def test_unknown_names_return_none():
    assert get_species("Not A Pokemon") is None
    assert get_move("") is None
