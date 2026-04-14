from src.pipelines.scan_props import _pick_best_line


def test_picks_max_odds_per_side():
    line_data = {
        'dk': {'over': 1.85, 'under': 1.95},
        'fd': {'over': 1.90, 'under': 1.92},
        'mgm': {'over': 1.88, 'under': 2.00},
    }
    best = _pick_best_line(line_data)
    assert best['over'] == (1.90, 'fd')
    assert best['under'] == (2.00, 'mgm')


def test_handles_missing_side():
    line_data = {
        'dk': {'over': 1.85},
        'fd': {'over': 1.90, 'under': 1.92},
    }
    best = _pick_best_line(line_data)
    assert best['over'] == (1.90, 'fd')
    assert best['under'] == (1.92, 'fd')


def test_handles_all_missing_under():
    line_data = {
        'dk': {'over': 1.85},
        'fd': {'over': 1.90},
    }
    best = _pick_best_line(line_data)
    assert best['over'] == (1.90, 'fd')
    assert best['under'] == (None, None)


def test_single_book():
    line_data = {'dk': {'over': 1.85, 'under': 1.95}}
    best = _pick_best_line(line_data)
    assert best['over'] == (1.85, 'dk')
    assert best['under'] == (1.95, 'dk')
