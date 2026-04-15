from src.pipelines.scan_props import _pick_sharp_pair, _pick_best_soft_line


SHARP = ['pinnacle', 'circasports']


def test_sharp_pair_prefers_pinnacle():
    line_data = {
        'pinnacle':    {'over': 1.95, 'under': 1.95},
        'circasports': {'over': 1.90, 'under': 2.00},
        'draftkings':  {'over': 2.10, 'under': 1.77},
    }
    assert _pick_sharp_pair(line_data, SHARP) == (1.95, 1.95, 'pinnacle')


def test_sharp_pair_falls_back_to_circa():
    line_data = {
        'circasports': {'over': 1.90, 'under': 2.00},
        'draftkings':  {'over': 2.10, 'under': 1.77},
    }
    assert _pick_sharp_pair(line_data, SHARP) == (1.90, 2.00, 'circasports')


def test_sharp_pair_none_when_no_sharp_book():
    line_data = {
        'draftkings': {'over': 2.10, 'under': 1.77},
        'fanduel':    {'over': 2.05, 'under': 1.80},
    }
    assert _pick_sharp_pair(line_data, SHARP) is None


def test_sharp_pair_ignores_one_sided_quote():
    # Pinnacle only has over; should fall through to circa.
    line_data = {
        'pinnacle':    {'over': 1.95},
        'circasports': {'over': 1.90, 'under': 2.00},
    }
    assert _pick_sharp_pair(line_data, SHARP) == (1.90, 2.00, 'circasports')


def test_best_soft_line_excludes_sharp_books():
    line_data = {
        'pinnacle':    {'over': 2.20, 'under': 1.80},  # should be excluded
        'draftkings':  {'over': 2.10, 'under': 1.77},
        'fanduel':     {'over': 2.05, 'under': 1.85},
    }
    best = _pick_best_soft_line(line_data, SHARP)
    assert best['over'] == (2.10, 'draftkings')
    assert best['under'] == (1.85, 'fanduel')


def test_best_soft_line_empty_when_only_sharp_books():
    line_data = {
        'pinnacle': {'over': 1.95, 'under': 1.95},
    }
    best = _pick_best_soft_line(line_data, SHARP)
    assert best['over'] == (None, None)
    assert best['under'] == (None, None)
