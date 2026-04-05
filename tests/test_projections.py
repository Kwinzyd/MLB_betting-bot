import pytest
from src.models.projections import ProjectionModel


@pytest.fixture
def model():
    return ProjectionModel()


def _make_pitcher_logs(n=15, ip=6.0, k=6, er=3):
    """Helper to create mock pitcher game logs."""
    return [
        {
            'date': f'2026-03-{i+1:02d}',
            'innings_pitched': ip,
            'strikeouts': k,
            'earned_runs': er,
            'hits_allowed': 5,
            'runs_allowed': er + 1,
            'walks': 2,
            'home_runs_allowed': 1,
            'pitches_thrown': 95,
        }
        for i in range(n)
    ]


def _make_batter_logs(n=20, hits=1, hr=0, tb=2):
    """Helper to create mock batter game logs."""
    return [
        {
            'date': f'2026-03-{i+1:02d}',
            'at_bats': 4,
            'hits': hits,
            'doubles': 0,
            'triples': 0,
            'home_runs': hr,
            'rbis': 1,
            'walks': 1,
            'strikeouts': 1,
            'total_bases': tb,
            'plate_appearances': 5,
        }
        for i in range(n)
    ]


def test_pitcher_strikeouts_basic(model):
    logs = _make_pitcher_logs(n=15, ip=6.0, k=7)
    # K/9 = (7/6)*9 = 10.5
    proj = model.project_pitcher_strikeouts(logs, opponent_k_rate=0.225, venue="Yankee Stadium", line=5.5)
    assert proj is not None
    assert proj['projected_mean'] > 0
    assert proj['prob_over'] > 0
    assert proj['prob_under'] > 0
    assert abs(proj['prob_over'] + proj['prob_under'] - 1.0) < 0.01


def test_pitcher_strikeouts_high_opp_k_rate(model):
    logs = _make_pitcher_logs(n=15, ip=6.0, k=6)
    # High opponent K rate should increase projection
    proj_high = model.project_pitcher_strikeouts(logs, opponent_k_rate=0.30, venue=None, line=5.5)
    proj_low = model.project_pitcher_strikeouts(logs, opponent_k_rate=0.18, venue=None, line=5.5)
    assert proj_high['projected_mean'] > proj_low['projected_mean']


def test_pitcher_strikeouts_coors_suppression(model):
    logs = _make_pitcher_logs(n=15, ip=6.0, k=6)
    proj_coors = model.project_pitcher_strikeouts(logs, opponent_k_rate=0.225, venue="Coors Field", line=5.5)
    proj_oracle = model.project_pitcher_strikeouts(logs, opponent_k_rate=0.225, venue="Oracle Park", line=5.5)
    # Oracle Park has higher SO factor (1.05) vs Coors (0.95)
    assert proj_oracle['projected_mean'] > proj_coors['projected_mean']


def test_pitcher_insufficient_logs(model):
    logs = _make_pitcher_logs(n=2)
    proj = model.project_pitcher_strikeouts(logs, 0.225, None, 5.5)
    assert proj is None


def test_pitcher_earned_runs(model):
    logs = _make_pitcher_logs(n=15, ip=6.0, er=3)
    # ERA = (3/6)*9 = 4.5
    proj = model.project_pitcher_earned_runs(logs, opponent_runs_per_game=4.5, venue=None, line=2.5)
    assert proj is not None
    assert proj['projected_mean'] > 0


def test_batter_hits_basic(model):
    logs = _make_batter_logs(n=20, hits=1)
    proj = model.project_batter_stat(logs, 'hits', pitcher_hand='R', batter_hand='L', venue=None, line=0.5)
    assert proj is not None
    assert proj['projected_mean'] > 0
    assert proj['market'] == 'batter_hits'


def test_batter_platoon_advantage(model):
    logs = _make_batter_logs(n=20, hits=1)
    # L batter vs R pitcher (favorable) vs L batter vs L pitcher (unfavorable)
    proj_favorable = model.project_batter_stat(logs, 'hits', 'R', 'L', None, 0.5)
    proj_unfavorable = model.project_batter_stat(logs, 'hits', 'L', 'L', None, 0.5)
    assert proj_favorable['projected_mean'] > proj_unfavorable['projected_mean']


def test_batter_switch_hitter_no_adjustment(model):
    logs = _make_batter_logs(n=20, hits=1)
    proj_vs_r = model.project_batter_stat(logs, 'hits', 'R', 'S', None, 0.5)
    proj_vs_l = model.project_batter_stat(logs, 'hits', 'L', 'S', None, 0.5)
    # Switch hitter should get same projection regardless of pitcher hand
    assert abs(proj_vs_r['projected_mean'] - proj_vs_l['projected_mean']) < 0.001


def test_batter_home_runs(model):
    logs = _make_batter_logs(n=20, hr=0, tb=1)
    proj = model.project_batter_stat(logs, 'home_runs', 'R', 'L', "Yankee Stadium", 0.5)
    assert proj is not None
    assert proj['market'] == 'batter_home_runs'


def test_batter_insufficient_logs(model):
    logs = _make_batter_logs(n=3)
    proj = model.project_batter_stat(logs, 'hits', 'R', 'L', None, 0.5)
    assert proj is None
