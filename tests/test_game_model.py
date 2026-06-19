"""Sanity tests for the team run model behind game markets."""
import pytest

from src.models.game_model import (
    project_team_runs,
    moneyline_probabilities,
    total_probabilities,
    run_line_cover_prob,
    game_market_probabilities,
)


class TestMoneyline:
    def test_symmetric_lambdas_favor_home_via_tie_split(self):
        ml = moneyline_probabilities(4.5, 4.5, home_tie_split=0.52)
        assert ml["home"] + ml["away"] == pytest.approx(1.0, abs=1e-9)
        assert ml["home"] > ml["away"]          # extra-innings edge to home
        assert ml["home"] == pytest.approx(0.5, abs=0.02)  # but only slightly

    def test_stronger_team_wins_more(self):
        ml = moneyline_probabilities(5.5, 3.5)
        assert ml["home"] > 0.6


class TestTotal:
    def test_halfline_sums_to_one_no_push(self):
        t = total_probabilities(4.5, 4.5, 8.5)
        assert t["push"] == 0.0
        assert t["over"] + t["under"] == pytest.approx(1.0, abs=1e-9)

    def test_higher_lambda_more_over(self):
        low = total_probabilities(3.5, 3.5, 8.5)["over"]
        high = total_probabilities(5.5, 5.5, 8.5)["over"]
        assert high > low

    def test_integer_line_has_push_and_conditional_sides(self):
        t = total_probabilities(4.5, 4.5, 9)
        assert t["push"] > 0.0
        assert t["over"] + t["under"] == pytest.approx(1.0, abs=1e-9)  # push-conditional


class TestRunLine:
    def test_home_minus_and_away_plus_are_complementary(self):
        ph = run_line_cover_prob(5.0, 4.0, "home", -1.5)
        pa = run_line_cover_prob(5.0, 4.0, "away", 1.5)
        assert ph + pa == pytest.approx(1.0, abs=1e-9)

    def test_covering_minus_1_5_harder_than_winning(self):
        ml = moneyline_probabilities(5.0, 4.0)["home"]
        cover = run_line_cover_prob(5.0, 4.0, "home", -1.5)
        assert cover < ml  # winning by 2+ is strictly harder than winning

    def test_bundle_picks_favorite_line(self):
        out = game_market_probabilities(5.2, 4.0, 9.0, run_line=1.5)
        assert out["run_line"]["home_line"] == -1.5   # home is favorite
        assert out["run_line"]["away_line"] == 1.5
        assert out["run_line"]["home"] + out["run_line"]["away"] == pytest.approx(1.0, abs=1e-9)


class TestProjectTeamRuns:
    def test_strong_offense_weak_pitching_scores_more(self):
        weak_opp = project_team_runs(off_rpg=5.2, opp_starter_runs9=5.5,
                                     opp_starter_ip=5.0, opp_bullpen_era=5.2, park_runs=1.05)
        strong_opp = project_team_runs(off_rpg=5.2, opp_starter_runs9=2.8,
                                       opp_starter_ip=6.5, opp_bullpen_era=3.2, park_runs=1.05)
        assert weak_opp > strong_opp

    def test_lambda_is_clamped(self):
        lam = project_team_runs(off_rpg=12.0, opp_starter_runs9=12.0,
                                opp_starter_ip=2.0, opp_bullpen_era=12.0, park_runs=1.30)
        assert 1.5 <= lam <= 9.0
