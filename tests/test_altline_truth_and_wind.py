"""Unit tests for the P0 audit fixes:

- `_shift_anchor_truth`: alt-line truth must stay tethered to the sharp anchor
  (it used to be the raw model prob, which let the model grade its own homework).
- `signed_wind_in_mph`: revives the previously-dead `wind_in_mph` HR signal and
  must carry the correct in(+)/out(-) sign.
"""
import pytest

from src.pipelines.scan_props import _shift_anchor_truth
from src.clients.weather import signed_wind_in_mph


class TestShiftAnchorTruth:
    def test_identity_at_anchor(self):
        # When the model prob at the alt-line equals the model prob at the anchor,
        # the shifted truth must equal the sharp anchor probability exactly.
        over, under = _shift_anchor_truth(0.60, 0.40, model_anchor_over=0.55,
                                          model_alt_over=0.55)
        assert over == pytest.approx(0.60, abs=1e-9)
        assert under == pytest.approx(0.40, abs=1e-9)

    def test_shift_follows_model_but_is_not_the_raw_model_prob(self):
        # Model says the over is much more likely at the alt-line (0.55 -> 0.75).
        # Truth moves UP from the sharp anchor 0.60 in the model's direction, but
        # it must NOT just become the raw model 0.75 — that was the bug (model
        # grading itself). It's an odds-ratio shift of the sharp anchor.
        over, under = _shift_anchor_truth(0.60, 0.40, model_anchor_over=0.55,
                                          model_alt_over=0.75)
        assert over > 0.60                      # moved with the model
        assert over != pytest.approx(0.75)      # but is sharp-anchored, not model
        assert over + under == pytest.approx(1.0, abs=1e-9)

    def test_reduces_to_model_only_when_sharp_agreed_at_anchor(self):
        # When sharp == model at the anchor, the carried disagreement is zero, so
        # the shifted truth equals the model prob at the alt-line.
        over, under = _shift_anchor_truth(0.55, 0.45, model_anchor_over=0.55,
                                          model_alt_over=0.75)
        assert over == pytest.approx(0.75, abs=1e-9)

    def test_normalized_to_one(self):
        over, under = _shift_anchor_truth(0.58, 0.42, 0.50, 0.30)
        assert over + under == pytest.approx(1.0, abs=1e-9)
        assert over < 0.58  # model pushed the over down

    def test_degenerate_anchor_returns_none(self):
        assert _shift_anchor_truth(0.6, 0.4, 0.0, 0.5) == (None, None)
        assert _shift_anchor_truth(0.6, 0.4, 1.0, 0.5) == (None, None)


class TestSignedWindInMph:
    def test_blowing_in_is_positive(self):
        # Wrigley outfield bearing 200; wind FROM 200 blows straight in.
        assert signed_wind_in_mph(15.0, 200.0, 200.0) == pytest.approx(15.0, abs=1e-6)

    def test_blowing_out_is_negative(self):
        # Wind from the opposite direction (20°) blows out -> negative.
        assert signed_wind_in_mph(15.0, 20.0, 200.0) == pytest.approx(-15.0, abs=1e-6)

    def test_crosswind_near_zero(self):
        # 90° off the bearing -> no in/out component.
        assert signed_wind_in_mph(15.0, 290.0, 200.0) == pytest.approx(0.0, abs=1e-6)

    def test_bad_input_is_neutral(self):
        assert signed_wind_in_mph(None, 200.0, 200.0) == 0.0
        assert signed_wind_in_mph(15.0, float("nan"), 200.0) == 0.0
