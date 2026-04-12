import pytest
from src.data.park_factors import compute_weather_adjustments, get_park_factor, STADIUM_META


# --- Helpers ---

def _open_stadium():
    """Wrigley Field meta - open air, high wind sensitivity."""
    return STADIUM_META["Wrigley Field"]


def _dome_stadium():
    """Tropicana Field meta - dome, zero wind sensitivity."""
    return STADIUM_META["Tropicana Field"]


def _retractable_stadium():
    """Globe Life Field meta - retractable roof, low wind sensitivity."""
    return STADIUM_META["Globe Life Field"]


# --- Temperature tests ---

class TestTemperatureAdjustments:
    def test_hot_weather_boosts_hr(self):
        weather = {"temp_f": 95.0, "wind_mph": 0, "wind_deg": 0}
        adj = compute_weather_adjustments(weather, _open_stadium())
        assert adj["hr"] > 1.0
        assert adj["runs"] > 1.0
        assert adj["hits"] > 1.0

    def test_cold_weather_suppresses_hr(self):
        weather = {"temp_f": 45.0, "wind_mph": 0, "wind_deg": 0}
        adj = compute_weather_adjustments(weather, _open_stadium())
        assert adj["hr"] < 1.0
        assert adj["runs"] < 1.0
        assert adj["hits"] < 1.0

    def test_cold_weather_boosts_strikeouts(self):
        weather = {"temp_f": 45.0, "wind_mph": 0, "wind_deg": 0}
        adj = compute_weather_adjustments(weather, _open_stadium())
        assert adj["so"] > 1.0

    def test_neutral_temp_no_effect(self):
        weather = {"temp_f": 72.0, "wind_mph": 0, "wind_deg": 0}
        adj = compute_weather_adjustments(weather, _open_stadium())
        assert adj["hr"] == 1.0
        assert adj["runs"] == 1.0
        assert adj["hits"] == 1.0
        assert adj["so"] == 1.0

    def test_temp_effect_is_capped(self):
        """Even at extreme temps, adjustment shouldn't exceed the ±30F cap."""
        weather_extreme = {"temp_f": 120.0, "wind_mph": 0, "wind_deg": 0}
        weather_capped = {"temp_f": 102.0, "wind_mph": 0, "wind_deg": 0}  # 72 + 30
        adj_extreme = compute_weather_adjustments(weather_extreme, _open_stadium())
        adj_capped = compute_weather_adjustments(weather_capped, _open_stadium())
        assert adj_extreme["hr"] == pytest.approx(adj_capped["hr"], abs=0.001)


# --- Wind tests ---

class TestWindAdjustments:
    def test_wind_blowing_out_boosts_offense(self):
        """Wind from behind home plate toward outfield = blowing out."""
        meta = _open_stadium()
        # Wrigley outfield_bearing = 200. Wind blowing OUT means wind comes from
        # opposite direction: 200 + 180 = 380 -> 20 degrees
        weather = {"temp_f": 72.0, "wind_mph": 15.0, "wind_deg": 20}
        adj = compute_weather_adjustments(weather, meta)
        assert adj["hr"] > 1.0
        assert adj["runs"] > 1.0

    def test_wind_blowing_in_suppresses_offense(self):
        """Wind from outfield toward home plate = blowing in."""
        meta = _open_stadium()
        # Wrigley outfield_bearing = 200. Wind FROM 200 = blowing in.
        weather = {"temp_f": 72.0, "wind_mph": 15.0, "wind_deg": 200}
        adj = compute_weather_adjustments(weather, meta)
        assert adj["hr"] < 1.0
        assert adj["runs"] < 1.0

    def test_crosswind_minimal_effect(self):
        """Wind perpendicular to outfield direction has near-zero effect."""
        meta = _open_stadium()
        # 90 degrees off from blowing-in direction
        crosswind_deg = (200 + 90) % 360  # 290
        weather = {"temp_f": 72.0, "wind_mph": 15.0, "wind_deg": crosswind_deg}
        adj = compute_weather_adjustments(weather, meta)
        assert adj["hr"] == pytest.approx(1.0, abs=0.01)

    def test_no_wind_no_effect(self):
        weather = {"temp_f": 72.0, "wind_mph": 0, "wind_deg": 0}
        adj = compute_weather_adjustments(weather, _open_stadium())
        assert adj["hr"] == 1.0
        assert adj["runs"] == 1.0

    def test_high_sensitivity_stronger_effect(self):
        """Wrigley (0.9 sensitivity) should show larger wind effect than Dodger (0.4)."""
        weather = {"temp_f": 72.0, "wind_mph": 15.0, "wind_deg": 20}
        adj_wrigley = compute_weather_adjustments(weather, STADIUM_META["Wrigley Field"])
        adj_dodger = compute_weather_adjustments(weather, STADIUM_META["Dodger Stadium"])
        wrigley_hr_delta = abs(adj_wrigley["hr"] - 1.0)
        dodger_hr_delta = abs(adj_dodger["hr"] - 1.0)
        assert wrigley_hr_delta > dodger_hr_delta


# --- Dome / roof tests ---

class TestDomeHandling:
    def test_dome_ignores_weather(self):
        weather = {"temp_f": 95.0, "wind_mph": 20.0, "wind_deg": 0}
        adj = compute_weather_adjustments(weather, _dome_stadium())
        assert adj["hr"] == 1.0
        assert adj["runs"] == 1.0
        assert adj["hits"] == 1.0
        assert adj["so"] == 1.0

    def test_retractable_low_wind_sensitivity(self):
        """Retractable roofs have low wind sensitivity even when open."""
        weather = {"temp_f": 72.0, "wind_mph": 15.0, "wind_deg": 30}
        adj = compute_weather_adjustments(weather, _retractable_stadium())
        # Should have some effect but small due to 0.2 sensitivity
        hr_delta = abs(adj["hr"] - 1.0)
        assert hr_delta < 0.03


# --- Integration: get_park_factor with weather ---

class TestGetParkFactorWithWeather:
    def test_weather_modifies_static_factors(self):
        """Hot weather at Coors should push already-high factors even higher."""
        static = get_park_factor("Coors Field")
        hot = get_park_factor("Coors Field", weather={"temp_f": 95.0, "wind_mph": 0, "wind_deg": 0})
        assert hot["hr"] > static["hr"]
        assert hot["runs"] > static["runs"]

    def test_no_weather_returns_static(self):
        static = get_park_factor("Coors Field")
        none_weather = get_park_factor("Coors Field", weather=None)
        assert static == none_weather

    def test_dome_weather_has_no_effect(self):
        static = get_park_factor("Tropicana Field")
        with_weather = get_park_factor(
            "Tropicana Field",
            weather={"temp_f": 100.0, "wind_mph": 20.0, "wind_deg": 0}
        )
        assert static == with_weather
