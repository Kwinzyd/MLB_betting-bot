# Static park factors for all 30 MLB stadiums (2024-2025 estimates)
# runs: overall run environment multiplier (1.0 = neutral)
# hr: home run factor
# hits: hits factor
# so: strikeout factor

PARK_FACTORS = {
    # --- American League ---
    "Angel Stadium": {"runs": 0.96, "hr": 1.00, "hits": 0.95, "so": 1.00},
    "Camden Yards": {"runs": 1.05, "hr": 1.12, "hits": 1.02, "so": 0.98},
    "Comerica Park": {"runs": 0.95, "hr": 0.92, "hits": 0.97, "so": 1.02},
    "Fenway Park": {"runs": 1.10, "hr": 0.95, "hits": 1.12, "so": 1.00},
    "Globe Life Field": {"runs": 0.95, "hr": 0.98, "hits": 0.94, "so": 1.02},
    "Guaranteed Rate Field": {"runs": 1.05, "hr": 1.10, "hits": 1.00, "so": 0.98},
    "Kauffman Stadium": {"runs": 0.97, "hr": 0.93, "hits": 1.00, "so": 1.00},
    "Minute Maid Park": {"runs": 1.02, "hr": 1.05, "hits": 1.00, "so": 0.99},
    "Oakland Coliseum": {"runs": 0.90, "hr": 0.88, "hits": 0.92, "so": 1.03},
    "Progressive Field": {"runs": 0.98, "hr": 1.00, "hits": 0.97, "so": 1.01},
    "T-Mobile Park": {"runs": 0.93, "hr": 0.90, "hits": 0.95, "so": 1.03},
    "Target Field": {"runs": 1.00, "hr": 1.02, "hits": 0.99, "so": 1.00},
    "Tropicana Field": {"runs": 0.92, "hr": 0.88, "hits": 0.94, "so": 1.04},
    "Rogers Centre": {"runs": 1.03, "hr": 1.08, "hits": 1.00, "so": 0.99},
    "Yankee Stadium": {"runs": 1.05, "hr": 1.15, "hits": 1.00, "so": 1.00},
    # --- National League ---
    "American Family Field": {"runs": 1.02, "hr": 1.05, "hits": 1.00, "so": 0.99},
    "Busch Stadium":            {"runs": 0.95, "hr": 0.92, "hits": 0.96, "so": 1.02},
    "Chase Field": {"runs": 1.06, "hr": 1.10, "hits": 1.03, "so": 0.97},
    "Citi Field": {"runs": 0.95, "hr": 0.95, "hits": 0.95, "so": 1.02},
    "Citizens Bank Park": {"runs": 1.08, "hr": 1.12, "hits": 1.04, "so": 0.98},
    "Coors Field": {"runs": 1.28, "hr": 1.20, "hits": 1.15, "so": 0.95},
    "Dodger Stadium": {"runs": 0.97, "hr": 1.02, "hits": 0.96, "so": 1.01},
    "Great American Ball Park": {"runs": 1.12, "hr": 1.18, "hits": 1.05, "so": 0.97},
    "loanDepot park": {"runs": 0.90, "hr": 0.85, "hits": 0.93, "so": 1.04},
    "Nationals Park": {"runs": 1.00, "hr": 1.03, "hits": 0.99, "so": 1.00},
    "Oracle Park": {"runs": 0.88, "hr": 0.82, "hits": 0.92, "so": 1.05},
    "Petco Park": {"runs": 0.93, "hr": 0.90, "hits": 0.94, "so": 1.03},
    "PNC Park": {"runs": 0.94, "hr": 0.90, "hits": 0.95, "so": 1.02},
    "Truist Park": {"runs": 1.00, "hr": 1.02, "hits": 0.99, "so": 1.00},
    "Wrigley Field": {"runs": 1.05, "hr": 1.08, "hits": 1.02, "so": 0.98},
}

DEFAULT_PARK_FACTOR = {"runs": 1.0, "hr": 1.0, "hits": 1.0, "so": 1.0}

# ---------------------------------------------------------------------------
# Weather model coefficients
# Every constant here is a tunable parameter. Adjust based on backtesting.
# See compute_weather_adjustments() for the physics rationale behind each.
# ---------------------------------------------------------------------------

# Temperature model
# Baseline: 72°F is "neutral" — adjustments are zero at this temperature.
# Research basis: ~10 ft of carry difference across a 25°F range → ~0.4%/°F on HR.
TEMP_NEUTRAL_F    = 72.0   # °F; adjustments are zero at this temperature
TEMP_DELTA_CAP_F  = 30.0   # °F; clamps extrapolation at extreme cold/heat

# Per-degree-F multiplier coefficients (full effect at ±TEMP_DELTA_CAP_F shown):
TEMP_COEFF_HR   = 0.004    # ±12%  at ±30°F  (air density → carry distance)
TEMP_COEFF_RUNS = 0.0025   # ± 7.5% at ±30°F (correlated with HR + singles)
TEMP_COEFF_HITS = 0.0015   # ± 4.5% at ±30°F (weaker carry effect on line drives)
TEMP_COEFF_SO   = 0.001    # ± 3%   at ±30°F (cold air → more pitch movement → more K)

# Wind model
WIND_SPEED_CAP_MPH = 20.0  # mph; gusts above this are treated as equally strong

# Max fractional effect at full outward wind + full park sensitivity + cap speed:
WIND_COEFF_HR   = 0.12     # ~+12% HR   (e.g. 20 mph out at Wrigley)
WIND_COEFF_RUNS = 0.08     # ~+ 8% runs
WIND_COEFF_HITS = 0.04     # ~+ 4% hits
WIND_COEFF_SO   = 0.02     # ~- 2% K   (wind in → harder to square up the ball)

# Stadium metadata: coordinates, roof type, and outfield compass bearing.
# outfield_bearing_deg: compass direction home plate faces toward center field.
#   Wind blowing FROM this direction = wind blowing IN (suppresses offense).
#   Wind blowing TOWARD this direction = wind blowing OUT (boosts offense).
# roof: "open" = fully exposed, "retractable" = may be open or closed,
#       "dome" = always closed (weather irrelevant).
# wind_sensitivity: 0.0-1.0 how much wind actually affects play.
#   Open parks with low walls (Wrigley) = high. Parks with tall enclosures = low.

STADIUM_META = {
    "Angel Stadium":            {"lat": 33.800, "lon": -117.883, "roof": "open",        "outfield_bearing": 200, "wind_sensitivity": 0.5},
    "Camden Yards":             {"lat": 39.284, "lon": -76.622,  "roof": "open",        "outfield_bearing": 218, "wind_sensitivity": 0.5},
    "Comerica Park":            {"lat": 42.339, "lon": -83.049,  "roof": "open",        "outfield_bearing": 215, "wind_sensitivity": 0.5},
    "Fenway Park":              {"lat": 42.346, "lon": -71.098,  "roof": "open",        "outfield_bearing": 200, "wind_sensitivity": 0.6},
    "Globe Life Field":         {"lat": 32.747, "lon": -97.084,  "roof": "retractable", "outfield_bearing": 210, "wind_sensitivity": 0.2},
    "Guaranteed Rate Field":    {"lat": 41.830, "lon": -87.634,  "roof": "open",        "outfield_bearing": 205, "wind_sensitivity": 0.6},
    "Kauffman Stadium":         {"lat": 39.051, "lon": -94.480,  "roof": "open",        "outfield_bearing": 220, "wind_sensitivity": 0.6},
    "Minute Maid Park":         {"lat": 29.757, "lon": -95.355,  "roof": "retractable", "outfield_bearing": 215, "wind_sensitivity": 0.2},
    "Oakland Coliseum":         {"lat": 37.752, "lon": -122.201, "roof": "open",        "outfield_bearing": 210, "wind_sensitivity": 0.5},
    "Progressive Field":        {"lat": 41.496, "lon": -81.685,  "roof": "open",        "outfield_bearing": 195, "wind_sensitivity": 0.5},
    "T-Mobile Park":            {"lat": 47.591, "lon": -122.333, "roof": "retractable", "outfield_bearing": 210, "wind_sensitivity": 0.2},
    "Target Field":             {"lat": 44.982, "lon": -93.278,  "roof": "open",        "outfield_bearing": 210, "wind_sensitivity": 0.6},
    "Tropicana Field":          {"lat": 27.768, "lon": -82.653,  "roof": "dome",        "outfield_bearing": 0,   "wind_sensitivity": 0.0},
    "Rogers Centre":            {"lat": 43.641, "lon": -79.389,  "roof": "retractable", "outfield_bearing": 200, "wind_sensitivity": 0.2},
    "Yankee Stadium":           {"lat": 40.829, "lon": -73.927,  "roof": "open",        "outfield_bearing": 205, "wind_sensitivity": 0.5},
    "American Family Field":    {"lat": 43.028, "lon": -87.971,  "roof": "retractable", "outfield_bearing": 215, "wind_sensitivity": 0.2},
    "Busch Stadium":            {"lat": 38.623, "lon": -90.193,  "roof": "open",        "outfield_bearing": 205, "wind_sensitivity": 0.5},
    "Chase Field":              {"lat": 33.446, "lon": -112.067, "roof": "retractable", "outfield_bearing": 200, "wind_sensitivity": 0.2},
    "Citi Field":               {"lat": 40.757, "lon": -73.846,  "roof": "open",        "outfield_bearing": 225, "wind_sensitivity": 0.5},
    "Citizens Bank Park":       {"lat": 39.906, "lon": -75.167,  "roof": "open",        "outfield_bearing": 220, "wind_sensitivity": 0.5},
    "Coors Field":              {"lat": 39.756, "lon": -104.994, "roof": "open",        "outfield_bearing": 225, "wind_sensitivity": 0.6},
    "Dodger Stadium":           {"lat": 34.074, "lon": -118.240, "roof": "open",        "outfield_bearing": 215, "wind_sensitivity": 0.4},
    "Great American Ball Park": {"lat": 39.097, "lon": -84.508,  "roof": "open",        "outfield_bearing": 205, "wind_sensitivity": 0.6},
    "loanDepot park":           {"lat": 25.778, "lon": -80.220,  "roof": "retractable", "outfield_bearing": 200, "wind_sensitivity": 0.2},
    "Nationals Park":           {"lat": 38.873, "lon": -77.008,  "roof": "open",        "outfield_bearing": 210, "wind_sensitivity": 0.5},
    "Oracle Park":              {"lat": 37.778, "lon": -122.389, "roof": "open",        "outfield_bearing": 205, "wind_sensitivity": 0.7},
    "Petco Park":               {"lat": 32.707, "lon": -117.157, "roof": "open",        "outfield_bearing": 200, "wind_sensitivity": 0.4},
    "PNC Park":                 {"lat": 40.447, "lon": -80.006,  "roof": "open",        "outfield_bearing": 210, "wind_sensitivity": 0.5},
    "Truist Park":              {"lat": 33.891, "lon": -84.468,  "roof": "open",        "outfield_bearing": 200, "wind_sensitivity": 0.5},
    "Wrigley Field":            {"lat": 41.948, "lon": -87.656,  "roof": "open",        "outfield_bearing": 200, "wind_sensitivity": 0.9},
}


def get_stadium_meta(venue: str) -> dict:
    """Look up stadium metadata (coords, roof, wind sensitivity)."""
    if not venue:
        return None
    if venue in STADIUM_META:
        return STADIUM_META[venue]
    venue_lower = venue.lower()
    for park_name, meta in STADIUM_META.items():
        if park_name.lower() in venue_lower or venue_lower in park_name.lower():
            return meta
    return None


def get_park_factor(venue: str, weather: dict = None) -> dict:
    """
    Look up park factors for a venue, optionally adjusted for live weather.

    If weather data is provided, the static factors are modified based on
    temperature and wind conditions via compute_weather_adjustments().

    Args:
        venue: Stadium name
        weather: Optional dict with temp_f, wind_mph, wind_deg keys
    """
    if not venue:
        return DEFAULT_PARK_FACTOR.copy()

    # Find static base factors
    base = None
    if venue in PARK_FACTORS:
        base = PARK_FACTORS[venue].copy()
    else:
        venue_lower = venue.lower()
        for park_name, factors in PARK_FACTORS.items():
            if park_name.lower() in venue_lower or venue_lower in park_name.lower():
                base = factors.copy()
                break
    if base is None:
        base = DEFAULT_PARK_FACTOR.copy()

    # Apply weather adjustments if available
    if weather:
        meta = get_stadium_meta(venue)
        if meta:
            adjustments = compute_weather_adjustments(weather, meta)
            base["runs"] *= adjustments["runs"]
            base["hr"]   *= adjustments["hr"]
            base["hits"] *= adjustments["hits"]
            base["so"]   *= adjustments["so"]

    # Derived total-bases factor: TB is dominated by singles/doubles, with a
    # meaningful HR component. Blend hits- and hr-factors (0.55/0.45) so TB props
    # aren't mis-priced by the pure HR factor at parks where the two diverge
    # (e.g. Fenway: hr 0.95 but hits 1.12). Computed after weather so it tracks
    # live conditions too.
    base["tb"] = round(0.55 * base["hits"] + 0.45 * base["hr"], 4)

    return base


def compute_weather_adjustments(weather: dict, stadium_meta: dict) -> dict:
    """
    Calculate multiplicative adjustments to park factors based on weather.

    Physics basis (from research document):
    - Air density decreases with temperature. A fly ball at 80°F travels ~10ft
      farther than at 55°F. This directly impacts HR and TB.
    - Wind blowing out increases carry, wind blowing in suppresses it.
    - Domed/closed stadiums are immune to weather.

    Returns multipliers centered around 1.0.
    """
    result = {"runs": 1.0, "hr": 1.0, "hits": 1.0, "so": 1.0}

    # Dome or closed retractable roof -> no weather effect
    if stadium_meta.get("roof") == "dome":
        return result

    temp_f = weather.get("temp_f")
    wind_mph = weather.get("wind_mph", 0)
    wind_deg = weather.get("wind_deg", 0)
    sensitivity = stadium_meta.get("wind_sensitivity", 0.5)

    # --- Temperature adjustment ---
    # Baseline: 72°F is "neutral". Each degree above/below shifts air density.
    # ~10ft of carry difference across a 25°F range -> roughly 0.4% per degree F
    # on HR probability. Runs and hits are affected at ~60% of the HR magnitude.
    if temp_f is not None:
        temp_delta = temp_f - TEMP_NEUTRAL_F
        temp_delta = max(-TEMP_DELTA_CAP_F, min(TEMP_DELTA_CAP_F, temp_delta))

        result["hr"]   *= 1.0 + (temp_delta * TEMP_COEFF_HR)
        result["runs"] *= 1.0 + (temp_delta * TEMP_COEFF_RUNS)
        result["hits"] *= 1.0 + (temp_delta * TEMP_COEFF_HITS)
        # Cold air is denser → more pitch movement → slightly more K
        result["so"]   *= 1.0 - (temp_delta * TEMP_COEFF_SO)

    # --- Wind adjustment ---
    # Compute whether wind is blowing OUT (toward outfield) or IN.
    # outfield_bearing is the compass direction from home plate to center field.
    # If wind is coming FROM the opposite direction (blowing toward CF), it's "out".
    if wind_mph > 0 and sensitivity > 0:
        outfield_bearing = stadium_meta.get("outfield_bearing", 200)

        # Wind direction is where the wind comes FROM.
        # Wind blowing OUT = wind coming from behind home plate toward outfield.
        # That means wind_deg ≈ outfield_bearing + 180 (mod 360) means blowing OUT.
        # wind_deg ≈ outfield_bearing means blowing IN.
        #
        # Compute the angular difference between wind direction and the
        # "blowing in" direction (wind coming from outfield toward home plate).
        blowing_in_dir = outfield_bearing  # wind from CF direction = blowing in
        angle_diff = abs(wind_deg - blowing_in_dir) % 360
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        # Map angle_diff to a -1 (blowing in) to +1 (blowing out) scale.
        # 0° diff = perfectly blowing in (-1.0)
        # 180° diff = perfectly blowing out (+1.0)
        # 90° diff = crosswind (0.0)
        wind_direction_factor = (angle_diff / 180.0) * 2.0 - 1.0

        # Scale by wind speed and park sensitivity
        effective_wind = min(wind_mph, WIND_SPEED_CAP_MPH) / WIND_SPEED_CAP_MPH
        wind_effect = wind_direction_factor * effective_wind * sensitivity

        result["hr"]   *= 1.0 + (wind_effect * WIND_COEFF_HR)
        result["runs"] *= 1.0 + (wind_effect * WIND_COEFF_RUNS)
        result["hits"] *= 1.0 + (wind_effect * WIND_COEFF_HITS)
        # Wind blowing in → harder to square up the ball → slightly more K
        result["so"]   *= 1.0 - (wind_effect * WIND_COEFF_SO)

    return result
