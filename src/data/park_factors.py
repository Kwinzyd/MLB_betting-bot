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
    "busch Stadium": {"runs": 0.95, "hr": 0.92, "hits": 0.96, "so": 1.02},
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


def get_park_factor(venue: str) -> dict:
    """Look up park factors for a venue, returning defaults if unknown."""
    if not venue:
        return DEFAULT_PARK_FACTOR
    # Try exact match first
    if venue in PARK_FACTORS:
        return PARK_FACTORS[venue]
    # Fuzzy match: check if venue name contains a known park name
    venue_lower = venue.lower()
    for park_name, factors in PARK_FACTORS.items():
        if park_name.lower() in venue_lower or venue_lower in park_name.lower():
            return factors
    return DEFAULT_PARK_FACTOR
