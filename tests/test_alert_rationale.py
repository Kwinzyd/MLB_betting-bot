from src.pipelines.alert_rationale import generate_alert_rationale, _facts
from src.clients.execution.telegram_venue import format_alert_message


class _StubClient:
    def __init__(self, text, available=True):
        self._text = text
        self.available = available

    async def complete_text(self, system, user, **kwargs):
        return self._text


_EDGE = {"edge_pct": 7.5, "ev": 0.05, "model_prob": 0.62, "book_implied": 0.55,
         "kelly": {"recommended_stake": 20.0}}
_CTX = {
    "player_name": "Gerrit Cole", "market": "pitcher_strikeouts", "side": "over",
    "line": 6.5, "home_team": "Yankees", "away_team": "Red Sox",
    "model_context": {"opp_k_rate": 0.27, "park_adj": 0.97,
                      "llm_injury_summary": "none", "weather": None},
}


def test_facts_includes_present_context_only():
    facts = _facts(_EDGE, _CTX)
    assert facts["player"] == "Gerrit Cole"
    assert facts["opp_k_rate"] == 0.27
    assert facts["park_adj"] == 0.97
    assert "weather" not in facts  # None values dropped


async def test_rationale_none_when_unavailable():
    assert await generate_alert_rationale(_EDGE, _CTX, client=_StubClient(None, available=False)) is None


async def test_rationale_returns_text():
    client = _StubClient('"Cole faces a high-strikeout Red Sox lineup in a pitcher-friendly park."')
    out = await generate_alert_rationale(_EDGE, _CTX, client=client)
    assert out.startswith("Cole faces")
    assert '"' not in out[:1]  # surrounding quotes stripped


async def test_rationale_none_on_empty_reply():
    assert await generate_alert_rationale(_EDGE, _CTX, client=_StubClient("")) is None


def test_format_alert_message_includes_rationale():
    msg = format_alert_message(
        player_name="Gerrit Cole", market="pitcher_strikeouts", side="over",
        line=6.5, odds=2.0, bookmaker="draftkings", edge=_EDGE,
        projection={"projected_mean": 7.5}, context={}, home_team="Yankees",
        away_team="Red Sox", venue="Yankee Stadium",
        rationale="High-K opponent in a neutral park.",
    )
    assert "High-K opponent" in msg


def test_format_alert_message_without_rationale():
    msg = format_alert_message(
        player_name="Gerrit Cole", market="pitcher_strikeouts", side="over",
        line=6.5, odds=2.0, bookmaker="draftkings", edge=_EDGE,
        projection={"projected_mean": 7.5}, context={}, home_team="Yankees",
        away_team="Red Sox", venue="Yankee Stadium",
    )
    assert "\U0001F9E0" not in msg  # no brain emoji / rationale line
