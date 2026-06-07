from datetime import datetime, timezone
from pytz import timezone as pytz_timezone


def get_eastern_local_date():
    """MLB games are commonly scheduled in Eastern time."""
    tz = pytz_timezone('US/Eastern')
    return datetime.now(tz).date()


def utcnow() -> datetime:
    """Naive UTC datetime, drop-in replacement for the deprecated datetime.utcnow().

    Returns a *naive* (tzinfo=None) value on purpose: every persisted timestamp in
    this codebase is a naive-UTC ISO string, and several pipelines compare those
    strings lexicographically (retention cutoffs, dedup windows). Emitting an
    aware datetime here would append a '+00:00' offset and silently break those
    string comparisons against existing rows. Use this everywhere instead of
    datetime.utcnow() (deprecated) or datetime.now(timezone.utc) (aware).
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def get_utc_now() -> datetime:
    return utcnow()


def get_utc_now_iso() -> str:
    return utcnow().isoformat()
