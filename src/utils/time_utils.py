from datetime import date as _date, datetime, timedelta, timezone
from pytz import timezone as pytz_timezone

_EASTERN = pytz_timezone('US/Eastern')


def get_eastern_local_date():
    """MLB games are commonly scheduled in Eastern time."""
    return datetime.now(_EASTERN).date()


def eastern_date_utc_window(d: _date) -> tuple[str, str]:
    """UTC ISO bounds [start, end) spanning the given Eastern calendar day.

    Games are stored with a UTC commence time, but "today's slate" is an
    Eastern-day concept. A 10pm ET game rolls past midnight UTC, so filtering
    `date LIKE '<eastern-day>%'` against the UTC string silently drops it. Use
    these bounds (`date >= start AND date < end`) to capture every game on the
    Eastern day regardless of its UTC date. Z-suffixed to match the stored
    Odds API format for clean lexicographic comparison.
    """
    start_et = _EASTERN.localize(datetime(d.year, d.month, d.day))
    end_et = start_et + timedelta(days=1)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return (
        start_et.astimezone(timezone.utc).strftime(fmt),
        end_et.astimezone(timezone.utc).strftime(fmt),
    )


def eastern_today_utc_window() -> tuple[str, str]:
    """UTC ISO bounds [start, end) of the current Eastern calendar day."""
    return eastern_date_utc_window(get_eastern_local_date())


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
