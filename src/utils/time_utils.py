from datetime import datetime
from pytz import timezone

def get_eastern_local_date():
    """MLB games are commonly scheduled in Eastern time."""
    tz = timezone('US/Eastern')
    return datetime.now(tz).date()

def get_utc_now():
    return datetime.utcnow()

def get_utc_now_iso():
    return datetime.utcnow().isoformat()
