from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

Interval = tuple[datetime, datetime]


def overlaps(a: Interval, b: Interval) -> bool:
    """Two time ranges overlap if each one starts before the other ends."""
    return a[0] < b[1] and b[0] < a[1]

def _round_up(dt: datetime, step: timedelta) -> datetime:
    """10:07 with a 30-minute step -> 10:30. Slots start on neat times."""
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    steps = -(-(dt - midnight) // step)  # ceiling division
    return midnight + steps * step

def find_free_slots(
    busy: list[Interval],
    *,
    start: datetime,
    duration: timedelta,
    tz: ZoneInfo,
    days: int = 7,
    day_start: time = time(10, 0),
    day_end: time = time(18, 0),
    step: timedelta = timedelta(minutes=30),
    limit: int = 3,
    skip_weekends: bool = True,
) -> list[Interval]:
    """First `limit` free slots of `duration`, in working hours, after `start`.
    Every datetime is timezone-aware; results are in the interviewer's timezone `tz`."""
    start = start.astimezone(tz)
    slots: list[Interval] = []
    for offset in range(days + 1):
        day = (start + timedelta(days=offset)).date()
        if skip_weekends and day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            continue
        candidate = datetime.combine(day, day_start, tzinfo=tz)
        day_close = datetime.combine(day, day_end, tzinfo=tz)
        candidate = max(candidate, _round_up(start, step))
        while candidate + duration <= day_close:
            slot = (candidate, candidate + duration)
            if not any(overlaps(slot, b) for b in busy):
                slots.append(slot)
                if len(slots) == limit:
                    return slots
            candidate += step
    return slots
