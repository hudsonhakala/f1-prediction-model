import fastf1 as f1
from datetime import datetime


SCHEDULE = f1.get_event_schedule(2026)
EVENTS_REMAINING = f1.get_events_remaining(datetime.now())

