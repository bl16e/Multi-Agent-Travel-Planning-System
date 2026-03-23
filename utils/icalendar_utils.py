from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from icalendar import Alarm, Calendar, Event

from utils.schemas import CalendarEventModel


def build_ics_calendar(
    calendar_name: str,
    events: list[CalendarEventModel],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    calendar = Calendar()
    calendar.add("prodid", "-//MASystem//Three Provinces Six Bureaus//EN")
    calendar.add("version", "2.0")
    calendar.add("x-wr-calname", calendar_name)

    for item in events:
        event = Event()
        event.add("uid", f"{uuid4()}@masystem.local")
        event.add("summary", item.title)
        event.add("dtstart", item.start_at)
        event.add("dtend", item.end_at)
        event.add("location", item.location)
        event.add("description", item.description)
        if item.url:
            event.add("url", str(item.url))

        for minutes in item.reminders_minutes:
            alarm = Alarm()
            alarm.add("action", "DISPLAY")
            alarm.add("description", f"Reminder: {item.title}")
            alarm.add("trigger", timedelta(minutes=-minutes))
            event.add_component(alarm)

        calendar.add_component(event)

    output_path.write_bytes(calendar.to_ical())
    return output_path
