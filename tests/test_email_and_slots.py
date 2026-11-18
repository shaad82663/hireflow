import socket
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from aiosmtpd.controller import Controller

from hireflow.integrations.email import (
    OutgoingEmail,
    SmtpEmailSender,
    latest_reply,
    reply_address,
    token_from_address,
)
from hireflow.scheduling.slots import find_free_slots, overlaps

IST = ZoneInfo("Asia/Kolkata")
MIN45 = timedelta(minutes=45)


# ---------------- email ----------------


class Collector:
    """A tiny SMTP server handler that keeps every received message."""

    def __init__(self):
        self.messages = []

    async def handle_DATA(self, server, session, envelope):
        self.messages.append(envelope.content.decode())
        return "250 OK"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))  # the OS picks a free port
        return s.getsockname()[1]


async def test_smtp_sender_talks_to_a_real_smtp_server():
    collector, port = Collector(), free_port()
    controller = Controller(collector, hostname="127.0.0.1", port=port)
    controller.start()  # a real SMTP server in a background thread
    try:
        sender = SmtpEmailSender("127.0.0.1", port, "jobs@hireflow.local")
        await sender.send(OutgoingEmail("asha@example.com", "Hi", "Body", reply_to="r@x.y"))
    finally:
        controller.stop()
    [raw] = collector.messages
    assert "To: asha@example.com" in raw and "Reply-To: r@x.y" in raw and "Body" in raw


def test_reply_address_round_trip():
    address = reply_address("Ab3_x-9")
    assert address == "interview+Ab3_x-9@hireflow.local"
    assert token_from_address(f"Jobs <{address}>") == "Ab3_x-9"
    assert token_from_address("someone@else.com") is None


def test_latest_reply_drops_the_quoted_email():
    text = "Option 2 works for me.\nThanks!\n\nOn Mon, 23 Nov, Recruiting wrote:\n> Hi Asha"
    assert latest_reply(text) == "Option 2 works for me.\nThanks!"
    assert latest_reply("Sure\n> quoted line") == "Sure"


# ---------------- slots ----------------

MONDAY_9AM = datetime(2026, 11, 23, 9, 0, tzinfo=IST)


def test_slots_start_in_working_hours_on_neat_times():
    slots = find_free_slots([], start=MONDAY_9AM + timedelta(minutes=67), duration=MIN45, tz=IST)
    assert [s[0].strftime("%H:%M") for s in slots] == ["10:30", "11:00", "11:30"]


def test_busy_time_is_avoided():
    busy = [(MONDAY_9AM.replace(hour=10), MONDAY_9AM.replace(hour=12))]
    slots = find_free_slots(busy, start=MONDAY_9AM, duration=MIN45, tz=IST, limit=2)
    assert [s[0].strftime("%H:%M") for s in slots] == ["12:00", "12:30"]
    assert not any(overlaps(s, busy[0]) for s in slots)


def test_weekends_and_day_end_are_skipped():
    friday_late = datetime(2026, 11, 27, 17, 20, tzinfo=IST)  # rounds up to 17:30
    [slot] = find_free_slots([], start=friday_late, duration=MIN45, tz=IST, limit=1)
    # 17:30-18:15 ends after 18:00, Saturday and Sunday are skipped -> Monday 10:00
    assert slot[0] == datetime(2026, 11, 30, 10, 0, tzinfo=IST)


def test_start_in_another_timezone_is_converted():
    utc_start = datetime(2026, 11, 23, 3, 30, tzinfo=ZoneInfo("UTC"))  # = 09:00 IST
    [slot] = find_free_slots(
        [], start=utc_start, duration=MIN45, tz=IST, limit=1, day_start=time(9, 0)
    )
    assert slot[0].hour == 9 and slot[0].tzinfo == IST