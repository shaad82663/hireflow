"""Sends real emails to the LOCAL Mailpit only, so you can open http://localhost:8025 and see them.

Safety: every test here is skipped unless
  1. SMTP_HOST is localhost / 127.0.0.1, and
  2. the Mailpit web API answers on that host (port 8025).
So if SMTP points anywhere external, nothing is sent.

Run:  uv run pytest tests/test_email_mailpit.py -v
"""

from uuid import uuid4

import httpx
import pytest

from hireflow.core.config import get_settings
from hireflow.integrations.email import (
    OutgoingEmail,
    SmtpEmailSender,
    reply_address,
    token_from_address,
)

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
MAILPIT_API = "http://localhost:8025/api/v1"


def _mailpit_ready() -> tuple[bool, str]:
    s = get_settings()
    if s.smtp_host not in LOCAL_HOSTS:
        return False, f"SMTP_HOST={s.smtp_host!r} is not local; refusing to send real email"
    try:
        info = httpx.get(f"{MAILPIT_API}/info", timeout=2)
        info.raise_for_status()
        if "Version" not in info.json():
            return False, "port 8025 answered, but it is not Mailpit"
    except (httpx.HTTPError, ValueError):
        return False, "Mailpit is not running (start it: docker compose up -d mailpit)"
    return True, ""


_ready, _why = _mailpit_ready()
pytestmark = pytest.mark.skipif(not _ready, reason=_why)


@pytest.fixture
def sender() -> SmtpEmailSender:
    s = get_settings()
    return SmtpEmailSender(s.smtp_host, s.smtp_port, s.email_from)


async def find_in_mailpit(subject: str) -> dict:
    """Look up the message we just sent by its unique subject."""
    async with httpx.AsyncClient(base_url=MAILPIT_API, timeout=5) as client:
        res = await client.get("/search", params={"query": f'subject:"{subject}"'})
        res.raise_for_status()
        messages = res.json()["messages"]
        assert len(messages) == 1, f"expected 1 message with subject {subject!r}, got {len(messages)}"
        # full message (includes the text body)
        full = await client.get(f"/message/{messages[0]['ID']}")
        full.raise_for_status()
        return full.json()


async def test_send_plain_email(sender):
    subject = f"HireFlow test {uuid4().hex[:6]}"
    await sender.send(
        OutgoingEmail(
            to="asha.rao@example.com",
            subject=subject,
            body="Hi Asha,\n\nThanks for applying to Senior Python Engineer.\n\nHireFlow",
        )
    )

    msg = await find_in_mailpit(subject)
    assert msg["To"][0]["Address"] == "asha.rao@example.com"
    assert msg["From"]["Address"] == "jobs@hireflow.local"
    assert "Thanks for applying" in msg["Text"]
    assert msg["ReplyTo"] == []


async def test_send_interview_email_with_reply_address(sender):
    token = uuid4().hex[:8]
    subject = f"HireFlow interview invite {token}"
    await sender.send(
        OutgoingEmail(
            to="ravi.kumar@example.com",
            subject=subject,
            body="Hi Ravi,\n\nCan you do a 30 minute call on Thursday at 3pm?\nJust reply to this email.",
            reply_to=reply_address(token),
        )
    )

    msg = await find_in_mailpit(subject)
    reply_to = msg["ReplyTo"][0]["Address"]
    assert reply_to == f"interview+{token}@hireflow.local"
    assert token_from_address(reply_to) == token  # a reply can be matched back to this interview
