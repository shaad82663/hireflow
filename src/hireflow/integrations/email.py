import re
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import make_msgid
from functools import lru_cache
from typing import Protocol

import aiosmtplib

from hireflow.core.config import get_settings

@dataclass
class OutgoingEmail:
    to: str
    subject: str
    body: str
    reply_to: str | None = None

class EmailSender(Protocol):  
    async def send(self, email: OutgoingEmail) -> None: ...

class SmtpEmailSender:

    def __init__(self, host: str, port: int, sender: str) -> None:
        self.host, self.port, self.sender = host, port, sender        

    async def send(self, email: OutgoingEmail) -> None:
        msg = EmailMessage()
        msg["From"] = self.sender
        msg["To"] = email.to
        msg["Subject"] = email.subject
        msg["Message-ID"] = make_msgid(domain="hireflow.local")
        if email.reply_to:
            msg["Reply-To"] = email.reply_to
        msg.set_content(email.body)
        await aiosmtplib.send(msg, hostname=self.host, port=self.port, timeout=15)

class OutboxEmailSender:
    """Keeps emails in a list instead of sending them. Used by tests."""

    def __init__(self) -> None:
        self.sent: list[OutgoingEmail] = []

    async def send(self, email: OutgoingEmail) -> None:
        self.sent.append(email)

@lru_cache
def get_email_sender() -> EmailSender:
    s = get_settings()
    return SmtpEmailSender(s.smtp_host, s.smtp_port, s.email_from)

# reply routing 

def reply_address(token: str) -> str:
    """Each conversation gets its own reply address, e.g. interview+Xy12ab@hireflow.local,
    so an incoming reply tells us exactly which interview it belongs to."""
    return f"interview+{token}@{get_settings().reply_domain}"

_TOKEN = re.compile(r"interview\+([A-Za-z0-9_-]+)@", re.IGNORECASE)

def token_from_address(address: str) -> str | None:
    match = _TOKEN.search(address)
    return match.group(1) if match else None

_QUOTE_HEADER = re.compile(r"^On .+wrote:\s*$", re.IGNORECASE)

def latest_reply(text: str) -> str:
    """Keep only what the person just wrote: cut at the quoted previous email."""
    lines = []
    for line in text.splitlines():
        if line.lstrip().startswith(">") or _QUOTE_HEADER.match(line.strip()):
            break
        lines.append(line)
    return "\n".join(lines).strip()