"""
Alert notification dispatcher.

Two things live here:
  - send_alert(): triggered price-level alerts (in-app DB write; email/Slack stubs).
  - notify_ops(): operational alerts (e.g. a data feed broke), sent to Slack and/or
    email using the same env vars scan_and_notify.py already uses.
"""
import json
import logging
import os
import smtplib
import urllib.request
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)


def notify_ops(subject: str, message: str) -> bool:
    """Send an operational alert to whatever channel(s) are configured.

    Env vars (shared with scan_and_notify.py):
      SLACK_WEBHOOK_URL: Slack incoming webhook
      SMTP_USERNAME / SMTP_PASSWORD / SMTP_TO: Gmail App Password + recipient(s)

    Always logs at ERROR first, so a failure is never fully silent even when no
    channel is configured. Returns True if at least one channel accepted it.
    """
    logger.error("OPS ALERT: %s: %s", subject, message)
    sent = False
    if _post_slack(f"*{subject}*\n{message}"):
        sent = True
    if _send_email(subject, message):
        sent = True
    return sent


def _post_slack(text: str) -> bool:
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook:
        return False
    try:
        payload = json.dumps({"text": text}).encode()
        req = urllib.request.Request(
            webhook, data=payload, headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception:
        logger.exception("Slack ops notification failed")
        return False


def _send_email(subject: str, body: str) -> bool:
    user = os.environ.get("SMTP_USERNAME")
    pw = os.environ.get("SMTP_PASSWORD")
    to = os.environ.get("SMTP_TO", user)
    if not user or not pw:
        return False
    try:
        msg = MIMEText(body)
        msg["Subject"] = f"[Small Cap Scanner] {subject}"
        msg["From"] = user
        msg["To"] = to
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(user, pw)
            server.sendmail(user, to.split(","), msg.as_string())
        return True
    except Exception:
        logger.exception("Email ops notification failed")
        return False


def send_alert(alert: dict) -> None:
    """Dispatch a triggered price alert to all configured channels."""
    _notify_inapp(alert)
    _notify_email(alert)
    _notify_slack(alert)


def _notify_inapp(alert: dict) -> None:
    # Alert is already written to price_alerts table by monitor.py before this is called.
    logger.info(
        "In-app alert stored: %s %s breached at $%.2f",
        alert["ticker"], alert["level_label"], alert["triggered_close"],
    )


def _notify_email(alert: dict) -> None:
    # TODO: wire up SMTP or SendGrid
    pass


def _notify_slack(alert: dict) -> None:
    # TODO: wire up Slack webhook
    pass
