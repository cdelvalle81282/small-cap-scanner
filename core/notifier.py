"""
Alert notification dispatcher.
Currently only in-app (DB write). Email and Slack are stubs for future wiring.
"""
import logging

logger = logging.getLogger(__name__)


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
