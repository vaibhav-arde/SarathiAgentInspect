"""Slack and Teams notification system.

Closed the feedback loop by sending evaluation summaries to
enterprise messaging platforms.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

try:
    from slack_sdk import WebClient
except ImportError:
    WebClient = None  # type: ignore

import httpx

if TYPE_CHECKING:
    from sarathi_agent_inspect.reporting.base import EvaluationSummary
from sarathi_agent_inspect.reporting.notifications_registry import BaseNotifier, NotifierRegistry

logger = logging.getLogger(__name__)


def _schedule_background_task(coro: Any) -> None:
    """Run a coroutine safely from sync code.

    If we're already inside an event loop, schedule the work in the background
    instead of raising from asyncio.run().
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coro)
        return

    task = loop.create_task(coro)

    def _log_task_error(completed_task: asyncio.Task[Any]) -> None:
        try:
            completed_task.result()
        except Exception as exc:
            logger.error(f"Background notification task failed: {exc}")

    task.add_done_callback(_log_task_error)


@NotifierRegistry.register("slack")
class SlackNotifier(BaseNotifier):
    """Sends evaluation summaries to Slack."""

    def __init__(self, token: str, channel: str) -> None:
        self.client = WebClient(token=token) if WebClient is not None else None
        self.channel = channel

    def send_summary(self, summary: EvaluationSummary, trend: dict[str, Any] | None = None) -> None:
        """Send a formatted Slack message with the run summary."""
        if not self.client:
            logger.warning("Slack SDK not installed. Skipping notification.")
            return

        status_emoji = "✅" if summary.pass_rate >= 0.8 else "⚠️"
        trend_text = ""
        if trend:
            direction = "📈" if trend.get("trend_direction") == "up" else "📉"
            trend_text = f"\n*Trend*: {direction} {trend.get('pass_rate_delta', 0) * 100:.1f}% change"

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"Sarathi Evaluation: {summary.metadata.run_id}"},
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*Status*: {status_emoji} *{(summary.pass_rate * 100):.1f}% Pass Rate*\n"
                        f"*Records*: {summary.total_records} "
                        f"({summary.passed_count} passed, {summary.failed_count} failed)\n"
                        f"*Cost*: ${summary.metadata.total_cost_usd:.4f}\n"
                        f"*Environment*: `{summary.metadata.environment}`"
                        f"{trend_text}"
                    ),
                },
            },
        ]

        try:
            self.client.chat_postMessage(channel=self.channel, blocks=blocks, text="Sarathi Evaluation Summary")
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")


@NotifierRegistry.register("teams")
class TeamsNotifier(BaseNotifier):
    """Sends evaluation summaries to MS Teams via Webhooks."""

    def __init__(self, webhook_url: str) -> None:
        self.webhook_url = webhook_url

    def send_summary(self, summary: EvaluationSummary, trend: dict[str, Any] | None = None) -> None:
        """Send a message to Teams (blocking wrapper for adaptive cards)."""
        _schedule_background_task(self._send_summary_async(summary, trend))

    async def _send_summary_async(
        self,
        summary: EvaluationSummary,
        trend: dict[str, Any] | None = None,
    ) -> None:
        """Send a message to Teams."""
        trend_facts: list[dict[str, str]] = []
        if trend:
            trend_facts.append(
                {
                    "title": "Trend",
                    "value": f"{trend.get('trend_direction', 'flat')} {trend.get('pass_rate_delta', 0) * 100:.1f}%",
                }
            )

        payload = {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": {
                        "type": "AdaptiveCard",
                        "body": [
                            {
                                "type": "TextBlock",
                                "text": f"Sarathi Evaluation: {summary.metadata.run_id}",
                                "weight": "Bolder",
                                "size": "Medium",
                            },
                            {
                                "type": "TextBlock",
                                "text": f"Pass Rate: {(summary.pass_rate * 100):.1f}%",
                                "color": "Good" if summary.pass_rate >= 0.8 else "Attention",
                            },
                            {
                                "type": "FactSet",
                                "facts": [
                                    {"title": "Total Records", "value": str(summary.total_records)},
                                    {"title": "Cost", "value": f"${summary.metadata.total_cost_usd:.4f}"},
                                    {"title": "Environment", "value": summary.metadata.environment},
                                    *trend_facts,
                                ],
                            },
                        ],
                        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                        "version": "1.4",
                    },
                }
            ],
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(self.webhook_url, json=payload)
                response.raise_for_status()
            except Exception as e:
                logger.error(f"Failed to send Teams notification: {e}")
