"""GitHub Actions integration.

Utilities for interacting with GitHub API, such as posting evaluation
summaries as comments on Pull Requests.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class GitHubPRCommenter:
    """Posts evaluation summaries to GitHub PRs."""

    def __init__(self, token: str | None = None, repo: str | None = None, pr_number: int | None = None) -> None:
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.repo = repo or os.getenv("GITHUB_REPOSITORY")
        self.pr_number = pr_number or int(os.getenv("GITHUB_PR_NUMBER", "0"))

    async def post_summary(self, summary: Any, trend: dict[str, Any] | None = None) -> bool:
        """Post the evaluation summary as a comment on the PR."""
        if not all([self.token, self.repo, self.pr_number]):
            logger.warning("GitHub credentials or PR info missing. Skipping comment.")
            return False

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github.v3+json",
        }

        url = f"https://api.github.com/repos/{self.repo}/issues/{self.pr_number}/comments"

        status_emoji = "✅" if summary.pass_rate >= 0.8 else "❌"
        trend_md = ""
        if trend:
            direction = "⬆️" if trend.get("trend_direction") == "up" else "⬇️"
            trend_md = f"| **Trend** | {direction} {trend.get('pass_rate_delta', 0) * 100:.1f}% |"

        body = f"""
## Sarathi Evaluation Summary {status_emoji}

| Metric | Value |
| :--- | :--- |
| **Pass Rate** | {summary.pass_rate * 100:.1f}% |
| **Total Records** | {summary.total_records} |
| **Passed** | {summary.passed_count} |
| **Failed** | {summary.failed_count} |
| **Total Cost** | ${summary.metadata.total_cost_usd:.4f} |
{trend_md}

---
*Environment: `{summary.metadata.environment}` | Run ID: `{summary.metadata.run_id}`*
"""

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json={"body": body.strip()}, headers=headers)
                response.raise_for_status()
                logger.info(f"Posted evaluation summary to PR #{self.pr_number}")
                return True
            except Exception as e:
                logger.error(f"Failed to post GitHub comment: {e}")
                return False
