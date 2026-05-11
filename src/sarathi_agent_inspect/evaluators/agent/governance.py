"""Agent Governance and Safety.

Detects infinite loops, retry loops, and measures task completion
to ensure agent safety and efficiency.
"""

from __future__ import annotations

import collections


class LoopDetector:
    """Detects repeating patterns in agent execution."""

    @staticmethod
    def detect_action_loops(actions: list[str], window_size: int = 3) -> bool:
        """Detect if the agent is repeating the same sequence of actions.

        Args:
            actions: List of tool names or action descriptions.
            window_size: Size of the sequence to check for repetition.
        """
        if len(actions) < window_size * 2:
            return False

        # Simple pattern matching for repeating subsequences
        for i in range(len(actions) - window_size * 2 + 1):
            window = actions[i : i + window_size]
            next_window = actions[i + window_size : i + window_size * 2]
            if window == next_window:
                return True

        return False


class TaskCompletionScorer:
    """Measures how close an agent got to the final goal."""

    @staticmethod
    def calculate_progress(current_state: str, target_state: str) -> float:
        """Heuristic for task progress.

        In a real scenario, this would compare specific world-state variables.
        Here we use a placeholder semantic similarity logic.
        """
        # Placeholder: 1.0 if identical, 0.0 if entirely different.
        if current_state == target_state:
            return 1.0
        return 0.5  # Default partial completion


class InfiniteLoopProtector:
    """Active monitor that flags runaway agent loops."""

    def __init__(self, max_steps: int = 20, max_repeats: int = 3) -> None:
        self.max_steps = max_steps
        self.max_repeats = max_repeats
        self.action_history: list[str] = []

    def should_terminate(self, next_action: str) -> bool:
        """Checks if the next action would exceed safety limits.

        This can be called DURING the agent loop to prevent infinite costs.
        """
        self.action_history.append(next_action)

        if len(self.action_history) > self.max_steps:
            return True

        # Check for simple repetition of the same action
        counts = collections.Counter(self.action_history[-5:])
        return counts[next_action] >= self.max_repeats
