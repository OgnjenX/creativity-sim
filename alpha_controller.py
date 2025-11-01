"""Alpha control for balancing exploration vs. exploitation.

This module provides the AlphaController class for dynamically adjusting
the alpha parameter during simulation using various control strategies.
"""

import math


class AlphaController:
    """Controls alpha as a continuous variable.

    Modes:
        - 'cosine': smooth periodic variation between [alpha_min, alpha_max]
        - 'adaptive': adjusts alpha based on recent change in EMA of creativity

    Args:
        mode: Control mode ('cosine' or 'adaptive').
        alpha_min: Minimum allowed alpha value (exploration-heavy).
        alpha_max: Maximum allowed alpha value (exploitation-heavy).
        period: Period of cosine cycle in steps (used only in 'cosine' mode).
        ema_beta: EMA smoothing factor for creativity tracking (used only in 'adaptive' mode).
        adapt_step_up: Step size to increase alpha when creativity improves.
        adapt_step_down: Step size to decrease alpha when creativity declines.
    """

    def __init__(
        self,
        mode: str = "cosine",
        alpha_min: float = 0.2,
        alpha_max: float = 0.8,
        period: int = 50,
        ema_beta: float = 0.9,
        adapt_step_up: float = 0.02,
        adapt_step_down: float = 0.04,
    ) -> None:
        self.mode = mode
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.period = max(1, period)
        self.ema_beta = ema_beta
        self.adapt_step_up = adapt_step_up
        self.adapt_step_down = adapt_step_down
        self._ema = None
        self._prev_ema = None
        self._alpha = (alpha_min + alpha_max) / 2.0

    @property
    def alpha(self) -> float:
        """Current alpha value."""
        return float(self._alpha)

    def update(self, step: int, creativity: float) -> float:
        """Update alpha based on the current mode and observed creativity.

        Args:
            step: Current simulation step.
            creativity: Latest creativity score.

        Returns:
            Updated alpha value.

        Note:
            In adaptive mode, the first call (typically step=0) initializes the EMA
            with the provided creativity value. Subsequent calls use the EMA
            to adjust alpha based on creativity trends.
        """
        if self.mode == "cosine":
            phase = 2 * math.pi * (step % self.period) / self.period
            self._alpha = self.alpha_min + 0.5 * (1 - math.cos(phase)) * (
                self.alpha_max - self.alpha_min
            )
        elif self.mode == "adaptive":
            # Update EMA
            if self._ema is None:
                self._ema = creativity
                self._prev_ema = creativity
            else:
                self._prev_ema = self._ema
                self._ema = self.ema_beta * self._ema + (1 - self.ema_beta) * creativity
            delta = self._ema - self._prev_ema
            # If creativity is improving, exploit a bit more (increase alpha),
            # else explore (decrease alpha)
            if delta > 0:
                self._alpha = min(self.alpha_max, self._alpha + self.adapt_step_up)
            else:
                self._alpha = max(self.alpha_min, self._alpha - self.adapt_step_down)
        else:
            # Fallback: keep alpha fixed mid-range
            self._alpha = (self.alpha_min + self.alpha_max) / 2.0
        return float(self._alpha)
