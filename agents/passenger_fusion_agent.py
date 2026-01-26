class FusionAgent:
    """
    Risk fusion agent.

    Combines physical and emotional risks into:
    - fused_risk ∈ [0, 1]
    - discrete safety state
    """

    def __init__(
        self,
        w_physical: float = 0.4,
        w_emotional: float = 0.6
    ):
        assert 0.0 <= w_physical <= 1.0
        assert 0.0 <= w_emotional <= 1.0
        assert abs(w_physical + w_emotional - 1.0) < 1e-6

        self.w_p = w_physical
        self.w_e = w_emotional

    def fuse(
        self,
        physical_risk: float,
        emotional_risk: float
    ) -> dict:
        """
        Args:
            physical_risk (float): ∈ [0, 1]
            emotional_risk (float): ∈ [0, 1]

        Returns:
            dict:
                {
                    "fused_risk": float,
                    "state": str
                }
        """

        # ---------------------------
        # 1. Weighted risk fusion
        # ---------------------------
        fused_risk = (
            self.w_p * physical_risk +
            self.w_e * emotional_risk
        )

        # Clamp for safety
        fused_risk = max(0.0, min(1.0, fused_risk))

        # ---------------------------
        # 2. Rule-aware escalation
        # ---------------------------
        if physical_risk > 0.75 and emotional_risk > 0.6:
            state = "CRITICAL"
        elif physical_risk > 0.75 or emotional_risk > 0.8:
            state = "WARNING"
        elif fused_risk > 0.6:
            state = "WARNING"
        else:
            state = "SAFE"

        return {
            "fused_risk": fused_risk,
            "state": state
        }