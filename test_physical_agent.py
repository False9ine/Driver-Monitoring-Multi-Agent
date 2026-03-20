import pytest
from agents.passenger_fusion_agent import FusionAgent


@pytest.fixture
def agent():
    return FusionAgent()


@pytest.mark.parametrize(
    "physical, emotional, expected_state",
    [
        (0.2, 0.3, "SAFE"),
        (0.8, 0.2, "WARNING"),
        (0.3, 0.85, "WARNING"),
        (0.85, 0.7, "CRITICAL"),
        (0.6, 0.65, "WARNING"),
        (0.0, 0.0, "SAFE"),
        (1.0, 1.0, "CRITICAL"),
    ]
)
def test_fusion_states(agent, physical, emotional, expected_state):
    result = agent.fuse(physical, emotional)

    assert 0.0 <= result["fused_risk"] <= 1.0
    assert result["state"] == expected_state
