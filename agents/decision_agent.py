import torch
from rl_training.dqn import DQN

class PassengerDecisionAgent:
    """
    RL-based passenger safeguarding decision agent.
    """

    def __init__(self):
        self.model = DQN()
        self.model.load_state_dict(
            torch.load("rl_training/passenger_decision_dqn.pth")
        )
        self.model.eval()

    def decide(self, physical_risk, emotional_risk):
        fused = 0.6 * physical_risk + 0.4 * emotional_risk

        state = torch.tensor(
            [physical_risk, emotional_risk, fused],
            dtype=torch.float32
        )

        with torch.no_grad():
            action = torch.argmax(self.model(state)).item()

        return [
            "NO_ACTION",
            "VERBAL_WARNING",
            "PROTECTIVE_INTERVENTION"
        ][action]