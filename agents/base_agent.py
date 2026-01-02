class BaseAgent:
    """
    Base class for all perception/decision agents.
    Ensures consistent interface across the team.
    """

    def update(self, landmarks, frame=None):
        """
        Parameters:
            landmarks : list of MediaPipe landmark points
            frame     : optional, raw frame if needed

        Returns:
            score (float in range 0–1)
        """
        raise NotImplementedError("Update method must be implemented")
