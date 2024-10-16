import copy

class Trajectory:
    def __init__(self):
        self._sequence = []
        self.logits = []

    def add_pair(self, state, action, logits=None):
        self._sequence.append((state, action))
        if logits is not None:
            self.logits.append(copy.deepcopy(logits.cpu().detach()))

    def concat(self, other):
        self._sequence = self._sequence + copy.deepcopy(other._sequence)
        self.logits = self.logits + copy.deepcopy(other.logits)
    
    def get_trajectory(self):
        return self._sequence
    
    def get_logits_sequence(self):
        return self.logits
    
    def get_action_sequence(self):
        return [pair[1] for pair in self._sequence]
    
    def get_state_sequence(self):
        return [pair[0] for pair in self._sequence]
    
    def __repr__(self):
        return f"Trajectory(sequence={self._sequence})"