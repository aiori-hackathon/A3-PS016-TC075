class EventA3Baseline:
    """3GPP Event A3 Handover Policy (Hysteresis + Time-To-Trigger)."""
    
    def __init__(self, hysteresis_db=2.0, time_to_trigger_s=0.08, dt=0.02):
        self.hysteresis_db = hysteresis_db
        self.ttt_s = time_to_trigger_s
        self.dt = dt
        self.ttt_timer = 0.0
        self.current_candidate = None

    def reset(self):
        self.ttt_timer = 0.0
        self.current_candidate = None

    def decide(self, serving_cell, rsrp_dict):
        """
        Evaluates Event A3 condition: RSRP_target > RSRP_serving + Hysteresis continuously for TTT.
        Returns: (should_handover: bool, target_cell: int)
        """
        serving_rsrp = rsrp_dict[serving_cell]
        candidates = {c: p for c, p in rsrp_dict.items() if c != serving_cell}
        best_candidate, best_rsrp = max(candidates.items(), key=lambda item: item[1])

        if best_rsrp > serving_rsrp + self.hysteresis_db:
            if self.current_candidate == best_candidate:
                self.ttt_timer += self.dt
            else:
                self.current_candidate = best_candidate
                self.ttt_timer = self.dt

            if self.ttt_timer >= self.ttt_s:
                target = self.current_candidate
                self.reset()
                return True, target
        else:
            self.reset()

        return False, serving_cell
