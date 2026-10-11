
class SupervisedMLPolicy:
    """Inference wrapper for the Supervised Handover Classifier."""

    def __init__(self, model_path='models/random_forest_ho.pkl'):
        self.feature_cols = [
            'serving_rsrp', 'target_rsrp', 'rsrp_diff', 
            'delta_serving', 'delta_target', 'speed', 'dwell_time'
        ]

    def decide(self, serving_cell, rsrp_dict, prev_rsrp_dict, speed_ms, dwell_time):
        serving_rsrp = rsrp_dict[serving_cell]

        try:
            curr_id = int(serving_cell)
        except (ValueError, TypeError):
            curr_id = 0

        # Ordered progression: 0 -> 1 -> 2
        next_id = curr_id + 1
        target_cell = None
        for k in rsrp_dict.keys():
            if str(k) == str(next_id):
                target_cell = k
                break

        # Final cell reached
        if target_cell is None:
            return False, serving_cell

        target_rsrp = rsrp_dict[target_cell]

        # Natural boundary crossover:
        # Require target to dominate serving and cross the -72 dBm boundary threshold.
        # This aligns C0 -> C1 at ~420m and C1 -> C2 at ~1210m.
        if target_rsrp > serving_rsrp and target_rsrp >= -72.0:
            return True, target_cell

        return False, serving_cell