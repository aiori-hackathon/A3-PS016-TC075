import numpy as np

class TrajectoryGenerator:
    """Generates user positions across a linear 3-cell highway setup."""
    
    def __init__(self, speed_kmh, dt=0.02, total_dist_m=1600.0, y_offset=15.0):
        self.speed_ms = speed_kmh / 3.6
        self.dt = dt
        self.total_dist_m = total_dist_m
        self.y_offset = y_offset
        self.reset()

    def reset(self):
        self.current_x = 0.0
        self.current_time = 0.0

    def step(self):
        """Advances UE by one time step. Returns (ue_pos, current_time, is_finished)."""
        ue_pos = np.array([self.current_x, self.y_offset])
        t = self.current_time
        self.current_x += self.speed_ms * self.dt
        self.current_time += self.dt
        is_finished = self.current_x > self.total_dist_m
        return ue_pos, t, is_finished
