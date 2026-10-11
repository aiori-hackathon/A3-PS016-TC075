import numpy as np
from src.radio_channel import RadioChannel
from src.mobility import TrajectoryGenerator

class HandoverSimulator:
    """Discrete-event simulation engine evaluating cellular handover performance."""
    
    def __init__(self, config):
        self.cfg = config
        self.channel = RadioChannel(
            carrier_freq_ghz=self.cfg['radio']['carrier_freq_ghz'],
            tx_power_dbm=self.cfg['radio']['tx_power_dbm'],
            bandwidth_mhz=self.cfg['radio']['bandwidth_mhz'],
            noise_floor_dbm=self.cfg['radio']['noise_floor_dbm'],
            shadowing_std_db=self.cfg['radio']['shadowing_std_db']
        )
        self.cell_coords = {i: np.array(coord) for i, coord in enumerate(self.cfg['cells']['coordinates'])}

    def run_trajectory(self, policy, speed_kmh, seed):
        rng = np.random.default_rng(seed)
        dt = self.cfg['simulation']['time_step_s']
        mobility = TrajectoryGenerator(speed_kmh=speed_kmh, dt=dt, 
                                       total_dist_m=self.cfg['simulation']['track_length_m'])
        
        serving_cell = 0
        dwell_time = 0.0
        prev_cell = -1
        last_ho_time = -999.0
        
        handovers = 0
        failures = 0
        ping_pongs = 0
        interruption_time_s = 0.0
        throughputs = []

        ue_pos, current_time, is_finished = mobility.step()
        prev_rsrp = {c: self.channel.compute_rsrp(ue_pos, self.cell_coords[c], rng) for c in self.cell_coords}

        while not is_finished:
            ue_pos, current_time, is_finished = mobility.step()
            rsrps = {c: self.channel.compute_rsrp(ue_pos, self.cell_coords[c], rng) for c in self.cell_coords}
            
            serving_rsrp = rsrps[serving_cell]
            neighbor_rsrps = [p for c, p in rsrps.items() if c != serving_cell]
            sinr_db = self.channel.compute_sinr(serving_rsrp, neighbor_rsrps)

            # Query policy
            if hasattr(policy, 'feature_cols'):  # ML Policy
                should_ho, target_cell = policy.decide(
                    serving_cell, rsrps, prev_rsrp, mobility.speed_ms, dwell_time
                )
            else:  # Baseline Policy
                should_ho, target_cell = policy.decide(serving_cell, rsrps)

            # Handover execution
            if should_ho and target_cell != serving_cell:
                handovers += 1
                interruption_time_s += self.cfg['simulation']['ho_interruption_s']

                # Failure condition
                if serving_rsrp < self.cfg['simulation']['min_serving_rsrp_dbm']:
                    failures += 1

                # Ping-pong condition
                if target_cell == prev_cell and (current_time - last_ho_time) <= self.cfg['simulation']['ping_pong_window_s']:
                    ping_pongs += 1

                prev_cell = serving_cell
                serving_cell = target_cell
                dwell_time = 0.0
                last_ho_time = current_time
                throughputs.append(0.0)  # Interruption penalty
            else:
                dwell_time += dt
                throughputs.append(self.channel.compute_throughput(sinr_db))

            prev_rsrp = rsrps

        return {
            'handovers': handovers,
            'failures': failures,
            'ping_pongs': ping_pongs,
            'interruption_ms': interruption_time_s * 1000.0,
            'avg_throughput_mbps': float(np.mean(throughputs)) if throughputs else 0.0
        }