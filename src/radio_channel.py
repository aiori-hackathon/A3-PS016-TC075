import numpy as np

class RadioChannel:
    """Simulates 3GPP-compliant path loss, log-normal shadowing, and SINR."""
    
    def __init__(self, carrier_freq_ghz=3.5, tx_power_dbm=43.0, 
                 bandwidth_mhz=20.0, noise_floor_dbm=-104.0, shadowing_std_db=4.0):
        self.carrier_freq_ghz = carrier_freq_ghz
        self.tx_power_dbm = tx_power_dbm
        self.bandwidth_mhz = bandwidth_mhz
        self.noise_floor_dbm = noise_floor_dbm
        self.shadowing_std_db = shadowing_std_db

    def compute_rsrp(self, ue_pos, cell_pos, rng):
        """Calculates RSRP using a log-distance path loss model with log-normal shadowing."""
        dist = np.linalg.norm(ue_pos - cell_pos)
        dist = max(dist, 10.0)  # Prevent division by zero near base station
        
        # 3GPP TR 38.901 urban macro simplified path loss
        pl_db = 20.0 * np.log10(self.carrier_freq_ghz * 1000.0) + 30.0 * np.log10(dist) - 27.55
        shadowing_db = rng.normal(0.0, self.shadowing_std_db)
        rsrp = self.tx_power_dbm - pl_db + shadowing_db
        return rsrp

    def compute_sinr(self, serving_rsrp, neighbor_rsrps):
        """Calculates SINR in dB given serving RSRP and co-channel neighbor interference."""
        interference_linear = sum(10.0 ** (p / 10.0) for p in neighbor_rsrps)
        noise_linear = 10.0 ** (self.noise_floor_dbm / 10.0)
        sinr_linear = (10.0 ** (serving_rsrp / 10.0)) / (interference_linear + noise_linear)
        return 10.0 * np.log10(max(sinr_linear, 1e-6))

    def compute_throughput(self, sinr_db):
        """Approximates user throughput (Mbps) using Shannon capacity."""
        sinr_linear = 10.0 ** (sinr_db / 10.0)
        return self.bandwidth_mhz * np.log2(1.0 + sinr_linear)
