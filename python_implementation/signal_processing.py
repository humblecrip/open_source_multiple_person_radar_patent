"""
Signal Processing Module
Implements Range FFT and basic signal processing operations
Based on firmware: dss_data_path.c
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftshift


class SignalProcessor:
    """
    Signal processing operations for radar data
    """

    def __init__(self, params):
        """
        Initialize signal processor

        Args:
            params: Dictionary of radar parameters from RadarDataLoader
        """
        self.params = params
        self.num_adc_samples = params['num_adc_samples']
        self.num_rx = params['num_rx']
        self.range_resolution = params['range_resolution']
        self.wavelength = params['wavelength']

    def range_fft(self, radar_cube, window_type='hanning'):
        """
        Perform 1D FFT along range dimension (ADC samples)

        Based on firmware: MmwDemo_processChirp() in dss_data_path.c

        Args:
            radar_cube: Complex radar data [num_frames, num_chirps, num_rx, num_adc_samples]
            window_type: Window function ('hanning', 'hamming', 'blackman', None)

        Returns:
            range_fft_output: Range FFT result [num_frames, num_chirps, num_rx, num_range_bins]
        """
        print("Performing Range FFT...")

        # Apply window function
        if window_type == 'hanning':
            window = np.hanning(self.num_adc_samples)
        elif window_type == 'hamming':
            window = np.hamming(self.num_adc_samples)
        elif window_type == 'blackman':
            window = np.blackman(self.num_adc_samples)
        else:
            window = np.ones(self.num_adc_samples)

        # Apply window along ADC samples dimension
        windowed_data = radar_cube * window[np.newaxis, np.newaxis, np.newaxis, :]

        # Perform FFT along range dimension (last axis)
        range_fft_output = fft(windowed_data, n=self.num_adc_samples, axis=-1)

        print(f"Range FFT output shape: {range_fft_output.shape}")

        return range_fft_output

    def compute_range_profile(self, range_fft_output):
        """
        Compute range profile (magnitude across range bins)

        Args:
            range_fft_output: Range FFT result

        Returns:
            range_profile: Magnitude of range FFT [num_frames, num_chirps, num_rx, num_range_bins]
        """
        range_profile = np.abs(range_fft_output)
        return range_profile

    def get_range_bins(self):
        """
        Get range bin values in meters

        Returns:
            range_bins: Array of range values [num_range_bins]
        """
        range_bins = np.arange(self.num_adc_samples) * self.range_resolution
        return range_bins

    def dc_removal(self, radar_cube, num_frames_avg=10):
        """
        Remove DC component (static clutter removal)

        Based on firmware: MmwDemo_dcRangeSignatureCompensation()

        Args:
            radar_cube: Complex radar data
            num_frames_avg: Number of frames to average for DC estimation

        Returns:
            radar_cube_dc_removed: Data with DC removed
        """
        print("Removing DC component...")

        # Estimate DC as mean over first few frames
        dc_estimate = np.mean(radar_cube[:num_frames_avg], axis=0, keepdims=True)

        # Subtract DC
        radar_cube_dc_removed = radar_cube - dc_estimate

        return radar_cube_dc_removed

    def clutter_removal(self, range_fft_output, alpha=0.1):
        """
        Adaptive clutter removal using exponential averaging

        Args:
            range_fft_output: Range FFT output [frames, chirps, rx, range_bins]
            alpha: Adaptation rate (0 < alpha < 1)

        Returns:
            clutter_removed: Data with clutter removed
        """
        print("Removing clutter...")

        num_frames = range_fft_output.shape[0]
        clutter_removed = np.zeros_like(range_fft_output)

        # Initialize clutter estimate with first frame
        clutter_estimate = range_fft_output[0].copy()

        for frame_idx in range(num_frames):
            # Remove clutter
            clutter_removed[frame_idx] = range_fft_output[frame_idx] - clutter_estimate

            # Update clutter estimate (exponential averaging)
            clutter_estimate = alpha * range_fft_output[frame_idx] + (1 - alpha) * clutter_estimate

        return clutter_removed


class BeamformingProcessor:
    """
    Beamforming and angle estimation
    Based on firmware: oddemo_heatmap.c
    """

    def __init__(self, params, num_azimuth_bins=64, angle_range_deg=60):
        """
        Initialize beamforming processor

        Args:
            params: Radar parameters
            num_azimuth_bins: Number of azimuth bins
            angle_range_deg: Angle range in degrees (±angle_range_deg)
        """
        self.params = params
        self.num_rx = params['num_rx']
        self.wavelength = params['wavelength']
        self.num_azimuth_bins = num_azimuth_bins
        self.angle_range_deg = angle_range_deg

        # Generate steering vectors
        self.steering_vectors = self._generate_steering_vectors()

    def _generate_steering_vectors(self):
        """
        Generate steering vectors for angle estimation

        Based on firmware: ODDemo_Heatmap_steeringVecGen()

        Returns:
            steering_vectors: [num_azimuth_bins, num_rx]
        """
        print("Generating steering vectors...")

        # Angle bins from -angle_range to +angle_range
        angles_deg = np.linspace(-self.angle_range_deg, self.angle_range_deg, self.num_azimuth_bins)
        angles_rad = np.deg2rad(angles_deg)

        # Antenna spacing (assuming uniform linear array)
        # For IWR6843, typical spacing is lambda/2
        d = self.wavelength / 2

        # Steering vectors: exp(-j * 2π * d * sin(θ) / λ)
        # For antenna array [0, 1, 2, ..., num_rx-1]
        antenna_indices = np.arange(self.num_rx)

        steering_vectors = np.zeros((self.num_azimuth_bins, self.num_rx), dtype=complex)

        for idx, angle in enumerate(angles_rad):
            phase_shift = 2 * np.pi * d * np.sin(angle) / self.wavelength
            steering_vectors[idx, :] = np.exp(-1j * phase_shift * antenna_indices)

        return steering_vectors

    def capon_beamforming(self, range_data, range_idx, gamma=0.01):
        """
        Capon (MVDR) beamforming for a specific range bin

        Based on firmware: ODDemo_Heatmap_aoaEstCaponBF_covInv() and
                          ODDemo_Heatmap_aoaEstCaponBF_heatmap()

        Args:
            range_data: Range FFT data for one frame [num_chirps, num_rx, num_range_bins]
            range_idx: Range bin index to process
            gamma: Diagonal loading factor for covariance matrix regularization

        Returns:
            azimuth_spectrum: Power spectrum across azimuth angles [num_azimuth_bins]
        """
        # Extract data for this range bin: [num_chirps, num_rx]
        x = range_data[:, :, range_idx]

        # Compute covariance matrix: R = (1/M) * X * X^H
        # X: [num_rx, num_chirps]
        X = x.T  # Transpose to [num_rx, num_chirps]
        num_chirps = X.shape[1]

        # Covariance matrix
        R = (X @ X.conj().T) / num_chirps

        # Diagonal loading for numerical stability
        R = R + gamma * np.eye(self.num_rx)

        # Invert covariance matrix
        try:
            R_inv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            # If inversion fails, use pseudo-inverse
            R_inv = np.linalg.pinv(R)

        # Compute Capon spectrum for each angle
        azimuth_spectrum = np.zeros(self.num_azimuth_bins)

        for az_idx in range(self.num_azimuth_bins):
            # Steering vector for this angle
            a = self.steering_vectors[az_idx, :]

            # Capon power: P(θ) = 1 / (a^H * R^-1 * a)
            denominator = np.abs(a.conj() @ R_inv @ a)

            if denominator > 1e-10:
                azimuth_spectrum[az_idx] = 1.0 / denominator
            else:
                azimuth_spectrum[az_idx] = 0.0

        return azimuth_spectrum

    def conventional_beamforming(self, range_data, range_idx):
        """
        Conventional (Bartlett) beamforming

        Args:
            range_data: Range FFT data for one frame [num_chirps, num_rx, num_range_bins]
            range_idx: Range bin index to process

        Returns:
            azimuth_spectrum: Power spectrum across azimuth angles [num_azimuth_bins]
        """
        # Extract data for this range bin
        x = range_data[:, :, range_idx]  # [num_chirps, num_rx]

        # Average across chirps
        x_avg = np.mean(x, axis=0)  # [num_rx]

        # Compute conventional beamforming spectrum
        azimuth_spectrum = np.zeros(self.num_azimuth_bins)

        for az_idx in range(self.num_azimuth_bins):
            # Steering vector
            a = self.steering_vectors[az_idx, :]

            # Conventional beamforming: |a^H * x|^2
            azimuth_spectrum[az_idx] = np.abs(a.conj() @ x_avg) ** 2

        return azimuth_spectrum

    def get_azimuth_angles(self):
        """
        Get azimuth angle values in degrees

        Returns:
            angles: Array of azimuth angles [num_azimuth_bins]
        """
        angles = np.linspace(-self.angle_range_deg, self.angle_range_deg, self.num_azimuth_bins)
        return angles


if __name__ == "__main__":
    # Test signal processing
    from radar_data_loader import RadarDataLoader

    loader = RadarDataLoader()
    params = loader.get_radar_params(bandwidth_ghz=2.5)

    # Create dummy data for testing
    radar_cube = np.random.randn(10, 1, 4, 200) + 1j * np.random.randn(10, 1, 4, 200)

    # Test Range FFT
    processor = SignalProcessor(params)
    range_fft_out = processor.range_fft(radar_cube)
    print(f"Range FFT output shape: {range_fft_out.shape}")

    # Test beamforming
    bf_processor = BeamformingProcessor(params)
    print(f"Steering vectors shape: {bf_processor.steering_vectors.shape}")
