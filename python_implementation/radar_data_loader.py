"""
Radar Data Loader for DCA1000 Binary Files
Extracts and parses raw ADC data from DCA1000 captures
"""

import numpy as np
import struct


class RadarDataLoader:
    """
    Load and parse DCA1000 radar data files

    Based on the dataset parameters:
    - Start frequency: 60 GHz
    - Chirp duration: 57 μs
    - ADC sample rate: 4000 ksps (4 MSps)
    - ADC samples: 200
    - Frame period: 50 ms
    - Number of frames: 1200
    - Chirps per frame: 1
    - Frequency slope: 40/50/60 MHz/μs (2/2.5/3 GHz bandwidth)
    """

    def __init__(self, num_rx=4, num_tx=1, num_adc_samples=200,
                 num_chirps_per_frame=1, num_frames=1200):
        """
        Initialize radar data loader

        Args:
            num_rx: Number of RX antennas (default: 4)
            num_tx: Number of TX antennas (default: 1)
            num_adc_samples: Number of ADC samples per chirp (default: 200)
            num_chirps_per_frame: Number of chirps per frame (default: 1)
            num_frames: Number of frames to load (default: 1200)
        """
        self.num_rx = num_rx
        self.num_tx = num_tx
        self.num_adc_samples = num_adc_samples
        self.num_chirps_per_frame = num_chirps_per_frame
        self.num_frames = num_frames
        self.num_virtual_antennas = num_rx * num_tx

    def load_bin_file(self, filepath):
        """
        Load DCA1000 binary file

        DCA1000 format: Interleaved I/Q samples, 16-bit signed integers
        Data order: [I0, Q0, I1, Q1, ...] for each RX channel

        Args:
            filepath: Path to .bin file

        Returns:
            radar_cube: Complex radar data [num_frames, num_chirps, num_rx, num_adc_samples]
        """
        print(f"Loading radar data from: {filepath}")

        # Read binary file
        with open(filepath, 'rb') as f:
            raw_data = np.fromfile(f, dtype=np.int16)

        print(f"Total samples read: {len(raw_data)}")

        # Calculate expected size
        # Each sample is I+Q (2 values), for num_rx channels
        samples_per_chirp = self.num_adc_samples * self.num_rx * 2  # *2 for I and Q
        total_samples = samples_per_chirp * self.num_chirps_per_frame * self.num_frames

        print(f"Expected samples: {total_samples}")
        print(f"Samples per chirp: {samples_per_chirp}")

        # Trim or pad if necessary
        if len(raw_data) > total_samples:
            raw_data = raw_data[:total_samples]
            print(f"Trimmed to {total_samples} samples")
        elif len(raw_data) < total_samples:
            # Adjust number of frames based on actual data
            actual_frames = len(raw_data) // samples_per_chirp // self.num_chirps_per_frame
            self.num_frames = actual_frames
            print(f"Adjusted frames to {actual_frames} based on file size")
            total_samples = samples_per_chirp * self.num_chirps_per_frame * self.num_frames
            raw_data = raw_data[:total_samples]

        # Reshape: [total_samples] -> [frames, chirps, rx, adc_samples, 2(I/Q)]
        try:
            reshaped = raw_data.reshape(
                self.num_frames,
                self.num_chirps_per_frame,
                self.num_rx,
                self.num_adc_samples,
                2  # I and Q
            )
        except ValueError as e:
            print(f"Reshape error: {e}")
            print(f"Trying alternative reshape...")
            # Alternative: assume data is organized differently
            reshaped = raw_data.reshape(-1, 2)
            reshaped = reshaped.reshape(
                self.num_frames,
                self.num_chirps_per_frame,
                self.num_rx,
                self.num_adc_samples,
                2
            )

        # Convert to complex: I + jQ
        radar_cube = reshaped[..., 0] + 1j * reshaped[..., 1]

        print(f"Radar cube shape: {radar_cube.shape}")
        print(f"Data type: {radar_cube.dtype}")

        return radar_cube

    def get_radar_params(self, bandwidth_ghz=2.5):
        """
        Get radar parameters for processing

        Args:
            bandwidth_ghz: Chirp bandwidth in GHz (2.0, 2.5, or 3.0)

        Returns:
            Dictionary of radar parameters
        """
        # Physical constants
        c = 3e8  # Speed of light (m/s)

        # Radar parameters
        start_freq = 60e9  # 60 GHz
        chirp_duration = 57e-6  # 57 μs
        adc_sample_rate = 4e6  # 4 MSps
        frame_period = 50e-3  # 50 ms

        # Derived parameters
        bandwidth = bandwidth_ghz * 1e9  # Convert to Hz
        freq_slope = bandwidth / chirp_duration  # Hz/s

        # Range resolution
        range_resolution = c / (2 * bandwidth)

        # Maximum range
        max_range = (c * adc_sample_rate * self.num_adc_samples) / (2 * freq_slope)

        # Wavelength at center frequency
        center_freq = start_freq + bandwidth / 2
        wavelength = c / center_freq

        params = {
            'start_freq': start_freq,
            'bandwidth': bandwidth,
            'chirp_duration': chirp_duration,
            'adc_sample_rate': adc_sample_rate,
            'frame_period': frame_period,
            'freq_slope': freq_slope,
            'range_resolution': range_resolution,
            'max_range': max_range,
            'wavelength': wavelength,
            'center_freq': center_freq,
            'num_adc_samples': self.num_adc_samples,
            'num_rx': self.num_rx,
            'num_frames': self.num_frames,
            'frame_rate': 1.0 / frame_period
        }

        return params

    def print_params(self, params):
        """Print radar parameters in readable format"""
        print("\n" + "="*60)
        print("RADAR PARAMETERS")
        print("="*60)
        print(f"Start Frequency:      {params['start_freq']/1e9:.1f} GHz")
        print(f"Bandwidth:            {params['bandwidth']/1e9:.2f} GHz")
        print(f"Center Frequency:     {params['center_freq']/1e9:.2f} GHz")
        print(f"Chirp Duration:       {params['chirp_duration']*1e6:.1f} μs")
        print(f"Frequency Slope:      {params['freq_slope']/1e12:.1f} MHz/μs")
        print(f"ADC Sample Rate:      {params['adc_sample_rate']/1e6:.1f} MSps")
        print(f"Frame Period:         {params['frame_period']*1e3:.1f} ms")
        print(f"Frame Rate:           {params['frame_rate']:.1f} Hz")
        print(f"Range Resolution:     {params['range_resolution']*100:.2f} cm")
        print(f"Maximum Range:        {params['max_range']:.2f} m")
        print(f"Wavelength:           {params['wavelength']*1000:.2f} mm")
        print(f"Number of RX:         {params['num_rx']}")
        print(f"Number of Samples:    {params['num_adc_samples']}")
        print(f"Number of Frames:     {params['num_frames']}")
        print("="*60 + "\n")


if __name__ == "__main__":
    # Test the loader
    loader = RadarDataLoader(
        num_rx=4,
        num_tx=1,
        num_adc_samples=200,
        num_chirps_per_frame=1,
        num_frames=1200
    )

    # Get parameters
    params = loader.get_radar_params(bandwidth_ghz=2.5)
    loader.print_params(params)

    # Test loading a file
    test_file = "/Users/andreachan/Documents/vital_sign/open_source_multiple_person_radar_patent/FMCW radar-based multi-person vital sign monitoring data/1_AsymmetricalPosition/1_Radar_Raw_Data/position_ (1)/adc_2_5GHZ_position1_ (1).bin"

    try:
        radar_cube = loader.load_bin_file(test_file)
        print(f"\nSuccessfully loaded radar data!")
        print(f"Shape: {radar_cube.shape}")
        print(f"Data range: [{np.min(np.abs(radar_cube)):.2f}, {np.max(np.abs(radar_cube)):.2f}]")
    except Exception as e:
        print(f"Error loading file: {e}")
