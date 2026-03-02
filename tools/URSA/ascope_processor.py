import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import pandas as pd
import datetime
import re
from pathlib import Path

# Add the functions directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
functions_dir = os.path.join(current_dir, "functions")
sys.path.append(functions_dir)

# Import from functions directory
from utils import load_config, ensure_output_dirs, load_and_preprocess_image
from preprocessing import (
    mask_sprocket_holes,
    detect_ascope_frames,
    verify_frames_visually,
)
from signal_processing import (
    detect_signal_in_frame,
    trim_signal_trace,
    adaptive_peak_preserving_smooth,
    verify_trace_quality,
)
from grid_detection import (
    detect_grid_lines_and_dotted,
    find_reference_line_blackhat,
    interpolate_regular_grid,
)
from echo_detection import (
    find_tx_pulse,
    detect_surface_echo,
    detect_bed_echo,
    detect_double_transmitter_pulse,
    detect_surface_echo_adaptive,
)


def export_ascope_database(base_filename, output_dir, frame_results, meta_info=None):
    """
    Export A-scope results as CSV and NPZ with metadata.
    """
    if not frame_results or len(frame_results) == 0:
        print("ERROR: No frame results to export.")
        return False

    try:
        # Create DataFrame from frame results
        df = pd.DataFrame(frame_results)
        csv_name = base_filename + "_pick.csv"
        npz_name = base_filename + "_pick.npz"
        csv_path = Path(output_dir) / csv_name
        npz_path = Path(output_dir) / npz_name

        # Export CSV with proper formatting
        df.to_csv(csv_path, index=False, float_format="%.6f", na_rep="NaN")
        print(f"INFO: Exported CSV database to {csv_path}")

        # Prepare metadata
        if meta_info is None:
            meta_info = {}
        meta_info["export_timestamp"] = str(datetime.datetime.now())
        meta_info["frame_count"] = len(df)
        meta_info["columns"] = list(df.columns)
        meta_info["filename"] = base_filename
        meta_info["ice_velocity_m_per_us"] = 168.0  # Standard ice velocity

        # Export NPZ with structured data
        np.savez(
            npz_path,
            frame=df["Frame"].values,
            cbd=df["CBD"].values,
            surface_time_us=df["Surface_Time_us"].values,
            bed_time_us=df["Bed_Time_us"].values,
            ice_thickness_m=df["Ice_Thickness_m"].values,
            surface_power_db=df["Surface_Power_dB"].values,
            bed_power_db=df["Bed_Power_dB"].values,
            transmitter_time_us=df["Transmitter_Time_us"].values,
            transmitter_power_db=df["Transmitter_Power_dB"].values,
            transmitter_x_pixel=df["Transmitter_X_pixel"].values,
            meta=meta_info,
        )
        print(f"INFO: Exported NPZ database to {npz_path}")

        # Display summary statistics
        valid_thickness = np.sum(~pd.isna(df["Ice_Thickness_m"]))
        valid_surface = np.sum(~pd.isna(df["Surface_Time_us"]))
        valid_bed = np.sum(~pd.isna(df["Bed_Time_us"]))

        print(
            f"INFO: Frame summary - Total: {len(df)}, Valid thickness: {valid_thickness}, Valid surface: {valid_surface}, Valid bed: {valid_bed}"
        )

        return True

    except Exception as e:
        print(f"ERROR: Database export failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    except Exception as e:
        print(f"ERROR: Database export failed: {e}")
        import traceback

        traceback.print_exc()
        return False


class AScope:
    """
    A-scope radar data processing class that orchestrates the entire processing pipeline.
    Enhanced with ice thickness calculation and data export functionality.
    """

    def __init__(self, config_path=None):
        # Set up configuration
        self.debug_mode = False  # Default until loaded from config
        if config_path is None:
            # Use default config path relative to the ascope directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, "config", "default_config.json")

        # Resolve the path (handle relative paths)
        resolved_config_path = os.path.abspath(os.path.expanduser(config_path))
        print(f"Loading configuration from: {resolved_config_path}")

        try:
            with open(resolved_config_path, "r") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print(
                f"ERROR: Processing configuration file not found at {resolved_config_path}"
            )
            raise

        # Initialize configuration sections
        self.processing_params = self.config.get("processing_params", {})
        self.physical_params = self.config.get("physical_params", {})
        self.output_config = self.config.get("output", {})
        self.debug_mode = self.output_config.get("debug_mode", False)
        self.output_dir = ensure_output_dirs(self.config)

        # Initialize frame results storage
        self.frame_results = []
        self.base_filename = None
        self.cbd_list = []

        if self.debug_mode:
            print(f"Debug mode enabled. Using config from: {resolved_config_path}")

    def set_output_directory(self, output_dir):
        """Override the output directory."""
        self.output_config["output_dir"] = output_dir
        self.output_dir = ensure_output_dirs(self.config)

    def set_debug_mode(self, debug_mode):
        """Set debug mode for additional outputs."""
        self.debug_mode = debug_mode
        self.output_config["debug_mode"] = debug_mode

    def _extract_cbd_sequence_from_filename(self, filename):
        """Extract CBD sequence from filename and validate frame count."""
        try:
            # Extract CBD range from filename (e.g., F103-C0467_0479.tiff)
            match = re.search(r"C(\d+)_(\d+)", filename)
            if not match:
                raise ValueError(
                    f"Could not extract CBD range from filename: {filename}"
                )

            cbd_start = int(match.group(1))  # 467
            cbd_end = int(match.group(2))  # 479
            cbd_list = list(range(cbd_start, cbd_end + 1))  # [467, 468, ..., 479]

            print(
                f"INFO: Extracted CBD sequence: {cbd_start} to {cbd_end} ({len(cbd_list)} CBDs expected)"
            )
            return cbd_list

        except Exception as e:
            print(f"ERROR: Failed to extract CBD sequence: {e}")
            return []

    def _calculate_ice_thickness(self, surface_time_us, bed_time_us):
        """
        Calculate ice thickness using one-way travel times.

        Args:
            surface_time_us (float): Surface echo time in microseconds (one-way)
            bed_time_us (float): Bed echo time in microseconds (one-way)

        Returns:
            float: Ice thickness in meters, or NaN if calculation not possible
        """
        if surface_time_us is None or bed_time_us is None:
            return np.nan

        if np.isnan(surface_time_us) or np.isnan(bed_time_us):
            return np.nan

        if bed_time_us <= surface_time_us:
            print(
                f"WARNING: Bed time ({bed_time_us:.2f} μs) <= surface time ({surface_time_us:.2f} μs)"
            )
            return np.nan

        # Ice thickness = (bed_time - surface_time) × ice_velocity
        # Using standard ice velocity of 168 m/μs
        ice_velocity_m_per_us = 168.0
        travel_time_us = bed_time_us - surface_time_us
        ice_thickness_m = travel_time_us * ice_velocity_m_per_us

        return ice_thickness_m

    def _export_results(self):
        """Export frame results as CSV and NPZ files."""
        if not self.frame_results:
            print("WARNING: No frame results to export")
            return False

        meta_info = {
            "ascope_file": self.base_filename,
            "processing_timestamp": str(datetime.datetime.now()),
            "ice_velocity_m_per_us": 168.0,
            "total_frames": len(self.frame_results),
            "cbd_sequence": self.cbd_list,
        }

        return export_ascope_database(
            self.base_filename, self.output_dir, self.frame_results, meta_info
        )

    def process_interactive(self):
        """Process an image with interactive input for the file path."""
        try:
            image, base_filename = load_and_preprocess_image()
            self._process_image_data(image, base_filename)
        except Exception as e:
            print(f"Error during interactive processing: {e}")
            import traceback

            traceback.print_exc()

    def process_image(self, file_path, output_dir=None):
        """
        Process a specific image file.

        Args:
            file_path (str): Path to the input image file
            output_dir (str, optional): Path to output directory
        """
        try:
            if output_dir:
                self.set_output_directory(output_dir)

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Image file not found: {file_path}")

            image, base_filename = load_and_preprocess_image(file_path)
            self._process_image_data(image, base_filename)

        except Exception as e:
            print(f"Error processing image {file_path}: {e}")
            import traceback

            traceback.print_exc()

    def _process_image_data(self, image, base_filename):
        """Internal method to process image data through the pipeline."""
        self.base_filename = base_filename
        self.frame_results = []  # Reset frame results for new image

        # Extract CBD sequence from filename
        self.cbd_list = self._extract_cbd_sequence_from_filename(base_filename)

        print(f"\n=== Processing A-scope Image: {base_filename} ===")

        # Step 1: Mask sprocket holes
        print("Step 1: Masking sprocket holes...")
        masked_image, mask = mask_sprocket_holes(image, self.config)

        # Step 2: Detect A-scope frames
        print("Step 2: Detecting A-scope frames...")
        frames = detect_ascope_frames(masked_image, self.config)

        # Validate frame count against CBD sequence
        if self.cbd_list and len(frames) != len(self.cbd_list):
            print(
                f"WARNING: Frame count ({len(frames)}) doesn't match expected CBD count ({len(self.cbd_list)})"
            )
            if len(frames) < len(self.cbd_list):
                # Adjust CBD list to match detected frames
                print(
                    f"INFO: Adjusting CBD list to match {len(frames)} detected frames"
                )
                self.cbd_list = self.cbd_list[: len(frames)]
            else:
                # Use detected frame count and extend CBD list if needed
                print(
                    f"INFO: Extending CBD list to match {len(frames)} detected frames"
                )
                while len(self.cbd_list) < len(frames):
                    self.cbd_list.append(
                        self.cbd_list[-1] + 1 if self.cbd_list else 467
                    )

        # Step 3: Verify frames visually
        print("Step 3: Creating frame verification plot...")
        verify_frames_visually(masked_image, frames, base_filename, self.config)

        # Step 4: Process each frame
        print(f"Step 4: Processing {len(frames)} individual frames...")
        for idx, (left, right) in enumerate(frames):
            print(
                f"\n--- Processing frame {idx + 1}/{len(frames)}: cols {left}-{right} ---"
            )
            self._process_frame(masked_image, left, right, base_filename, idx + 1)

        # Step 5: Export data automatically
        print(f"\nStep 5: Exporting database (CSV + NPZ)...")
        self._export_results()

        print(f"\n=== Processing Complete for {base_filename} ===")

    def _process_frame(self, masked_image, left, right, base_filename, frame_idx):
        """Process an individual A-scope frame and store results."""
        # Initialize frame result structure
        frame_result = {
            "Frame": frame_idx,
            "CBD": self.cbd_list[frame_idx - 1]
            if frame_idx <= len(self.cbd_list)
            else np.nan,
            "Surface_Time_us": np.nan,
            "Bed_Time_us": np.nan,
            "Ice_Thickness_m": np.nan,
            "Surface_Power_dB": np.nan,
            "Bed_Power_dB": np.nan,
            "Transmitter_Time_us": np.nan,
            "Transmitter_Power_dB": np.nan,
            "Transmitter_X_pixel": np.nan,
        }

        try:
            # Extract the frame
            frame_img = masked_image[:, left:right].copy()
            h, w = frame_img.shape

            if w <= 0 or h <= 0:
                print("Warning: Frame has zero width or height. Skipping.")
                self.frame_results.append(frame_result)
                return

            # 1. Detect Signal Trace
            signal_x, signal_y = detect_signal_in_frame(frame_img, self.config)
            if signal_x is None:
                print(f"Signal detection failed for frame {frame_idx}. Skipping.")
                self.frame_results.append(frame_result)
                return

            signal_y = adaptive_peak_preserving_smooth(signal_y, self.config)

            # 2. Clean Signal Trace
            signal_x_clean, signal_y_clean = trim_signal_trace(
                frame_img, signal_x, signal_y, self.config
            )

            if len(signal_x_clean) == 0:
                print(
                    f"No valid signal trace left after cleaning for frame {frame_idx}. Skipping."
                )
                self.frame_results.append(frame_result)
                return

            print(f"Cleaned signal length: {len(signal_x_clean)} points")

            # Verify the signal trace
            trace_quality_score = verify_trace_quality(
                frame_img, signal_x_clean, signal_y_clean
            )
            print(
                f"Trace quality score for frame {frame_idx}: {trace_quality_score:.2f}"
            )

            # 3. Initial Tx Pulse Detection (for grid calibration)
            tx_pulse_col, tx_idx_in_clean = find_tx_pulse(
                signal_x_clean, signal_y_clean, self.config
            )

            if tx_pulse_col is None:
                print("Warning: Tx pulse detection failed. Using default X-anchor.")
                tx_pulse_col = w * 0.15  # Fallback anchor near start
                tx_idx_in_clean = (
                    int(len(signal_x_clean) * 0.15) if len(signal_x_clean) > 0 else 0
                )

            print(
                f"Initial Tx pulse estimated at col {tx_pulse_col:.1f} (index {tx_idx_in_clean} in clean signal)"
            )

            # 4. Find Reference Line (Y-axis anchor)
            ref_row = find_reference_line_blackhat(
                frame_img, base_filename, frame_idx, self.config
            )

            if ref_row is None:
                print(
                    "Error: Reference line detection failed critically. Skipping frame."
                )
                self.frame_results.append(frame_result)
                return

            print(
                f"Reference row ({self.physical_params['y_ref_dB']} dB) estimated at y={ref_row}"
            )

            # 5. Detect Grid Lines (for interpolation)
            qa_path = (
                f"{self.output_dir}/{base_filename}_frame{frame_idx:02d}_grid_QA.png"
            )
            h_peaks_initial, v_peaks_initial, h_minor_peaks, v_minor_peaks = (
                detect_grid_lines_and_dotted(
                    frame_img, self.config, qa_plot_path=qa_path, ref_row_for_qa=ref_row
                )
            )

            # When calling interpolate_regular_grid, use the dynamic range values
            y_range_dB = 8.25 * 10  # Dynamic range based on signal extent
            y_major, y_minor = interpolate_regular_grid(
                h,
                h_peaks_initial,
                ref_row,
                self.physical_params["y_major_dB"],
                self.physical_params["y_minor_per_major"],
                y_range_dB,
                is_y_axis=True,
                config=self.config,
            )

            # For X-axis (time in μs), use range of 17.5 μs (0 to 17.5 μs)
            x_range_us = self.physical_params.get(
                "x_range_factor"
            ) * self.physical_params.get("x_major_us")
            x_major, x_minor = interpolate_regular_grid(
                w,
                v_peaks_initial,
                tx_pulse_col,  # ✅ Fixed: Use tx_pulse_col (number) not tx_analysis (dict)
                self.physical_params["x_major_us"],
                self.physical_params["x_minor_per_major"],
                x_range_us,
                is_y_axis=False,
                config=self.config,
            )

            print(f"Saved grid QA plot: {qa_path}")

            # 6. Calculate Calibration Factors
            px_per_us_echo, px_per_db_echo = self._calculate_calibration_factors(
                x_major, y_major, w, h
            )

            # 7. Calibrate Signal to get power_vals and time_vals
            power_vals, time_vals = None, None
            if tx_pulse_col is not None and ref_row is not None:
                power_vals = (
                    self.physical_params["y_ref_dB"]
                    - (signal_y_clean - ref_row) / px_per_db_echo
                )
                time_vals = (signal_x_clean - tx_pulse_col) / px_per_us_echo
            else:
                print(
                    "Warning: Cannot calibrate signal power due to missing anchors or factors."
                )
                self.frame_results.append(frame_result)
                return

            # 8. Detection with Double TX Pulse Support
            print("INFO: Starting enhanced echo detection...")

            # Call enhanced double transmitter pulse detection
            tx_analysis = detect_double_transmitter_pulse(
                signal_x_clean, signal_y_clean, power_vals, time_vals, self.config
            )

            if tx_analysis["is_double_pulse"]:
                print(
                    f"INFO: Double transmitter pulse detected (confidence: {tx_analysis['confidence']:.2f})"
                )
                print(
                    f"INFO: TX components at: {[time_vals[i] for i in tx_analysis['tx_peaks']]} μs"
                )

                # Update tx_idx_in_clean if enhanced detection found something better
                if tx_analysis.get("recommended_tx_idx") is not None:
                    tx_idx_in_clean = tx_analysis["recommended_tx_idx"]
                    print(
                        f"INFO: Updated TX index to {tx_idx_in_clean} based on enhanced detection"
                    )

            # Call surface detection
            surf_idx_in_clean = detect_surface_echo_adaptive(
                power_vals, time_vals, tx_analysis, self.config
            )

            # Call bed detection
            bed_idx_in_clean = detect_bed_echo(
                power_vals, time_vals, surf_idx_in_clean, px_per_us_echo, self.config
            )

            # 9. Store Frame Results with Ice Thickness Calculation
            if (
                surf_idx_in_clean is not None
                and surf_idx_in_clean < len(time_vals)
                and time_vals is not None
                and power_vals is not None
            ):
                surface_time = time_vals[surf_idx_in_clean]
                surface_power = power_vals[surf_idx_in_clean]
                frame_result["Surface_Time_us"] = surface_time
                frame_result["Surface_Power_dB"] = surface_power

                print(
                    f"Surface detected: {surface_time:.2f} μs, {surface_power:.1f} dB"
                )

            if (
                bed_idx_in_clean is not None
                and bed_idx_in_clean < len(time_vals)
                and time_vals is not None
                and power_vals is not None
            ):
                bed_time = time_vals[bed_idx_in_clean]
                bed_power = power_vals[bed_idx_in_clean]
                frame_result["Bed_Time_us"] = bed_time
                frame_result["Bed_Power_dB"] = bed_power

                print(f"Bed detected: {bed_time:.2f} μs, {bed_power:.1f} dB")

            # Calculate ice thickness if both surface and bed are available
            if not np.isnan(frame_result["Surface_Time_us"]) and not np.isnan(
                frame_result["Bed_Time_us"]
            ):
                ice_thickness = self._calculate_ice_thickness(
                    frame_result["Surface_Time_us"], frame_result["Bed_Time_us"]
                )
                frame_result["Ice_Thickness_m"] = ice_thickness

                if not np.isnan(ice_thickness):
                    print(f"Ice thickness calculated: {ice_thickness:.1f} m")

            # Store Transmitter Pulse Results
            if (
                tx_idx_in_clean is not None
                and tx_idx_in_clean < len(time_vals)
                and time_vals is not None
                and power_vals is not None
                and tx_pulse_col is not None
            ):
                tx_time = time_vals[tx_idx_in_clean]
                tx_power = power_vals[tx_idx_in_clean]

                frame_result["Transmitter_Time_us"] = tx_time
                frame_result["Transmitter_Power_dB"] = tx_power
                frame_result["Transmitter_X_pixel"] = tx_pulse_col

                print(
                    f"Transmitter pulse stored: {tx_time:.2f} μs, {tx_power:.1f} dB, X-pixel: {tx_pulse_col:.1f}"
                )

            # 10. Plot Combined Results (renamed to "picked.png")
            self._plot_combined_results(
                frame_img,
                signal_x_clean,
                signal_y_clean,
                x_major,
                y_major,
                x_minor,
                y_minor,
                ref_row,
                base_filename,
                frame_idx,
                tx_pulse_col,
                tx_idx_in_clean,
                surf_idx_in_clean,
                bed_idx_in_clean,
                power_vals,
                time_vals,
                px_per_us_echo,
                px_per_db_echo,
            )

        except Exception as e:
            print(f"ERROR: Frame {frame_idx} processing failed: {e}")
            import traceback

            traceback.print_exc()

        # Store frame result (even if processing failed)
        self.frame_results.append(frame_result)

    def _process_frame_for_override(
        self, masked_image, left, right, base_filename, frame_idx
    ):
        """
        Process frame up to automatic echo detection for interactive override.

        Returns:
            dict: Frame data containing all necessary components for override
        """
        try:
            # Extract the frame
            frame_img = masked_image[:, left:right].copy()
            h, w = frame_img.shape

            if w <= 0 or h <= 0:
                print("Warning: Frame has zero width or height.")
                return None

            # 1-7: Same processing as normal frame processing up to signal calibration
            from functions.signal_processing import (
                detect_signal_in_frame,
                trim_signal_trace,
                adaptive_peak_preserving_smooth,
            )
            from functions.grid_detection import (
                detect_grid_lines_and_dotted,
                find_reference_line_blackhat,
                interpolate_regular_grid,
            )
            from functions.echo_detection import (
                find_tx_pulse,
                detect_surface_echo_adaptive,
                detect_bed_echo,
            )

            # Signal detection
            signal_x, signal_y = detect_signal_in_frame(frame_img, self.config)
            if signal_x is None:
                print(f"Signal detection failed for frame {frame_idx}")
                return None

            signal_y = adaptive_peak_preserving_smooth(signal_y, self.config)
            signal_x_clean, signal_y_clean = trim_signal_trace(
                frame_img, signal_x, signal_y, self.config
            )

            if len(signal_x_clean) == 0:
                print(f"No valid signal trace for frame {frame_idx}")
                return None

            # TX pulse and reference line detection
            tx_pulse_col, tx_idx_in_clean = find_tx_pulse(
                signal_x_clean, signal_y_clean, self.config
            )
            ref_row = find_reference_line_blackhat(
                frame_img, base_filename, frame_idx, self.config
            )

            if tx_pulse_col is None or ref_row is None:
                print(f"Critical detection failed for frame {frame_idx}")
                return None

            # Grid detection and calibration
            h_peaks_initial, v_peaks_initial, h_minor_peaks, v_minor_peaks = (
                detect_grid_lines_and_dotted(frame_img, self.config)
            )

            y_range_dB = 8.25 * 10
            y_major, y_minor = interpolate_regular_grid(
                h,
                h_peaks_initial,
                ref_row,
                self.physical_params["y_major_dB"],
                self.physical_params["y_minor_per_major"],
                y_range_dB,
                is_y_axis=True,
                config=self.config,
            )

            x_range_us = self.physical_params.get(
                "x_range_factor"
            ) * self.physical_params.get("x_major_us")
            x_major, x_minor = interpolate_regular_grid(
                w,
                v_peaks_initial,
                tx_pulse_col,
                self.physical_params["x_major_us"],
                self.physical_params["x_minor_per_major"],
                x_range_us,
                is_y_axis=False,
                config=self.config,
            )

            # Calculate calibration factors
            px_per_us_echo, px_per_db_echo = self._calculate_calibration_factors(
                x_major, y_major, w, h
            )

            # Calibrate signal
            power_vals = (
                self.physical_params["y_ref_dB"]
                - (signal_y_clean - ref_row) / px_per_db_echo
            )
            time_vals = (signal_x_clean - tx_pulse_col) / px_per_us_echo

            # Get automatic echo detection results
            from functions.echo_detection import (
                detect_double_transmitter_pulse,
                detect_surface_echo_adaptive,
                detect_bed_echo,
            )

            tx_analysis = detect_double_transmitter_pulse(
                signal_x_clean, signal_y_clean, power_vals, time_vals, self.config
            )
            surface_idx = detect_surface_echo_adaptive(
                power_vals, time_vals, tx_analysis, self.config
            )
            bed_idx = detect_bed_echo(
                power_vals, time_vals, surface_idx, px_per_us_echo, self.config
            )

            # Use enhanced TX detection if available
            if tx_analysis.get("recommended_tx_idx") is not None:
                tx_idx_in_clean = tx_analysis["recommended_tx_idx"]

            return {
                "frame_img": frame_img,
                "signal_x_clean": signal_x_clean,
                "signal_y_clean": signal_y_clean,
                "power_vals": power_vals,
                "time_vals": time_vals,
                "tx_idx": tx_idx_in_clean,
                "surface_idx": surface_idx,
                "bed_idx": bed_idx,
                "calibration_data": {
                    "px_per_us_echo": px_per_us_echo,
                    "px_per_db_echo": px_per_db_echo,
                    "ref_row": ref_row,
                    "tx_pulse_col": tx_pulse_col,
                },
            }

        except Exception as e:
            print(f"ERROR: Frame processing for override failed: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _calculate_calibration_factors(self, x_major, y_major, w, h):
        """Calculate px_per_us and px_per_db calibration factors."""
        px_per_us, px_per_db = None, None

        # Calculate px_per_us
        if len(x_major) >= 2:
            x_spacings = np.diff(x_major)
            positive_spacings = x_spacings[x_spacings > 0]
            if len(positive_spacings) > 0:
                median_x_spacing = np.median(positive_spacings)
                if median_x_spacing > 0 and self.physical_params["x_major_us"] > 0:
                    px_per_us = abs(
                        median_x_spacing / self.physical_params["x_major_us"]
                    )

        # Calculate px_per_db
        if len(y_major) >= 2:
            y_spacings = np.diff(y_major)
            positive_spacings = y_spacings[y_spacings > 0]
            if len(positive_spacings) > 0:
                median_y_spacing = np.median(positive_spacings)
                if median_y_spacing > 0 and self.physical_params["y_major_dB"] > 0:
                    px_per_db = abs(
                        median_y_spacing / self.physical_params["y_major_dB"]
                    )

        # Use fallbacks if calculation failed
        if px_per_us is None or px_per_us <= 0:
            usable_width_fraction = self.physical_params.get(
                "usable_width_fraction", 0.8
            )
            num_x_intervals = (
                self.physical_params.get("x_range_us", 30)
                / self.physical_params.get("x_major_us", 3)
                if self.physical_params.get("x_major_us", 3) > 0
                else 5
            )
            px_per_us = (
                (w * usable_width_fraction) / num_x_intervals
                if num_x_intervals > 0
                else w / 5.0
            )

        if px_per_db is None or px_per_db <= 0:
            usable_height_fraction = self.physical_params.get(
                "usable_height_fraction", 0.8
            )
            num_y_intervals = (
                self.physical_params.get("y_range_dB", 60)
                / self.physical_params.get("y_major_dB", 10)
                if self.physical_params.get("y_major_dB", 10) > 0
                else 6
            )
            px_per_db = (
                (h * usable_height_fraction) / num_y_intervals
                if num_y_intervals > 0
                else h / 6.0
            )

        return px_per_us, px_per_db

    def _save_frame_with_manual_picks(
        self,
        frame_data,
        manual_tx,
        manual_surface,
        manual_bed,
        overrides,
        base_filename,
        frame_idx,
    ):
        """
        Save frame results with manual pick overrides and update main database files.
        """
        try:
            # Get CBD
            if (
                hasattr(self, "cbd_list")
                and self.cbd_list
                and frame_idx <= len(self.cbd_list)
            ):
                correct_cbd = self.cbd_list[frame_idx - 1]
            else:
                # Fallback: extract from filename if cbd_list not available
                import re

                match = re.search(r"C(\d+)_(\d+)", base_filename)
                if match:
                    cbd_start = int(match.group(1))
                    correct_cbd = cbd_start + frame_idx - 1
                else:
                    correct_cbd = np.nan

            # Create frame result with manual picks
            frame_result = {
                "Frame": frame_idx,
                "CBD": correct_cbd,
                "Surface_Time_us": np.nan,
                "Bed_Time_us": np.nan,
                "Ice_Thickness_m": np.nan,
                "Surface_Power_dB": np.nan,
                "Bed_Power_dB": np.nan,
                "Transmitter_Time_us": np.nan,
                "Transmitter_Power_dB": np.nan,
                "Transmitter_X_pixel": np.nan,
            }

            # Extract data
            power_vals = frame_data["power_vals"]
            time_vals = frame_data["time_vals"]

            # Store results using manual picks
            if manual_surface is not None and manual_surface < len(time_vals):
                frame_result["Surface_Time_us"] = time_vals[manual_surface]
                frame_result["Surface_Power_dB"] = power_vals[manual_surface]

            if manual_bed is not None and manual_bed < len(time_vals):
                frame_result["Bed_Time_us"] = time_vals[manual_bed]
                frame_result["Bed_Power_dB"] = power_vals[manual_bed]

            if manual_tx is not None and manual_tx < len(time_vals):
                frame_result["Transmitter_Time_us"] = time_vals[manual_tx]
                frame_result["Transmitter_Power_dB"] = power_vals[manual_tx]
                frame_result["Transmitter_X_pixel"] = frame_data["calibration_data"][
                    "tx_pulse_col"
                ]

            # Calculate ice thickness
            if not np.isnan(frame_result["Surface_Time_us"]) and not np.isnan(
                frame_result["Bed_Time_us"]
            ):
                ice_thickness = self._calculate_ice_thickness(
                    frame_result["Surface_Time_us"], frame_result["Bed_Time_us"]
                )
                frame_result["Ice_Thickness_m"] = ice_thickness

            print(f"INFO: Updated frame {frame_idx} with manual picks:")
            print(f"  CBD: {frame_result['CBD']}")
            if manual_tx is not None and manual_tx < len(time_vals):
                print(
                    f"  Transmitter: {time_vals[manual_tx]:.2f} μs {'(Manual)' if overrides.get('transmitter') else '(Auto)'}"
                )
            if not np.isnan(frame_result["Surface_Time_us"]):
                print(
                    f"  Surface: {frame_result['Surface_Time_us']:.2f} μs {'(Manual)' if overrides.get('surface') else '(Auto)'}"
                )
            if not np.isnan(frame_result["Bed_Time_us"]):
                print(
                    f"  Bed: {frame_result['Bed_Time_us']:.2f} μs {'(Manual)' if overrides.get('bed') else '(Auto)'}"
                )
            if not np.isnan(frame_result["Ice_Thickness_m"]):
                print(f"  Ice thickness: {frame_result['Ice_Thickness_m']:.1f} m")

            # Update the main CSV file
            main_csv_path = Path(self.output_dir) / f"{base_filename}_pick.csv"
            main_npz_path = Path(self.output_dir) / f"{base_filename}_pick.npz"

            if main_csv_path.exists():
                # Load existing CSV data
                existing_df = pd.read_csv(main_csv_path)

                # Update only the specific columns, preserve CBD
                frame_mask = existing_df["Frame"] == frame_idx
                if frame_mask.any():
                    # Update only the data columns that were manually changed
                    update_columns = [
                        "Surface_Time_us",
                        "Bed_Time_us",
                        "Ice_Thickness_m",
                        "Surface_Power_dB",
                        "Bed_Power_dB",
                        "Transmitter_Time_us",
                        "Transmitter_Power_dB",
                        "Transmitter_X_pixel",
                    ]

                    for col in update_columns:
                        if col in frame_result:
                            existing_df.loc[frame_mask, col] = frame_result[col]

                    # Ensure CBD is preserved/corrected
                    existing_df.loc[frame_mask, "CBD"] = correct_cbd

                    print(f"INFO: Updated frame {frame_idx} in existing CSV database")
                else:
                    # Append new row if frame doesn't exist
                    new_row_df = pd.DataFrame([frame_result])
                    existing_df = pd.concat(
                        [existing_df, new_row_df], ignore_index=True
                    )
                    existing_df = existing_df.sort_values("Frame").reset_index(
                        drop=True
                    )
                    print(f"INFO: Added frame {frame_idx} to CSV database")

                # Save updated CSV
                existing_df.to_csv(
                    main_csv_path, index=False, float_format="%.6f", na_rep="NaN"
                )
                print(f"INFO: Updated main CSV database: {main_csv_path}")

                # Update the main NPZ file
                if main_npz_path.exists():
                    # Load existing NPZ metadata
                    try:
                        npz_data = np.load(main_npz_path, allow_pickle=True)
                        meta_info = (
                            npz_data["meta"].item() if "meta" in npz_data else {}
                        )
                    except:
                        meta_info = {}

                    # Update metadata
                    meta_info["last_manual_override"] = str(datetime.datetime.now())
                    meta_info["manual_override_frame"] = frame_idx
                    meta_info["override_types"] = [k for k, v in overrides.items() if v]
                else:
                    meta_info = {
                        "ascope_file": base_filename,
                        "processing_timestamp": str(datetime.datetime.now()),
                        "ice_velocity_m_per_us": 168.0,
                        "manual_override_frame": frame_idx,
                        "override_types": [k for k, v in overrides.items() if v],
                    }

                # Save updated NPZ with corrected data from CSV
                np.savez(
                    main_npz_path,
                    frame=existing_df["Frame"].values,
                    cbd=existing_df["CBD"].values,
                    surface_time_us=existing_df["Surface_Time_us"].values,
                    bed_time_us=existing_df["Bed_Time_us"].values,
                    ice_thickness_m=existing_df["Ice_Thickness_m"].values,
                    surface_power_db=existing_df["Surface_Power_dB"].values,
                    bed_power_db=existing_df["Bed_Power_dB"].values,
                    meta=meta_info,
                )
                print(f"INFO: Updated main NPZ database: {main_npz_path}")

            else:
                print("WARNING: Main CSV file not found, creating new database")
                # Create new database with single frame
                df = pd.DataFrame([frame_result])
                df.to_csv(main_csv_path, index=False, float_format="%.6f", na_rep="NaN")

                meta_info = {
                    "ascope_file": base_filename,
                    "processing_timestamp": str(datetime.datetime.now()),
                    "ice_velocity_m_per_us": 168.0,
                    "manual_override_frame": frame_idx,
                    "override_types": [k for k, v in overrides.items() if v],
                }

                np.savez(
                    main_npz_path,
                    frame=df["Frame"].values,
                    cbd=df["CBD"].values,
                    surface_time_us=df["Surface_Time_us"].values,
                    bed_time_us=df["Bed_Time_us"].values,
                    ice_thickness_m=df["Ice_Thickness_m"].values,
                    surface_power_db=df["Surface_Power_dB"].values,
                    bed_power_db=df["Bed_Power_dB"].values,
                    meta=meta_info,
                )

            # Reconstruct grid data and replace the existing picked.png file
            try:
                from functions.grid_detection import (
                    detect_grid_lines_and_dotted,
                    interpolate_regular_grid,
                )

                frame_img = frame_data["frame_img"]
                h, w = frame_img.shape
                ref_row = frame_data["calibration_data"]["ref_row"]
                tx_pulse_col = frame_data["calibration_data"]["tx_pulse_col"]

                # Re-detect grid lines for proper plotting
                h_peaks_initial, v_peaks_initial, h_minor_peaks, v_minor_peaks = (
                    detect_grid_lines_and_dotted(frame_img, self.config)
                )

                # Reconstruct grid data
                y_range_dB = 8.25 * 10
                y_major, y_minor = interpolate_regular_grid(
                    h,
                    h_peaks_initial,
                    ref_row,
                    self.physical_params["y_major_dB"],
                    self.physical_params["y_minor_per_major"],
                    y_range_dB,
                    is_y_axis=True,
                    config=self.config,
                )

                x_range_us = self.physical_params.get(
                    "x_range_factor"
                ) * self.physical_params.get("x_major_us")
                x_major, x_minor = interpolate_regular_grid(
                    w,
                    v_peaks_initial,
                    tx_pulse_col,
                    self.physical_params["x_major_us"],
                    self.physical_params["x_minor_per_major"],
                    x_range_us,
                    is_y_axis=False,
                    config=self.config,
                )

            except Exception as e:
                print(f"WARNING: Could not reconstruct grid data: {e}")
                # Use empty grid data as fallback
                y_major, y_minor, x_major, x_minor = [], [], [], []

            # Replace the existing picked.png file
            self._plot_combined_results(
                frame_data["frame_img"],
                frame_data["signal_x_clean"],
                frame_data["signal_y_clean"],
                x_major,  # Reconstructed grid data
                y_major,  # Reconstructed grid data
                x_minor,  # Reconstructed grid data
                y_minor,  # Reconstructed grid data
                frame_data["calibration_data"]["ref_row"],
                base_filename,
                frame_idx,
                frame_data["calibration_data"]["tx_pulse_col"],
                manual_tx,
                manual_surface,
                manual_bed,
                power_vals,
                time_vals,
                frame_data["calibration_data"]["px_per_us_echo"],
                frame_data["calibration_data"]["px_per_db_echo"],
            )

            print(
                f"INFO: Replaced frame plot: {base_filename}_frame{frame_idx:02d}_picked.png"
            )
            return True

        except Exception as e:
            print(f"ERROR: Failed to save manual picks: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _plot_combined_results(
        self,
        frame_img,
        signal_x_clean,
        signal_y_clean,
        x_major,
        y_major,
        x_minor,
        y_minor,
        ref_row,
        base_filename,
        frame_idx,
        tx_pulse_col,
        tx_idx_in_clean,
        surf_idx_in_clean,
        bed_idx_in_clean,
        power_vals,
        time_vals,
        px_per_us,
        px_per_db,
    ):
        """Generate and save a combined plot with debug view and calibrated view."""
        h, w = frame_img.shape

        # RENAMED: Use "picked.png" instead of "combined_annotated.png"
        plot_filename = (
            f"{self.output_dir}/{base_filename}_frame{frame_idx:02d}_picked.png"
        )

        # Create the figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # --- Plot 1: Debug View (Image + Overlays) ---
        ax_debug = axes[0]
        ax_debug.imshow(frame_img, cmap="gray", aspect="auto")

        # Plot signal trace
        if signal_x_clean is not None and len(signal_x_clean) > 0:
            ax_debug.plot(
                signal_x_clean,
                signal_y_clean,
                "r-",
                linewidth=1,
                label="Detected Trace",
            )

        # Plot grid lines
        grid_line_color = "#00BFFF"  # Deep sky blue
        major_alpha, minor_alpha = 0.6, 0.3

        for y in y_major:
            ax_debug.axhline(
                y,
                color=grid_line_color,
                linestyle="-",
                alpha=major_alpha,
                linewidth=0.8,
            )

        for y in y_minor:
            ax_debug.axhline(
                y,
                color=grid_line_color,
                linestyle=":",
                alpha=minor_alpha,
                linewidth=0.8,
            )

        for x in x_major:
            ax_debug.axvline(
                x,
                color=grid_line_color,
                linestyle="-",
                alpha=major_alpha,
                linewidth=0.8,
            )

        for x in x_minor:
            ax_debug.axvline(
                x,
                color=grid_line_color,
                linestyle=":",
                alpha=minor_alpha,
                linewidth=1,
            )

        # Highlight Reference Line and Tx Column
        if ref_row is not None:
            ax_debug.axhline(
                ref_row,
                color="lime",
                linestyle=":",
                linewidth=0.8,
                label=f"Ref Line ({self.physical_params['y_ref_dB']} dB)",
            )

        if tx_pulse_col is not None:
            ax_debug.axvline(
                tx_pulse_col,
                color="orange",
                linestyle=":",
                linewidth=1.5,
                label="Tx Time Zero",
            )

        # Mark detected echoes
        valid_signal = (
            signal_x_clean is not None
            and len(signal_x_clean) > 0
            and signal_y_clean is not None
            and len(signal_y_clean) == len(signal_x_clean)
        )

        if valid_signal:
            if tx_idx_in_clean is not None and tx_idx_in_clean < len(signal_x_clean):
                ax_debug.plot(
                    signal_x_clean[tx_idx_in_clean],
                    signal_y_clean[tx_idx_in_clean],
                    "bo",
                    ms=6,
                    label="Tx",
                )

            if surf_idx_in_clean is not None and surf_idx_in_clean < len(
                signal_x_clean
            ):
                ax_debug.plot(
                    signal_x_clean[surf_idx_in_clean],
                    signal_y_clean[surf_idx_in_clean],
                    "go",
                    ms=6,
                    label="Surf",
                )

            if bed_idx_in_clean is not None and bed_idx_in_clean < len(signal_x_clean):
                ax_debug.plot(
                    signal_x_clean[bed_idx_in_clean],
                    signal_y_clean[bed_idx_in_clean],
                    "mo",
                    ms=6,
                    label="Bed",
                )

        ax_debug.set_title(f"A-scope Frame {frame_idx} (Debug View)")
        ax_debug.set_ylim(h, 0)
        ax_debug.set_xlim(0, w)
        ax_debug.axis("on")
        ax_debug.set_xticks([])
        ax_debug.set_yticks([])
        ax_debug.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, -0.15))

        # --- Plot 2: Calibrated View (Time vs Power) ---
        ax_calib = axes[1]

        if time_vals is not None and power_vals is not None and len(time_vals) > 0:
            ax_calib.plot(time_vals, power_vals, "r-", linewidth=1.2)
        else:
            ax_calib.text(
                0.5,
                0.5,
                "Calibration Failed",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax_calib.transAxes,
            )

        # Plot physical grid
        plot_min_db = -65
        plot_max_db = 2.5

        # Define major and minor grid lines
        major_db_ticks = np.arange(
            self.physical_params["y_ref_dB"],
            plot_max_db + 1,
            self.physical_params["y_major_dB"],
        )
        minor_db_per_major = self.physical_params["y_major_dB"] / (
            self.physical_params["y_minor_per_major"] + 1
        )
        minor_db_ticks = np.arange(
            plot_min_db, plot_max_db + minor_db_per_major, minor_db_per_major
        )

        # Draw horizontal grid lines
        for db in major_db_ticks:
            ax_calib.axhline(
                db,
                color=grid_line_color,
                linestyle="-",
                alpha=major_alpha,
                linewidth=1.0,
            )

        for db in minor_db_ticks:
            ax_calib.axhline(
                db,
                color=grid_line_color,
                linestyle=":",
                alpha=minor_alpha,
                linewidth=0.8,
            )

        # Define time range
        plot_min_time = -1  # Start slightly before Tx
        plot_max_time = self.physical_params["x_range_us"] + 2
        if time_vals is not None and len(time_vals) > 0:
            plot_max_time = max(plot_max_time, np.ceil(time_vals.max()) + 2)

        major_time_ticks = np.arange(
            0, plot_max_time, self.physical_params["x_major_us"]
        )

        minor_time_per_major = self.physical_params["x_major_us"] / (
            self.physical_params["x_minor_per_major"] + 1
        )
        minor_time_ticks = np.arange(plot_min_time, plot_max_time, minor_time_per_major)

        # Draw vertical grid lines
        for t in major_time_ticks:
            ax_calib.axvline(
                t,
                color=grid_line_color,
                linestyle="-",
                alpha=major_alpha,
                linewidth=1.0,
            )

        for t in minor_time_ticks:
            ax_calib.axvline(
                t,
                color=grid_line_color,
                linestyle=":",
                alpha=minor_alpha,
                linewidth=0.8,
            )

        # Highlight reference line
        ax_calib.axhline(
            self.physical_params["y_ref_dB"],
            color="lime",
            linestyle=":",
            linewidth=1,
            label=f"Reference ({self.physical_params['y_ref_dB']} dB)",
        )

        # Set ticks and labels
        from matplotlib.ticker import FixedLocator

        # Y-axis labels (every 20 dB)
        y_label_step = 20
        y_major_labels = np.arange(
            self.physical_params["y_ref_dB"], plot_max_db + 1, y_label_step
        )

        ax_calib.set_yticks(y_major_labels)
        ax_calib.set_yticklabels([f"{int(db)}" for db in y_major_labels], fontsize=10)
        ax_calib.yaxis.set_minor_locator(FixedLocator(minor_db_ticks))

        # X-axis labels
        ax_calib.set_xticks(major_time_ticks)
        ax_calib.set_xticklabels([f"{int(t)}" for t in major_time_ticks], fontsize=10)
        ax_calib.xaxis.set_minor_locator(FixedLocator(minor_time_ticks))

        ax_calib.set_xlabel("One-way travel time (µs)")
        ax_calib.set_ylabel("Power (dB)")
        ax_calib.set_title(f"Calibrated A-scope Frame {frame_idx}")

        # Set plot limits
        ax_calib.set_ylim(plot_min_db, plot_max_db)
        ax_calib.set_xlim(
            plot_min_time,
            plot_max_time - 1 if plot_max_time > plot_min_time + 1 else plot_max_time,
        )

        # Annotate echoes on calibrated plot
        if valid_signal and time_vals is not None and len(time_vals) > 0:
            if tx_idx_in_clean is not None and tx_idx_in_clean < len(time_vals):
                ax_calib.plot(
                    time_vals[tx_idx_in_clean],
                    power_vals[tx_idx_in_clean],
                    "o",
                    color="blue",
                    label="Tx",
                    markersize=6,
                )

            if surf_idx_in_clean is not None and surf_idx_in_clean < len(time_vals):
                ax_calib.plot(
                    time_vals[surf_idx_in_clean],
                    power_vals[surf_idx_in_clean],
                    "o",
                    color="green",
                    label="Surface",
                    markersize=6,
                )

            if bed_idx_in_clean is not None and bed_idx_in_clean < len(time_vals):
                ax_calib.plot(
                    time_vals[bed_idx_in_clean],
                    power_vals[bed_idx_in_clean],
                    "o",
                    color="magenta",
                    label="Bed",
                    markersize=6,
                )

        # Add legend to calibrated plot
        handles, labels = ax_calib.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax_calib.legend(
                by_label.values(), by_label.keys(), loc="upper right", fontsize=9
            )

        # Add annotations with arrows for detected points
        if valid_signal and time_vals is not None and len(time_vals) > 0:
            # Annotate Tx point
            if tx_idx_in_clean is not None and tx_idx_in_clean < len(time_vals):
                tx_time = time_vals[tx_idx_in_clean]
                tx_power = power_vals[tx_idx_in_clean]
                tx_label = (
                    f"transmitter pulse\n(~{tx_power:.1f} dB at {tx_time:.1f} μs)"
                )

                ax_calib.annotate(
                    tx_label,
                    xy=(tx_time, tx_power),
                    xytext=(-50, -40),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7),
                    arrowprops=dict(
                        arrowstyle="->", connectionstyle="arc3,rad=0.3", color="blue"
                    ),
                )

            # Annotate Surface echo point
            if surf_idx_in_clean is not None and surf_idx_in_clean < len(time_vals):
                surf_time = time_vals[surf_idx_in_clean]
                surf_power = power_vals[surf_idx_in_clean]
                surf_label = f"surface\n(~{surf_power:.1f} dB at {surf_time:.1f} μs)"

                ax_calib.annotate(
                    surf_label,
                    xy=(surf_time, surf_power),
                    xytext=(-60, -15),
                    textcoords="offset points",
                    ha="right",
                    va="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7),
                    arrowprops=dict(
                        arrowstyle="->",
                        connectionstyle="arc3,rad=0.3",
                        color="green",
                    ),
                )

            # Annotate Bed echo point
            if bed_idx_in_clean is not None and bed_idx_in_clean < len(time_vals):
                bed_time = time_vals[bed_idx_in_clean]
                bed_power = power_vals[bed_idx_in_clean]
                bed_label = f"bed\n(~{bed_power:.1f} dB at {bed_time:.1f} μs)"

                ax_calib.annotate(
                    bed_label,
                    xy=(bed_time, bed_power),
                    xytext=(60, -15),
                    textcoords="offset points",
                    ha="left",
                    va="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7),
                    arrowprops=dict(
                        arrowstyle="->",
                        connectionstyle="arc3,rad=-0.3",
                        color="magenta",
                    ),
                )
        plt.tight_layout(pad=1.5)
        plt.savefig(plot_filename, dpi=self.output_config.get("plot_dpi", 200))
        print(f"Saved combined plot: {plot_filename}")
        plt.close(fig)
