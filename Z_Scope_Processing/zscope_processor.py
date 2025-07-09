import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import cv2
import matplotlib.pyplot as plt

from functions.image_utils import load_and_preprocess_image
from functions.artifact_detection import (
    detect_film_artifact_boundaries,
    detect_zscope_boundary,
)
from functions.feature_detection import (
    detect_transmitter_pulse,
    detect_calibration_pip,
)
from functions.calibration_utils import calculate_pixels_per_microsecond
from functions.visualization_utils import create_time_calibrated_zscope
from functions.echo_tracing import detect_surface_echo, detect_bed_echo


class ZScopeProcessor:
    def __init__(
        self,
        config_path="config/default_config.json",
        physics_path="config/physical_constants.json",
    ):
        processor_script_dir = Path(__file__).resolve().parent
        config_path_obj = Path(config_path)
        physics_path_obj = Path(physics_path)

        if not config_path_obj.is_absolute():
            resolved_config_path = processor_script_dir / config_path_obj
        else:
            resolved_config_path = config_path_obj

        if not physics_path_obj.is_absolute():
            resolved_physics_path = processor_script_dir / physics_path_obj
        else:
            resolved_physics_path = physics_path_obj

        with open(resolved_config_path, "r") as f:
            self.config = json.load(f)

        with open(resolved_physics_path, "r") as f:
            self.physics_constants = json.load(f)

        # Initialize instance variables
        self.image_np = None
        self.base_filename = None
        self.data_top_abs = None
        self.data_bottom_abs = None
        self.transmitter_pulse_y_abs = None
        self.best_pip_details = None
        self.pixels_per_microsecond = None
        self.calibrated_fig = None
        self.calibrated_ax = None
        self.detected_surface_y_abs = None
        self.detected_bed_y_abs = None
        self.time_axis = None
        self.output_dir = None
        self.last_pip_details = None
        self.calculated_ticks = None  # For storing CBD tick positions
        self._parameters_were_optimized = False  # Track if parameters were updated

    def save_calpip_state(self, state_path):
        import numpy as np

        def make_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(v) for v in obj]
            else:
                return obj

        if self.best_pip_details is not None:
            serializable_pip = make_json_serializable(self.best_pip_details)
            with open(state_path, "w") as f:
                json.dump(serializable_pip, f, indent=4)
            print(f"INFO: Calpip state saved to {state_path}")
        else:
            print("WARNING: No calpip details to save.")

    def load_calpip_state(self, state_path):
        state_file = Path(state_path)
        if state_file.exists():
            with open(state_file, "r") as f:
                self.best_pip_details = json.load(f)
            self.last_pip_details = self.best_pip_details
            print(f"INFO: Calpip state loaded from {state_path}")
        else:
            print(f"WARNING: Calpip state file {state_path} does not exist.")

    def export_enhanced_csv_with_coordinates(
        self, output_dir, nav_df=None, cbd_tick_xs=None
    ):
        """
        Export comprehensive 7-column CSV with full-resolution data and coordinate interpolation.

        Args:
            output_dir: Output directory path
            nav_df: Navigation DataFrame with CBD coordinates
            cbd_tick_xs: List of CBD tick mark x-positions

        Returns:
            DataFrame with enhanced data or None if failed
        """
        output_path = Path(output_dir) / f"{self.base_filename}_thickness.csv"

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if (
            self.detected_surface_y_abs is None
            or self.detected_bed_y_abs is None
            or len(self.detected_surface_y_abs) != len(self.detected_bed_y_abs)
        ):
            print("WARNING: Enhanced CSV not exported due to missing echo data.")
            return None

        print(f"INFO: Starting enhanced CSV export to {output_path}")

        # Create full-resolution x-pixel array
        x_pixels = np.arange(len(self.detected_surface_y_abs))

        # Convert to one-way travel times (microseconds)
        surface_time_us = self._convert_pixels_to_one_way_time(
            self.detected_surface_y_abs
        )
        bed_time_us = self._convert_pixels_to_one_way_time(self.detected_bed_y_abs)

        # Calculate ice thickness in meters
        ice_thickness_meters = self._calculate_ice_thickness_meters(
            self.detected_surface_y_abs, self.detected_bed_y_abs
        )

        # Initialize coordinate arrays
        cbd_numbers = np.full(len(x_pixels), np.nan, dtype=object)
        latitudes = np.full(len(x_pixels), np.nan)
        longitudes = np.full(len(x_pixels), np.nan)

        # Coordinate interpolation if navigation data is available
        if nav_df is not None and cbd_tick_xs is not None and len(cbd_tick_xs) > 0:
            try:
                print(
                    f"INFO: Interpolating coordinates for {len(cbd_tick_xs)} CBD positions"
                )
                cbd_numbers, latitudes, longitudes = (
                    self._interpolate_coordinates_full_resolution(
                        x_pixels, cbd_tick_xs, nav_df
                    )
                )
                print(
                    f"INFO: Successfully interpolated coordinates for {np.sum(~np.isnan(latitudes))} pixels"
                )
            except Exception as e:
                print(f"WARNING: Coordinate interpolation failed: {e}")
        else:
            print(
                "INFO: No navigation data or CBD positions available for coordinate interpolation"
            )

        # Create the required 7-column DataFrame
        df = pd.DataFrame(
            {
                "X (pixel)": x_pixels,
                "Latitude": latitudes,
                "Longitude": longitudes,
                "CBD": cbd_numbers,
                "Surface Depth (μs)": surface_time_us,
                "Bed Depth (μs)": bed_time_us,
                "Ice Thickness (m)": ice_thickness_meters,
            }
        )

        # Export with high precision
        try:
            df.to_csv(output_path, index=False, float_format="%.6f")
            print(f"SUCCESS: Enhanced 7-column CSV exported to: {output_path}")
            print(f"INFO: Data summary: {len(df)} rows, {len(df.columns)} columns")
            print(
                f"INFO: Coordinate coverage: {np.sum(~np.isnan(latitudes))}/{len(x_pixels)} pixels"
            )
            return df
        except Exception as e:
            print(f"ERROR: Failed to save CSV file: {e}")
            return None

    def _convert_pixels_to_one_way_time(self, y_pixels):
        """
        Convert pixel positions to one-way travel time in microseconds.
        """
        if self.transmitter_pulse_y_abs is None or self.pixels_per_microsecond is None:
            return np.full_like(y_pixels, np.nan)

        # Convert to relative pixel position from transmitter pulse
        y_relative = y_pixels - self.transmitter_pulse_y_abs

        # Convert to two-way travel time
        two_way_time_us = y_relative / self.pixels_per_microsecond

        # Convert to one-way travel time
        one_way_time_us = two_way_time_us / 2.0

        return one_way_time_us

    def _calculate_ice_thickness_meters(self, surface_y_pixels, bed_y_pixels):
        """
        Calculate ice thickness in meters using proper one-way travel times.
        """
        if self.transmitter_pulse_y_abs is None or self.pixels_per_microsecond is None:
            return np.full_like(surface_y_pixels, np.nan)

        # Get one-way travel times
        surface_time_us = self._convert_pixels_to_one_way_time(surface_y_pixels)
        bed_time_us = self._convert_pixels_to_one_way_time(bed_y_pixels)

        # Calculate travel time difference (one-way through ice)
        ice_travel_time_us = bed_time_us - surface_time_us

        # Convert to meters using physical constants
        c0 = self.physics_constants.get("speed_of_light_vacuum_mps", 299792458)
        epsilon_r = self.physics_constants.get("ice_relative_permittivity_real", 3.17)
        firn_correction = self.physics_constants.get("firn_correction_meters", 0.0)

        # Calculate ice velocity and thickness
        ice_velocity = c0 / np.sqrt(epsilon_r)  # m/s
        time_in_seconds = ice_travel_time_us * 1e-6  # Convert μs to seconds
        ice_thickness = (time_in_seconds * ice_velocity) + firn_correction

        return ice_thickness

    def _interpolate_coordinates_full_resolution(self, x_pixels, cbd_tick_xs, nav_df):
        """
        Interpolate Bingham coordinates for all x-pixels with full resolution.
        """
        import re

        # Initialize output arrays
        cbd_numbers = np.full(len(x_pixels), np.nan, dtype=object)
        latitudes = np.full(len(x_pixels), np.nan)
        longitudes = np.full(len(x_pixels), np.nan)

        # Extract CBD range from filename
        cbd_match = re.search(r"C(\d+)_(\d+)", self.base_filename)
        if not cbd_match:
            print(
                f"Warning: Could not extract CBD range from filename: {self.base_filename}"
            )
            return cbd_numbers, latitudes, longitudes

        cbd_start = int(cbd_match.group(1))
        cbd_end = int(cbd_match.group(2))

        # Create CBD sequence (descending order: left to right)
        if cbd_start > cbd_end:
            cbd_sequence = list(range(cbd_start, cbd_end - 1, -1))
        else:
            cbd_sequence = list(range(cbd_start, cbd_end + 1))
            cbd_sequence.reverse()

        # Match CBD tick positions with known coordinates
        valid_cbd_data = []

        for i, tick_x in enumerate(cbd_tick_xs):
            if i < len(cbd_sequence):
                cbd_num = cbd_sequence[i]

                # Find navigation data for this CBD
                nav_row = nav_df[nav_df["CBD"] == cbd_num]
                if not nav_row.empty:
                    valid_cbd_data.append(
                        {
                            "cbd": cbd_num,
                            "x_pos": tick_x,
                            "lat": nav_row["LAT (bingham)"].values[0],
                            "lon": nav_row["LON (bingham)"].values[0],
                        }
                    )

        if len(valid_cbd_data) < 2:
            print("Warning: Need at least 2 valid CBD positions for interpolation")
            return cbd_numbers, latitudes, longitudes

        # Extract coordinate arrays for interpolation
        tick_x_coords = np.array([d["x_pos"] for d in valid_cbd_data])
        tick_lats = np.array([d["lat"] for d in valid_cbd_data])
        tick_lons = np.array([d["lon"] for d in valid_cbd_data])

        # Create interpolation functions
        try:
            lat_interp = interp1d(
                tick_x_coords,
                tick_lats,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            lon_interp = interp1d(
                tick_x_coords,
                tick_lons,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )

            # Interpolate for all x-pixels
            interpolated_lats = lat_interp(x_pixels)
            interpolated_lons = lon_interp(x_pixels)

            # Only keep interpolations within reasonable bounds
            x_min, x_max = np.min(tick_x_coords), np.max(tick_x_coords)
            valid_range = (x_pixels >= x_min) & (x_pixels <= x_max)

            latitudes[valid_range] = interpolated_lats[valid_range]
            longitudes[valid_range] = interpolated_lons[valid_range]

            # Mark CBD positions where tick marks exist
            for data in valid_cbd_data:
                closest_pixel_idx = np.argmin(np.abs(x_pixels - data["x_pos"]))
                cbd_numbers[closest_pixel_idx] = data["cbd"]

            print(f"Interpolated coordinates for {np.sum(valid_range)} pixels")

        except Exception as e:
            print(f"Error in coordinate interpolation: {e}")

        return cbd_numbers, latitudes, longitudes

    def save_optimized_parameters(self, output_dir):
        """Save optimized parameters for use in subsequent images."""
        params_file = Path(output_dir) / "optimized_echo_params.json"

        # Extract current echo tracing parameters
        echo_params = self.config.get("echo_tracing_params", {})

        # Add metadata
        optimized_data = {
            "timestamp": str(pd.Timestamp.now()),
            "source_image": self.base_filename,
            "echo_tracing_params": echo_params,
            "optimization_method": "user_guided_search_parameters",
            "flight_sequence": True,
        }

        with open(params_file, "w") as f:
            json.dump(optimized_data, f, indent=4)

        print(f"INFO: Saved optimized parameters to {params_file}")

    def load_previous_optimized_parameters(self, output_dir):
        """Load optimized parameters from previous image processing."""
        params_file = Path(output_dir) / "optimized_echo_params.json"

        if params_file.exists():
            try:
                with open(params_file, "r") as f:
                    optimized_data = json.load(f)

                # Update current configuration with optimized parameters
                if "echo_tracing_params" in optimized_data:
                    self.config["echo_tracing_params"].update(
                        optimized_data["echo_tracing_params"]
                    )
                    print(f"INFO: Loaded optimized parameters from previous processing")
                    print(
                        f"INFO: Source: {optimized_data.get('source_image', 'unknown')}"
                    )
                    return True
            except Exception as e:
                print(f"WARNING: Could not load optimized parameters: {e}")

        return False

    def _show_automatic_results_for_approval(self, valid_data_crop):
        """
        Display automatic detection results and get user approval.

        Returns:
            bool: True if user is satisfied, False if they want to optimize parameters
        """
        fig, ax = plt.subplots(figsize=(24, 12))

        # Display image with automatic detection results
        enhanced = cv2.createCLAHE(clipLimit=3.0).apply(valid_data_crop)
        ax.imshow(enhanced, cmap="gray", aspect="auto")

        # Plot automatic traces
        x_coords = np.arange(len(self.detected_surface_y_abs))

        # Surface trace
        surface_y_rel = self.detected_surface_y_abs - self.data_top_abs
        valid_surface = np.isfinite(surface_y_rel)
        if np.any(valid_surface):
            ax.plot(
                x_coords[valid_surface],
                surface_y_rel[valid_surface],
                "cyan",
                linewidth=2,
                label="Automatic Surface Detection",
                alpha=0.8,
            )

        # Bed trace
        bed_y_rel = self.detected_bed_y_abs - self.data_top_abs
        valid_bed = np.isfinite(bed_y_rel)
        if np.any(valid_bed):
            ax.plot(
                x_coords[valid_bed],
                bed_y_rel[valid_bed],
                "orange",
                linewidth=2,
                label="Automatic Bed Detection",
                alpha=0.8,
            )

        # Calculate and display quality metrics
        surface_coverage = (
            np.sum(valid_surface) / len(surface_y_rel) * 100
            if len(surface_y_rel) > 0
            else 0
        )
        bed_coverage = (
            np.sum(valid_bed) / len(bed_y_rel) * 100 if len(bed_y_rel) > 0 else 0
        )

        ax.set_title(
            f"Automatic Echo Detection Results\n"
            f"Surface Coverage: {surface_coverage:.1f}% | Bed Coverage: {bed_coverage:.1f}%\n"
            f"Review results and decide if parameter optimization is needed",
            fontsize=16,
            fontweight="bold",
        )
        ax.legend()

        # Add instruction text
        ax.text(
            0.02,
            0.02,
            "AUTOMATIC DETECTION REVIEW:\n"
            "• Cyan = Automatic surface detection\n"
            "• Orange = Automatic bed detection\n"
            "• Close window to continue",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9),
        )

        plt.tight_layout()
        plt.show()

        # Get user satisfaction feedback
        while True:
            user_input = (
                input(
                    "\nAre you satisfied with the automatic echo detection results?\n"
                    "Enter 'y' to proceed to CBD selection, 'n' to optimize parameters: "
                )
                .strip()
                .lower()
            )

            if user_input in ["y", "yes"]:
                return True
            elif user_input in ["n", "no"]:
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no")

    def _run_user_guided_calibration(self, valid_data_crop):
        """
        Run user-guided calibration workflow.

        Returns:
            dict: Optimized parameters or None if cancelled
        """
        from functions.interactive_tools import EchoPointSelector
        from functions.echo_calibration import analyze_echo_points
        from functions.echo_validation import EchoValidationInterface

        # Phase 1: User selects representative points
        print(
            "Please select 5 representative points each for surface and bed echoes..."
        )
        print("Click directly on clear echo peaks for more accurate parameter tuning.")

        point_selector = EchoPointSelector(
            valid_data_crop,
            title="Select Echo Calibration Points - Surface and Bed Echoes",
        )

        surface_points, bed_points = point_selector.start_selection()

        if not surface_points or not bed_points:
            print("Insufficient point selections - cancelling guided calibration")
            return None

        print(
            f"Selected {len(surface_points)} surface points and {len(bed_points)} bed points"
        )

        # Phase 2: Analyze points and optimize parameters
        print("Analyzing selected points to optimize detection parameters...")

        optimized_params = analyze_echo_points(
            valid_data_crop, surface_points, bed_points
        )

        return optimized_params

    def _update_search_parameters(self, optimized_params):
        """Update configuration with optimized search parameters only."""
        echo_tracing_config = self.config.get("echo_tracing_params", {})

        # Apply optimized surface search parameters only
        if "surface_detection" in optimized_params:
            surface_config = echo_tracing_config.get("surface_detection", {})

            # ONLY update search window parameters
            search_params = optimized_params["surface_detection"]
            if "search_start_offset_px" in search_params:
                surface_config["search_start_offset_px"] = search_params[
                    "search_start_offset_px"
                ]
            if "search_depth_px" in search_params:
                surface_config["search_depth_px"] = search_params["search_depth_px"]

            echo_tracing_config["surface_detection"] = surface_config
            print(f"Updated surface search parameters: {search_params}")

        # Apply optimized bed search parameters only
        if "bed_detection" in optimized_params:
            bed_config = echo_tracing_config.get("bed_detection", {})

            # ONLY update search window parameters
            search_params = optimized_params["bed_detection"]
            if "search_start_offset_px" in search_params:
                bed_config["search_start_offset_px"] = search_params[
                    "search_start_offset_px"
                ]
            if "search_depth_px" in search_params:
                bed_config["search_depth_px"] = search_params["search_depth_px"]

            echo_tracing_config["bed_detection"] = bed_config
            print(f"Updated bed search parameters: {search_params}")

        # Update the main config
        self.config["echo_tracing_params"] = echo_tracing_config
        self._parameters_were_optimized = True

        print("Search window parameters optimized based on your point selections!")
        print("Peak detection and enhancement parameters remain at default values.")

    def _run_default_echo_detection(self):
        """Run default automatic echo detection without user guidance."""
        if (
            self.image_np is not None
            and self.data_top_abs is not None
            and self.data_bottom_abs is not None
            and self.transmitter_pulse_y_abs is not None
            and self.best_pip_details is not None
            and self.pixels_per_microsecond is not None
        ):
            valid_data_crop = self.image_np[self.data_top_abs : self.data_bottom_abs, :]
            crop_height, crop_width = valid_data_crop.shape
            tx_pulse_y_rel = self.transmitter_pulse_y_abs - self.data_top_abs
            z_boundary_y_abs_for_echo_search = self.data_bottom_abs
            z_boundary_y_rel = z_boundary_y_abs_for_echo_search - self.data_top_abs

            # Use DEFAULT configuration parameters (no optimization)
            echo_tracing_config = self.config.get("echo_tracing_params", {})
            surface_config = echo_tracing_config.get("surface_detection", {})

            surface_y_rel = detect_surface_echo(
                valid_data_crop, tx_pulse_y_rel, surface_config
            )

            if np.any(np.isfinite(surface_y_rel)):
                self.detected_surface_y_abs = surface_y_rel + self.data_top_abs

                bed_config = echo_tracing_config.get("bed_detection", {})
                bed_y_rel = detect_bed_echo(
                    valid_data_crop, surface_y_rel, z_boundary_y_rel, bed_config
                )

                if np.any(np.isfinite(bed_y_rel)):
                    self.detected_bed_y_abs = bed_y_rel + self.data_top_abs
                else:
                    self.detected_bed_y_abs = np.full(valid_data_crop.shape[1], np.nan)
            else:
                self.detected_surface_y_abs = np.full(valid_data_crop.shape[1], np.nan)
                self.detected_bed_y_abs = np.full(valid_data_crop.shape[1], np.nan)
        else:
            width_for_nan_fallback = 100
            if self.image_np is not None:
                width_for_nan_fallback = self.image_np.shape[1]
            self.detected_surface_y_abs = np.full(width_for_nan_fallback, np.nan)
            self.detected_bed_y_abs = np.full(width_for_nan_fallback, np.nan)

    def process_image(
        self, image_path_str, output_dir_str, approx_x_pip, nav_df=None, nav_path=None
    ):
        """
        Main image processing method with adaptive parameter learning and automatic-first workflow.
        """
        image_path_obj = Path(image_path_str)
        self.base_filename = image_path_obj.stem
        self.output_dir = Path(output_dir_str)

        # NEW: Load optimized parameters from previous processing
        if self.load_previous_optimized_parameters(self.output_dir):
            print(
                "INFO: Using optimized parameters from previous image in flight sequence"
            )
        else:
            print("INFO: Using default parameters for echo detection")

        output_params_config = self.config.get("output_params", {})
        debug_subdir_name = output_params_config.get(
            "debug_output_directory", "debug_output"
        )

        current_output_params = {
            "debug_output_directory": str(self.output_dir / debug_subdir_name),
            "figure_save_dpi": output_params_config.get("figure_save_dpi", 300),
        }

        Path(current_output_params["debug_output_directory"]).mkdir(
            parents=True, exist_ok=True
        )

        print(f"\n--- Processing Z-scope Image: {self.base_filename} ---")

        print("\nStep 1: Loading and preprocessing image...")
        self.image_np = load_and_preprocess_image(
            image_path_str, self.config.get("preprocessing_params", {})
        )

        if self.image_np is None:
            print(
                f"ERROR: Failed to load or preprocess image {image_path_str}. Aborting."
            )
            return False

        img_height, img_width = self.image_np.shape
        print(f"INFO: Image dimensions: {img_width}x{img_height}")

        print("\nStep 2: Detecting film artifact boundaries...")
        artifact_params = self.config.get("artifact_detection_params", {})
        self.data_top_abs, self.data_bottom_abs = detect_film_artifact_boundaries(
            self.image_np,
            self.base_filename,
            top_exclude_ratio=artifact_params.get("top_exclude_ratio", 0.05),
            bottom_exclude_ratio=artifact_params.get("bottom_exclude_ratio", 0.05),
            gradient_smooth_kernel=artifact_params.get("gradient_smooth_kernel", 15),
            gradient_threshold_factor=artifact_params.get(
                "gradient_threshold_factor", 1.5
            ),
            safety_margin=artifact_params.get("safety_margin", 20),
            visualize=artifact_params.get("visualize_film_artifact_boundaries", False),
        )

        print(
            f"INFO: Film artifact boundaries determined: Top={self.data_top_abs}, Bottom={self.data_bottom_abs}"
        )

        print("\nStep 3: Detecting transmitter pulse...")
        tx_pulse_params_config = self.config.get("transmitter_pulse_params", {})
        self.transmitter_pulse_y_abs = detect_transmitter_pulse(
            self.image_np,
            self.base_filename,
            self.data_top_abs,
            self.data_bottom_abs,
            tx_pulse_params=tx_pulse_params_config,
        )

        print(
            f"INFO: Transmitter pulse detected at Y-pixel (absolute): {self.transmitter_pulse_y_abs}"
        )

        print(f"\nStep 4: Detecting calibration pip around X-pixel {approx_x_pip}...")
        if approx_x_pip is None:
            print(
                "ERROR: Approximate X-position for calibration pip not provided. Cannot detect pip."
            )
            return False

        pip_detection_strip_config = self.config.get("pip_detection_params", {}).get(
            "approach_1", {}
        )

        strip_center_for_z_boundary = approx_x_pip
        z_boundary_vslice_width = pip_detection_strip_config.get(
            "z_boundary_vslice_width_px", 10
        )

        v_slice_x_start = max(
            0, strip_center_for_z_boundary - z_boundary_vslice_width // 2
        )
        v_slice_x_end = min(
            img_width, strip_center_for_z_boundary + z_boundary_vslice_width // 2
        )

        if v_slice_x_start >= v_slice_x_end:
            print(
                f"WARNING: Cannot extract vertical slice for Z-boundary detection at X={strip_center_for_z_boundary}. Using full width."
            )
            vertical_slice_for_z = self.image_np
        else:
            vertical_slice_for_z = self.image_np[:, v_slice_x_start:v_slice_x_end]

        z_boundary_params_config = self.config.get(
            "zscope_boundary_detection_params", {}
        )
        z_boundary_y_for_pip = detect_zscope_boundary(
            vertical_slice_for_z, self.data_top_abs, self.data_bottom_abs
        )

        print(
            f"INFO: Z-scope boundary for pip strip detected at Y-pixel (absolute): {z_boundary_y_for_pip}"
        )

        pip_detection_main_config = self.config.get("pip_detection_params", {})
        self.best_pip_details = detect_calibration_pip(
            self.image_np,
            self.base_filename,
            approx_x_pip,
            self.data_top_abs,
            self.data_bottom_abs,
            z_boundary_y_for_pip,
            pip_detection_params=pip_detection_main_config,
        )

        calpip_state_path = self.output_dir / "calpip_state.json"
        if not self.best_pip_details:
            print("WARNING: Calibration pip not detected in this image.")
            if hasattr(self, "last_pip_details") and self.last_pip_details:
                print("INFO: Reusing calibration pip details from previous image.")
                self.best_pip_details = self.last_pip_details
            else:
                self.load_calpip_state(calpip_state_path)
                if self.best_pip_details:
                    print("INFO: Loaded calibration pip details from saved state.")
                else:
                    print(
                        "ERROR: No previous calibration pip available to reuse. Cannot calibrate this image."
                    )
                    return False
        else:
            self.last_pip_details = self.best_pip_details
            self.save_calpip_state(calpip_state_path)

        print("\nStep 5: Visualizing calibration pip detection results...")
        pip_visualization_params_config = pip_detection_main_config.get(
            "visualization_params", {}
        )

        if not self.best_pip_details:
            print(
                "ERROR: Calibration pip detection failed. Cannot perform time calibration."
            )
            return False

        print("\nStep 6: Calculating pixels per microsecond...")
        pip_interval_us = self.physics_constants.get(
            "calibration_pip_interval_microseconds", 2.0
        )

        try:
            self.pixels_per_microsecond = calculate_pixels_per_microsecond(
                self.best_pip_details["mean_spacing"], pip_interval_us
            )
        except ValueError as e:
            print(f"ERROR calculating pixels_per_microsecond: {e}")
            return False

        print(
            f"INFO: Calculated pixels per microsecond: {self.pixels_per_microsecond:.3f}"
        )

        print("\nStep 6.5: Automatic Echo Detection with Optional User Guidance...")

        # Phase 1: Run automatic detection with current parameters
        print("Running automatic echo detection with current parameters...")

        valid_data_crop = self.image_np[self.data_top_abs : self.data_bottom_abs, :]
        crop_height, crop_width = valid_data_crop.shape
        tx_pulse_y_rel = self.transmitter_pulse_y_abs - self.data_top_abs
        z_boundary_y_abs_for_echo_search = self.data_bottom_abs
        z_boundary_y_rel = z_boundary_y_abs_for_echo_search - self.data_top_abs

        # Get current echo tracing configuration
        echo_tracing_config = self.config.get("echo_tracing_params", {})
        surface_config = echo_tracing_config.get("surface_detection", {})
        bed_config = echo_tracing_config.get("bed_detection", {})

        # Automatic surface detection
        surface_y_rel = detect_surface_echo(
            valid_data_crop, tx_pulse_y_rel, surface_config
        )

        if np.any(np.isfinite(surface_y_rel)):
            self.detected_surface_y_abs = surface_y_rel + self.data_top_abs
            print(
                f"Automatic surface detection: {np.sum(np.isfinite(surface_y_rel))}/{len(surface_y_rel)} valid points"
            )

            # Automatic bed detection
            bed_y_rel = detect_bed_echo(
                valid_data_crop, surface_y_rel, z_boundary_y_rel, bed_config
            )

            if np.any(np.isfinite(bed_y_rel)):
                self.detected_bed_y_abs = bed_y_rel + self.data_top_abs
                print(
                    f"Automatic bed detection: {np.sum(np.isfinite(bed_y_rel))}/{len(bed_y_rel)} valid points"
                )
            else:
                self.detected_bed_y_abs = np.full(valid_data_crop.shape[1], np.nan)
                print("WARNING: No valid bed echoes detected automatically")
        else:
            self.detected_surface_y_abs = np.full(valid_data_crop.shape[1], np.nan)
            self.detected_bed_y_abs = np.full(valid_data_crop.shape[1], np.nan)
            print("WARNING: No valid surface echoes detected automatically")

        # Phase 2: Show results and get user feedback
        user_satisfied = self._show_automatic_results_for_approval(valid_data_crop)

        if not user_satisfied:
            print(
                "User requested parameter optimization - starting guided calibration..."
            )

            # Run user-guided calibration
            optimized_params = self._run_user_guided_calibration(valid_data_crop)

            if optimized_params:
                # Update configuration and re-run detection
                self._update_search_parameters(optimized_params)

                # Re-run automatic detection with optimized parameters
                surface_y_rel = detect_surface_echo(
                    valid_data_crop,
                    tx_pulse_y_rel,
                    echo_tracing_config.get("surface_detection", {}),
                )
                bed_y_rel = detect_bed_echo(
                    valid_data_crop,
                    surface_y_rel,
                    z_boundary_y_rel,
                    echo_tracing_config.get("bed_detection", {}),
                )

                # Update final results
                if np.any(np.isfinite(surface_y_rel)):
                    self.detected_surface_y_abs = surface_y_rel + self.data_top_abs
                if np.any(np.isfinite(bed_y_rel)):
                    self.detected_bed_y_abs = bed_y_rel + self.data_top_abs

                print("Echo detection completed with user-optimized parameters")
            else:
                print("User-guided calibration cancelled - using automatic results")
        else:
            print("User satisfied with automatic results - proceeding to CBD selection")

        print("Echo detection phase completed successfully!")

        print("\nStep 7: Creating time-calibrated Z-scope visualization...")
        time_vis_params_config = self.config.get(
            "time_calibration_visualization_params", {}
        )

        self.calibrated_fig, self.calibrated_ax, self.time_axis = (
            create_time_calibrated_zscope(
                self.image_np,
                self.base_filename,
                self.best_pip_details,
                self.transmitter_pulse_y_abs,
                self.data_top_abs,
                self.data_bottom_abs,
                self.pixels_per_microsecond,
                time_vis_params=time_vis_params_config,
                physics_constants=self.physics_constants,
                output_params=current_output_params,
                surface_y_abs=self.detected_surface_y_abs,
                bed_y_abs=self.detected_bed_y_abs,
                nav_df=nav_df,
                nav_path=nav_path,
                main_output_dir=self.output_dir,
                processor_ref=self,
            )
        )

        if self.calibrated_fig is None:
            print("ERROR: Failed to create time-calibrated Z-scope plot.")
            return False

        # NEW: Save optimized parameters if they were updated
        if (
            hasattr(self, "_parameters_were_optimized")
            and self._parameters_were_optimized
        ):
            self.save_optimized_parameters(self.output_dir)

        print(f"\n--- Processing for {self.base_filename} complete. ---")
        print(
            f"INFO: Main calibrated plot saved to {self.output_dir / (self.base_filename + '_picked.png')}"
        )

        return True
