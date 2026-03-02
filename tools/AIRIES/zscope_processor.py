import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import cv2

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
from functions.echo_tracing import (
    detect_surface_echo,
    detect_bed_echo,
    region_constrained_surface_detection,
    region_constrained_bed_detection,
)

# Import the enhanced semi-automatic processing module
from semi_auto_processor import SemiAutoProcessor
from session_manager import SessionManager
from enhanced_visualization import EnhancedVisualization


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
        self.calculated_ticks = None
        self._parameters_were_optimized = False

        # Enhanced semi-automatic processing configuration
        self.use_semi_automatic_picker = self.config.get(
            "semi_automatic_params", {}
        ).get("enabled", True)

        # Region-based processing data
        self.regions_data = None
        self.session_metadata = None
        self.template_matching_stats = None

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

    def run_semi_automatic_echo_detection(
        self, valid_data_crop, tx_pulse_y_rel, z_boundary_y_rel
    ):
        """Run enhanced region-based semi-automatic echo detection workflow."""
        print("INFO: Starting enhanced region-based semi-automatic echo detection...")

        # Create semi-automatic processor with enhanced capabilities
        semi_auto = SemiAutoProcessor(self)

        # Run the complete region-based workflow
        results = semi_auto.run_semi_automatic_picking(
            valid_data_crop, tx_pulse_y_rel, z_boundary_y_rel
        )

        # Process results
        if results and results.get("surface_picks") is not None:
            self.detected_surface_y_abs = results["surface_picks"] + self.data_top_abs
            print(
                f"INFO: Semi-automatic surface detection: {np.sum(np.isfinite(results['surface_picks']))}/{len(results['surface_picks'])} valid points"
            )
        else:
            self.detected_surface_y_abs = np.full(valid_data_crop.shape[1], np.nan)
            print("WARNING: No surface picks from semi-automatic detection")

        if results and results.get("bed_picks") is not None:
            self.detected_bed_y_abs = results["bed_picks"] + self.data_top_abs
            print(
                f"INFO: Semi-automatic bed detection: {np.sum(np.isfinite(results['bed_picks']))}/{len(results['bed_picks'])} valid points"
            )
        else:
            self.detected_bed_y_abs = np.full(valid_data_crop.shape[1], np.nan)
            print("WARNING: No bed picks from semi-automatic detection")

        # Store session data and metadata
        if results and results.get("session_data"):
            self.session_metadata = results["session_data"]
            print("INFO: Session data stored successfully")

        # Export comprehensive session data
        session_path = self.output_dir / f"{self.base_filename}_picking_session.json"
        semi_auto.export_session_data(session_path)

        # Mark parameters as optimized for next image in sequence
        self._parameters_were_optimized = True

        return True

    def run_region_based_automatic_echo_detection(
        self, valid_data_crop, tx_pulse_y_rel, z_boundary_y_rel, regions_manager=None
    ):
        """Run region-based automatic detection using existing regions."""
        if regions_manager is None:
            print(
                "WARNING: No regions manager provided, falling back to traditional detection"
            )
            return self.run_automatic_echo_detection(
                valid_data_crop, tx_pulse_y_rel, z_boundary_y_rel
            )

        print("INFO: Running region-based automatic echo detection...")

        # Get echo tracing configuration
        echo_tracing_config = self.config.get("echo_tracing_params", {})
        surface_config = echo_tracing_config.get("surface_detection", {})
        bed_config = echo_tracing_config.get("bed_detection", {})

        # Add template matching parameters
        template_params = self.config.get("semi_automatic_params", {}).get(
            "template_params", {}
        )
        surface_config["template_params"] = template_params
        bed_config["template_params"] = template_params

        # Process surface echoes with regions
        surface_regions = regions_manager.get_regions_by_type("surface")
        if surface_regions:
            print(f"INFO: Processing {len(surface_regions)} surface regions")
            surface_y_rel = self._process_regions_for_echo_type(
                valid_data_crop,
                surface_regions,
                "surface",
                tx_pulse_y_rel,
                surface_config,
            )
        else:
            print("INFO: No surface regions defined, using traditional detection")
            surface_y_rel = detect_surface_echo(
                valid_data_crop, tx_pulse_y_rel, surface_config
            )

        if np.any(np.isfinite(surface_y_rel)):
            self.detected_surface_y_abs = surface_y_rel + self.data_top_abs
            print(
                f"Region-based surface detection: {np.sum(np.isfinite(surface_y_rel))}/{len(surface_y_rel)} valid points"
            )
        else:
            self.detected_surface_y_abs = np.full(valid_data_crop.shape[1], np.nan)
            print("WARNING: No valid surface echoes detected with regions")

        # Process bed echoes with regions
        bed_regions = regions_manager.get_regions_by_type("bed")
        if bed_regions:
            print(f"INFO: Processing {len(bed_regions)} bed regions")
            bed_y_rel = self._process_regions_for_echo_type(
                valid_data_crop, bed_regions, "bed", surface_y_rel, bed_config
            )
        else:
            print("INFO: No bed regions defined, using traditional detection")
            bed_y_rel = detect_bed_echo(
                valid_data_crop, surface_y_rel, z_boundary_y_rel, bed_config
            )

        if np.any(np.isfinite(bed_y_rel)):
            self.detected_bed_y_abs = bed_y_rel + self.data_top_abs
            print(
                f"Region-based bed detection: {np.sum(np.isfinite(bed_y_rel))}/{len(bed_y_rel)} valid points"
            )
        else:
            self.detected_bed_y_abs = np.full(valid_data_crop.shape[1], np.nan)
            print("WARNING: No valid bed echoes detected with regions")

        return True

    def _process_regions_for_echo_type(
        self, radar_data, regions, echo_type, reference_picks, config
    ):
        """Process multiple regions for a specific echo type."""
        full_detections = np.full(radar_data.shape[1], np.nan)

        for region in regions:
            if not region or "control_points" not in region:
                continue

            manual_picks = region["control_points"]
            if len(manual_picks) < 2:
                continue

            # Create region mask
            x_start, x_end = region["bounds"]
            region_mask = np.zeros(radar_data.shape[1], dtype=bool)
            region_mask[x_start : x_end + 1] = True

            # Use region-constrained detection
            if echo_type == "surface":
                region_detections = region_constrained_surface_detection(
                    radar_data, reference_picks, region_mask, manual_picks, config
                )
            else:  # bed
                region_detections = region_constrained_bed_detection(
                    radar_data,
                    reference_picks,
                    reference_picks,
                    region_mask,
                    manual_picks,
                    config,
                )

            # Merge into full detections
            valid_detections = np.isfinite(region_detections)
            full_detections[valid_detections] = region_detections[valid_detections]

        return full_detections

    def run_automatic_echo_detection(
        self, valid_data_crop, tx_pulse_y_rel, z_boundary_y_rel
    ):
        """Run traditional automatic echo detection workflow."""
        print("INFO: Running traditional automatic echo detection...")

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

        return True

    def save_enhanced_session_metadata(self, output_dir):
        """Save enhanced session metadata with proper type conversion."""
        if not self.session_metadata:
            return

        metadata_path = Path(output_dir) / f"{self.base_filename}_session_metadata.json"

        def safe_convert(obj):
            """Convert numpy types to JSON-serializable format."""
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: safe_convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [safe_convert(v) for v in obj]
            else:
                return obj

        # Create metadata with safe conversion
        enhanced_metadata = {
            "image_info": {
                "filename": str(self.base_filename),
                "dimensions": {
                    "width": int(self.image_np.shape[1])
                    if self.image_np is not None
                    else None,
                    "height": int(self.image_np.shape[0])
                    if self.image_np is not None
                    else None,
                },
                "data_bounds": {
                    "top": int(self.data_top_abs)
                    if self.data_top_abs is not None
                    else None,
                    "bottom": int(self.data_bottom_abs)
                    if self.data_bottom_abs is not None
                    else None,
                },
            },
            "calibration_info": {
                "transmitter_pulse_y": int(self.transmitter_pulse_y_abs)
                if self.transmitter_pulse_y_abs is not None
                else None,
                "pixels_per_microsecond": float(self.pixels_per_microsecond)
                if self.pixels_per_microsecond is not None
                else None,
                "pip_details": safe_convert(self.best_pip_details)
                if self.best_pip_details
                else None,
            },
            "detection_results": safe_convert(
                {
                    "surface_coverage": float(
                        np.sum(np.isfinite(self.detected_surface_y_abs))
                        / len(self.detected_surface_y_abs)
                        * 100
                    )
                    if self.detected_surface_y_abs is not None
                    else 0.0,
                    "bed_coverage": float(
                        np.sum(np.isfinite(self.detected_bed_y_abs))
                        / len(self.detected_bed_y_abs)
                        * 100
                    )
                    if self.detected_bed_y_abs is not None
                    else 0.0,
                    "total_pixels": int(len(self.detected_surface_y_abs))
                    if self.detected_surface_y_abs is not None
                    else 0,
                }
            ),
            "workflow_info": {
                "detection_method": "region_based_semi_automatic"
                if self.use_semi_automatic_picker
                else "automatic",
                "template_matching_enabled": True,
                "timestamp": str(pd.Timestamp.now()),
                "parameters_optimized": bool(self._parameters_were_optimized),
            },
            "session_data": safe_convert(self.session_metadata)
            if self.session_metadata
            else None,
        }

        try:
            with open(metadata_path, "w") as f:
                json.dump(enhanced_metadata, f, indent=2)
            print(f"INFO: Enhanced session metadata saved to {metadata_path}")
        except Exception as e:
            print(f"WARNING: Failed to save session metadata: {e}")

    def save_visualization_data(self, output_path: str):
        """Save all data needed to recreate the exact visualization including lat/lon."""
        import numpy as np
        import pandas as pd

        # Extract CBD sequence from filename if not available as attribute
        cbd_sequence = None
        if hasattr(self, "cbd_sequence"):
            cbd_sequence = self.cbd_sequence
        else:
            # Extract from filename as fallback
            import re

            cbd_match = re.search(r"C(\d+)_(\d+)", self.base_filename)
            if cbd_match:
                cbd_start = int(cbd_match.group(1))
                cbd_end = int(cbd_match.group(2))
                if cbd_start > cbd_end:
                    cbd_sequence = list(range(cbd_start, cbd_end - 1, -1))
                else:
                    cbd_sequence = list(range(cbd_start, cbd_end + 1))
                    cbd_sequence.reverse()

        # Extract CBD tick positions and their coordinates
        cbd_coordinates = {}
        cbd_tick_positions = None

        if hasattr(self, "calculated_ticks") and self.calculated_ticks is not None:
            cbd_tick_positions = self.calculated_ticks

            # Extract coordinate information if available
            if hasattr(self, "nav_df") and self.nav_df is not None:
                # Match CBD ticks with navigation data
                try:
                    for i, tick_x in enumerate(self.calculated_ticks):
                        if i < len(cbd_sequence):
                            cbd_num = cbd_sequence[i]
                            # Find navigation data for this CBD
                            nav_row = self.nav_df[self.nav_df["CBD"] == cbd_num]
                            if not nav_row.empty:
                                cbd_coordinates[cbd_num] = {
                                    "x_pixel": float(tick_x),
                                    "latitude": float(
                                        nav_row["LAT (bingham)"].values[0]
                                    ),
                                    "longitude": float(
                                        nav_row["LON (bingham)"].values[0]
                                    ),
                                }
                except Exception as e:
                    print(f"Warning: Could not extract CBD coordinates: {e}")

        viz_data = {
            "image_shape": self.image_np.shape if self.image_np is not None else None,
            "axis_setup": {
                "pixels_per_microsecond": self.pixels_per_microsecond,
                "cbd_tick_positions": cbd_tick_positions,
                "data_top_abs": self.data_top_abs,
                "data_bottom_abs": self.data_bottom_abs,
                "transmitter_pulse_y_abs": self.transmitter_pulse_y_abs,
            },
            "picks": {
                "surface_y_abs": self.detected_surface_y_abs,
                "bed_y_abs": self.detected_bed_y_abs,
                "transmitter_y_abs": self.transmitter_pulse_y_abs,
            },
            # Add coordinate information
            "coordinate_data": {
                "cbd_coordinates": cbd_coordinates,
                "cbd_sequence": cbd_sequence,
                "has_navigation": len(cbd_coordinates) > 0,
            },
            "plot_settings": {
                "figure_size": (16, 10),
                "colormap": "gray",
            },
            "metadata": {
                "filename": self.base_filename,
                "processing_timestamp": str(pd.Timestamp.now()),
                "cbd_sequence": cbd_sequence,
                "physics_constants": self.physics_constants,
            },
        }

        # Save as NPZ for easy loading
        viz_path = output_path.replace(".jpg", "_visualization_data.npz").replace(
            ".png", "_visualization_data.npz"
        )
        np.savez(viz_path, **viz_data)

        # Print diagnostic information
        coord_count = len(cbd_coordinates)
        tick_count = len(cbd_tick_positions) if cbd_tick_positions is not None else 0
        print(f"📊 Visualization data saved: {viz_path}")
        print(f"   CBD tick positions: {tick_count}")
        print(f"   CBD coordinates: {coord_count}")

        return viz_path

    def load_session_for_batch_processing(self, session_path):
        """Load previous session data for batch processing workflow."""
        if not Path(session_path).exists():
            return False

        try:
            with open(session_path, "r") as f:
                session_data = json.load(f)

            # Extract regions data if available
            if "regions" in session_data:
                self.regions_data = session_data["regions"]
                print(
                    f"INFO: Loaded {len(self.regions_data)} regions from previous session"
                )

            # Extract template parameters
            if "template_params" in session_data:
                # Update configuration with learned parameters
                if "semi_automatic_params" not in self.config:
                    self.config["semi_automatic_params"] = {}
                self.config["semi_automatic_params"]["template_params"] = session_data[
                    "template_params"
                ]
                print("INFO: Updated template parameters from previous session")

            return True

        except Exception as e:
            print(f"WARNING: Failed to load session data: {e}")
            return False

    def export_enhanced_csv_with_coordinates(
        self, output_dir, nav_df=None, cbd_tick_xs=None
    ):
        """
        Export enhanced CSV with region-based metadata and coordinate interpolation.
        """
        output_path = Path(output_dir) / f"{self.base_filename}_picked.csv"

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

        # Create the enhanced 7-column DataFrame
        if (
            hasattr(self, "transmitter_pulse_intensities")
            and self.transmitter_pulse_intensities is not None
        ):
            transmitter_intensity = self.transmitter_pulse_intensities
            transmitter_time_us = self.transmitter_pulse_time_us
            transmitter_y_pixel = np.full(len(x_pixels), self.transmitter_pulse_y_abs)
        else:
            transmitter_intensity = np.full(len(x_pixels), np.nan)
            transmitter_time_us = np.full(len(x_pixels), np.nan)
            transmitter_y_pixel = np.full(len(x_pixels), np.nan)

        df = pd.DataFrame(
            {
                "X (pixel)": x_pixels,
                "Latitude": latitudes,
                "Longitude": longitudes,
                "CBD": cbd_numbers,
                "Surface Depth (μs)": surface_time_us,
                "Bed Depth (μs)": bed_time_us,
                "Ice Thickness (m)": ice_thickness_meters,
                "Transmitter Depth (μs)": transmitter_time_us,
                "Transmitter Intensity": transmitter_intensity,
                "Transmitter Y_pixel": transmitter_y_pixel,
            }
        )

        # Add metadata header if region-based detection was used
        metadata_lines = []
        if self.use_semi_automatic_picker and self.session_metadata:
            metadata_lines.extend(
                [
                    f"# Processing Method: Region-Based Semi-Automatic Picker",
                    f"# Timestamp: {pd.Timestamp.now().isoformat()}",
                    f"# Image: {self.base_filename}",
                ]
            )

            if hasattr(self, "regions_data") and self.regions_data:
                metadata_lines.append(f"# Regions Used: {len(self.regions_data)}")

            metadata_lines.append("#")  # Separator line

        # Export with high precision
        try:
            with open(output_path, "w") as f:
                # Write metadata header
                for line in metadata_lines:
                    f.write(line + "\n")

            # Append the CSV data
            df.to_csv(output_path, mode="a", index=False, float_format="%.6f")

            print(f"SUCCESS: Enhanced 7-column CSV exported to: {output_path}")
            print(f"INFO: Data summary: {len(df)} rows, {len(df.columns)} columns")
            print(
                f"INFO: Coordinate coverage: {np.sum(~np.isnan(latitudes))}/{len(x_pixels)} pixels"
            )

            if metadata_lines:
                print(
                    f"INFO: Metadata header included with {len(metadata_lines) - 1} lines"
                )

            return df

        except Exception as e:
            print(f"ERROR: Failed to save CSV file: {e}")
            return None

    def _convert_pixels_to_one_way_time(self, y_pixels):
        """Convert pixel positions to one-way travel time in microseconds."""
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
        """Calculate ice thickness in meters using proper one-way travel times."""
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
        """Interpolate Bingham coordinates for all x-pixels with full resolution."""
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
        """Save enhanced optimized parameters including template matching data."""
        params_file = Path(output_dir) / "optimized_echo_params.json"

        # Extract current echo tracing parameters
        echo_params = self.config.get("echo_tracing_params", {})
        semi_auto_params = self.config.get("semi_automatic_params", {})

        # Enhanced optimization data
        optimized_data = {
            "timestamp": str(pd.Timestamp.now()),
            "source_image": self.base_filename,
            "echo_tracing_params": echo_params,
            "semi_automatic_params": semi_auto_params,
            "optimization_method": "region_based_semi_automatic_picker",
            "flight_sequence": True,
            "template_matching_enabled": True,
            "regions_count": len(self.regions_data) if self.regions_data else 0,
            "performance_metrics": {
                "surface_coverage": float(
                    np.sum(np.isfinite(self.detected_surface_y_abs))
                    / len(self.detected_surface_y_abs)
                    * 100
                )
                if self.detected_surface_y_abs is not None
                else 0.0,
                "bed_coverage": float(
                    np.sum(np.isfinite(self.detected_bed_y_abs))
                    / len(self.detected_bed_y_abs)
                    * 100
                )
                if self.detected_bed_y_abs is not None
                else 0.0,
            },
        }

        with open(params_file, "w") as f:
            json.dump(optimized_data, f, indent=4)

        print(f"INFO: Enhanced optimized parameters saved to {params_file}")

    def load_previous_optimized_parameters(self, output_dir):
        """Load enhanced optimized parameters from previous image processing."""
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

                # Update semi-automatic parameters
                if "semi_automatic_params" in optimized_data:
                    if "semi_automatic_params" not in self.config:
                        self.config["semi_automatic_params"] = {}
                    self.config["semi_automatic_params"].update(
                        optimized_data["semi_automatic_params"]
                    )

                print(
                    f"INFO: Loaded enhanced optimized parameters from previous processing"
                )
                print(f"INFO: Source: {optimized_data.get('source_image', 'unknown')}")
                print(
                    f"INFO: Template matching: {optimized_data.get('template_matching_enabled', False)}"
                )

                return True

            except Exception as e:
                print(f"WARNING: Could not load optimized parameters: {e}")
                return False

        return False

    def process_image(
        self, image_path_str, output_dir_str, approx_x_pip, nav_df=None, nav_path=None
    ):
        """
        Enhanced main image processing method with region-based semi-automatic echo detection.
        """
        image_path_obj = Path(image_path_str)
        self.base_filename = image_path_obj.stem
        self.output_dir = Path(output_dir_str)

        # Load optimized parameters from previous processing
        if self.load_previous_optimized_parameters(self.output_dir):
            print(
                "INFO: Using enhanced optimized parameters from previous image in flight sequence"
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
            vertical_slice_for_z,
            self.data_top_abs,
            self.data_bottom_abs,
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

        print("\nStep 7: Enhanced Echo Detection...")

        # Prepare data for echo detection
        valid_data_crop = self.image_np[self.data_top_abs : self.data_bottom_abs, :]
        tx_pulse_y_rel = self.transmitter_pulse_y_abs - self.data_top_abs
        z_boundary_y_abs_for_echo_search = self.data_bottom_abs
        z_boundary_y_rel = z_boundary_y_abs_for_echo_search - self.data_top_abs

        # Choose detection method based on configuration
        if self.config.get("semi_automatic_params", {}).get("enabled", False):
            print("Running enhanced region-based semi-automatic echo detection...")
            success = self.run_semi_automatic_echo_detection(
                valid_data_crop, tx_pulse_y_rel, z_boundary_y_rel
            )
        else:
            print("Running traditional automatic echo detection...")
            success = self.run_automatic_echo_detection(
                valid_data_crop, tx_pulse_y_rel, z_boundary_y_rel
            )

        if not success:
            print("ERROR: Echo detection failed")
            return False

        print("Enhanced echo detection phase completed successfully!")

        # Extract transmitter pulse intensity data
        print("\nStep 7b: Extracting transmitter pulse intensity data...")
        if self.transmitter_pulse_y_abs is not None:
            # Get transmitter pulse intensities across the image width
            tx_y_pixel = int(self.transmitter_pulse_y_abs)
            image_width = self.image_np.shape[1]

            # Extract intensity values along the transmitter pulse line
            self.transmitter_pulse_intensities = self.image_np[tx_y_pixel, :]

            # Convert pixel position to time for transmitter pulse
            self.transmitter_pulse_time_us = np.zeros(
                image_width
            )  # Transmitter is at time zero

            print(
                f"INFO: Extracted transmitter pulse intensities across {image_width} pixels"
            )
            print(
                f"INFO: Transmitter pulse intensity range: {np.min(self.transmitter_pulse_intensities):.1f} - {np.max(self.transmitter_pulse_intensities):.1f}"
            )
        else:
            print(
                "WARNING: No transmitter pulse detected, cannot extract intensity data"
            )
            self.transmitter_pulse_intensities = None
            self.transmitter_pulse_time_us = None

        # Save enhanced session metadata
        self.save_enhanced_session_metadata(self.output_dir)

        print("\nStep 8: Creating time-calibrated Z-scope visualization...")
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
                processor_ref=self,  # Pass processor reference for CBD tick storage
            )
        )

        if self.calibrated_fig is None:
            print("ERROR: Failed to create time-calibrated Z-scope plot.")
            return False

        # Save optimized parameters if they were updated
        if (
            hasattr(self, "_parameters_were_optimized")
            and self._parameters_were_optimized
        ):
            self.save_optimized_parameters(self.output_dir)

        print(f"\n--- Processing for {self.base_filename} complete. ---")
        print(
            f"INFO: Main calibrated plot saved to {self.output_dir / (self.base_filename + '_picked.png')}"
        )
        # Save visualization data for radiometric calibrator
        viz_data_path = self.save_visualization_data(
            str(self.output_dir / (self.base_filename + "_picked.png"))
        )

        return True
