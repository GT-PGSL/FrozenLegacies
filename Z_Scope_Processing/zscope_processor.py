import json
from pathlib import Path
import numpy as np
import pandas as pd

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

    def export_ice_measurements(self, output_dir):
        output_path = Path(output_dir) / f"{self.base_filename}_ice_measurements.csv"
        if (
            self.detected_surface_y_abs is not None
            and self.detected_bed_y_abs is not None
            and len(self.detected_surface_y_abs) == len(self.detected_bed_y_abs)
        ):
            df = pd.DataFrame(
                {
                    "x_pixel": np.arange(len(self.detected_surface_y_abs)),
                    "surface_y": self.detected_surface_y_abs,
                    "bed_y": self.detected_bed_y_abs,
                    "ice_thickness": self.detected_bed_y_abs
                    - self.detected_surface_y_abs,
                }
            )
            df.to_csv(output_path, index=False)
            print(f"INFO: Ice measurements CSV saved to {output_path}")
        else:
            print(
                "WARNING: Ice measurements not saved due to missing or mismatched data."
            )

    def process_image(
        self, image_path_str, output_dir_str, approx_x_pip, nav_df=None, nav_path=None
    ):
        image_path_obj = Path(image_path_str)
        self.base_filename = image_path_obj.stem
        self.output_dir = Path(output_dir_str)

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

        print("\nStep 6.5: Automatic echo tracing...")
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
            echo_tracing_config = self.config.get("echo_tracing_params", {})
            surface_config = echo_tracing_config.get("surface_detection", {})
            surface_y_rel = detect_surface_echo(
                valid_data_crop,
                tx_pulse_y_rel,
                surface_config,
            )
            if np.any(np.isfinite(surface_y_rel)):
                self.detected_surface_y_abs = surface_y_rel + self.data_top_abs
                bed_config = echo_tracing_config.get("bed_detection", {})
                bed_y_rel = detect_bed_echo(
                    valid_data_crop,
                    surface_y_rel,
                    z_boundary_y_rel,
                    bed_config,
                )
                if np.any(np.isfinite(bed_y_rel)):
                    self.detected_bed_y_abs = bed_y_rel + self.data_top_abs
                else:
                    if valid_data_crop is not None:
                        self.detected_bed_y_abs = np.full(
                            valid_data_crop.shape[1], np.nan
                        )
            else:
                if valid_data_crop is not None:
                    self.detected_surface_y_abs = np.full(
                        valid_data_crop.shape[1], np.nan
                    )
                    self.detected_bed_y_abs = np.full(valid_data_crop.shape[1], np.nan)
        else:
            width_for_nan_fallback = 100
            if self.image_np is not None:
                width_for_nan_fallback = self.image_np.shape[1]
            self.detected_surface_y_abs = np.full(width_for_nan_fallback, np.nan)
            self.detected_bed_y_abs = np.full(width_for_nan_fallback, np.nan)

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
            )
        )
        if self.calibrated_fig is None:
            print("ERROR: Failed to create time-calibrated Z-scope plot.")
            return False

        self.export_ice_measurements(self.output_dir)

        print(f"\n--- Processing for {self.base_filename} complete. ---")
        print(
            f"INFO: Main calibrated plot saved to {self.output_dir / (self.base_filename + '_picked.png')}"
        )
        return True
