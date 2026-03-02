import numpy as np
import pandas as pd
from pathlib import Path
import json
from scipy.interpolate import PchipInterpolator, UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from zscope_picker_gui import ZScopePickerGUI
from regions_manager import RegionsManager
from session_manager import SessionManager


class SemiAutoProcessor:
    def __init__(self, processor_ref):
        self.processor_ref = processor_ref
        self.config = processor_ref.config
        self.physics_constants = processor_ref.physics_constants

        # Processing state
        self.regions_manager = RegionsManager()
        self.radar_data = None
        self.user_session_data = None

        ### NEW: Layered data structure
        self.layer_names = ["surface", "bed"]
        self.layers = {
            layer: {
                "picks": None,  # 1D array of pick y-pixels (np.nan for gap)
                "intensity": None,  # 1D array of pixel intensity (NaN for gap)
                "undo_stack": [],
                "redo_stack": [],
            }
            for layer in self.layer_names
        }
        self.active_layer = "surface"  # Default; 'tab' key toggles

        # Template matching parameters
        self.template_params = {
            "window_size": (20, 10),  # (width, height) for echo templates
            "confidence_threshold": 0.7,
            "max_search_range": 30,  # pixels above/below seed curve
            "template_update_rate": 0.3,
        }

    def run_semi_automatic_picking(
        self, radar_data, transmitter_pulse_y_rel, z_boundary_y_rel
    ):
        print("INFO: Starting region-based semi-automatic picking workflow...")
        self.radar_data = radar_data
        self.regions_manager.initialize_radar_data(radar_data)
        print("Step 1: Starting interactive region-based refinement...")
        self.run_interactive_refinement(radar_data)
        print("Step 2: Applying final processing...")
        self.apply_final_processing()
        # Return picks as dict for compatibility
        return {
            "surface_picks": self.layers["surface"]["picks"],
            "bed_picks": self.layers["bed"]["picks"],
            "surface_intensity": self.layers["surface"]["intensity"],
            "bed_intensity": self.layers["bed"]["intensity"],
            "session_data": self.user_session_data,
        }

    def run_interactive_refinement(self, radar_data):
        print("INFO: Starting interactive region-based refinement interface...")
        gui = ZScopePickerGUI(
            radar_data,
            self.processor_ref,
            title="ZScope Region-Based Picker",
            workflow_mode="region_based",
            regions_manager=self.regions_manager,
            semi_auto_processor=self,
        )
        # Initialize layer arrays
        for layer in self.layer_names:
            gui.__setattr__(f"{layer}_picks", np.full(radar_data.shape[1], np.nan))
        session_results = gui.start_picking()
        if session_results:
            self.user_session_data = session_results
            for layer in self.layer_names:
                picks = session_results.get(f"{layer}_picks")
                if picks is not None:
                    print(
                        f"INFO: Retrieved {np.sum(np.isfinite(picks))} valid {layer} picks from GUI"
                    )
                    self.layers[layer]["picks"] = np.array(picks)
        # Store regions
        regions_data = (
            session_results.get("regions_data", {}) if session_results else {}
        )
        if regions_data and self.regions_manager:
            try:
                if "regions" in regions_data:
                    self.regions_manager.regions = regions_data["regions"]
                    print("INFO: Regions data updated successfully")
            except Exception as e:
                print(f"WARNING: Could not update regions data safely: {e}")
        print("INFO: Interactive refinement completed")

    ### NEW: Helper to extract intensity for all picks in a layer
    def extract_intensities_for_layer(self, layer):
        picks = self.layers[layer]["picks"]
        if picks is None or self.radar_data is None:
            self.layers[layer]["intensity"] = None
            return
        width = self.radar_data.shape[1]
        intensities = np.full(width, np.nan)
        for i in range(width):
            pick_y = picks[i]
            if np.isfinite(pick_y):
                row = int(round(pick_y))
                if 0 <= row < self.radar_data.shape[0]:
                    intensities[i] = self.radar_data[row, i]
        self.layers[layer]["intensity"] = intensities

    ### MANUAL OR AUTO PICK OPERATIONS: update only the active layer
    def add_manual_pick(self, x, y):
        # Example: add a manual pick at (x, y) for the active layer
        picks = self.layers[self.active_layer]["picks"]
        undo_stack = self.layers[self.active_layer]["undo_stack"]
        redo_stack = self.layers[self.active_layer]["redo_stack"]
        if picks is None:
            picks = np.full(self.radar_data.shape[1], np.nan)
            self.layers[self.active_layer]["picks"] = picks
        undo_stack.append(picks.copy())
        picks[x] = y
        redo_stack.clear()
        # Update intensity right after pick
        self.extract_intensities_for_layer(self.active_layer)

    def undo(self):
        undo_stack = self.layers[self.active_layer]["undo_stack"]
        redo_stack = self.layers[self.active_layer]["redo_stack"]
        picks = self.layers[self.active_layer]["picks"]
        if undo_stack:
            redo_stack.append(picks.copy())
            self.layers[self.active_layer]["picks"] = undo_stack.pop()
            self.extract_intensities_for_layer(self.active_layer)

    def redo(self):
        redo_stack = self.layers[self.active_layer]["redo_stack"]
        undo_stack = self.layers[self.active_layer]["undo_stack"]
        picks = self.layers[self.active_layer]["picks"]
        if redo_stack:
            undo_stack.append(picks.copy())
            self.layers[self.active_layer]["picks"] = redo_stack.pop()
            self.extract_intensities_for_layer(self.active_layer)

    def switch_layer(self):
        current_idx = self.layer_names.index(self.active_layer)
        self.active_layer = self.layer_names[(current_idx + 1) % len(self.layer_names)]
        print(f"INFO: Switched to layer {self.active_layer}")

    def process_region_with_templates(self, region_id, echo_type="surface"):
        region = self.regions_manager.get_region(region_id)
        if not region:
            return {"detections": [], "confidence_scores": []}
        manual_picks = region["control_points"]
        if len(manual_picks) < 2:
            print(
                f"WARNING: Region {region_id} needs at least 2 control points for template matching"
            )
            return {"detections": [], "confidence_scores": []}
        print(
            f"INFO: Processing region {region_id} with {len(manual_picks)} control points..."
        )
        template = self.create_echo_template(manual_picks, echo_type)
        seed_curve = self.generate_seed_curve(manual_picks, region["bounds"])
        detections = self.template_based_detection(
            region, seed_curve, template, echo_type
        )
        constrained_detections = self.apply_region_constraints(detections, echo_type)
        # Update layer immediately if desired
        x_start, x_end = region["bounds"]
        picks_array = self.layers[echo_type]["picks"]
        for i, det in enumerate(constrained_detections["detections"]):
            if x_start + i < len(picks_array):
                picks_array[x_start + i] = det
        self.extract_intensities_for_layer(echo_type)
        return constrained_detections

    def create_echo_template(self, manual_picks, echo_type):
        if not manual_picks or len(manual_picks) < 2:
            return None
        templates = []
        window_width, window_height = self.template_params["window_size"]
        for pick in manual_picks:
            x, y = int(pick["x"]), int(pick["y"])
            y_start = max(0, y - window_height // 2)
            y_end = min(self.radar_data.shape[0], y + window_height // 2)
            x_start = max(0, x - window_width // 2)
            x_end = min(self.radar_data.shape[1], x + window_width // 2)
            local_window = self.radar_data[y_start:y_end, x_start:x_end]
            characteristics = {
                "intensity_mean": np.mean(local_window),
                "intensity_std": np.std(local_window),
                "gradient_strength": np.mean(np.abs(np.gradient(local_window, axis=0))),
                "pattern": local_window.copy(),
                "location": (x, y),
            }
            templates.append(characteristics)
        composite_template = {
            "intensity_mean": np.mean([t["intensity_mean"] for t in templates]),
            "intensity_std": np.mean([t["intensity_std"] for t in templates]),
            "gradient_strength": np.mean([t["gradient_strength"] for t in templates]),
            "pattern_size": (window_width, window_height),
            "num_samples": len(templates),
            "echo_type": echo_type,
        }
        return composite_template

    def generate_seed_curve(self, manual_picks, region_bounds):
        if len(manual_picks) < 2:
            return None
        sorted_picks = sorted(manual_picks, key=lambda p: p["x"])
        x_coords = [p["x"] for p in sorted_picks]
        y_coords = [p["y"] for p in sorted_picks]
        interpolator = PchipInterpolator(x_coords, y_coords)
        x_start, x_end = region_bounds
        x_range = np.arange(x_start, x_end + 1)
        min_x, max_x = min(x_coords), max(x_coords)
        valid_range = (x_range >= min_x) & (x_range <= max_x)
        seed_curve = np.full(len(x_range), np.nan)
        if np.any(valid_range):
            valid_x = x_range[valid_range]
            seed_curve[valid_range] = interpolator(valid_x)
        return seed_curve

    def template_based_detection(self, region, seed_curve, template, echo_type):
        if seed_curve is None or template is None:
            return {"detections": [], "confidence_scores": []}
        x_start, x_end = region["bounds"]
        detections = []
        confidence_scores = []
        search_range = self.template_params["max_search_range"]
        for i, x in enumerate(range(x_start, x_end + 1)):
            if i >= len(seed_curve) or np.isnan(seed_curve[i]):
                detections.append(np.nan)
                confidence_scores.append(0.0)
                continue
            seed_y = int(seed_curve[i])
            y_search_start = max(0, seed_y - search_range)
            y_search_end = min(self.radar_data.shape[0], seed_y + search_range)
            best_y = seed_y
            best_confidence = 0.0
            for y in range(y_search_start, y_search_end):
                confidence = self.calculate_template_similarity(x, y, template)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_y = y
            detections.append(
                best_y
                if best_confidence > self.template_params["confidence_threshold"]
                else np.nan
            )
            confidence_scores.append(best_confidence)
        return {
            "detections": np.array(detections),
            "confidence_scores": np.array(confidence_scores),
            "region_id": region["id"],
            "echo_type": echo_type,
        }

    def calculate_template_similarity(self, x, y, template):
        if template is None:
            return 0.0
        window_width, window_height = template["pattern_size"]
        y_start = max(0, y - window_height // 2)
        y_end = min(self.radar_data.shape[0], y + window_height // 2)
        x_start = max(0, x - window_width // 2)
        x_end = min(self.radar_data.shape[1], x + window_width // 2)
        local_window = self.radar_data[y_start:y_end, x_start:x_end]
        if local_window.size == 0:
            return 0.0
        local_intensity = np.mean(local_window)
        local_gradient = np.mean(np.abs(np.gradient(local_window, axis=0)))
        intensity_diff = abs(local_intensity - template["intensity_mean"])
        intensity_similarity = 1.0 / (1.0 + intensity_diff)
        gradient_diff = abs(local_gradient - template["gradient_strength"])
        gradient_similarity = 1.0 / (1.0 + gradient_diff)
        similarity = 0.6 * intensity_similarity + 0.4 * gradient_similarity
        return min(1.0, max(0.0, similarity))

    def update_region_with_new_pick(self, region_id, new_pick, echo_type):
        self.regions_manager.add_control_point(region_id, new_pick)
        updated_detections = self.process_region_with_templates(region_id, echo_type)
        region = self.regions_manager.get_region(region_id)
        if region:
            if "auto_detections" in region and region["auto_detections"]:
                blended_detections = self.blend_detections(
                    region["auto_detections"],
                    updated_detections["detections"],
                    self.template_params["template_update_rate"],
                )
                region["auto_detections"] = blended_detections
            else:
                region["auto_detections"] = updated_detections["detections"]
            region["confidence_scores"] = updated_detections["confidence_scores"]
        return updated_detections

    def blend_detections(self, old_detections, new_detections, blend_factor):
        old_array = np.array(old_detections)
        new_array = np.array(new_detections)
        valid_old = np.isfinite(old_array)
        valid_new = np.isfinite(new_array)
        valid_both = valid_old & valid_new
        blended = new_array.copy()
        blended[valid_both] = (1 - blend_factor) * old_array[
            valid_both
        ] + blend_factor * new_array[valid_both]
        return blended

    def apply_region_constraints(self, detections, echo_type):
        detection_array = detections["detections"]
        confidence_array = detections["confidence_scores"]
        valid_mask = np.isfinite(detection_array)
        if np.sum(valid_mask) > 2:
            smoothed = gaussian_filter1d(
                detection_array[valid_mask], sigma=1.0, mode="nearest"
            )
            detection_array[valid_mask] = smoothed
        max_slope = 5.0
        for i in range(1, len(detection_array)):
            if np.isfinite(detection_array[i - 1]) and np.isfinite(detection_array[i]):
                slope = abs(detection_array[i] - detection_array[i - 1])
                if slope > max_slope:
                    detection_array[i] = (
                        detection_array[i - 1]
                        + np.sign(detection_array[i] - detection_array[i - 1])
                        * max_slope
                    )
                    confidence_array[i] *= 0.5
        return {
            "detections": detection_array,
            "confidence_scores": confidence_array,
            "region_id": detections["region_id"],
            "echo_type": detections["echo_type"],
        }

    ### FINAL DATA COMPILATION
    def compile_final_picks(self):
        if self.radar_data is None:
            return
        width = self.radar_data.shape[1]
        for layer in self.layer_names:
            self.layers[layer]["picks"] = np.full(width, np.nan)
        # Get picks from session
        if self.user_session_data:
            for layer in self.layer_names:
                session_picks = self.user_session_data.get(f"{layer}_picks")
                if session_picks is not None and len(session_picks) == width:
                    self.layers[layer]["picks"] = np.array(session_picks)
        # Fallback: attempt region extraction if session is empty
        regions_data = self.regions_manager.export_regions_data()
        for layer in self.layer_names:
            for region in regions_data.get("regions", {}).get(layer, []):
                if "auto_detections" in region and region["auto_detections"]:
                    detections = np.array(region["auto_detections"])
                    bounds = region.get("bounds", (0, width - 1))
                    start_x, end_x = int(bounds[0]), int(bounds[1])
                    for i, detection in enumerate(detections):
                        x_pos = start_x + i
                        if 0 <= x_pos < width and np.isfinite(detection):
                            self.layers[layer]["picks"][x_pos] = detection

    ### FINAL PROCESSING + intensity extraction
    def apply_final_processing(self):
        print("INFO: Starting final processing with enhanced data compilation...")
        self.compile_final_picks()
        valid_any = False
        for layer in self.layer_names:
            picks = self.layers[layer]["picks"]
            if picks is not None and np.sum(np.isfinite(picks)) > 0:
                valid_any = True
                print(
                    f"INFO: Final processing - {np.sum(np.isfinite(picks))} valid {layer} picks"
                )
            else:
                width = (
                    self.radar_data.shape[1] if self.radar_data is not None else 38004
                )
                self.layers[layer]["picks"] = np.full(width, np.nan)
                print(f"WARNING: No {layer} picks available, using empty array")
            # Always extract intensities (new)
            self.extract_intensities_for_layer(layer)
        if valid_any:
            self.apply_geological_constraints()
            self.apply_final_smoothing()
            print("INFO: Applied geological constraints and smoothing")
        else:
            print(
                "WARNING: No valid picks to process - skipping constraints and smoothing"
            )
        print("INFO: Final processing completed")

    def apply_geological_constraints(self):
        surface = self.layers["surface"]["picks"]
        bed = self.layers["bed"]["picks"]
        if surface is None or bed is None:
            return
        min_thickness_pixels = 50
        valid_surface = np.isfinite(surface)
        valid_bed = np.isfinite(bed)
        for i in range(len(surface)):
            if valid_surface[i] and valid_bed[i]:
                thickness = bed[i] - surface[i]
                if thickness < min_thickness_pixels:
                    bed[i] = surface[i] + min_thickness_pixels
        print("INFO: Applied geological constraints")

    def apply_final_smoothing(self):
        for layer in self.layer_names:
            picks = self.layers[layer]["picks"]
            if picks is None:
                continue
            valid_mask = np.isfinite(picks)
            if np.any(valid_mask):
                valid_data = picks[valid_mask]
                smoothed_data = gaussian_filter1d(valid_data, sigma=1.0, mode="nearest")
                picks[valid_mask] = smoothed_data
        print("INFO: Applied final smoothing")

    def make_json_serializable(self, obj, visited=None):
        """Convert numpy types to JSON-serializable types with circular reference protection."""
        if visited is None:
            visited = set()

        # Check for circular references using object id
        obj_id = id(obj)
        if obj_id in visited:
            # Return a placeholder for circular references
            return f"<circular_reference_to_{type(obj).__name__}>"

        # Add current object to visited set for complex types
        if isinstance(obj, (dict, list, tuple)):
            visited.add(obj_id)

        try:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                # Create new dict to avoid modifying original
                result = {}
                for k, v in obj.items():
                    # Skip problematic keys that might contain circular references
                    if k in [
                        "region",
                        "active_region",
                        "current_region",
                        "gui",
                        "processor_ref",
                        "semi_auto_processor",
                        "regions_manager",
                        "canvas",
                        "fig",
                        "ax",
                        "radar_data",
                        "session_manager",
                    ]:
                        result[k] = f"<skipped_{k}>"
                    else:
                        try:
                            result[k] = self.make_json_serializable(v, visited.copy())
                        except Exception as e:
                            result[k] = f"<serialization_error_{k}>"
                return result
            elif isinstance(obj, (list, tuple)):
                result = []
                for i, v in enumerate(obj):
                    try:
                        result.append(self.make_json_serializable(v, visited.copy()))
                    except Exception as e:
                        result.append(f"<serialization_error_item_{i}>")
                return result
            elif hasattr(obj, "isoformat"):  # datetime objects
                return obj.isoformat()
            elif hasattr(obj, "__dict__"):
                # Handle custom objects by converting to safe representation
                return f"<object_{type(obj).__name__}>"
            elif pd.isna(obj):
                return None
            else:
                return obj
        except Exception as e:
            # Fallback for any serialization errors
            return f"<serialization_error_{type(obj).__name__}>"
        finally:
            # Remove from visited set when done
            if isinstance(obj, (dict, list, tuple)) and obj_id in visited:
                visited.discard(obj_id)

    def export_session_data(self, output_path):
        """Export session data with safe serialization."""
        if self.user_session_data is None:
            print("WARNING: No session data to export")
            return

        try:
            # Create a clean copy of session data without circular references
            safe_session_data = {
                "surface_picks": self.user_session_data.get("surface_picks", []),
                "bed_picks": self.user_session_data.get("bed_picks", []),
                "workflow_completed": self.user_session_data.get(
                    "workflow_completed", False
                ),
                "session_metadata": self.user_session_data.get("session_metadata", {}),
            }

            # Get clean regions data
            regions_data = []
            if self.regions_manager:
                for region_id in self.regions_manager.get_all_region_ids():
                    region = self.regions_manager.get_region(region_id)
                    if region:
                        # Create clean region data without circular references
                        clean_region = {
                            "id": region.get("id"),
                            "echo_type": region.get("echo_type"),
                            "bounds": region.get("bounds"),
                            "control_points": region.get("control_points", []),
                            "auto_detections": region.get("auto_detections", []),
                            "confidence_scores": region.get("confidence_scores", []),
                            "status": region.get("status"),
                        }
                        regions_data.append(clean_region)

            session_export = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "workflow_type": "region_based",
                "final_picks": {
                    "surface": self.make_json_serializable(
                        self.layers["surface"]["picks"]
                    )
                    if self.layers["surface"]["picks"] is not None
                    else [],
                    "bed": self.make_json_serializable(self.layers["bed"]["picks"])
                    if self.layers["bed"]["picks"] is not None
                    else [],
                },
                "regions": regions_data,
                "template_params": self.template_params,
                "session_data": safe_session_data,
            }

            # Use safe serialization
            safe_export = self.make_json_serializable(session_export)

            with open(output_path, "w") as f:
                json.dump(safe_export, f, indent=2)
            print(f"INFO: Session data exported to {output_path}")

        except Exception as e:
            print(f"ERROR: Failed to export session data: {e}")
            # Try to save minimal data
            try:
                minimal_export = {
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "workflow_type": "region_based",
                    "error": str(e),
                    "final_picks": {
                        "surface": self.layers["surface"]["picks"].tolist()
                        if self.layers["surface"]["picks"] is not None
                        and hasattr(self.layers["surface"]["picks"], "tolist")
                        else [],
                        "bed": self.layers["bed"]["picks"].tolist()
                        if self.layers["bed"]["picks"] is not None
                        and hasattr(self.layers["bed"]["picks"], "tolist")
                        else [],
                    },
                }
                with open(output_path, "w") as f:
                    json.dump(minimal_export, f, indent=2)
                print(f"INFO: Minimal session data exported to {output_path}")
            except Exception as e2:
                print(f"ERROR: Even minimal export failed: {e2}")

    def load_session_data(self, session_path):
        """Load previous session data."""
        if not Path(session_path).exists():
            print(f"WARNING: Session file {session_path} does not exist")
            return False

        try:
            with open(session_path, "r") as f:
                session_data = json.load(f)

            # Restore picks
            if session_data.get("final_picks", {}).get("surface"):
                self.layers["surface"]["picks"] = np.array(
                    session_data["final_picks"]["surface"]
                )
            if session_data.get("final_picks", {}).get("bed"):
                self.layers["bed"]["picks"] = np.array(
                    session_data["final_picks"]["bed"]
                )

            # Restore regions
            if "regions" in session_data:
                for region_data in session_data["regions"]:
                    self.regions_manager.load_region(region_data)

            # Restore template parameters
            if "template_params" in session_data:
                self.template_params.update(session_data["template_params"])

            print(f"INFO: Loaded session data from {session_path}")
            return True

        except Exception as e:
            print(f"ERROR: Failed to load session data: {e}")
            return False
