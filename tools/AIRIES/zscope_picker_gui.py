import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from matplotlib.backend_bases import cursors
import numpy as np
import cv2
from pathlib import Path
import json
from scipy.interpolate import UnivariateSpline, PchipInterpolator, interp1d
from scipy.ndimage import gaussian_filter1d
from matplotlib.backend_bases import MouseEvent, KeyEvent
from enhanced_visualization import EnhancedVisualization
from session_manager import SessionManager
from datetime import datetime


class ZScopePickerGUI:
    def __init__(
        self,
        radar_data,
        processor_ref,
        title="ZScope Region-Based Picker",
        workflow_mode="region_based",
        regions_manager=None,
        semi_auto_processor=None,
    ):
        self.radar_data = radar_data
        self.processor_ref = processor_ref
        self.title = title
        self.workflow_mode = workflow_mode

        # Store new parameters for region-based workflow
        self.regions_manager = regions_manager
        self.semi_auto_processor = semi_auto_processor
        self.region_based_mode = (
            workflow_mode == "region_based" and regions_manager is not None
        )

        # Initialize GUI state
        self.fig, self.ax = plt.subplots(figsize=(16, 10))
        self.canvas = self.fig.canvas

        # Display modes
        self.display_mode = 1  # 1=raw, 2=smoothed, 3=difference
        self.bnw_mode = True
        self.picks_hidden = False
        self.annotations_hidden = False
        self.active_layer = "surface"  # Default active layer

        # Navigation parameters
        self.pan_factor = 0.4
        self.zoom_factor = 0.5

        # Control points and picks
        self.surface_control_points = []
        self.bed_control_points = []
        self.surface_picks = np.full(radar_data.shape[1], np.nan)
        self.bed_picks = np.full(radar_data.shape[1], np.nan)

        # Session management (initialize with processor reference)
        try:
            output_dir = getattr(processor_ref, "output_dir", Path("."))
            self.session_manager = SessionManager(output_dir)
            self.enhanced_viz = EnhancedVisualization(
                getattr(processor_ref, "config", {}).get("enhanced_visualization", {})
            )

            # Create new session when starting
            image_filename = getattr(processor_ref, "base_filename", "unknown")
            self.session_id = self.session_manager.create_new_session(
                image_filename=image_filename,
                session_name=f"picking_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
        except Exception as e:
            print(f"WARNING: Session manager initialization failed: {e}")
            self.session_manager = None
            self.enhanced_viz = None
            self.session_id = None

        # Region-based workflow state
        self.workflow_state = (
            "region_selection"  # region_selection, manual_picking, auto_detection
        )
        self.current_picking_mode = "surface"  # surface or bed
        self.regions = []
        self.current_region = None
        self.echo_characteristics = {"surface": {}, "bed": {}}

        # Display region boundaries - remove cutoff region with corrected detection
        self.display_boundaries = self.calculate_display_boundaries()

        # Gap regions (where no echoes exist)
        self.gap_regions = []

        # Action history for undo functionality
        self.action_history = []
        self.max_history = 20

        # Mouse interaction state
        self.dragging = False
        self.dragging_point = None
        self.dragging_list = None
        self.dragging_type = None

        # Interpolation method
        self.current_method = "constrained_spline"

        # Preprocessing for different display modes
        self.prepare_display_data()

        # Set up event handlers
        self.setup_event_handlers()
        self.setup_mouse_handlers()

        # Initialize region-based workflow
        self.initialize_region_workflow()

        # Initial display
        self.update_display()

    def start_picking(self):
        """Start the picking workflow and return session results."""
        if self.region_based_mode and self.regions_manager:
            print("INFO: Starting region-based picking workflow")

            # Show the GUI and wait for user interaction
            plt.show()

            final_surface_picks = self.surface_picks.copy()
            final_bed_picks = self.bed_picks.copy()

            # Update regions manager with final picks
            if self.regions_manager:
                try:
                    # Create clean region data without circular references
                    for i, region in enumerate(self.regions):
                        region_id = f"{region['mode']}_{i}"
                        if region["mode"] == "surface":
                            start_x, end_x = (
                                int(region["start_x"]),
                                int(region["end_x"]),
                            )
                            region_picks = final_surface_picks[start_x : end_x + 1]

                            clean_region = {
                                "id": region_id,
                                "echo_type": region["mode"],
                                "bounds": (start_x, end_x),
                                "auto_detections": region_picks.tolist(),
                                "confidence_scores": np.ones_like(
                                    region_picks
                                ).tolist(),
                                "control_points": [
                                    {"x": p.get("x"), "y": p.get("y")}
                                    for p in region.get("picks", [])
                                ],
                                "status": "completed",
                            }
                            self.regions_manager.regions[region["mode"]].append(
                                clean_region
                            )

                        elif region["mode"] == "bed":
                            start_x, end_x = (
                                int(region["start_x"]),
                                int(region["end_x"]),
                            )
                            region_picks = final_bed_picks[start_x : end_x + 1]

                            clean_region = {
                                "id": region_id,
                                "echo_type": region["mode"],
                                "bounds": (start_x, end_x),
                                "auto_detections": region_picks.tolist(),
                                "confidence_scores": np.ones_like(
                                    region_picks
                                ).tolist(),
                                "control_points": [
                                    {"x": p.get("x"), "y": p.get("y")}
                                    for p in region.get("picks", [])
                                ],
                                "status": "completed",
                            }
                            self.regions_manager.regions[region["mode"]].append(
                                clean_region
                            )

                    print("INFO: Regions manager updated without circular references")

                except Exception as e:
                    print(f"WARNING: Regions manager update failed: {e}")

            return {
                "surface_picks": final_surface_picks,
                "bed_picks": final_bed_picks,
                "regions_data": self.regions_manager.export_regions_data()
                if self.regions_manager
                else {},
                "workflow_completed": True,
                "session_metadata": {
                    "total_control_points": len(self.surface_control_points)
                    + len(self.bed_control_points),
                    "regions_created": len(self.regions),
                    "workflow_mode": self.workflow_mode,
                },
            }

    def calculate_display_boundaries(self):
        """Calculate display boundaries - FIXED to show full image extent."""
        # For region-based picker, we want to show the FULL radar image
        # Don't crop anything unless absolutely necessary

        if hasattr(self.processor_ref, "data_top_abs") and hasattr(
            self.processor_ref, "data_bottom_abs"
        ):
            # Get processor boundaries but DON'T use them for cropping
            processor_top = self.processor_ref.data_top_abs
            processor_bottom = self.processor_ref.data_bottom_abs
            print(
                f"INFO: Processor boundaries available - Top: {processor_top}, Bottom: {processor_bottom}"
            )

            # Show full image extent instead of cropping
            boundaries = {
                "top": 0,  # Start from very top of image
                "bottom": self.radar_data.shape[0],  # Go to very bottom
                "left": 0,
                "right": self.radar_data.shape[1],
            }
        else:
            # Fallback - still show full image
            boundaries = {
                "top": 0,
                "bottom": self.radar_data.shape[0],
                "left": 0,
                "right": self.radar_data.shape[1],
            }

        print(f"INFO: Display boundaries set to show full image: {boundaries}")
        return boundaries

    def auto_detect_echogram_boundaries(self):
        """Automatically detect the echogram boundaries to exclude CBD region - CORRECTED."""
        # Calculate horizontal gradient to find top/bottom boundaries
        horizontal_gradient = np.gradient(self.radar_data, axis=0)
        gradient_magnitude = np.abs(horizontal_gradient)

        # Find boundaries based on gradient changes
        row_gradients = np.mean(gradient_magnitude, axis=1)

        # Smooth the gradient profile
        smoothed_gradients = gaussian_filter1d(row_gradients, sigma=5)

        # Find top boundary (start of echogram data)
        top_boundary = 0
        threshold = np.mean(smoothed_gradients) * 0.8  # Lower threshold for top
        for i in range(20, len(smoothed_gradients) // 3):
            if smoothed_gradients[i] > threshold:
                top_boundary = max(0, i - 10)  # Add small buffer
                break

        # Find bottom boundary (end of echogram, start of CBD region)
        bottom_boundary = len(smoothed_gradients) - 1
        # Look for the CBD region (typically has lower gradient activity)
        # Start from bottom and work upward to find where echogram ends
        for i in range(len(smoothed_gradients) - 20, len(smoothed_gradients) // 2, -1):
            # Look for sustained low gradient region (CBD area)
            window_size = 50
            if i - window_size > 0:
                window_gradient = smoothed_gradients[i - window_size : i]
                if np.mean(window_gradient) < np.mean(smoothed_gradients) * 0.3:
                    bottom_boundary = i
                    break

        # Ensure we have reasonable boundaries
        if bottom_boundary - top_boundary < 100:
            # Fallback to safer boundaries
            top_boundary = max(0, int(0.05 * len(smoothed_gradients)))
            bottom_boundary = min(
                len(smoothed_gradients) - 1, int(0.85 * len(smoothed_gradients))
            )

        print(
            f"INFO: Auto-detected boundaries - Top: {top_boundary}, Bottom: {bottom_boundary}"
        )

        return top_boundary, bottom_boundary

    def crop_display_data(self, data):
        """Crop data to display boundaries, excluding cutoff region."""
        return data

    def prepare_display_data(self):
        """Prepare display data without cropping."""
        # Raw data - use full extent
        self.raw_data = self.radar_data.copy()

        # Smoothed data using existing preprocessing
        if self.radar_data.dtype != np.uint8:
            radar_uint8 = cv2.normalize(
                self.radar_data, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
        else:
            radar_uint8 = self.radar_data.copy()

        smoothed_full = cv2.createCLAHE(clipLimit=3.0).apply(radar_uint8)
        self.smoothed_data = smoothed_full

        # Difference data (enhanced edges)
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        diff_full = cv2.filter2D(self.radar_data, -1, kernel)
        self.diff_data = diff_full

    def initialize_region_workflow(self):
        """Initialize the region-based workflow."""
        self.workflow_state = "region_selection"
        self.current_picking_mode = "surface"
        print("INFO: Region-based workflow initialized")
        print("INFO: Starting with surface echo region selection")
        print("INFO: Click two points to define the first picking region")

    def setup_event_handlers(self):
        """Set up keyboard event handlers - removed scroll event."""
        try:
            self.canvas.mpl_disconnect("key_press_event")
        except:
            pass

        # Only connect keyboard events - no scroll events for zoom
        self.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.ensure_canvas_focus()

    def setup_mouse_handlers(self):
        """Enhanced mouse event handlers for region-based workflow."""
        try:
            self.canvas.mpl_disconnect("button_press_event")
            self.canvas.mpl_disconnect("button_release_event")
            self.canvas.mpl_disconnect("motion_notify_event")
        except:
            pass

        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_motion)

    def safe_set_cursor(self, cursor_type):
        """Safely set cursor using proper matplotlib cursor constants."""
        try:
            if cursor_type == "default":
                self.canvas.set_cursor(cursors.Cursor.POINTER)
            elif cursor_type == "hand":
                self.canvas.set_cursor(cursors.Cursor.HAND)
            elif cursor_type == "move":
                self.canvas.set_cursor(cursors.Cursor.MOVE)
            else:
                self.canvas.set_cursor(cursors.Cursor.POINTER)
        except (AttributeError, ValueError):
            pass

    def reset_mouse_state(self):
        """Reset all mouse interaction state variables."""
        self.dragging = False
        self.dragging_point = None
        self.dragging_list = None
        self.dragging_type = None
        self.safe_set_cursor("default")

    def ensure_canvas_focus(self):
        """Ensure the canvas has proper focus for event handling."""
        try:
            self.canvas.setFocus()
            if hasattr(self.canvas, "activateWindow"):
                self.canvas.activateWindow()
        except AttributeError:
            pass
        plt.figure(self.fig.number)

    def update_display_and_status_bar(self):
        """Update display to show current active layer"""
        if hasattr(self, "ax") and self.ax:
            # Update title or add text to show current layer
            current_title = self.ax.get_title()
            if "Active Layer:" not in current_title:
                self.ax.set_title(
                    f"{current_title} - Active Layer: {self.active_layer.title()}"
                )
            else:
                # Update existing title
                base_title = current_title.split(" - Active Layer:")[0]
                self.ax.set_title(
                    f"{base_title} - Active Layer: {self.active_layer.title()}"
                )

            # Refresh the plot
            if hasattr(self, "fig") and self.fig:
                self.fig.canvas.draw()

    def on_key_press(self, event):
        """Handle keyboard navigation and commands for region-based workflow."""
        if event.key is None:
            return
        self.canvas.draw_idle()

        # Navigation
        if event.key == "left":
            self.pan_view("left")
        elif event.key == "right":
            self.pan_view("right")
        elif event.key == "up":
            self.pan_view("up")
        elif event.key == "down":
            self.pan_view("down")
        elif event.key == "+":
            self.zoom_view("in")
        elif event.key == "-":
            self.zoom_view("out")

        # Display modes
        elif event.key == "d":
            self.cycle_display_mode()
        elif event.key == "p":
            self.toggle_picks_visibility()
        if event.key == "tab":
            self.active_layer = "bed" if self.active_layer == "surface" else "surface"
            print(f"INFO: Switched to {self.active_layer} layer")
            self.update_display_and_status_bar()

        # Region-based workflow commands
        elif event.key == "r":
            self.toggle_region_selection()
        elif event.key == "space":
            self.trigger_automatic_detection()
        elif event.key == "b":  # Switch to bed detection mode
            self.switch_to_bed_detection()
        elif event.key == "s":  # Switch to surface detection mode
            self.switch_to_surface_detection()
        elif event.key == "y":  # Complete picking session
            self.finalize_picking()

        # Other commands
        elif event.key == "l":
            self.apply_spline_interpolation()
        elif event.key == "i":
            self.cycle_interpolation_method()
        elif event.key == "g":
            self.mark_gap_region()
        elif event.key == "backspace":
            self.undo_last_action()
        elif event.key == "escape":
            self.cancel_current_operation()

        try:
            self.canvas.setFocus()
        except AttributeError:
            pass

    def switch_to_bed_detection(self):
        """Switch to bed detection mode."""
        self.current_picking_mode = "bed"
        self.workflow_state = "region_selection"
        print("INFO: Switched to bed detection mode")
        print("INFO: Click two points to define bed picking regions")
        self.update_display()

    def switch_to_surface_detection(self):
        """Switch to surface detection mode."""
        self.current_picking_mode = "surface"
        self.workflow_state = "region_selection"
        print("INFO: Switched to surface detection mode")
        print("INFO: Click two points to define surface picking regions")
        self.update_display()

    def trigger_automatic_detection(self):
        """Trigger automatic detection based on manual picks."""
        if self.workflow_state == "manual_picking":
            self.run_automatic_detection()

            # CRITICAL: Ensure gaps are preserved after automatic detection
            self.validate_region_gaps()

            self.workflow_state = "auto_detection"
            print("INFO: Automatic detection triggered with region constraints")
            self.update_display()

    def cancel_current_operation(self):
        """Cancel current operation and reset to region selection."""
        self.current_region = None
        self.workflow_state = "region_selection"
        print("INFO: Operation cancelled, returning to region selection")
        self.update_display()

    def on_mouse_press(self, event):
        """Handle mouse press events for region-based workflow."""
        self.reset_mouse_state()

        if event.inaxes != self.ax:
            return False

        if event.xdata is None or event.ydata is None:
            return False

        # Adjust coordinates for cropped display
        adjusted_x = event.xdata
        adjusted_y = event.ydata

        try:
            if event.button == 1:  # Left click
                if self.workflow_state == "region_selection":
                    self.handle_region_selection(adjusted_x, adjusted_y)
                elif self.workflow_state == "manual_picking":
                    self.handle_manual_picking(adjusted_x, adjusted_y)
                elif self.workflow_state == "auto_detection":
                    # Check if clicking near existing control point for refinement
                    dragging_point = self.find_nearest_control_point(
                        adjusted_x, adjusted_y
                    )

                    if dragging_point is not None:
                        self.start_dragging(dragging_point)
                    else:
                        self.handle_refinement_picking(adjusted_x, adjusted_y)

            elif event.button == 3:  # Right click
                if self.workflow_state in ["manual_picking", "auto_detection"]:
                    success = self.remove_control_point(adjusted_x, adjusted_y)
                    if success:
                        print("INFO: Control point removed successfully")
                    else:
                        print("INFO: No control point found near click location")

        except Exception as e:
            print(f"WARNING: Mouse press event error: {e}")
            self.reset_mouse_state()

        return True

    def handle_region_selection(self, x, y):
        """Handle region selection clicks."""
        if self.current_region is None:
            # Start new region
            self.current_region = {
                "start_x": x,
                "end_x": None,
                "mode": self.current_picking_mode,
                "picks": [],
            }

            print(f"INFO: Started {self.current_picking_mode} region at x={int(x)}")
        else:
            # Complete region
            self.current_region["end_x"] = x

            # Ensure start < end
            if self.current_region["start_x"] > self.current_region["end_x"]:
                self.current_region["start_x"], self.current_region["end_x"] = (
                    self.current_region["end_x"],
                    self.current_region["start_x"],
                )

            self.regions.append(self.current_region)
            print(
                f"INFO: Completed {self.current_picking_mode} region from x={int(self.current_region['start_x'])} to x={int(self.current_region['end_x'])}"
            )

            # Switch to manual picking mode
            self.workflow_state = "manual_picking"
            print(
                "INFO: Region defined. Now click on representative echo points within the region"
            )

            self.current_region = None

        self.update_display()

    def handle_manual_picking(self, x, y):
        """Handle manual picking clicks."""
        # Find the active region for this pick
        active_region = None
        for region in self.regions:
            if (
                region["mode"] == self.current_picking_mode
                and region["start_x"] <= x <= region["end_x"]
            ):
                active_region = region
                break

        if active_region is None:
            print(
                "INFO: Click is outside defined regions. Define a region first with 'r' key"
            )
            return

        # Add manual pick
        pick = {"x": int(x), "y": int(y), "region": active_region}
        active_region["picks"].append(pick)

        # Add to control points
        control_point = {"x": int(x), "y": int(y)}
        if self.current_picking_mode == "surface":
            self.surface_control_points.append(control_point)
        else:
            self.bed_control_points.append(control_point)

        print(f"INFO: Added {self.current_picking_mode} pick at ({int(x)}, {int(y)})")

        # Analyze echo characteristics
        self.analyze_echo_characteristics(pick, active_region)

        # Update interpolation
        self.update_interpolation()
        self.update_display()

    def handle_refinement_picking(self, x, y):
        """Handle refinement picking in auto-detection mode."""
        # Similar to manual picking but updates existing detection
        self.handle_manual_picking(x, y)

        # Re-run automatic detection with new pick
        self.run_automatic_detection()
        print("INFO: Added refinement pick and updated automatic detection")

    def analyze_echo_characteristics(self, pick, region):
        """Analyze characteristics of manually picked echo points."""
        x, y = pick["x"], pick["y"]

        # Adjust coordinates for cropped display
        display_y = int(y)
        display_x = int(x)

        # Ensure coordinates are within bounds
        if (
            display_y < 5
            or display_y >= self.raw_data.shape[0] - 5
            or display_x < 10
            or display_x >= self.raw_data.shape[1] - 10
        ):
            return

        # Extract local window around pick
        local_window = self.raw_data[
            display_y - 5 : display_y + 5, display_x - 10 : display_x + 10
        ]

        # Calculate characteristics
        characteristics = {
            "brightness": float(np.mean(local_window)),
            "contrast": float(np.std(local_window)),
            "gradient_strength": float(np.mean(np.abs(np.gradient(local_window)))),
            "depth": int(y),
        }

        # Store in region characteristics
        mode = self.current_picking_mode
        if mode not in region:
            region[mode + "_characteristics"] = []
        region[mode + "_characteristics"].append(characteristics)

        print(
            f"INFO: Analyzed echo characteristics - Brightness: {characteristics['brightness']:.1f}, Contrast: {characteristics['contrast']:.1f}"
        )

    def run_automatic_detection(self):
        """Run automatic detection based on analyzed characteristics."""
        for region in self.regions:
            if region["mode"] != self.current_picking_mode:
                continue

            characteristics_key = self.current_picking_mode + "_characteristics"
            if characteristics_key not in region or not region[characteristics_key]:
                continue

            # Calculate average characteristics
            chars = region[characteristics_key]
            avg_brightness = np.mean([c["brightness"] for c in chars])
            avg_contrast = np.mean([c["contrast"] for c in chars])
            avg_gradient = np.mean([c["gradient_strength"] for c in chars])

            # Detect similar echoes ONLY in this specific region
            detected_points = self.detect_similar_echoes(
                region, avg_brightness, avg_contrast, avg_gradient
            )

            # Update picks array with region-constrained detections
            self.update_picks_from_detection(detected_points)

            print(
                f"INFO: Detected {len(detected_points)} points in {region['mode']} region {region['start_x']}-{region['end_x']}"
            )

    def detect_similar_echoes(self, region, avg_brightness, avg_contrast, avg_gradient):
        """Detect echoes with similar characteristics ONLY within the defined region."""
        start_x = int(region["start_x"])
        end_x = int(region["end_x"])

        # Ensure bounds are within data AND within the specific region
        start_x = max(10, start_x)
        end_x = min(self.raw_data.shape[1] - 10, end_x)

        detected_points = []

        # CRITICAL: Only search within THIS region, not globally
        for x in range(start_x, end_x, 2):  # Sample every 2 pixels for efficiency
            best_match_y = None
            best_score = 0

            # Search for best matching echo in vertical column
            for y in range(10, self.raw_data.shape[0] - 10):
                local_window = self.raw_data[y - 5 : y + 5, x - 10 : x + 10]

                # Calculate similarity score
                brightness_score = 1 - min(
                    1.0, abs(np.mean(local_window) - avg_brightness) / 128
                )
                contrast_score = (
                    1
                    - min(1.0, abs(np.std(local_window) - avg_contrast) / avg_contrast)
                    if avg_contrast > 0
                    else 0
                )
                gradient_score = (
                    1
                    - min(
                        1.0,
                        abs(np.mean(np.abs(np.gradient(local_window))) - avg_gradient)
                        / avg_gradient,
                    )
                    if avg_gradient > 0
                    else 0
                )

                total_score = (brightness_score + contrast_score + gradient_score) / 3

                if total_score > best_score:
                    best_score = total_score
                    best_match_y = y

            if best_match_y and best_score > 0.6:
                detected_points.append(
                    {"x": x, "y": best_match_y, "confidence": best_score}
                )

        return detected_points

    def update_picks_from_detection(self, detected_points):
        """Update picks array from automatic detection results"""
        for point in detected_points:
            x, y = point["x"], point["y"]

            # CRITICAL: Only update if point is within a defined region
            point_in_region = False
            for region in self.regions:
                if (
                    region["mode"] == self.current_picking_mode
                    and region["start_x"] <= x <= region["end_x"]
                ):
                    point_in_region = True
                    break

            # Only update if point is within a region boundary
            if point_in_region and 0 <= x < len(self.surface_picks):
                if self.current_picking_mode == "surface":
                    self.surface_picks[x] = y
                else:
                    self.bed_picks[x] = y

    def start_dragging(self, dragging_point):
        """Start dragging an existing control point."""
        self.dragging = True
        self.dragging_point = dragging_point

        if dragging_point in self.surface_control_points:
            self.dragging_list = self.surface_control_points
            self.dragging_type = "surface"
        elif dragging_point in self.bed_control_points:
            self.dragging_list = self.bed_control_points
            self.dragging_type = "bed"

        self.safe_set_cursor("hand")

    def on_mouse_release(self, event):
        """Handle mouse release events."""
        try:
            if self.dragging:
                self.dragging = False
                self.dragging_point = None
                self.dragging_list = None
                self.dragging_type = None
                self.safe_set_cursor("default")
                self.update_interpolation()
        except Exception as e:
            print(f"WARNING: Mouse release event error: {e}")
            self.reset_mouse_state()

        return True

    def on_mouse_motion(self, event):
        """Handle mouse motion for dragging points."""
        try:
            if (
                self.dragging
                and self.dragging_point is not None
                and event.inaxes == self.ax
            ):
                if event.xdata is not None and event.ydata is not None:
                    # Adjust coordinates for cropped display
                    adjusted_x = event.xdata
                    adjusted_y = event.ydata

                    old_x, old_y = self.dragging_point["x"], self.dragging_point["y"]

                    self.dragging_point["x"] = int(adjusted_x)
                    self.dragging_point["y"] = int(adjusted_y)

                    if (
                        old_x != self.dragging_point["x"]
                        or old_y != self.dragging_point["y"]
                    ):
                        if self.dragging_list is not None:
                            self.dragging_list.sort(key=lambda p: p["x"])

                        self.update_display()
        except Exception as e:
            print(f"WARNING: Mouse motion event error: {e}")

        return True

    def find_nearest_control_point(self, x, y, threshold=25):
        """Find the nearest control point within threshold distance."""
        if x is None or y is None:
            return None

        min_dist = float("inf")
        nearest_point = None
        all_points = []

        for cp in self.surface_control_points:
            all_points.append(cp)
        for cp in self.bed_control_points:
            all_points.append(cp)

        for cp in all_points:
            dist = np.sqrt((cp["x"] - x) ** 2 + (cp["y"] - y) ** 2)
            if dist < min_dist and dist < threshold:
                min_dist = dist
                nearest_point = cp

        return nearest_point

    def pan_view(self, direction):
        """Pan the view in the specified direction."""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        if direction == "left":
            new_xlim = (
                xlim[0] - self.pan_factor * x_range,
                xlim[1] - self.pan_factor * x_range,
            )
        elif direction == "right":
            new_xlim = (
                xlim[0] + self.pan_factor * x_range,
                xlim[1] + self.pan_factor * x_range,
            )
        elif direction == "up":
            new_ylim = (
                ylim[0] - self.pan_factor * y_range,
                ylim[1] - self.pan_factor * y_range,
            )
        elif direction == "down":
            new_ylim = (
                ylim[0] + self.pan_factor * y_range,
                ylim[1] + self.pan_factor * y_range,
            )

        if direction in ["left", "right"]:
            self.ax.set_xlim(new_xlim)
        else:
            self.ax.set_ylim(new_ylim)

        self.canvas.draw_idle()

    def zoom_view(self, direction):
        """Zoom the view in or out using keyboard controls only."""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2

        if direction == "in":
            factor = self.zoom_factor
        else:
            factor = 1 / self.zoom_factor

        x_range = (xlim[1] - xlim[0]) * factor
        y_range = (ylim[1] - ylim[0]) * factor

        new_xlim = (x_center - x_range / 2, x_center + x_range / 2)
        new_ylim = (y_center - y_range / 2, y_center + y_range / 2)

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.canvas.draw_idle()

    def cycle_display_mode(self):
        """Cycle through display modes."""
        self.display_mode = (self.display_mode % 3) + 1
        self.prepare_display_data()  # Reprocess data for new mode
        self.update_display()

    def toggle_bnw_mode(self):
        """Toggle between black/white and color display."""
        self.bnw_mode = not self.bnw_mode
        self.update_display()

    def toggle_picks_visibility(self):
        """Toggle visibility of pick lines."""
        self.picks_hidden = not self.picks_hidden
        self.update_display()

    def toggle_annotations_visibility(self):
        """Toggle visibility of annotations."""
        self.annotations_hidden = not self.annotations_hidden
        self.update_display()

    def toggle_region_selection(self):
        """Toggle or start region selection mode."""
        if self.workflow_state != "region_selection":
            self.workflow_state = "region_selection"
            print("INFO: Switched to region selection mode")
        else:
            print("INFO: Already in region selection mode")
        self.update_display()

    def remove_control_point(self, x, y, threshold=35):
        """Remove the nearest control point with improved detection."""
        if x is None or y is None:
            return False

        min_dist = float("inf")
        nearest_point = None
        point_list = None
        point_type = None

        for cp in list(self.surface_control_points):
            dist = np.sqrt((cp["x"] - x) ** 2 + (cp["y"] - y) ** 2)
            if dist < min_dist and dist < threshold:
                min_dist = dist
                nearest_point = cp
                point_list = self.surface_control_points
                point_type = "surface"

        for cp in list(self.bed_control_points):
            dist = np.sqrt((cp["x"] - x) ** 2 + (cp["y"] - y) ** 2)
            if dist < min_dist and dist < threshold:
                min_dist = dist
                nearest_point = cp
                point_list = self.bed_control_points
                point_type = "bed"

        if nearest_point is not None and point_list is not None:
            try:
                self.save_action(
                    "remove_control_point",
                    {
                        "point": nearest_point.copy(),
                        "point_type": point_type,
                        "x": nearest_point["x"],
                        "y": nearest_point["y"],
                    },
                )

                point_list.remove(nearest_point)
                print(
                    f"INFO: Removed {point_type} control point at ({nearest_point['x']}, {nearest_point['y']})"
                )

                # Re-run automatic detection if in auto-detection mode
                if self.workflow_state == "auto_detection":
                    self.run_automatic_detection()

                self.update_interpolation()
                self.update_display()
                return True

            except ValueError as e:
                print(f"WARNING: Failed to remove control point: {e}")
                return False

        return False

    def interpolate_trace(
        self,
        control_points,
        picks_array,
        method="constrained_spline",
        trace_type="surface",
    ):
        """Interpolate trace ONLY within defined regions - no inter-region interpolation."""

        # First, clear the entire picks array to ensure no interpolation between regions
        picks_array.fill(np.nan)

        # Group control points by their associated regions
        for region in self.regions:
            if region["mode"] != trace_type:
                continue

            # Find control points that belong to this region
            region_points = []
            region_start = int(region["start_x"])
            region_end = int(region["end_x"])

            for cp in control_points:
                if region_start <= cp["x"] <= region_end:
                    region_points.append(cp)

            if len(region_points) < 2:
                continue  # Need at least 2 points to interpolate

            # Sort points by x-coordinate within the region
            region_points.sort(key=lambda p: p["x"])

            x_coords = np.array([cp["x"] for cp in region_points])
            y_coords = np.array([cp["y"] for cp in region_points])

            try:
                # Perform interpolation using selected method
                if method == "constrained_spline":
                    y_interp = self.constrained_spline_interpolation(x_coords, y_coords)
                elif method == "pchip":
                    y_interp = self.pchip_interpolation(x_coords, y_coords)
                elif method == "linear_smoothed":
                    y_interp = self.linear_smoothed_interpolation(x_coords, y_coords)
                else:
                    y_interp = self.linear_interpolation(x_coords, y_coords)

                # Apply geological constraints
                y_interp = self.apply_trace_constraints(y_interp, trace_type)

                # Only update picks array WITHIN the region bounds
                x_range = np.arange(min(x_coords), max(x_coords) + 1)

                # Ensure we stay within region boundaries
                for i, x_pos in enumerate(x_range):
                    if region_start <= x_pos <= region_end and i < len(y_interp):
                        if 0 <= x_pos < len(picks_array):
                            picks_array[x_pos] = y_interp[i]

            except Exception as e:
                print(
                    f"WARNING: {method} interpolation failed for {trace_type} in region {region_start}-{region_end}: {e}"
                )

    def constrained_spline_interpolation(self, x_coords, y_coords):
        """Constrained spline interpolation with reduced oscillation."""
        interp = PchipInterpolator(x_coords, y_coords)
        x_range = np.arange(min(x_coords), max(x_coords) + 1)
        return interp(x_range)

    def pchip_interpolation(self, x_coords, y_coords):
        """PCHIP interpolation - shape-preserving and monotonic."""
        interp = PchipInterpolator(x_coords, y_coords)
        x_range = np.arange(min(x_coords), max(x_coords) + 1)
        return interp(x_range)

    def update_display(self):
        """Update the display with current data and picks."""
        try:
            self.ax.clear()

            # Select data based on display mode
            if self.display_mode == 1:
                data_to_display = self.raw_data
                mode_name = "Raw"
            elif self.display_mode == 2:
                data_to_display = self.smoothed_data
                mode_name = "Smoothed"
            else:
                data_to_display = self.diff_data
                mode_name = "Difference"

            # Display radar image
            if self.bnw_mode:
                cmap = "gray"
            else:
                cmap = "viridis"

            im = self.ax.imshow(
                data_to_display,
                cmap=cmap,
                aspect="auto",
                interpolation="nearest",
                extent=[0, data_to_display.shape[1], data_to_display.shape[0], 0],
            )

            # Display regions
            self.display_regions()

            # Display control points and picks
            if not self.picks_hidden:
                self.display_picks_and_points()

            # Display annotations
            if not self.annotations_hidden:
                self.display_annotations()

            # Set title and labels
            workflow_state_text = self.workflow_state.replace("_", " ").title()
            title = f"{self.title} - {mode_name} - {workflow_state_text} ({self.current_picking_mode.title()})"
            self.ax.set_title(title, fontsize=12, fontweight="bold")

            self.ax.set_xlabel("X Pixel (Distance)")
            self.ax.set_ylabel("Y Pixel (Depth)")

            # Add status text
            self.add_status_text()

            # Refresh canvas
            self.canvas.draw_idle()

        except Exception as e:
            print(f"WARNING: Display update failed: {e}")

    def display_regions(self):
        """Display defined regions with enhanced visual boundaries."""
        if not hasattr(self, "regions"):
            return

        for i, region in enumerate(self.regions):
            start_x = region["start_x"]
            end_x = region["end_x"]

            # Choose color based on region mode
            if region["mode"] == "surface":
                color = "cyan"
                alpha = 0.15
            else:
                color = "orange"
                alpha = 0.15

            # Add region boundary rectangles
            height = self.raw_data.shape[0]
            rect = patches.Rectangle(
                (start_x, 0),
                end_x - start_x,
                height,
                linewidth=3,
                edgecolor=color,
                facecolor=color,
                alpha=alpha,
                linestyle="-",
            )
            self.ax.add_patch(rect)

            # Add thick boundary lines
            self.ax.axvline(x=start_x, color=color, linewidth=3, alpha=0.8)
            self.ax.axvline(x=end_x, color=color, linewidth=3, alpha=0.8)

            # Add region label
            label_y = height * 0.05
            self.ax.text(
                start_x + (end_x - start_x) / 2,
                label_y,
                f"{region['mode'].title()} Region {i + 1}",
                color=color,
                fontweight="bold",
                fontsize=10,
                ha="center",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="white",
                    alpha=0.9,
                    edgecolor=color,
                ),
            )

    def display_picks_and_points(self):
        """Display control points and interpolated picks - region-wise only."""
        # Display surface control points
        if self.surface_control_points:
            surface_x = [cp["x"] for cp in self.surface_control_points]
            surface_y = [cp["y"] for cp in self.surface_control_points]

            self.ax.scatter(
                surface_x,
                surface_y,
                c="red",
                s=50,
                marker="o",
                alpha=0.8,
                edgecolors="black",
                linewidths=1,
                label="Surface Control Points",
                zorder=5,
            )

        # Display bed control points
        if self.bed_control_points:
            bed_x = [cp["x"] for cp in self.bed_control_points]
            bed_y = [cp["y"] for cp in self.bed_control_points]

            self.ax.scatter(
                bed_x,
                bed_y,
                c="blue",
                s=50,
                marker="s",
                alpha=0.8,
                edgecolors="black",
                linewidths=1,
                label="Bed Control Points",
                zorder=5,
            )

        # Display interpolated picks - ONLY within regions
        self._display_region_wise_picks("surface", "red", "Surface Echo")
        self._display_region_wise_picks("bed", "blue", "Bed Echo")

        # Add legend if there are points or picks
        if self.surface_control_points or self.bed_control_points:
            self.ax.legend(loc="upper right", framealpha=0.8, fontsize=8)

    def _display_region_wise_picks(self, echo_type, color, label):
        """Display picks only within defined regions."""
        picks_array = self.surface_picks if echo_type == "surface" else self.bed_picks

        # Plot each region separately to show gaps between regions
        first_region = True
        for region in self.regions:
            if region["mode"] != echo_type:
                continue

            start_x = int(region["start_x"])
            end_x = int(region["end_x"])

            # Get picks for this region only
            region_mask = np.zeros_like(picks_array, dtype=bool)
            region_mask[start_x : end_x + 1] = True
            region_valid = np.isfinite(picks_array) & region_mask

            if np.any(region_valid):
                x_indices = np.where(region_valid)[0]
                y_values = picks_array[region_valid]

                # Only add label for first region to avoid duplicate legend entries
                region_label = label if first_region else ""
                first_region = False

                self.ax.plot(
                    x_indices,
                    y_values,
                    color=color,
                    linewidth=2,
                    alpha=0.7,
                    label=region_label,
                    zorder=3,
                )

    def get_regions_for_echo_type(self, echo_type):
        """Get all regions for a specific echo type, sorted by x-position."""
        regions = [r for r in self.regions if r["mode"] == echo_type]
        return sorted(regions, key=lambda r: r["start_x"])

    def check_region_gaps(self, echo_type):
        """Check for gaps between regions of the same type."""
        regions = self.get_regions_for_echo_type(echo_type)

        if len(regions) < 2:
            return []

        gaps = []
        for i in range(len(regions) - 1):
            current_end = regions[i]["end_x"]
            next_start = regions[i + 1]["start_x"]

            if next_start > current_end + 1:
                gaps.append(
                    {
                        "start": current_end + 1,
                        "end": next_start - 1,
                        "width": next_start - current_end - 1,
                    }
                )

        return gaps

    def validate_region_gaps(self):
        """Validate that gaps between regions remain as NaN values."""
        # Get all region boundaries for current picking mode
        current_regions = [
            r for r in self.regions if r["mode"] == self.current_picking_mode
        ]

        if len(current_regions) < 2:
            return  # No gaps to validate

        # Sort regions by start position
        current_regions.sort(key=lambda r: r["start_x"])

        # Check gaps between consecutive regions
        for i in range(len(current_regions) - 1):
            current_end = int(current_regions[i]["end_x"])
            next_start = int(current_regions[i + 1]["start_x"])

            if next_start > current_end + 1:
                # There should be a gap - ensure it's filled with NaN
                gap_start = current_end + 1
                gap_end = next_start - 1

                if self.current_picking_mode == "surface":
                    self.surface_picks[gap_start : gap_end + 1] = np.nan
                else:
                    self.bed_picks[gap_start : gap_end + 1] = np.nan

                print(f"INFO: Preserved gap from x={gap_start} to x={gap_end}")

    def update_interpolation(self, method=None):
        """Update interpolation with gap validation."""
        if method is None:
            method = self.current_method

        # Surface interpolation
        if len(self.surface_control_points) >= 2:
            self.interpolate_trace(
                self.surface_control_points,
                self.surface_picks,
                method=method,
                trace_type="surface",
            )

        # Bed interpolation
        if len(self.bed_control_points) >= 2:
            self.interpolate_trace(
                self.bed_control_points, self.bed_picks, method=method, trace_type="bed"
            )

        # CRITICAL: Validate and preserve gaps between regions
        self.validate_region_gaps()
        self.update_display()

    def validate_region_associations(self):
        """Validate that all control points are properly associated with regions."""
        orphaned_points = []

        # Check surface control points
        for cp in self.surface_control_points:
            found_region = False
            for region in self.regions:
                if (
                    region["mode"] == "surface"
                    and region["start_x"] <= cp["x"] <= region["end_x"]
                ):
                    found_region = True
                    break
            if not found_region:
                orphaned_points.append(("surface", cp))

        # Check bed control points
        for cp in self.bed_control_points:
            found_region = False
            for region in self.regions:
                if (
                    region["mode"] == "bed"
                    and region["start_x"] <= cp["x"] <= region["end_x"]
                ):
                    found_region = True
                    break
            if not found_region:
                orphaned_points.append(("bed", cp))

        if orphaned_points:
            print(
                f"WARNING: Found {len(orphaned_points)} control points outside defined regions"
            )
            for point_type, cp in orphaned_points:
                print(
                    f"  {point_type} point at ({cp['x']}, {cp['y']}) has no associated region"
                )

        return len(orphaned_points) == 0

    def display_annotations(self):
        """Display workflow annotations and status."""
        if hasattr(self, "gap_regions") and self.gap_regions:
            for gap in self.gap_regions:
                start_x = gap["start"] - self.display_boundaries["left"]
                end_x = gap["end"] - self.display_boundaries["left"]

                # Add gap region overlay
                rect = patches.Rectangle(
                    (start_x, 0),
                    end_x - start_x,
                    self.raw_data.shape[0],
                    linewidth=1,
                    edgecolor="yellow",
                    facecolor="yellow",
                    alpha=0.3,
                    linestyle=":",
                )
                self.ax.add_patch(rect)

    def add_status_text(self):
        """Add status information to the display."""
        status_lines = []

        # Workflow status
        status_lines.append(f"Mode: {self.workflow_state.replace('_', ' ').title()}")
        status_lines.append(f"Echo Type: {self.current_picking_mode.title()}")

        # Region status
        if hasattr(self, "regions"):
            surface_regions = sum(1 for r in self.regions if r["mode"] == "surface")
            bed_regions = sum(1 for r in self.regions if r["mode"] == "bed")
            status_lines.append(
                f"Regions: {surface_regions} Surface, {bed_regions} Bed"
            )

        # Control points status
        status_lines.append(
            f"Control Points: {len(self.surface_control_points)} Surface, {len(self.bed_control_points)} Bed"
        )

        # Display status text
        status_text = "\n".join(status_lines)
        self.ax.text(
            0.02,
            0.98,
            status_text,
            transform=self.ax.transAxes,
            fontsize=10,
            fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        )

        # Add keyboard shortcuts reminder
        shortcuts_text = (
            "Keys: [r] Regions [Space] Auto-detect [s] Surface [b] Bed\n"
            "[+/-] Zoom [←→↑↓] Pan [d] Display [c] B&W [y] Finish"
        )
        self.ax.text(
            0.02,
            0.02,
            shortcuts_text,
            transform=self.ax.transAxes,
            fontsize=8,
            fontfamily="monospace",
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

    def apply_trace_constraints(self, y_interp, trace_type):
        """Apply geological constraints to interpolated trace."""
        if len(y_interp) < 2:
            return y_interp

        # Apply smoothing to reduce noise
        smoothed = gaussian_filter1d(y_interp, sigma=1.0)

        # Apply slope constraints
        max_slope = 5.0  # Maximum pixels per pixel slope
        for i in range(1, len(smoothed)):
            slope = abs(smoothed[i] - smoothed[i - 1])
            if slope > max_slope:
                # Limit slope
                direction = 1 if smoothed[i] > smoothed[i - 1] else -1
                smoothed[i] = smoothed[i - 1] + direction * max_slope

        return smoothed

    def linear_interpolation(self, x_coords, y_coords):
        """Simple linear interpolation."""
        interp = interp1d(
            x_coords,
            y_coords,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        x_range = np.arange(min(x_coords), max(x_coords) + 1)
        return interp(x_range)

    def linear_smoothed_interpolation(self, x_coords, y_coords):
        """Linear interpolation with smoothing."""
        y_linear = self.linear_interpolation(x_coords, y_coords)
        return gaussian_filter1d(y_linear, sigma=2.0)

    def cycle_interpolation_method(self):
        """Cycle through available interpolation methods."""
        methods = ["constrained_spline", "pchip", "linear_smoothed", "linear"]
        current_index = methods.index(self.current_method)
        self.current_method = methods[(current_index + 1) % len(methods)]
        print(f"INFO: Switched to {self.current_method} interpolation")
        self.update_interpolation()

    def apply_spline_interpolation(self):
        """Apply spline interpolation to current picks."""
        self.update_interpolation(method="constrained_spline")
        print("INFO: Applied constrained spline interpolation")

    def mark_gap_region(self):
        """Mark a region as a gap (no echoes)."""
        print("INFO: Gap region marking not yet implemented")
        # TODO: Implement gap region marking functionality

    def save_action(self, action_type, action_data):
        """Save action to history for undo functionality."""
        if len(self.action_history) >= self.max_history:
            self.action_history.pop(0)

        self.action_history.append(
            {
                "type": action_type,
                "data": action_data,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def undo_last_action(self):
        """Undo the last action."""
        if not self.action_history:
            print("INFO: No actions to undo")
            return

        last_action = self.action_history.pop()
        action_type = last_action["type"]
        action_data = last_action["data"]

        try:
            if action_type == "add_control_point":
                # Remove the added point
                point_type = action_data["point_type"]
                point = action_data["point"]

                if point_type == "surface" and point in self.surface_control_points:
                    self.surface_control_points.remove(point)
                elif point_type == "bed" and point in self.bed_control_points:
                    self.bed_control_points.remove(point)

                print(f"INFO: Undid add {point_type} control point")

            elif action_type == "remove_control_point":
                # Re-add the removed point
                point_type = action_data["point_type"]
                point = action_data["point"]

                if point_type == "surface":
                    self.surface_control_points.append(point)
                    self.surface_control_points.sort(key=lambda p: p["x"])
                elif point_type == "bed":
                    self.bed_control_points.append(point)
                    self.bed_control_points.sort(key=lambda p: p["x"])

                print(f"INFO: Undid remove {point_type} control point")

            self.update_interpolation()
            self.update_display()

        except Exception as e:
            print(f"WARNING: Failed to undo action: {e}")

    def finalize_picking(self):
        """Finalize the picking session and close GUI."""
        print("INFO: Finalizing picking session...")

        # Update session manager with final results
        if self.session_manager:
            try:
                # Create clean regions data without circular references
                clean_regions_data = {"surface": [], "bed": []}

                for region in self.regions:
                    clean_region = {
                        "mode": region["mode"],
                        "bounds": (int(region["start_x"]), int(region["end_x"])),
                        "control_points": [
                            {"x": int(p.get("x", 0)), "y": int(p.get("y", 0))}
                            for p in region.get("picks", [])
                        ],
                        "status": "completed",
                    }
                    clean_regions_data[region["mode"]].append(clean_region)

                # Use safe update method
                self.session_manager.update_regions_data(clean_regions_data)

                # Add final processing event
                self.session_manager.add_processing_event(
                    "session_finalized",
                    {
                        "total_surface_points": len(self.surface_control_points),
                        "total_bed_points": len(self.bed_control_points),
                        "regions_created": len(self.regions),
                        "workflow_completed": True,
                    },
                )

                print("INFO: Session data updated successfully")

            except Exception as e:
                print(f"WARNING: Failed to update session: {e}")

        # Close the plot
        plt.close(self.fig)
        print("INFO: Picking session completed")
