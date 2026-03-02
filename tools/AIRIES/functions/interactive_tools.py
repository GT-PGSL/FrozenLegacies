import matplotlib.pyplot as plt
import numpy as np
import cv2


class ClickSelector:
    """
    An interactive tool to select a point (specifically an X-coordinate) on an image.

    When instantiated with an image, it displays the image in a Matplotlib window.
    The user can click on the image. The class captures the X and Y coordinates of the click.
    The window closes automatically after the first click.

    Attributes:
        image (np.ndarray): The image to be displayed.
        selected_x (int or None): The X-coordinate of the point clicked by the user.
                                  None if no click has occurred or window closed.
        selected_y (int or None): The Y-coordinate of the point clicked by the user.
                                  None if no click has occurred or window closed.
        fig (matplotlib.figure.Figure): The Matplotlib figure object.
        ax (matplotlib.axes.Axes): The Matplotlib axes object displaying the image.
    """

    def __init__(self, image_to_display, title="Click on the target location"):
        """
        Initializes the ClickSelector and displays the image for selection.

        Args:
            image_to_display (np.ndarray): The image (as a NumPy array) on which the user will click.
            title (str, optional): The title for the Matplotlib window.
                                   Defaults to "Click on the target location".
        """
        self.image = image_to_display
        self.selected_x = None
        self.selected_y = None

        # Determine figure size
        img_height, img_width = self.image.shape[:2]
        fig_height_inches = 6
        aspect_ratio = img_width / img_height
        fig_width_inches = min(24, fig_height_inches * aspect_ratio)

        # If image is very tall and narrow, this might result in too narrow a figure,
        # so ensure a minimum width too, e.g., 8 inches.
        fig_width_inches = max(8, fig_width_inches)

        self.fig, self.ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches))
        self.ax.imshow(self.image, cmap="gray", aspect="auto")
        self.ax.set_title(title, fontsize=12)
        self.ax.set_xlabel("X-pixel coordinate")
        self.ax.set_ylabel("Y-pixel coordinate")

        # Connect the click event to the onclick method
        self.cid = self.fig.canvas.mpl_connect("button_press_event", self._onclick)

        print(
            "INFO: Displaying image for selection. Click the desired location in the pop-up window."
        )
        print("      The window will close automatically after your click.")
        plt.show()  # This will block until the window is closed

    def _onclick(self, event):
        """
        Handles the mouse click event on the Matplotlib figure.

        Stores the click coordinates and closes the figure.

        Args:
            event (matplotlib.backend_bases.MouseEvent): The Matplotlib mouse event.
        """
        # Check if the click was within the axes
        if event.inaxes == self.ax:
            if event.xdata is not None and event.ydata is not None:
                self.selected_x = int(round(event.xdata))
                self.selected_y = int(round(event.ydata))
                print(f"INFO: User selected X={self.selected_x}, Y={self.selected_y}")
            else:
                print(
                    "INFO: Click was outside image data area. No coordinates captured."
                )
        else:
            print("INFO: Click was outside the main axes. No coordinates captured.")

        # Disconnect the event handler and close the figure
        if hasattr(self, "cid") and self.cid is not None:
            self.fig.canvas.mpl_disconnect(self.cid)
            self.cid = None  # Prevent multiple disconnects if somehow called again
        plt.close(self.fig)


class ThreePointSelector:
    """
    Interactive tool to select exactly 3 CBD tick marks for automatic count determination.
    User selects: first tick, second tick (for spacing), and last tick (for validation).
    After selection, shows all calculated CBD picks for debugging and quality evaluation.
    """

    def __init__(
        self, image_to_display, title="Select 3 CBD tick marks for validation"
    ):
        self.image = image_to_display
        self.selected_points = []
        self.point_labels = ["First", "Second", "Last"]
        self.current_point = 0

        # Store calculated results for visualization
        self.calculated_ticks = []
        self.cbd_sequence = []
        self.tick_count = 0

        # Use same preprocessing as version 3.3 CBDTickSelector
        height, width = image_to_display.shape
        sprocket_removal_ratio = 0.08
        search_height_ratio = 0.12

        sprocket_height = int(height * sprocket_removal_ratio)
        search_start = sprocket_height
        search_height = int(height * search_height_ratio)
        search_end = search_start + search_height

        # Extract clean region BELOW the sprocket holes
        clean_region = image_to_display[search_start:search_end, :]

        # Enhanced preprocessing
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))
        enhanced_region = clahe.apply(clean_region)

        # Apply strong unsharp masking
        gaussian_blur = cv2.GaussianBlur(enhanced_region, (0, 0), 3.0)
        unsharp_mask = cv2.addWeighted(enhanced_region, 2.0, gaussian_blur, -1.0, 0)
        final_region = np.clip(unsharp_mask, 0, 255).astype(np.uint8)

        # Create figure with zoomed display
        self.fig, self.ax = plt.subplots(figsize=(24, 8))

        # Display with correct coordinate mapping
        self.ax.imshow(
            final_region,
            cmap="gray",
            aspect="auto",
            extent=[0, width, search_end, search_start],
        )

        # Initialize crosshair lines
        self.crosshair_v = self.ax.axvline(
            x=0,
            color="yellow",
            linestyle="-",
            linewidth=1,
            alpha=0.8,
            visible=False,
            zorder=20,
        )
        self.crosshair_h = self.ax.axhline(
            y=0,
            color="yellow",
            linestyle="-",
            linewidth=1,
            alpha=0.8,
            visible=False,
            zorder=20,
        )

        # Connect mouse motion event for crosshair
        self.motion_cid = self.fig.canvas.mpl_connect(
            "motion_notify_event", self._on_mouse_move
        )

        # Enhanced title and styling
        self.ax.set_title(
            f"{title} - Click {self.point_labels[0]} tick mark",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        self.ax.set_xlabel("X Position (pixels)", fontsize=14)
        self.ax.set_ylabel("Y Position (pixels)", fontsize=14)

        # Enhanced grid for precision
        self.ax.grid(True, alpha=0.6, linestyle="-", linewidth=0.8, color="cyan")

        # Set limits to match the clean view
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(search_end, search_start)

        # Store parameters for coordinate adjustment
        self.search_start = search_start
        self.search_end = search_end
        self.sprocket_height = sprocket_height

        # Enhanced instructions
        self.ax.text(
            0.98,
            0.98,
            "3-POINT CBD SELECTION:\n"
            "• Click FIRST tick (leftmost)\n"
            "• Click SECOND tick (for spacing)\n"
            "• Click LAST tick (rightmost)\n"
            "• Yellow crosshair provides precise alignment",
            transform=self.ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.95),
        )

        # Connect click event
        self.cid = self.fig.canvas.mpl_connect("button_press_event", self._onclick)

        print("INFO: Select 3 CBD tick marks in order:")
        print("  1. First tick mark (leftmost)")
        print("  2. Second tick mark (for spacing calculation)")
        print("  3. Last tick mark (rightmost)")
        print("  4. Yellow crosshair provides precise alignment feedback")

        plt.tight_layout()
        plt.show()

    def _on_mouse_move(self, event):
        """Handle mouse movement to update crosshair position."""
        if event.inaxes == self.ax:
            # Update crosshair position
            self.crosshair_v.set_xdata([event.xdata])
            self.crosshair_h.set_ydata([event.ydata])
            self.crosshair_v.set_visible(True)
            self.crosshair_h.set_visible(True)
            self.fig.canvas.draw_idle()
        else:
            self.crosshair_v.set_visible(False)
            self.crosshair_h.set_visible(False)
            self.fig.canvas.draw_idle()

    def _onclick(self, event):
        if (
            event.inaxes == self.ax
            and event.xdata is not None
            and len(self.selected_points) < 3
        ):
            x, y = int(round(event.xdata)), int(round(event.ydata))
            self.selected_points.append((x, y))

            # Enhanced visual markers
            colors = ["red", "blue", "green"]
            color = colors[self.current_point]
            label = self.point_labels[self.current_point]

            # Improved visual markers
            self.ax.plot(
                x,
                y,
                "o",
                color=color,
                markersize=12,
                markeredgewidth=2,
                markeredgecolor="white",
                markerfacecolor=color,
                label=f"{label} CBD Tick",
                zorder=15,
                alpha=0.8,
            )

            # Vertical reference line
            self.ax.axvline(
                x=x,
                color=color,
                linestyle="-",
                alpha=0.6,
                linewidth=2,
                zorder=14,
            )

            # Enhanced annotation
            self.ax.annotate(
                f"{label} CBD Tick\nX: {int(x)}",
                xy=(x, y),
                xytext=(15, 15),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc=color, alpha=0.7),
                fontsize=12,
                color="white",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=color, lw=2),
                zorder=16,
            )

            print(f"INFO: Selected {label} tick at X={x}")

            self.current_point += 1

            if self.current_point < 3:
                self.ax.set_title(
                    f"Select 3 CBD tick marks - Click {self.point_labels[self.current_point]} tick mark",
                    fontsize=16,
                    fontweight="bold",
                    pad=20,
                )
                self.fig.canvas.draw()
            else:
                print("INFO: All 3 tick marks selected successfully")
                self.ax.set_title(
                    "All 3 CBD tick marks selected - Processing and visualizing results...",
                    fontsize=16,
                    fontweight="bold",
                    pad=20,
                )
                self.ax.legend(fontsize=12, loc="upper right")
                self.fig.canvas.draw()

                # NEW: Calculate and visualize all CBD picks
                self._calculate_and_show_all_cbd_picks()

    def _calculate_and_show_all_cbd_picks(self):
        """Calculate all CBD tick positions using ALL THREE control points for accurate interpolation."""
        if len(self.selected_points) != 3:
            return

        import re

        # Get all three control points
        first_x, second_x, last_x = [p[0] for p in self.selected_points]

        # Calculate spacing from first two points
        spacing = abs(second_x - first_x)

        # Estimate total tick count based on distance and spacing
        total_distance = abs(last_x - first_x)
        estimated_count = round(total_distance / spacing) + 1

        print(f"INFO: Using THREE-POINT interpolation for accurate positioning")
        print(
            f"INFO: Control points: First={first_x}, Second={second_x}, Last={last_x}"
        )
        print(f"INFO: Estimated tick count: {estimated_count}")

        # ===  THREE-POINT INTERPOLATION ===
        if estimated_count >= 3:
            # Define control indices for the three selected points
            first_idx = 0
            second_idx = 1
            last_idx = estimated_count - 1

            # Create tick positions array
            self.calculated_ticks = []

            for i in range(estimated_count):
                if i == first_idx:
                    tick_x = first_x
                elif i == second_idx:
                    tick_x = second_x
                elif i == last_idx:
                    tick_x = last_x
                else:
                    # Interpolate between control points
                    if i < second_idx:
                        # Between first and second point
                        fraction = i / second_idx
                        tick_x = first_x + fraction * (second_x - first_x)
                    else:
                        # Between second and last point
                        fraction = (i - second_idx) / (last_idx - second_idx)
                        tick_x = second_x + fraction * (last_x - second_x)

                self.calculated_ticks.append(int(tick_x))
        else:
            # Fallback for fewer than 3 ticks
            self.calculated_ticks = [first_x, second_x, last_x][:estimated_count]

        self.tick_count = len(self.calculated_ticks)

        print(
            f"SUCCESS: Generated {self.tick_count} CBD tick positions using 3-point control"
        )
        print(f"INFO: Tick positions: {self.calculated_ticks}")

        # === VISUALIZE ALL CBD PICKS WITH ENHANCED MARKERS ===

        # Plot all calculated tick positions
        for i, tick_x in enumerate(self.calculated_ticks):
            if 0 <= tick_x < self.ax.get_xlim()[1]:  # Within image bounds
                if i < 3:
                    # Skip the first three ticks (already plotted as red, blue, green)
                    continue
                else:
                    # Plot calculated tick as green line with enhanced visibility
                    self.ax.axvline(
                        x=tick_x,
                        color="lime",  # Brighter green for better visibility
                        linestyle="-",
                        alpha=0.8,
                        linewidth=2.0,  # Thicker line
                        zorder=13,
                    )

                    # Add tick labels for every other tick
                    if i % 2 == 0:
                        y_label_pos = (
                            self.search_start
                            + (self.search_end - self.search_start) * 0.15
                        )
                        self.ax.text(
                            tick_x,
                            y_label_pos,
                            f"T{i + 1}",
                            ha="center",
                            va="bottom",
                            fontsize=10,
                            color="darkgreen",
                            fontweight="bold",
                            zorder=17,
                            bbox=dict(
                                boxstyle="round,pad=0.2", fc="lightgreen", alpha=0.9
                            ),
                        )

        # === CONTROL POINT VISUALIZATION ===
        control_colors = ["red", "blue", "green"]
        control_labels = ["1st (Control)", "2nd (Spacing)", "3rd (Endpoint)"]

        for i, (tick_x, color, label) in enumerate(
            zip([first_x, second_x, last_x], control_colors, control_labels)
        ):
            # Add enhanced control point markers
            y_marker_pos = (
                self.search_start + (self.search_end - self.search_start) * 0.8
            )
            self.ax.plot(
                tick_x,
                y_marker_pos,
                marker="v",
                color=color,
                markersize=15,
                markeredgewidth=2,
                markeredgecolor="white",
                zorder=20,
                label=f"{label}",
            )

        # Update title with enhanced results summary
        self.ax.set_title(
            f"3-Point CBD Calibration Complete - {self.tick_count} ticks calculated\n"
            f"Using THREE control points for accurate positioning. Close window to continue.",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        # Add enhanced result summary text
        result_text = (
            f"THREE-POINT INTERPOLATION RESULTS:\n"
            f"• Control Point 1: X={first_x} (Fixed)\n"
            f"• Control Point 2: X={second_x} (Spacing)\n"
            f"• Control Point 3: X={last_x} (Endpoint)\n"
            f"• Total ticks: {self.tick_count}\n"
            f"• Method: 3-Point Linear Interpolation\n"
            f"• Lime lines show calculated CBD positions"
        )

        self.ax.text(
            0.02,
            0.98,
            result_text,
            transform=self.ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.95),
        )

        # Clean up event connections but keep window open for inspection
        if hasattr(self, "motion_cid") and self.motion_cid is not None:
            self.fig.canvas.mpl_disconnect(self.motion_cid)
            self.motion_cid = None

        if hasattr(self, "cid") and self.cid is not None:
            self.fig.canvas.mpl_disconnect(self.cid)
            self.cid = None

        # Draw final result
        self.fig.canvas.draw()

        print(
            "INFO: CBD interpolation complete. Close the window to continue processing."
        )

    @property
    def tick_positions(self):
        """Return just the X coordinates of the selected ticks."""
        return (
            [point[0] for point in self.selected_points]
            if len(self.selected_points) == 3
            else None
        )

    def calculate_tick_count_and_positions(self, filename):
        """
        Calculate total tick count and all tick positions using 3-point selection with enhanced control.
        """
        if len(self.selected_points) != 3:
            return None, 0, []

        import re

        # Extract CBD range from filename
        cbd_match = re.search(r"C(\d+)_(\d+)", filename)
        if not cbd_match:
            print(f"ERROR: Could not extract CBD range from filename: {filename}")
            return None, 0, []

        cbd_start = int(cbd_match.group(1))  # 468
        cbd_end = int(cbd_match.group(2))  # 481
        expected_cbd_count = abs(cbd_end - cbd_start) + 1  # 14 ticks for 468-481

        # Get all three control points
        first_x, second_x, last_x = [p[0] for p in self.selected_points]

        # Use the calculated ticks from visualization (which used 3-point interpolation)
        if hasattr(self, "calculated_ticks") and self.calculated_ticks:
            calculated_ticks = self.calculated_ticks
            actual_tick_count = len(calculated_ticks)

            print(f"INFO: Using 3-point interpolated positions:")
            print(f"  Control points: {first_x}, {second_x}, {last_x}")
            print(f"  Expected {expected_cbd_count} ticks from filename")
            print(f"  Calculated {actual_tick_count} ticks from 3-point interpolation")

            # Create CBD sequence (descending: 481, 480, 479, ..., 468)
            if actual_tick_count == expected_cbd_count:
                cbd_sequence = list(range(cbd_end, cbd_start - 1, -1))
                print(
                    f"✅ Perfect match: {actual_tick_count} ticks = {expected_cbd_count} expected CBDs"
                )
            else:
                print(
                    f"⚠️  Count adjustment: Using {actual_tick_count} ticks vs {expected_cbd_count} expected"
                )
                cbd_sequence = list(range(cbd_end, cbd_end - actual_tick_count, -1))

            return calculated_ticks, actual_tick_count, cbd_sequence
        else:
            print("ERROR: No calculated ticks available from 3-point interpolation")
            return None, 0, []


class EchoPointSelector:
    """
    Interactive tool to select multiple points for echo trace refinement.
    """

    def __init__(self, image_to_display, title="Select echo points"):
        self.image = image_to_display
        self.selected_points = []

        # Determine figure size
        img_height, img_width = self.image.shape[:2]
        fig_height_inches = 6
        aspect_ratio = img_width / img_height
        fig_width_inches = min(24, fig_height_inches * aspect_ratio)
        fig_width_inches = max(8, fig_width_inches)

        self.fig, self.ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches))
        self.ax.imshow(self.image, cmap="gray", aspect="auto")
        self.ax.set_title(title, fontsize=12)
        self.ax.set_xlabel("X-pixel coordinate")
        self.ax.set_ylabel("Y-pixel coordinate")

        # Connect the click event
        self.cid = self.fig.canvas.mpl_connect("button_press_event", self._onclick)

        print("INFO: Click on echo points. Press 'q' to finish selection.")
        plt.show()

    def _onclick(self, event):
        if event.inaxes == self.ax:
            if event.xdata is not None and event.ydata is not None:
                x = int(round(event.xdata))
                y = int(round(event.ydata))
                self.selected_points.append((x, y))

                # Mark the selected point
                self.ax.plot(x, y, "ro", markersize=6)
                self.ax.text(
                    x,
                    y - 10,
                    f"{len(self.selected_points)}",
                    ha="center",
                    va="bottom",
                    color="red",
                    fontweight="bold",
                )

                print(
                    f"INFO: Selected point {len(self.selected_points)} at X={x}, Y={y}"
                )
                self.fig.canvas.draw()

    def get_points(self):
        """Return the list of selected points."""
        return self.selected_points


def get_manual_feature_annotations(
    default_features,
    pixels_per_microsecond,
    transmitter_pulse_y_abs,
    prompt_message="Do you want to manually annotate radar features? (yes/no): ",
):
    """
    Prompts the user to manually input or confirm pixel coordinates for radar features.

    It iterates through a dictionary of default features, allowing the user to
    update the 'pixel_abs' (absolute Y-coordinate) for each. If updated, the
    corresponding 'time_us' is recalculated.

    Args:
        default_features (dict): A dictionary where keys are feature identifiers (e.g., 'i')
                                 and values are dictionaries containing:
                                     'name' (str): Display name (e.g., "Ice Surface").
                                     'pixel_abs' (int): Default absolute Y-pixel coordinate.
                                     'color' (str): Color for visualization.
                                     (Optionally 'time_us' can be pre-filled or will be calculated).
        pixels_per_microsecond (float): Calibration factor (pixels / µs) used to calculate time.
        transmitter_pulse_y_abs (int): Absolute Y-coordinate of the transmitter pulse (0 µs reference).
        prompt_message (str, optional): The message to display when asking if the user wants to annotate.

    Returns:
        tuple: (updated_features_dict, bool)
               - updated_features_dict (dict): The dictionary of features, potentially updated by the user.
               - user_did_annotate (bool): True if the user chose to annotate, False otherwise.
    """
    updated_features = default_features.copy()  # Work on a copy
    user_did_annotate = False

    while True:
        annotate_choice = input(prompt_message).strip().lower()
        if annotate_choice in ["yes", "y", "no", "n"]:
            break
        print("Invalid input. Please enter 'yes' (or 'y') or 'no' (or 'n').")

    if annotate_choice in ["yes", "y"]:
        user_did_annotate = True
        print("\n--- Manual Feature Annotation ---")
        print(
            "For each feature, enter the absolute Y-pixel coordinate from the original image."
        )
        print("Press Enter to keep the current default value.")

        for key, feature_details in updated_features.items():
            current_pixel = feature_details.get("pixel_abs", "Not set")
            prompt_text = (
                f"Enter Y-pixel for '{feature_details['name']}' "
                f"(current: {current_pixel}): "
            )

            # We usually don't ask to re-input the transmitter pulse if it's auto-detected
            if (
                key == "t" and "pixel_abs" in feature_details
            ):  # Assuming 't' is key for Tx pulse
                print(
                    f"INFO: Transmitter Pulse ('{feature_details['name']}') is set to {feature_details['pixel_abs']}."
                )
                # Ensure time is 0 for Tx pulse if not already set
                updated_features[key]["time_us"] = 0.0
                continue

            while True:
                try:
                    user_input = input(prompt_text).strip()
                    if not user_input:  # User pressed Enter, keep default
                        print(f"Keeping current value for '{feature_details['name']}'.")
                        # Ensure time is calculated if pixel_abs exists
                        if (
                            "pixel_abs" in feature_details
                            and pixels_per_microsecond > 0
                        ):
                            updated_features[key]["time_us"] = (
                                feature_details["pixel_abs"] - transmitter_pulse_y_abs
                            ) / pixels_per_microsecond
                        break

                    pixel_abs_val = int(user_input)
                    updated_features[key]["pixel_abs"] = pixel_abs_val
                    if (
                        pixels_per_microsecond > 0
                    ):  # Avoid division by zero if not calibrated
                        updated_features[key]["time_us"] = (
                            pixel_abs_val - transmitter_pulse_y_abs
                        ) / pixels_per_microsecond
                    else:
                        updated_features[key]["time_us"] = float(
                            "nan"
                        )  # Indicate time cannot be calculated

                    print(
                        f"Set '{feature_details['name']}' to Y-pixel {pixel_abs_val} (Time: {updated_features[key]['time_us']:.1f} µs)."
                    )
                    break
                except ValueError:
                    print(
                        "Invalid input. Please enter a whole number for the pixel coordinate."
                    )
                except Exception as e:
                    print(f"An error occurred: {e}. Please try again.")
        print("--- End of Manual Feature Annotation ---\n")
    else:
        print("INFO: Skipping manual feature annotation.")
        # Ensure times are calculated for default features if not already present
        for key, feature_details in updated_features.items():
            if (
                "pixel_abs" in feature_details
                and "time_us" not in feature_details
                and pixels_per_microsecond > 0
            ):
                updated_features[key]["time_us"] = (
                    feature_details["pixel_abs"] - transmitter_pulse_y_abs
                ) / pixels_per_microsecond
            elif "pixel_abs" in feature_details and pixels_per_microsecond <= 0:
                updated_features[key]["time_us"] = float("nan")

    return updated_features, user_did_annotate
