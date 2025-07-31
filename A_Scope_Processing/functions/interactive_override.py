import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class ManualPickOverride:
    """Interactive manual override for automatic peak detection results."""

    def __init__(
        self,
        frame_img,
        signal_x_clean,
        signal_y_clean,
        power_vals,
        time_vals,
        tx_idx,
        surface_idx,
        bed_idx,
        base_filename,
        frame_idx,
        config,
    ):
        self.frame_img = frame_img
        self.signal_x_clean = signal_x_clean
        self.signal_y_clean = signal_y_clean
        self.power_vals = power_vals
        self.time_vals = time_vals

        # Original automatic picks
        self.auto_tx_idx = tx_idx
        self.auto_surface_idx = surface_idx
        self.auto_bed_idx = bed_idx

        # Current picks (will be updated by manual selection)
        self.manual_tx_idx = tx_idx
        self.manual_surface_idx = surface_idx
        self.manual_bed_idx = bed_idx

        self.base_filename = base_filename
        self.frame_idx = frame_idx
        self.config = config

        # Track what has been manually overridden
        self.overrides = {"transmitter": False, "surface": False, "bed": False}

        self.fig = None
        self.ax_debug = None
        self.ax_calib = None
        self.selection_mode = None  # 't', 's', or 'b'

    def start_interactive_session(self):
        """Start the interactive manual override session."""
        print("\n" + "=" * 60)
        print("INTERACTIVE MANUAL PEAK OVERRIDE")
        print("=" * 60)
        print(f"Frame: {self.frame_idx} ({self.base_filename})")
        print("\nAutomatic detection results:")
        if self.auto_tx_idx is not None and self.auto_tx_idx < len(self.time_vals):
            print(f"  Transmitter: {self.time_vals[self.auto_tx_idx]:.2f} μs")
        if self.auto_surface_idx is not None and self.auto_surface_idx < len(
            self.time_vals
        ):
            print(f"  Surface: {self.time_vals[self.auto_surface_idx]:.2f} μs")
        if self.auto_bed_idx is not None and self.auto_bed_idx < len(self.time_vals):
            print(f"  Bed: {self.time_vals[self.auto_bed_idx]:.2f} μs")

        print("\nInstructions:")
        print("  Press 't' to manually redefine transmitter pulse")
        print("  Press 's' to manually redefine surface echo")
        print("  Press 'b' to manually redefine bed echo")
        print("  Press 'r' to reset to automatic picks")
        print("  Press 'Enter' to finish and save results")
        print("  Click on the calibrated plot (right) to select peaks")

        self._create_interactive_plot()
        return (
            self.manual_tx_idx,
            self.manual_surface_idx,
            self.manual_bed_idx,
            self.overrides,
        )

    def _create_interactive_plot(self):
        """Create the interactive plot interface."""
        self.fig, (self.ax_debug, self.ax_calib) = plt.subplots(1, 2, figsize=(20, 8))

        # Left plot: Debug view (same as before)
        self._plot_debug_view()

        # Right plot: Calibrated view for interaction
        self._plot_calibrated_view()

        # Connect event handlers
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        plt.tight_layout()
        plt.show()

    def _plot_debug_view(self):
        """Plot the debug view (left subplot)."""
        h, w = self.frame_img.shape
        self.ax_debug.imshow(self.frame_img, cmap="gray", aspect="auto")

        # Plot signal trace
        if self.signal_x_clean is not None and len(self.signal_x_clean) > 0:
            self.ax_debug.plot(
                self.signal_x_clean,
                self.signal_y_clean,
                "r-",
                linewidth=1,
                label="Signal Trace",
            )

        # Mark current picks
        self._update_debug_markers()

        self.ax_debug.set_title(f"Frame {self.frame_idx} - Debug View")
        self.ax_debug.set_ylim(h, 0)
        self.ax_debug.set_xlim(0, w)
        self.ax_debug.legend(fontsize=10)

    def _plot_calibrated_view(self):
        """Plot the calibrated view (right subplot) for interaction."""
        if self.time_vals is not None and self.power_vals is not None:
            self.ax_calib.clear()
            self.ax_calib.plot(
                self.time_vals, self.power_vals, "r-", linewidth=1.5, label="Signal"
            )

            # Add grid
            self._add_calibrated_grid()

            # Mark current picks
            self._update_calibrated_markers()

            self.ax_calib.set_xlabel("One-way travel time (μs)", fontsize=12)
            self.ax_calib.set_ylabel("Power (dB)", fontsize=12)
            self.ax_calib.set_title(
                "Calibrated A-scope - Click to Override Picks", fontsize=12
            )
            self.ax_calib.grid(True, alpha=0.3)
            self.ax_calib.legend(fontsize=10)

            # Set reasonable limits
            if len(self.time_vals) > 0:
                self.ax_calib.set_xlim(-1, max(15, np.max(self.time_vals) * 0.8))
            self.ax_calib.set_ylim(-65, 5)

        # Add status text
        self._update_status_text()

    def _add_calibrated_grid(self):
        """Add grid lines to calibrated plot."""
        # Time grid (vertical lines)
        for t in range(0, 20, 3):  # Every 3 μs
            self.ax_calib.axvline(t, color="lightblue", linestyle="-", alpha=0.5)

        # Power grid (horizontal lines)
        for p in range(-60, 10, 10):  # Every 10 dB
            self.ax_calib.axhline(p, color="lightblue", linestyle="-", alpha=0.5)

    def _update_debug_markers(self):
        """Update markers on debug view."""

        children_to_remove = []
        for child in self.ax_debug.get_children():
            try:
                if hasattr(child, "get_label") and callable(child.get_label):
                    label = child.get_label()
                    label_str = str(label) if label is not None else ""
                    if label_str in [
                        "TX (Auto)",
                        "TX (Manual)",
                        "Surface (Auto)",
                        "Surface (Manual)",
                        "Bed (Auto)",
                        "Bed (Manual)",
                    ]:
                        children_to_remove.append(child)
            except (TypeError, AttributeError, ValueError):
                continue

        # Remove identified markers
        for child in children_to_remove:
            try:
                child.remove()
            except:
                pass

        # Rest of your existing marker plotting code remains the same...
        # Plot current markers
        if self.manual_tx_idx is not None and self.manual_tx_idx < len(
            self.signal_x_clean
        ):
            color = "blue" if not self.overrides["transmitter"] else "cyan"
            label = "TX (Auto)" if not self.overrides["transmitter"] else "TX (Manual)"
            self.ax_debug.plot(
                self.signal_x_clean[self.manual_tx_idx],
                self.signal_y_clean[self.manual_tx_idx],
                "o",
                color=color,
                markersize=8,
                label=label,
            )

        if self.manual_surface_idx is not None and self.manual_surface_idx < len(
            self.signal_x_clean
        ):
            color = "green" if not self.overrides["surface"] else "lime"
            label = (
                "Surface (Auto)"
                if not self.overrides["surface"]
                else "Surface (Manual)"
            )
            self.ax_debug.plot(
                self.signal_x_clean[self.manual_surface_idx],
                self.signal_y_clean[self.manual_surface_idx],
                "o",
                color=color,
                markersize=8,
                label=label,
            )

        if self.manual_bed_idx is not None and self.manual_bed_idx < len(
            self.signal_x_clean
        ):
            color = "red" if not self.overrides["bed"] else "orange"
            label = "Bed (Auto)" if not self.overrides["bed"] else "Bed (Manual)"
            self.ax_debug.plot(
                self.signal_x_clean[self.manual_bed_idx],
                self.signal_y_clean[self.manual_bed_idx],
                "o",
                color=color,
                markersize=8,
                label=label,
            )

        self.ax_debug.legend(fontsize=8)

    def _update_calibrated_markers(self):
        """Update markers on calibrated view."""

        children_to_remove = []
        for child in self.ax_calib.get_children():
            try:
                if hasattr(child, "get_label") and callable(child.get_label):
                    label = child.get_label()
                    label_str = str(label) if label is not None else ""
                    if (
                        "TX" in label_str
                        or "Surface" in label_str
                        or "Bed" in label_str
                    ):
                        children_to_remove.append(child)
            except (TypeError, AttributeError, ValueError):
                continue

        # Remove identified markers
        for child in children_to_remove:
            try:
                child.remove()
            except:
                pass

        # Rest of your existing marker plotting code remains the same...
        # Plot current markers
        if self.manual_tx_idx is not None and self.manual_tx_idx < len(self.time_vals):
            color = "blue" if not self.overrides["transmitter"] else "cyan"
            label = "TX (Auto)" if not self.overrides["transmitter"] else "TX (Manual)"
            self.ax_calib.plot(
                self.time_vals[self.manual_tx_idx],
                self.power_vals[self.manual_tx_idx],
                "o",
                color=color,
                markersize=10,
                label=label,
            )

        if self.manual_surface_idx is not None and self.manual_surface_idx < len(
            self.time_vals
        ):
            color = "green" if not self.overrides["surface"] else "lime"
            label = (
                "Surface (Auto)"
                if not self.overrides["surface"]
                else "Surface (Manual)"
            )
            self.ax_calib.plot(
                self.time_vals[self.manual_surface_idx],
                self.power_vals[self.manual_surface_idx],
                "o",
                color=color,
                markersize=10,
                label=label,
            )

        if self.manual_bed_idx is not None and self.manual_bed_idx < len(
            self.time_vals
        ):
            color = "red" if not self.overrides["bed"] else "orange"
            label = "Bed (Auto)" if not self.overrides["bed"] else "Bed (Manual)"
            self.ax_calib.plot(
                self.time_vals[self.manual_bed_idx],
                self.power_vals[self.manual_bed_idx],
                "o",
                color=color,
                markersize=10,
                label=label,
            )

    def _update_status_text(self):
        """Update status text on the plot with error handling."""
        try:
            status_lines = []
            if self.selection_mode:
                mode_text = str(self.selection_mode).upper()  # Ensure string conversion
                status_lines.append(
                    f"MODE: Select {mode_text} peak (click on right plot)"
                )
            else:
                status_lines.append("Press 't', 's', or 'b' to select peak type")

            # Show current overrides
            overrides_text = []
            for peak_type, is_manual in self.overrides.items():
                if is_manual:
                    overrides_text.append(
                        str(peak_type).capitalize()
                    )  # Ensure string conversion

            if overrides_text:
                status_lines.append(f"Manual overrides: {', '.join(overrides_text)}")
            else:
                status_lines.append("No manual overrides")

            status_text = "\n".join(status_lines)

            children_to_remove = []
            for child in self.ax_calib.get_children():
                try:
                    # Check if this is a text object and if it contains our status text
                    if hasattr(child, "get_text") and callable(child.get_text):
                        text_content = child.get_text()
                        # Convert to string safely
                        text_str = str(text_content) if text_content is not None else ""
                        # Check for our status text markers
                        if (
                            "MODE:" in text_str
                            or "Manual overrides:" in text_str
                            or "Press 't'" in text_str
                            or "No manual overrides" in text_str
                        ):
                            children_to_remove.append(child)
                except (TypeError, AttributeError, ValueError):
                    # Skip any problematic objects
                    continue

            # Remove the identified text objects
            for child in children_to_remove:
                try:
                    child.remove()
                except:
                    pass  # Ignore removal errors

            # Add new status text box
            self.ax_calib.text(
                0.02,
                0.98,
                status_text,
                transform=self.ax_calib.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                fontsize=10,
            )

        except Exception as e:
            print(f"ERROR updating status text: {e}")
            # Continue without status text if there's an error
            pass

    def _on_key_press(self, event):
        """Handle key press events with proper error handling."""
        # Defensive programming: ensure event.key is a string
        if not hasattr(event, "key") or event.key is None:
            return

        key = event.key
        if not isinstance(key, str):
            print(f"WARNING: Unexpected key type: {type(key)}")
            return

        try:
            if key == "t":
                self.selection_mode = "transmitter"
                print("Select transmitter pulse peak (click on calibrated plot)")
            elif key == "s":
                self.selection_mode = "surface"
                print("Select surface echo peak (click on calibrated plot)")
            elif key == "b":
                self.selection_mode = "bed"
                print("Select bed echo peak (click on calibrated plot)")
            elif key == "r":
                self._reset_to_automatic()
            elif key == "enter":
                self._finish_session()
            else:
                return  # Ignore other keys

            # Update display after valid key press
            self._plot_calibrated_view()
            if hasattr(self.fig, "canvas"):
                self.fig.canvas.draw()

        except Exception as e:
            print(f"ERROR in key press handler: {e}")
            import traceback

            traceback.print_exc()

    def _on_click(self, event):
        """Handle mouse click events with proper validation."""
        try:
            # Validate event and axes
            if (
                not hasattr(event, "inaxes")
                or not hasattr(event, "xdata")
                or event.inaxes != self.ax_calib
                or self.selection_mode is None
            ):
                return

            # Validate click coordinates
            if event.xdata is None or not isinstance(event.xdata, (int, float)):
                print("WARNING: Invalid click coordinates")
                return

            # Validate time values array
            if (
                not hasattr(self, "time_vals")
                or self.time_vals is None
                or len(self.time_vals) == 0
            ):
                print("ERROR: No time values available for peak selection")
                return

            # Find closest point to click
            click_time = float(event.xdata)
            distances = np.abs(self.time_vals - click_time)
            closest_idx = np.argmin(distances)

            # Validate index
            if closest_idx < 0 or closest_idx >= len(self.time_vals):
                print("ERROR: Invalid peak index selected")
                return

            # Update the appropriate pick
            if self.selection_mode == "transmitter":
                self.manual_tx_idx = closest_idx
                self.overrides["transmitter"] = True
                print(
                    f"Transmitter pulse manually set to {self.time_vals[closest_idx]:.2f} μs"
                )
            elif self.selection_mode == "surface":
                self.manual_surface_idx = closest_idx
                self.overrides["surface"] = True
                print(
                    f"Surface echo manually set to {self.time_vals[closest_idx]:.2f} μs"
                )
            elif self.selection_mode == "bed":
                self.manual_bed_idx = closest_idx
                self.overrides["bed"] = True
                print(f"Bed echo manually set to {self.time_vals[closest_idx]:.2f} μs")

            # Clear selection mode and update plots
            self.selection_mode = None
            self._update_debug_markers()
            self._update_calibrated_markers()
            self._plot_calibrated_view()

            if hasattr(self.fig, "canvas"):
                self.fig.canvas.draw()

        except Exception as e:
            print(f"ERROR in click handler: {e}")
            import traceback

            traceback.print_exc()

    def _reset_to_automatic(self):
        """Reset all picks to automatic detection results."""
        self.manual_tx_idx = self.auto_tx_idx
        self.manual_surface_idx = self.auto_surface_idx
        self.manual_bed_idx = self.auto_bed_idx
        self.overrides = {"transmitter": False, "surface": False, "bed": False}
        self.selection_mode = None

        print("Reset to automatic detection results")
        self._update_debug_markers()
        self._update_calibrated_markers()
        self._plot_calibrated_view()
        self.fig.canvas.draw()

    def _finish_session(self):
        """Finish the interactive session."""
        plt.close(self.fig)

        print("\n" + "=" * 60)
        print("MANUAL OVERRIDE SESSION COMPLETED")
        print("=" * 60)
        print("Final results:")
        if self.manual_tx_idx is not None and self.manual_tx_idx < len(self.time_vals):
            status = "(Manual)" if self.overrides["transmitter"] else "(Auto)"
            print(
                f"  Transmitter: {self.time_vals[self.manual_tx_idx]:.2f} μs {status}"
            )
        if self.manual_surface_idx is not None and self.manual_surface_idx < len(
            self.time_vals
        ):
            status = "(Manual)" if self.overrides["surface"] else "(Auto)"
            print(
                f"  Surface: {self.time_vals[self.manual_surface_idx]:.2f} μs {status}"
            )
        if self.manual_bed_idx is not None and self.manual_bed_idx < len(
            self.time_vals
        ):
            status = "(Manual)" if self.overrides["bed"] else "(Auto)"
            print(f"  Bed: {self.time_vals[self.manual_bed_idx]:.2f} μs {status}")
        print("=" * 60)
