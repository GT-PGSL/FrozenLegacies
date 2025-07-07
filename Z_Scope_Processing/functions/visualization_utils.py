import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import cv2
import pandas as pd
import re
from pathlib import Path
from scipy.signal import find_peaks
from scipy import ndimage
from .calibration_utils import convert_time_to_depth, convert_depth_to_time


class CBDTickSelector:
    """Enhanced interactive CBD tick mark selector with local image recognition refinement."""

    def __init__(
        self,
        image_full,
        expected_count=13,
        sprocket_removal_ratio=0.08,
        search_height_ratio=0.12,
    ):
        self.image_full = image_full
        self.expected_count = expected_count
        self.sprocket_removal_ratio = sprocket_removal_ratio
        self.search_height_ratio = search_height_ratio
        self.selected_points = []
        self.calculated_ticks = []
        self.refined_ticks = []
        self.fig = None
        self.ax = None

    def _refine_tick_positions_with_local_detection(
        self, approximate_positions, search_radius=25
    ):
        """
        Refine tick positions using local image recognition around approximate locations.

        Args:
            approximate_positions: List of approximate x-coordinates from uniform spacing
            search_radius: Pixel radius to search around each approximate position

        Returns:
            List of refined x-coordinates for actual tick marks
        """
        height, width = self.image_full.shape

        # Define the same search region used for manual selection
        sprocket_height = int(height * self.sprocket_removal_ratio)
        search_start = sprocket_height
        search_height = int(height * self.search_height_ratio)
        search_end = search_start + search_height

        # Extract the clean region for analysis
        clean_region = self.image_full[search_start:search_end, :]

        # Apply same preprocessing as in manual selection
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))
        enhanced_region = clahe.apply(clean_region)

        refined_positions = []

        for i, approx_x in enumerate(approximate_positions):
            # Skip the first two positions (manually selected)
            if i < 2:
                refined_positions.append(approx_x)
                continue

            # Define local search window
            x_start = max(0, approx_x - search_radius)
            x_end = min(width, approx_x + search_radius)

            # Extract local region
            local_region = enhanced_region[:, x_start:x_end]

            if local_region.shape[1] == 0:
                # Fallback to approximate position if region is invalid
                refined_positions.append(approx_x)
                continue

            # Method 1: Vertical line detection using morphology
            kernel_height = min(local_region.shape[0] // 2, 20)
            if kernel_height >= 3:
                vertical_kernel = cv2.getStructuringElement(
                    cv2.MORPH_RECT, (1, kernel_height)
                )
                morph_result = cv2.morphologyEx(
                    255 - local_region, cv2.MORPH_OPEN, vertical_kernel
                )
                vertical_profile = np.mean(morph_result, axis=0)
            else:
                vertical_profile = np.mean(255 - local_region, axis=0)

            # Method 2: Gradient-based edge detection
            if local_region.shape[1] > 3:
                sobel_x = cv2.Sobel(local_region, cv2.CV_64F, 1, 0, ksize=3)
                gradient_magnitude = np.abs(sobel_x)
                gradient_profile = np.mean(gradient_magnitude, axis=0)
            else:
                gradient_profile = np.zeros(local_region.shape[1])

            # Method 3: Template matching for vertical lines
            if local_region.shape[0] >= 5 and local_region.shape[1] >= 3:
                template_height = min(local_region.shape[0] // 2, 15)
                template = np.ones((template_height, 3), dtype=np.uint8) * 128
                template[:, 1] = 0  # Dark center line

                try:
                    template_result = cv2.matchTemplate(
                        local_region, template, cv2.TM_CCOEFF_NORMED
                    )
                    if template_result.size > 0:
                        template_profile = np.mean(template_result, axis=0)
                        # Pad to match other profiles
                        if len(template_profile) < len(vertical_profile):
                            pad_width = len(vertical_profile) - len(template_profile)
                            template_profile = np.pad(
                                template_profile, (0, pad_width), mode="constant"
                            )
                    else:
                        template_profile = np.zeros(len(vertical_profile))
                except:
                    template_profile = np.zeros(len(vertical_profile))
            else:
                template_profile = np.zeros(len(vertical_profile))

            # Normalize profiles
            def safe_normalize(profile):
                if len(profile) == 0 or np.max(profile) == 0:
                    return profile
                return profile / np.max(profile)

            vertical_norm = safe_normalize(vertical_profile)
            gradient_norm = safe_normalize(gradient_profile)
            template_norm = safe_normalize(template_profile)

            # Ensure all profiles have the same length
            min_length = min(len(vertical_norm), len(gradient_norm), len(template_norm))
            if min_length > 0:
                vertical_norm = vertical_norm[:min_length]
                gradient_norm = gradient_norm[:min_length]
                template_norm = template_norm[:min_length]

                # Combine profiles with weights
                combined_profile = (
                    0.4 * vertical_norm + 0.4 * gradient_norm + 0.2 * template_norm
                )

                # Find the best peak in the local region
                if len(combined_profile) > 0:
                    try:
                        peaks, _ = find_peaks(
                            combined_profile,
                            distance=max(3, len(combined_profile) // 10),
                            prominence=np.std(combined_profile) * 0.2,
                        )

                        if len(peaks) > 0:
                            # Choose the peak closest to the center (original approximate position)
                            center_idx = len(combined_profile) // 2
                            best_peak = peaks[np.argmin(np.abs(peaks - center_idx))]
                            refined_x = x_start + best_peak
                        else:
                            # No peaks found, use approximate position
                            refined_x = approx_x
                    except:
                        refined_x = approx_x
                else:
                    refined_x = approx_x
            else:
                refined_x = approx_x

            # Ensure refined position is within image bounds
            refined_x = max(0, min(width - 1, refined_x))
            refined_positions.append(int(refined_x))

        return refined_positions

    def start_selection(self, title="Select First Two CBD Tick Marks"):
        """Start interactive selection with proper TOP sprocket hole removal."""
        height, width = self.image_full.shape

        # REMOVE the top portion where sprocket holes are located
        sprocket_height = int(height * self.sprocket_removal_ratio)

        # START the search region BELOW the sprocket holes
        search_start = sprocket_height
        search_height = int(height * self.search_height_ratio)
        search_end = search_start + search_height

        # Extract the clean region BELOW the sprocket holes
        clean_region = self.image_full[search_start:search_end, :]

        # Create figure optimized for the focused view
        self.fig, self.ax = plt.subplots(figsize=(24, 8))

        # Enhanced preprocessing specifically for tick mark visibility
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))
        enhanced_region = clahe.apply(clean_region)

        # Apply strong unsharp masking to make tick marks very prominent
        gaussian_blur = cv2.GaussianBlur(enhanced_region, (0, 0), 3.0)
        unsharp_mask = cv2.addWeighted(enhanced_region, 2.0, gaussian_blur, -1.0, 0)

        # Final region for display
        final_region = np.clip(unsharp_mask, 0, 255).astype(np.uint8)

        # Display with correct coordinate mapping
        self.ax.imshow(
            final_region,
            cmap="gray",
            aspect="auto",
            extent=[0, width, search_end, search_start],
        )

        self.ax.set_title(
            f"{title}\n"
            "ENHANCED VIEW - Sprocket holes at TOP removed + Local Image Recognition\n"
            "CBD tick marks shown in clean region below sprocket holes\n"
            "Click on the leftmost CBD tick mark first, then the next one to the right",
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

        # Clear, focused instructions - positioned to avoid covering tick marks
        self.ax.text(
            0.98,
            0.98,
            "ENHANCED CBD TICK MARK SELECTION:\n"
            "• Sprocket holes at TOP completely removed\n"
            f"• Search starts at Y={search_start} (below sprocket holes)\n"
            "• Uses uniform spacing + local image recognition\n"
            "• Click on leftmost CBD tick mark\n"
            "• Click on next tick mark to the right\n"
            "• CBD tick marks are the dark vertical lines\n"
            "• Close window when done",
            transform=self.ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.95),
        )

        # Connect click event
        self.cid = self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        # Store parameters for coordinate adjustment
        self.search_start = search_start
        self.search_end = search_end
        self.sprocket_height = sprocket_height

        plt.tight_layout()
        plt.show()

        return self.selected_points

    def _on_click(self, event):
        """Handle mouse click events with proper coordinate adjustment."""
        if event.inaxes != self.ax:
            return

        if len(self.selected_points) < 2:
            x, y = event.xdata, event.ydata
            self.selected_points.append((int(x), int(y)))

            # Enhanced visual markers
            color = "red" if len(self.selected_points) == 1 else "blue"
            label = (
                "First CBD Tick"
                if len(self.selected_points) == 1
                else "Second CBD Tick"
            )

            # Large, highly visible markers
            self.ax.plot(
                x,
                y,
                "o",
                color=color,
                markersize=25,
                markeredgewidth=5,
                markeredgecolor="white",
                markerfacecolor=color,
                label=label,
                zorder=15,
            )

            # Prominent vertical line
            self.ax.axvline(
                x=x, color=color, linestyle="-", alpha=0.95, linewidth=5, zorder=14
            )

            # Enhanced annotation with arrow
            self.ax.annotate(
                f"{label}\nX: {int(x)}, Y: {int(y)}",
                xy=(x, y),
                xytext=(25, 25),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=1.0", fc=color, alpha=0.95),
                fontsize=16,
                color="white",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=color, lw=4),
                zorder=16,
            )

            if len(self.selected_points) == 2:
                self._calculate_all_ticks()
                self._show_calculated_ticks()

            self.ax.legend(fontsize=14, loc="upper right")
            self.fig.canvas.draw()

    def _calculate_all_ticks(self):
        """Calculate all tick positions with local refinement."""
        if len(self.selected_points) != 2:
            return

        x1, y1 = self.selected_points[0]
        x2, y2 = self.selected_points[1]

        # Calculate spacing
        spacing = abs(x2 - x1)
        start_x = min(x1, x2)

        # Generate approximate positions using uniform spacing
        approximate_ticks = []
        for i in range(self.expected_count):
            tick_x = start_x + i * spacing
            if 0 <= tick_x < self.image_full.shape[1]:
                approximate_ticks.append(int(tick_x))

        # Refine positions using local image recognition
        self.calculated_ticks = approximate_ticks
        self.refined_ticks = self._refine_tick_positions_with_local_detection(
            approximate_ticks
        )

        print(
            f"Calculated {len(self.calculated_ticks)} approximate tick positions with spacing of {spacing} pixels"
        )
        print(f"Approximate positions: {self.calculated_ticks}")
        print(f"Refined positions: {self.refined_ticks}")

        # Calculate refinement statistics
        if len(self.calculated_ticks) == len(self.refined_ticks):
            adjustments = [
                abs(refined - approx)
                for refined, approx in zip(self.refined_ticks, self.calculated_ticks)
            ]
            print(f"Average adjustment: {np.mean(adjustments):.1f} pixels")
            print(f"Max adjustment: {np.max(adjustments):.1f} pixels")

    def _show_calculated_ticks(self):
        """Show both approximate and refined tick positions on the plot."""
        for i, (approx_x, refined_x) in enumerate(
            zip(self.calculated_ticks, self.refined_ticks)
        ):
            if i < 2:  # Skip the manually selected ones
                continue

            # Show approximate position (gray, dashed)
            if 0 <= approx_x < self.image_full.shape[1]:
                self.ax.axvline(
                    x=approx_x,
                    color="gray",
                    linestyle="--",
                    alpha=0.5,
                    linewidth=2,
                    zorder=11,
                )

            # Show refined position (green, solid)
            if 0 <= refined_x < self.image_full.shape[1]:
                self.ax.axvline(
                    x=refined_x,
                    color="green",
                    linestyle="-",
                    alpha=0.9,
                    linewidth=3,
                    zorder=13,
                )

                # Add tick labels for every other tick
                if i % 2 == 0:
                    self.ax.text(
                        refined_x,
                        self.search_start
                        + (self.search_end - self.search_start) * 0.15,
                        f"T{i + 1}",
                        ha="center",
                        va="bottom",
                        fontsize=14,
                        color="green",
                        fontweight="bold",
                        zorder=17,
                        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.95),
                    )

        # Update title with comprehensive results
        spacing = abs(self.selected_points[1][0] - self.selected_points[0][0])
        avg_adjustment = np.mean(
            [abs(r - a) for r, a in zip(self.refined_ticks, self.calculated_ticks)]
        )

        self.ax.set_title(
            f"CBD Tick Mark Selection Complete (ENHANCED)\n"
            f"{len(self.refined_ticks)} ticks: {spacing}px spacing + local refinement\n"
            f"Gray=Approximate, Green=Refined (avg adjustment: {avg_adjustment:.1f}px)\n"
            f"Sprocket holes removed (Y=0 to Y={self.sprocket_height})",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        self.fig.canvas.draw()

    def get_tick_positions(self):
        """Return the refined tick positions."""
        return self.refined_ticks if self.refined_ticks else self.calculated_ticks


def manual_cbd_tick_selection(
    image_full,
    expected_count=13,
    sprocket_removal_ratio=0.08,
    search_height_ratio=0.12,
    debug=False,
):
    """
    Enhanced manual CBD tick mark selection with local image recognition refinement.

    Args:
        image_full: Full radar image array
        expected_count: Expected number of CBD tick marks (default: 13)
        sprocket_removal_ratio: Ratio of top region to remove (sprocket holes) (default: 0.08)
        search_height_ratio: Height ratio for CBD search region below sprockets (default: 0.12)
        debug: Enable debug output

    Returns:
        List of x-coordinates for all CBD tick marks (refined positions)
    """
    print("Starting ENHANCED manual CBD tick mark selection...")
    print(
        "This version uses uniform spacing as a guide + local image recognition for refinement."
    )
    print("Please select the first two CBD tick marks in the clean interface.")

    selector = CBDTickSelector(
        image_full, expected_count, sprocket_removal_ratio, search_height_ratio
    )
    selected_points = selector.start_selection()

    if len(selected_points) < 2:
        print("Warning: Less than 2 points selected. Using fallback method.")
        return []

    tick_positions = selector.get_tick_positions()

    if debug:
        print(f"Enhanced manual selection complete:")
        print(f"  Selected points: {selected_points}")
        print(
            f"  Calculated spacing: {abs(selected_points[1][0] - selected_points[0][0])} pixels"
        )
        print(
            f"  Generated {len(tick_positions)} refined tick positions: {tick_positions}"
        )
        if hasattr(selector, "refined_ticks") and selector.refined_ticks:
            adjustments = [
                abs(r - a)
                for r, a in zip(selector.refined_ticks, selector.calculated_ticks)
            ]
            print(f"  Average refinement adjustment: {np.mean(adjustments):.1f} pixels")

    return tick_positions


def validate_manual_selection_enhanced(
    image_full, tick_positions, base_filename, output_dir
):
    """Create enhanced validation plot with complete sprocket hole removal and focused view."""
    if not tick_positions:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 10))

    # Show clean, focused region with detected ticks
    height, width = image_full.shape
    search_height = max(30, int(height * 0.12))  # Search region height
    sprocket_height = int(height * 0.08)  # Sprocket removal height

    # Extract clean region
    clean_region = image_full[sprocket_height : sprocket_height + search_height, :]

    # Apply same enhancement as in selection interface
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))
    enhanced_region = clahe.apply(clean_region)

    gaussian_blur = cv2.GaussianBlur(enhanced_region, (0, 0), 3.0)
    final_region = cv2.addWeighted(enhanced_region, 2.0, gaussian_blur, -1.0, 0)
    final_region = np.clip(final_region, 0, 255).astype(np.uint8)

    ax1.imshow(
        final_region,
        cmap="gray",
        aspect="auto",
        extent=[0, width, sprocket_height + search_height, sprocket_height],
    )

    # Plot tick marks with enhanced visibility
    for i, tick_x in enumerate(tick_positions):
        if 0 <= tick_x < width:
            ax1.axvline(x=tick_x, color="red", linewidth=4, alpha=0.9, zorder=10)
            if i % 2 == 0:  # Label every other tick
                ax1.text(
                    tick_x,
                    sprocket_height + search_height * 0.2,
                    f"CBD{i + 1}",
                    ha="center",
                    va="bottom",
                    fontsize=14,
                    color="red",
                    fontweight="bold",
                    zorder=11,
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.95),
                )

    ax1.set_title(
        f"ENHANCED Manual CBD Tick Selection Results: {base_filename}\n"
        f"{len(tick_positions)} ticks detected with local image recognition refinement\n"
        "Sprocket holes completely removed - Focused view with refined positions",
        fontsize=16,
    )
    ax1.set_xlabel("X Position (pixels)", fontsize=14)
    ax1.set_ylabel("Y Position (pixels)", fontsize=14)
    ax1.grid(True, alpha=0.4, color="cyan")

    # Enhanced spacing analysis
    if len(tick_positions) > 1:
        spacings = np.diff(tick_positions)
        bars = ax2.bar(
            range(len(spacings)),
            spacings,
            alpha=0.8,
            color="steelblue",
            edgecolor="navy",
            linewidth=2,
        )

        # Add value labels on bars
        for i, (bar, spacing) in enumerate(zip(bars, spacings)):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{spacing:.0f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax2.axhline(
            y=np.mean(spacings),
            color="red",
            linestyle="--",
            linewidth=3,
            label=f"Mean Spacing: {np.mean(spacings):.1f} pixels",
        )
        ax2.set_title(
            f"Spacing Between Adjacent CBD Ticks (Enhanced with Local Refinement)\n"
            f"Standard Deviation: {np.std(spacings):.1f} pixels (Lower is better)",
            fontsize=16,
        )
        ax2.set_xlabel("Tick Pair Index", fontsize=14)
        ax2.set_ylabel("Spacing (pixels)", fontsize=14)
        ax2.legend(fontsize=14)
        ax2.grid(True, alpha=0.4)

        # Enhanced statistics
        uniformity = (
            (1 - np.std(spacings) / np.mean(spacings)) * 100
            if np.mean(spacings) > 0
            else 0
        )
        ax2.text(
            0.02,
            0.98,
            f"ENHANCED SPACING STATISTICS:\n"
            f"Min: {np.min(spacings):.1f} px\n"
            f"Max: {np.max(spacings):.1f} px\n"
            f"Mean: {np.mean(spacings):.1f} px\n"
            f"Std Dev: {np.std(spacings):.1f} px\n"
            f"Uniformity: {uniformity:.1f}%\n"
            f"Range: {np.max(spacings) - np.min(spacings):.1f} px\n"
            f"Method: Local Image Recognition",
            transform=ax2.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.95),
        )
    else:
        ax2.text(
            0.5,
            0.5,
            "Not enough ticks for spacing analysis",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=18,
        )
        ax2.set_title("Spacing Analysis", fontsize=16)

    plt.tight_layout()
    plt.savefig(
        output_dir / f"{base_filename}_enhanced_local_refinement_validation.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def align_cbd_labels_with_manual_selection(
    ax, cbd_tick_xs, cbd_list, nav_df, base_filename
):
    """Enhanced alignment function for manual selection results with local refinement."""

    if len(cbd_tick_xs) != len(cbd_list):
        print(f"Info: Adjusting CBD list to match {len(cbd_tick_xs)} detected ticks")

        if len(cbd_list) > len(cbd_tick_xs):
            # Take the first N CBDs
            cbd_list = cbd_list[: len(cbd_tick_xs)]
        elif len(cbd_list) < len(cbd_tick_xs):
            # Extend CBD list if needed (though this should be rare)
            last_cbd = cbd_list[-1]
            direction = -1 if len(cbd_list) > 1 and cbd_list[1] < cbd_list[0] else 1

            while len(cbd_list) < len(cbd_tick_xs):
                last_cbd += direction
                cbd_list.append(last_cbd)

    # Get navigation data for each CBD
    cbd_data = []
    for cbd in cbd_list:
        row = nav_df[nav_df["CBD"] == cbd]
        if not row.empty:
            cbd_data.append(
                {
                    "cbd": cbd,
                    "lat_stanford": row["LAT (stanford)"].values[0],
                    "lon_stanford": row["LON (stanford)"].values[0],
                    "lat_bingham": row["LAT (bingham)"].values[0],
                    "lon_bingham": row["LON (bingham)"].values[0],
                }
            )
        else:
            print(f"Warning: CBD {cbd} not found in navigation data")
            cbd_data.append(
                {
                    "cbd": cbd,
                    "lat_stanford": np.nan,
                    "lon_stanford": np.nan,
                    "lat_bingham": np.nan,
                    "lon_bingham": np.nan,
                }
            )

    # Create formatted labels
    labels = []
    for data in cbd_data:
        if np.isnan(data["lat_stanford"]):
            labels.append(f"{data['cbd']}\nN/A\nN/A")
        else:
            labels.append(
                f"{data['cbd']}\n"
                f"{data['lat_stanford']:.3f},{data['lon_stanford']:.3f}\n"
                f"{data['lat_bingham']:.3f},{data['lon_bingham']:.3f}"
            )

    # Set ticks and labels with precise alignment
    ax.set_xticks(cbd_tick_xs)
    ax.set_xticklabels(labels, rotation=0, fontsize=8, ha="center")

    # Add subtle vertical lines for visual confirmation (blue for refined positions)
    for x_pos in cbd_tick_xs:
        ax.axvline(x=x_pos, color="blue", linestyle=":", alpha=0.4, linewidth=1)

    return cbd_tick_xs, labels


def create_time_calibrated_zscope(
    image_full,
    base_filename,
    best_pip,
    transmitter_pulse_y_abs,
    data_top_abs,
    data_bottom_abs,
    pixels_per_microsecond,
    time_vis_params=None,
    physics_constants=None,
    output_params=None,
    approx_x_click=None,
    visualization_params=None,
    surface_y_abs=None,
    bed_y_abs=None,
    nav_df=None,
    nav_path=None,
    main_output_dir=None,
    processor_ref=None,
):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
        }
    )

    if time_vis_params is None:
        time_vis_params = {}
    if output_params is None:
        output_params = {}

    output_dir_name = output_params.get("debug_output_directory", "debug_output")
    output_dir = Path(output_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_dpi = output_params.get("figure_save_dpi", 600)
    legend_loc = time_vis_params.get("legend_location", "upper right")

    img_height = data_bottom_abs - data_top_abs
    img_width = image_full.shape[1]

    # Margin above transmitter pulse for context
    margin_above_tx_us = 10
    margin_above_tx_px = int(margin_above_tx_us * pixels_per_microsecond)
    cropped_top = max(0, transmitter_pulse_y_abs - margin_above_tx_px)

    # Dynamic bottom based on bed echo
    if bed_y_abs is not None and np.any(np.isfinite(bed_y_abs)):
        max_bed_y = np.nanpercentile(bed_y_abs, 95)
        margin_us = 20
        margin_px = int(margin_us * pixels_per_microsecond)
        cropped_bottom = min(data_bottom_abs, int(max_bed_y + margin_px))
    else:
        cropped_bottom = data_bottom_abs

    valid_data_crop = image_full[cropped_top:cropped_bottom, :]

    colormap = "Grays_r"
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_data = clahe.apply(
        cv2.normalize(valid_data_crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    )

    fig, ax = plt.subplots(figsize=(24, 10))

    # Store processor reference for CBD tick storage
    if processor_ref is not None:
        ax._processor_ref = processor_ref

    ax.imshow(
        enhanced_data,
        cmap=colormap,
        aspect="auto",
        extent=[0, img_width, cropped_bottom, cropped_top],
    )

    ax.set_ylim(cropped_bottom, cropped_top)
    ax.yaxis.set_ticks([])
    ax.set_ylabel("")

    # Calibration pip and transmitter pulse overlays
    if best_pip and "x_position" in best_pip:
        ax.axvline(
            x=best_pip["x_position"],
            color="#DCC462",
            linestyle=":",
            linewidth=2.5,
            alpha=0.9,
            label="Picked calpip",
            zorder=5,
        )

    ax.axhline(
        y=transmitter_pulse_y_abs,
        color="#0072B2",
        linestyle="-",
        linewidth=3,
        alpha=0.9,
        label="Transmitter Pulse",
        zorder=5,
    )

    ax.text(
        60,
        transmitter_pulse_y_abs,
        "0 µs (Tx Pulse)",
        color="#0072B2",
        fontsize=14,
        fontweight="bold",
        va="center",
        path_effects=[
            path_effects.withStroke(linewidth=3, foreground="white", alpha=0.7)
        ],
        zorder=6,
    )

    # Grid lines (time)
    total_time_range_us = (
        cropped_bottom - transmitter_pulse_y_abs
    ) / pixels_per_microsecond
    min_time_us = (cropped_top - transmitter_pulse_y_abs) / pixels_per_microsecond

    major_grid_interval_us = time_vis_params.get("major_grid_time_interval_us", 10)
    minor_grid_interval_us = time_vis_params.get("minor_grid_time_interval_us", 2)

    t_us_vals = np.arange(
        min_time_us,
        total_time_range_us + minor_grid_interval_us,
        minor_grid_interval_us,
    )

    for t_us in t_us_vals:
        pixel_y_abs_coord = transmitter_pulse_y_abs + t_us * pixels_per_microsecond
        if cropped_top <= pixel_y_abs_coord <= cropped_bottom:
            is_major_grid = round(t_us, 6) % major_grid_interval_us == 0

            ax.axhline(
                y=pixel_y_abs_coord,
                color="white",
                linestyle="-" if is_major_grid else "--",
                alpha=0.8 if is_major_grid else 0.2,
                linewidth=2 if is_major_grid else 1.5,
                zorder=2,
            )

            if is_major_grid:
                ax.text(
                    60,
                    pixel_y_abs_coord,
                    f"{int(round(t_us))} µs",
                    color="white",
                    fontsize=12,
                    alpha=0.9,
                    va="center",
                    path_effects=[
                        path_effects.withStroke(
                            linewidth=3, foreground="black", alpha=0.7
                        )
                    ],
                    zorder=3,
                )

    # --- MANUAL CBD TICK MARK SELECTION WITH LOCAL REFINEMENT ---
    detection_method = output_params.get("cbd_detection_method", "manual")

    if detection_method == "manual":
        print("Using manual CBD tick mark selection with local image recognition...")
        cbd_tick_xs = manual_cbd_tick_selection(
            image_full,
            expected_count=13,
            sprocket_removal_ratio=output_params.get("sprocket_removal_ratio", 0.08),
            search_height_ratio=output_params.get("search_height_ratio", 0.12),
            debug=output_params.get("debug_tick_detection", False),
        )
    else:
        # Fallback to automated methods if needed
        print("Manual selection not specified, using automated detection...")
        cbd_tick_xs = []

    # Load navigation data
    if nav_df is None and nav_path is not None:
        nav_df = pd.read_csv(nav_path)

    if nav_df is None:
        flight_match = re.search(r"F(\d+)", base_filename)
        if flight_match:
            flight_num = flight_match.group(1)
            nav_file_guess = f"merged_{flight_num}_nav.csv"
            if Path(nav_file_guess).exists():
                nav_df = pd.read_csv(nav_file_guess)

    if nav_df is not None and len(cbd_tick_xs) > 0:
        cbd_match = re.search(r"C(\d+)_(\d+)", base_filename)
        if cbd_match:
            cbd_start = int(cbd_match.group(1))
            cbd_end = int(cbd_match.group(2))

            # Create CBD list in descending order (largest to smallest, left to right)
            if cbd_start > cbd_end:
                cbd_list = list(range(cbd_start, cbd_end - 1, -1))
            else:
                cbd_list = list(range(cbd_start, cbd_end + 1))
                cbd_list.reverse()  # Reverse to get largest first

            # Use enhanced alignment with local refinement
            cbd_tick_xs, cbd_labels = align_cbd_labels_with_manual_selection(
                ax, cbd_tick_xs, cbd_list, nav_df, base_filename
            )

            ax.set_xlabel(
                "CBD\nLat, Lon (Stanford)\nLat, Lon (Bingham)",
                fontsize=14,
                fontweight="bold",
            )

            # Create enhanced validation plot
            if output_params.get("create_validation_plot", True):
                validate_manual_selection_enhanced(
                    image_full, cbd_tick_xs, base_filename, output_dir
                )

    # Surface, Bed, and Ice Thickness Labels (existing code remains the same)
    if surface_y_abs is not None and np.any(np.isfinite(surface_y_abs)):
        x_coords = np.arange(len(surface_y_abs))
        valid = np.isfinite(surface_y_abs)
        ax.plot(
            x_coords[valid],
            surface_y_abs[valid],
            color="#009E73",
            linestyle="--",
            linewidth=3,
            label="Surface",
            zorder=6,
        )

    if bed_y_abs is not None and np.any(np.isfinite(bed_y_abs)):
        x_coords = np.arange(len(bed_y_abs))
        valid = np.isfinite(bed_y_abs)
        ax.plot(
            x_coords[valid],
            bed_y_abs[valid],
            color="#D55E00",
            linestyle="--",
            linewidth=3,
            label="Bed",
            zorder=6,
        )

    # Ice thickness calculations
    if (
        surface_y_abs is not None
        and bed_y_abs is not None
        and np.any(np.isfinite(surface_y_abs))
        and np.any(np.isfinite(bed_y_abs))
        and physics_constants is not None
    ):
        valid_surface = surface_y_abs[np.isfinite(surface_y_abs)]
        valid_bed = bed_y_abs[np.isfinite(bed_y_abs)]

        if len(valid_surface) > 0 and len(valid_bed) > 0:
            avg_surface_y = np.mean(valid_surface)
            avg_bed_y = np.mean(valid_bed)

            c0 = physics_constants.get("speed_of_light_vacuum_mps")
            epsilon_r_ice = physics_constants.get("ice_relative_permittivity_real")
            firn_corr_m = physics_constants.get("firn_correction_meters")

            def abs_y_to_depth(y_abs):
                y_rel = y_abs - transmitter_pulse_y_abs
                time_us = y_rel / pixels_per_microsecond
                one_way_time_us = time_us / 2.0
                return convert_time_to_depth(
                    one_way_time_us, c0, epsilon_r_ice, firn_corr_m
                )

            avg_surface_depth = abs_y_to_depth(avg_surface_y)
            avg_bed_depth = abs_y_to_depth(avg_bed_y)
            ice_thickness = avg_bed_depth - avg_surface_depth

            x_pos = 0.80 * img_width

            ax.annotate(
                f"Surface: {avg_surface_depth:.1f} m",
                xy=(x_pos, avg_surface_y),
                xytext=(10, -5),
                textcoords="offset points",
                color="#009E73",
                fontsize=14,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3", fc="white", ec="#009E73", alpha=0.8
                ),
                zorder=10,
            )

            ax.annotate(
                f"Bed: {avg_bed_depth:.1f} m",
                xy=(x_pos, avg_bed_y),
                xytext=(10, 5),
                textcoords="offset points",
                color="#D55E00",
                fontsize=14,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3", fc="white", ec="#D55E00", alpha=0.8
                ),
                zorder=10,
            )

            ax.annotate(
                f"Ice thickness: {ice_thickness:.1f} m",
                xy=(x_pos, (avg_surface_y + avg_bed_y) / 2),
                xytext=(10, 0),
                textcoords="offset points",
                color="black",
                fontsize=14,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
                zorder=10,
            )

    ax.legend(
        loc=legend_loc,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        fontsize=14,
        fancybox=True,
        borderpad=1.2,
    )

    # Depth and time axes (existing code remains the same)
    def abs_y_to_depth(y_abs_coord):
        y_rel = y_abs_coord - transmitter_pulse_y_abs
        time_us = y_rel / pixels_per_microsecond
        one_way_time_us = time_us / 2.0
        c0 = physics_constants.get("speed_of_light_vacuum_mps")
        epsilon_r_ice = physics_constants.get("ice_relative_permittivity_real")
        firn_corr_m = physics_constants.get("firn_correction_meters")
        return convert_time_to_depth(one_way_time_us, c0, epsilon_r_ice, firn_corr_m)

    def depth_to_abs_y(depth_m):
        c0 = physics_constants.get("speed_of_light_vacuum_mps")
        epsilon_r_ice = physics_constants.get("ice_relative_permittivity_real")
        firn_corr_m = physics_constants.get("firn_correction_meters")
        one_way_time_us = convert_depth_to_time(depth_m, c0, epsilon_r_ice, firn_corr_m)
        two_way_time_us = one_way_time_us * 2.0
        y_rel = two_way_time_us * pixels_per_microsecond
        return transmitter_pulse_y_abs + y_rel

    depth_ax = ax.secondary_yaxis("left", functions=(abs_y_to_depth, depth_to_abs_y))
    depth_ax.set_ylabel("Depth Below Phase Center (m)", fontsize=16, fontweight="bold")
    depth_ax.tick_params(
        axis="y",
        which="both",
        colors="black",
        labelcolor="black",
        direction="out",
        length=8,
        width=2,
        labelsize=14,
    )
    depth_ax.yaxis.label.set_color("black")

    def abs_y_to_time_us(y_abs_coord):
        return (y_abs_coord - transmitter_pulse_y_abs) / pixels_per_microsecond

    def time_us_to_abs_y(t_us_val):
        return (t_us_val * pixels_per_microsecond) + transmitter_pulse_y_abs

    time_ax = ax.secondary_yaxis(
        "right", functions=(abs_y_to_time_us, time_us_to_abs_y)
    )
    time_ax.set_ylabel("Two-way Travel Time (µs)", fontsize=16, fontweight="bold")
    time_ax.set_ylim(cropped_bottom, cropped_top)
    time_ax.tick_params(
        axis="y",
        which="both",
        colors="black",
        labelcolor="black",
        direction="out",
        length=8,
        width=2,
        labelsize=14,
    )
    time_ax.yaxis.label.set_color("black")

    plt.subplots_adjust(left=0.12, right=0.88, bottom=0.10, top=0.95)

    ax.set_title(
        f"Picked Z-scope: {base_filename}",
        fontsize=20,
        fontweight="bold",
    )

    if main_output_dir is not None:
        main_plot_path = Path(main_output_dir) / f"{base_filename}_picked.png"
    else:
        main_plot_path = output_dir / f"{base_filename}_picked.png"

    fig.savefig(main_plot_path, dpi=save_dpi)
    plt.close(fig)

    # Store CBD tick positions for export
    if "cbd_tick_xs" in locals() and cbd_tick_xs and hasattr(ax, "_processor_ref"):
        ax._processor_ref.calculated_ticks = cbd_tick_xs

    return fig, ax, time_ax
