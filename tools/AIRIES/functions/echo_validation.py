import matplotlib.pyplot as plt
import numpy as np
import cv2


class EchoValidationInterface:
    """Interface for validating and iteratively refining automatic echo detection results."""

    def __init__(self, image, surface_trace, bed_trace, optimized_params):
        self.image = image
        self.surface_trace = surface_trace.copy()
        self.bed_trace = bed_trace.copy()
        self.optimized_params = optimized_params
        self.problem_regions = []
        self.refinement_iteration = 0
        self.max_iterations = 3

    def validate_results(self):
        """Show results and allow iterative refinement until user is satisfied."""

        while self.refinement_iteration < self.max_iterations:
            print(f"\n=== VALIDATION ITERATION {self.refinement_iteration + 1} ===")

            # Show current results
            user_satisfied = self._show_current_results()

            if user_satisfied:
                print("User satisfied with results - proceeding to CBD selection")
                break

            # Get problem regions for refinement
            problem_regions = self._get_problem_regions()

            if not problem_regions:
                print("No problem regions selected - proceeding with current results")
                break

            # Perform refinement
            self._refine_problem_regions(problem_regions)
            self.refinement_iteration += 1

        if self.refinement_iteration >= self.max_iterations:
            print(f"Maximum refinement iterations ({self.max_iterations}) reached")

        return self.problem_regions

    def _show_current_results(self):
        """Display current detection results and get user feedback."""

        fig, ax = plt.subplots(figsize=(24, 12))

        # Display image with detected traces
        enhanced = cv2.createCLAHE(clipLimit=3.0).apply(self.image)
        ax.imshow(enhanced, cmap="gray", aspect="auto")

        # Plot detected traces
        x_coords = np.arange(len(self.surface_trace))

        # Surface trace
        valid_surface = np.isfinite(self.surface_trace)
        if np.any(valid_surface):
            ax.plot(
                x_coords[valid_surface],
                self.surface_trace[valid_surface],
                "cyan",
                linewidth=2,
                label="Detected Surface",
                alpha=0.8,
            )

        # Bed trace
        valid_bed = np.isfinite(self.bed_trace)
        if np.any(valid_bed):
            ax.plot(
                x_coords[valid_bed],
                self.bed_trace[valid_bed],
                "orange",
                linewidth=2,
                label="Detected Bed",
                alpha=0.8,
            )

        # Add quality metrics
        surface_coverage = np.sum(valid_surface) / len(self.surface_trace) * 100
        bed_coverage = np.sum(valid_bed) / len(self.bed_trace) * 100

        ax.set_title(
            f"Echo Detection Results - Iteration {self.refinement_iteration + 1}\n"
            f"Surface Coverage: {surface_coverage:.1f}% | Bed Coverage: {bed_coverage:.1f}%\n"
            "Review the results and decide if refinement is needed",
            fontsize=16,
            fontweight="bold",
        )
        ax.legend()

        # Add instruction text
        ax.text(
            0.02,
            0.02,
            "RESULT REVIEW:\n"
            "• Cyan = Surface echoes\n"
            "• Orange = Bed echoes\n"
            "• Close window to continue",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.9),
        )

        plt.tight_layout()
        plt.show()

        # Get user satisfaction feedback
        while True:
            user_input = (
                input("\nAre you satisfied with these echo detection results? (y/n): ")
                .strip()
                .lower()
            )

            if user_input in ["y", "yes"]:
                return True
            elif user_input in ["n", "no"]:
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no")

    def _get_problem_regions(self):
        """Allow user to select problem regions for refinement."""

        fig, ax = plt.subplots(figsize=(24, 12))

        # Display image with current traces
        enhanced = cv2.createCLAHE(clipLimit=3.0).apply(self.image)
        ax.imshow(enhanced, cmap="gray", aspect="auto")

        # Plot current traces
        x_coords = np.arange(len(self.surface_trace))

        valid_surface = np.isfinite(self.surface_trace)
        if np.any(valid_surface):
            ax.plot(
                x_coords[valid_surface],
                self.surface_trace[valid_surface],
                "cyan",
                linewidth=2,
                label="Current Surface",
                alpha=0.7,
            )

        valid_bed = np.isfinite(self.bed_trace)
        if np.any(valid_bed):
            ax.plot(
                x_coords[valid_bed],
                self.bed_trace[valid_bed],
                "orange",
                linewidth=2,
                label="Current Bed",
                alpha=0.7,
            )

        ax.set_title(
            "Select Problem Regions for Refinement\n"
            "Click and drag to select regions that need improvement",
            fontsize=16,
            fontweight="bold",
        )
        ax.legend()

        # Add region selector for problem areas
        from matplotlib.widgets import RectangleSelector

        self.problem_regions = []

        def on_problem_region_select(eclick, erelease):
            x1, x2 = sorted([eclick.xdata, erelease.xdata])
            y1, y2 = sorted([eclick.ydata, erelease.ydata])

            if x1 is not None and x2 is not None:
                self.problem_regions.append(
                    {
                        "x_range": (int(x1), int(x2)),
                        "y_range": (int(y1), int(y2)),
                        "needs_retuning": True,
                    }
                )

                print(f"Marked problem region: X={int(x1)}-{int(x2)}")

                # Draw the selected region
                from matplotlib.patches import Rectangle

                rect = Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="red",
                    alpha=0.3,
                    label=f"Problem Region {len(self.problem_regions)}",
                )
                ax.add_patch(rect)
                ax.legend()
                fig.canvas.draw()

        problem_selector = RectangleSelector(
            ax,
            on_problem_region_select,
            useblit=True,
            button=[1],
            minspanx=10,
            minspany=10,
        )

        # Add instruction text
        ax.text(
            0.02,
            0.98,
            "PROBLEM REGION SELECTION:\n"
            "• Click and drag to select problematic areas\n"
            "• Red boxes show selected regions\n"
            "• Close window when done selecting",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.9),
        )

        plt.tight_layout()
        plt.show()

        return self.problem_regions

    def _refine_problem_regions(self, problem_regions):
        """
        Refine detection in selected problem regions with slope-aware adjustments.

        This enhanced version analyzes the slope characteristics of problem regions
        and applies adaptive parameter adjustments to improve bed echo detection
        in sloping areas.

        Args:
            problem_regions: List of problem region dictionaries with x_range and y_range
        """
        from functions.echo_tracing import detect_surface_echo, detect_bed_echo

        print(
            f"Refining {len(problem_regions)} problem regions with slope-aware parameters..."
        )
        print("Note: Using enhanced slope-aware parameter optimization")

        for i, region in enumerate(problem_regions):
            x_start, x_end = region["x_range"]
            print(f"Refining region {i + 1}: X={x_start}-{x_end}")

            # Extract problem region
            region_data = self.image[:, x_start:x_end]

            if region_data.shape[1] > 0:
                # Analyze slope characteristics in problem region
                region_surface = self.surface_trace[x_start:x_end]
                region_bed = self.bed_trace[x_start:x_end]

                # Calculate slope metrics for adaptive parameter adjustment
                slope_metrics = self._calculate_slope_metrics(
                    region_surface, region_bed
                )

                # Apply slope-aware parameter adjustments
                adjusted_surface_params, adjusted_bed_params = (
                    self._get_slope_adjusted_parameters(
                        slope_metrics, region_surface, region_bed
                    )
                )

                # Re-run detection with slope-optimized parameters
                try:
                    # Estimate transmitter pulse position for this region
                    tx_pulse_y_rel = self._estimate_tx_pulse_position(region_data)

                    # Re-detect surface echo with adjusted parameters
                    region_surface_refined = detect_surface_echo(
                        region_data, tx_pulse_y_rel, adjusted_surface_params
                    )

                    # Re-detect bed echo with adjusted parameters
                    region_bed_refined = detect_bed_echo(
                        region_data,
                        region_surface_refined,
                        region_data.shape[0] - 50,  # Z-boundary estimate
                        adjusted_bed_params,
                    )

                    # Update the main traces with refined results
                    self.surface_trace[x_start:x_end] = region_surface_refined
                    self.bed_trace[x_start:x_end] = region_bed_refined

                    print(
                        f"Region {i + 1} refinement completed with slope-aware adjustments"
                    )
                    print(f"  Slope angle: {slope_metrics['slope_angle']:.1f}°")
                    print(
                        f"  Applied {slope_metrics['adjustment_level']} parameter adjustments"
                    )

                except Exception as e:
                    print(f"Error refining region {i + 1}: {e}")
                    # Fall back to original aggressive parameter adjustment
                    self._apply_fallback_refinement(region, region_data, i)

    def _calculate_slope_metrics(self, surface_trace, bed_trace):
        """
        Calculate slope characteristics for a given region.

        Args:
            surface_trace: Surface echo trace for the region
            bed_trace: Bed echo trace for the region

        Returns:
            dict: Dictionary containing slope metrics and adjustment recommendations
        """
        metrics = {
            "slope_angle": 0.0,
            "surface_variability": 0.0,
            "bed_variability": 0.0,
            "adjustment_level": "conservative",
            "prominence_factor": 0.85,
            "search_expansion": 1.2,
        }

        # Analyze surface slope if sufficient valid data
        valid_surface = np.where(np.isfinite(surface_trace))[0]
        if len(valid_surface) >= 3:
            try:
                # Calculate slope angle from surface trace
                surface_slope = np.polyfit(
                    valid_surface, surface_trace[valid_surface], 1
                )[0]
                surface_angle = abs(np.arctan(surface_slope) * 180 / np.pi)

                # Calculate variability metrics
                surface_variability = np.std(surface_trace[valid_surface])

                # Update metrics based on surface characteristics
                metrics["slope_angle"] = surface_angle
                metrics["surface_variability"] = surface_variability

            except (np.linalg.LinAlgError, ValueError):
                # Fall back to conservative estimates if fitting fails
                pass

        # Analyze bed slope if sufficient valid data
        valid_bed = np.where(np.isfinite(bed_trace))[0]
        if len(valid_bed) >= 3:
            try:
                bed_slope = np.polyfit(valid_bed, bed_trace[valid_bed], 1)[0]
                bed_angle = abs(np.arctan(bed_slope) * 180 / np.pi)
                bed_variability = np.std(bed_trace[valid_bed])

                # Use the steeper angle for parameter adjustment
                metrics["slope_angle"] = max(metrics["slope_angle"], bed_angle)
                metrics["bed_variability"] = bed_variability

            except (np.linalg.LinAlgError, ValueError):
                pass

        # Determine adjustment level based on slope angle
        if metrics["slope_angle"] > 20:
            metrics["adjustment_level"] = "aggressive"
            metrics["prominence_factor"] = 0.5  # Very sensitive
            metrics["search_expansion"] = 2.5  # Large search windows
        elif metrics["slope_angle"] > 12:
            metrics["adjustment_level"] = "moderate"
            metrics["prominence_factor"] = 0.65  # More sensitive
            metrics["search_expansion"] = 1.8  # Expanded search
        elif metrics["slope_angle"] > 6:
            metrics["adjustment_level"] = "gentle"
            metrics["prominence_factor"] = 0.75  # Slightly more sensitive
            metrics["search_expansion"] = 1.4  # Some expansion
        else:
            metrics["adjustment_level"] = "conservative"
            metrics["prominence_factor"] = 0.85  # Minimal change
            metrics["search_expansion"] = 1.2  # Slight expansion

        return metrics

    def _get_slope_adjusted_parameters(self, slope_metrics, surface_trace, bed_trace):
        """
        Generate slope-adjusted parameters for surface and bed detection.

        Args:
            slope_metrics: Dictionary containing slope analysis results
            surface_trace: Surface echo trace for context
            bed_trace: Bed echo trace for context

        Returns:
            tuple: (adjusted_surface_params, adjusted_bed_params)
        """
        # Base parameters from optimization
        base_surface_params = self.optimized_params.get("surface_detection", {}).copy()
        base_bed_params = self.optimized_params.get("bed_detection", {}).copy()

        # Apply slope-specific adjustments
        prominence_factor = slope_metrics["prominence_factor"]
        search_expansion = slope_metrics["search_expansion"]

        # Adjust surface detection parameters
        adjusted_surface_params = base_surface_params.copy()
        if "peak_prominence" in adjusted_surface_params:
            adjusted_surface_params["peak_prominence"] = max(
                10, adjusted_surface_params["peak_prominence"] * prominence_factor
            )

        if "search_depth_px" in adjusted_surface_params:
            adjusted_surface_params["search_depth_px"] = int(
                adjusted_surface_params["search_depth_px"] * search_expansion
            )

        # Enhance image processing for difficult slopes
        if slope_metrics["adjustment_level"] in ["aggressive", "moderate"]:
            adjusted_surface_params["enhancement_clahe_clip"] = min(
                8.0, adjusted_surface_params.get("enhancement_clahe_clip", 3.0) * 1.4
            )

        # Adjust bed detection parameters
        adjusted_bed_params = base_bed_params.copy()
        if "peak_prominence" in adjusted_bed_params:
            adjusted_bed_params["peak_prominence"] = max(
                15, adjusted_bed_params["peak_prominence"] * prominence_factor
            )

        if "search_start_offset_from_surface_px" in adjusted_bed_params:
            adjusted_bed_params["search_start_offset_from_surface_px"] = int(
                adjusted_bed_params["search_start_offset_from_surface_px"]
                * search_expansion
            )

        if "search_end_offset_from_z_boundary_px" in adjusted_bed_params:
            adjusted_bed_params["search_end_offset_from_z_boundary_px"] = max(
                30,
                int(adjusted_bed_params["search_end_offset_from_z_boundary_px"] * 1.5),
            )

        # Apply enhanced image processing for steep slopes
        if slope_metrics["adjustment_level"] == "aggressive":
            adjusted_bed_params["enhancement_clahe_clip"] = min(
                8.0, adjusted_bed_params.get("enhancement_clahe_clip", 3.0) * 1.6
            )
            # Use more aggressive smoothing parameters
            adjusted_bed_params["max_gradient"] = 0.4  # Allow steeper gradients
            adjusted_bed_params["adaptive_min_window"] = (
                15  # Smaller windows for detail
            )

        return adjusted_surface_params, adjusted_bed_params

    def _estimate_tx_pulse_position(self, region_data):
        """
        Estimate transmitter pulse position for a region.

        Args:
            region_data: Image data for the region

        Returns:
            int: Estimated Y-position of transmitter pulse relative to region
        """
        # Use a simple approach: assume TX pulse is near the top of the usable data
        # This is a fallback when we don't have access to the global TX pulse position

        # Look for the strongest horizontal feature in the top third of the region
        top_third = region_data[: region_data.shape[0] // 3, :]

        if top_third.shape[0] > 0:
            # Calculate horizontal intensity profile
            horizontal_profile = np.mean(top_third, axis=1)

            # Find the peak (likely TX pulse location)
            if len(horizontal_profile) > 0:
                tx_estimate = np.argmax(horizontal_profile)
                return min(
                    tx_estimate + 20, region_data.shape[0] - 50
                )  # Add small offset

        # Fallback to a reasonable estimate
        return min(50, region_data.shape[0] // 4)

    def _apply_fallback_refinement(self, region, region_data, region_index):
        """
        Apply fallback refinement when slope-aware method fails.

        Args:
            region: Region dictionary with x_range and y_range
            region_data: Image data for the region
            region_index: Index of the region being processed
        """
        print(f"Applying fallback refinement for region {region_index + 1}")

        x_start, x_end = region["x_range"]

        # Apply more aggressive but generic parameters
        adjusted_surface_params = self.optimized_params.get(
            "surface_detection", {}
        ).copy()
        adjusted_bed_params = self.optimized_params.get("bed_detection", {}).copy()

        # Generic aggressive adjustments
        adjusted_surface_params["peak_prominence"] = max(
            10, adjusted_surface_params.get("peak_prominence", 20) * 0.6
        )
        adjusted_bed_params["peak_prominence"] = max(
            15, adjusted_bed_params.get("peak_prominence", 30) * 0.6
        )

        # Increase enhancement
        adjusted_surface_params["enhancement_clahe_clip"] = min(
            8.0, adjusted_surface_params.get("enhancement_clahe_clip", 3.0) * 1.5
        )
        adjusted_bed_params["enhancement_clahe_clip"] = min(
            8.0, adjusted_bed_params.get("enhancement_clahe_clip", 3.0) * 1.5
        )

        try:
            from functions.echo_tracing import detect_surface_echo, detect_bed_echo

            tx_pulse_y_rel = self._estimate_tx_pulse_position(region_data)

            # Re-run detection with fallback parameters
            region_surface = detect_surface_echo(
                region_data, tx_pulse_y_rel, adjusted_surface_params
            )
            region_bed = detect_bed_echo(
                region_data,
                region_surface,
                region_data.shape[0] - 50,
                adjusted_bed_params,
            )

            # Update traces
            self.surface_trace[x_start:x_end] = region_surface
            self.bed_trace[x_start:x_end] = region_bed

            print(f"Fallback refinement completed for region {region_index + 1}")

        except Exception as e:
            print(f"Fallback refinement also failed for region {region_index + 1}: {e}")
