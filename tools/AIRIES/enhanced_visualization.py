# enhanced_visualization.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
from matplotlib.colors import LinearSegmentedColormap
import cv2
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime


class EnhancedVisualization:
    """
    Enhanced visualization components for region-based semi-automatic picker.
    Provides advanced plotting, real-time feedback, and quality assessment visualizations.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Default visualization parameters
        self.viz_params = {
            "figure_size": (16, 10),
            "dpi": 100,
            "region_colors": {
                "surface": "cyan",
                "bed": "orange",
                "active": "red",
                "completed": "green",
            },
            "pick_colors": {
                "manual": "red",
                "automatic": "yellow",
                "high_confidence": "lime",
                "low_confidence": "orange",
            },
            "alpha_values": {
                "region_overlay": 0.3,
                "pick_points": 0.8,
                "confidence_overlay": 0.6,
            },
        }

        # Custom colormaps
        self._create_custom_colormaps()

    def _create_custom_colormaps(self):
        """Create custom colormaps for specialized visualizations."""
        # Confidence colormap (red -> yellow -> green)
        confidence_colors = ["red", "orange", "yellow", "lightgreen", "green"]
        self.confidence_cmap = LinearSegmentedColormap.from_list(
            "confidence", confidence_colors
        )

        # Template similarity colormap
        similarity_colors = ["darkblue", "blue", "cyan", "yellow", "red"]
        self.similarity_cmap = LinearSegmentedColormap.from_list(
            "similarity", similarity_colors
        )

    def _extract_layer_data(self, processor_ref, layer_name):
        """Extract picks and intensity data for a specific layer."""
        if hasattr(processor_ref, "layers") and layer_name in processor_ref.layers:
            return {
                "picks": processor_ref.layers[layer_name]["picks"],
                "intensity": processor_ref.layers[layer_name]["intensity"],
            }
        else:
            # Fallback for old structure
            if layer_name == "surface":
                return {
                    "picks": getattr(processor_ref, "final_surface_picks", None),
                    "intensity": getattr(processor_ref, "surface_intensity", None),
                }
            elif layer_name == "bed":
                return {
                    "picks": getattr(processor_ref, "final_bed_picks", None),
                    "intensity": getattr(processor_ref, "bed_intensity", None),
                }
        return {"picks": None, "intensity": None}

    def create_enhanced_picking_display(
        self,
        radar_data: np.ndarray,
        regions_data: Dict[str, List[Dict]],
        session_metadata: Optional[Dict] = None,
        processor_ref: Optional[Any] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create enhanced picking display with regions and real-time feedback.

        Args:
            radar_data: 2D radar image data
            regions_data: Regions data from RegionsManager
            session_metadata: Optional session metadata

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        fig, ax = plt.subplots(
            figsize=self.viz_params["figure_size"], dpi=self.viz_params["dpi"]
        )

        # Display radar image with enhanced contrast
        enhanced_image = self._enhance_image_for_display(radar_data)
        im = ax.imshow(
            enhanced_image, cmap="gray", aspect="auto", interpolation="nearest"
        )

        # Add active layer indicator
        if processor_ref is not None and hasattr(processor_ref, "active_layer"):
            ax.text(
                0.02,
                0.98,
                f"Active Layer: {processor_ref.active_layer.title()}",
                transform=ax.transAxes,
                fontsize=14,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                zorder=20,
            )

        # Add region overlays
        self._add_region_overlays(ax, regions_data, radar_data.shape[1])

        # Add pick overlays
        self._add_pick_overlays(ax, regions_data, processor_ref=processor_ref)

        # Add confidence visualization
        self._add_confidence_visualization(ax, regions_data)

        # Customize display
        ax.set_xlabel("X Pixel")
        ax.set_ylabel("Y Pixel")
        ax.set_title(
            "Enhanced Region-Based Echo Picker", fontsize=14, fontweight="bold"
        )

        # Add legend
        self._add_picking_legend(ax)

        # Add session info if available
        if session_metadata:
            self._add_session_info_panel(fig, session_metadata)

        plt.tight_layout()
        return fig, ax

    def create_template_analysis_plot(
        self,
        template_data: Dict,
        similarity_scores: np.ndarray,
        region_bounds: Tuple[int, int],
    ) -> plt.Figure:
        """
        Create template analysis visualization.

        Args:
            template_data: Template characteristics
            similarity_scores: Similarity scores across region
            region_bounds: (x_start, x_end) for region

        Returns:
            plt.Figure: Template analysis figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Template characteristics plot
        ax1 = axes[0, 0]
        template_metrics = [
            template_data.get("intensity_mean", 0),
            template_data.get("gradient_vertical", 0),
            template_data.get("center_intensity", 0),
        ]
        metric_names = ["Intensity\nMean", "Gradient\nStrength", "Center\nIntensity"]

        bars = ax1.bar(
            metric_names, template_metrics, color=["skyblue", "lightgreen", "salmon"]
        )
        ax1.set_title("Template Characteristics", fontweight="bold")
        ax1.set_ylabel("Value")

        # Add value labels on bars
        for bar, value in zip(bars, template_metrics):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(template_metrics) * 0.01,
                f"{value:.1f}",
                ha="center",
                va="bottom",
            )

        # Similarity score distribution
        ax2 = axes[0, 1]
        ax2.hist(
            similarity_scores[~np.isnan(similarity_scores)],
            bins=30,
            color="lightblue",
            alpha=0.7,
            edgecolor="black",
        )
        ax2.axvline(
            np.nanmean(similarity_scores),
            color="red",
            linestyle="--",
            label=f"Mean: {np.nanmean(similarity_scores):.3f}",
        )
        ax2.set_title("Similarity Score Distribution", fontweight="bold")
        ax2.set_xlabel("Similarity Score")
        ax2.set_ylabel("Frequency")
        ax2.legend()

        # Similarity across region
        ax3 = axes[1, 0]
        x_range = np.arange(region_bounds[0], region_bounds[1] + 1)
        ax3.plot(x_range, similarity_scores, color="blue", linewidth=2)
        ax3.fill_between(x_range, similarity_scores, alpha=0.3, color="lightblue")
        ax3.set_title("Similarity Across Region", fontweight="bold")
        ax3.set_xlabel("X Pixel")
        ax3.set_ylabel("Similarity Score")
        ax3.grid(True, alpha=0.3)

        # Template quality metrics
        ax4 = axes[1, 1]
        quality_metrics = {
            "Sample Count": template_data.get("num_samples", 0),
            "Intensity Range": template_data.get("intensity_range", [0, 0])[1]
            - template_data.get("intensity_range", [0, 0])[0],
            "Gradient Consistency": 1.0
            / (1.0 + template_data.get("gradient_tolerance", 1.0)),
        }

        metric_values = list(quality_metrics.values())
        metric_labels = list(quality_metrics.keys())

        colors = ["gold", "lightcoral", "lightseagreen"]
        wedges, texts, autotexts = ax4.pie(
            metric_values,
            labels=metric_labels,
            colors=colors,
            autopct="%1.1f",
            startangle=90,
        )
        ax4.set_title("Template Quality Metrics", fontweight="bold")

        plt.tight_layout()
        return fig

    def create_confidence_heatmap(
        self,
        radar_data: np.ndarray,
        confidence_scores: np.ndarray,
        region_bounds: Tuple[int, int],
    ) -> plt.Figure:
        """
        Create confidence score heatmap overlay.

        Args:
            radar_data: 2D radar image data
            confidence_scores: Confidence scores for region
            region_bounds: (x_start, x_end) for region

        Returns:
            plt.Figure: Confidence heatmap figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Original radar image
        enhanced_image = self._enhance_image_for_display(radar_data)
        ax1.imshow(enhanced_image, cmap="gray", aspect="auto", interpolation="nearest")
        ax1.set_title("Original Radar Data", fontweight="bold")
        ax1.set_ylabel("Y Pixel")

        # Confidence overlay
        confidence_overlay = np.full(radar_data.shape, np.nan)
        x_start, x_end = region_bounds

        for i, x in enumerate(range(x_start, x_end + 1)):
            if i < len(confidence_scores) and not np.isnan(confidence_scores[i]):
                # Create vertical column of confidence values
                confidence_overlay[:, x] = confidence_scores[i]

        # Display confidence heatmap
        im2 = ax2.imshow(
            confidence_overlay,
            cmap=self.confidence_cmap,
            aspect="auto",
            interpolation="nearest",
            vmin=0,
            vmax=1,
            alpha=self.viz_params["alpha_values"]["confidence_overlay"],
        )

        # Overlay original image with transparency
        ax2.imshow(
            enhanced_image,
            cmap="gray",
            aspect="auto",
            interpolation="nearest",
            alpha=0.4,
        )

        ax2.set_title("Confidence Score Overlay", fontweight="bold")
        ax2.set_xlabel("X Pixel")
        ax2.set_ylabel("Y Pixel")

        # Add colorbar
        cbar = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label("Confidence Score", rotation=270, labelpad=20)

        plt.tight_layout()
        return fig

    def create_quality_assessment_dashboard(
        self, session_summary: Dict, regions_data: Dict[str, List[Dict]]
    ) -> plt.Figure:
        """
        Create comprehensive quality assessment dashboard.

        Args:
            session_summary: Session summary data
            regions_data: Regions data from RegionsManager

        Returns:
            plt.Figure: Quality assessment dashboard
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # Coverage metrics
        ax1 = fig.add_subplot(gs[0, :2])
        coverage_data = session_summary["quality_metrics"]
        coverage_metrics = ["Surface Coverage", "Bed Coverage"]
        coverage_values = [
            coverage_data.get("surface_coverage", 0),
            coverage_data.get("bed_coverage", 0),
        ]

        bars = ax1.bar(
            coverage_metrics, coverage_values, color=["skyblue", "salmon"], alpha=0.8
        )
        ax1.set_ylim(0, 100)
        ax1.set_ylabel("Coverage (%)")
        ax1.set_title("Echo Coverage Metrics", fontweight="bold")

        # Add percentage labels
        for bar, value in zip(bars, coverage_values):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Region distribution
        ax2 = fig.add_subplot(gs[0, 2:])
        region_summary = session_summary["regions_summary"]
        region_labels = ["Surface Regions", "Bed Regions"]
        region_counts = [
            region_summary.get("surface_regions", 0),
            region_summary.get("bed_regions", 0),
        ]

        if sum(region_counts) > 0:
            wedges, texts, autotexts = ax2.pie(
                region_counts,
                labels=region_labels,
                colors=["lightblue", "lightcoral"],
                autopct="%1.0f",
                startangle=90,
            )
            ax2.set_title("Region Distribution", fontweight="bold")
        else:
            ax2.text(
                0.5,
                0.5,
                "No regions defined",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=12,
            )
            ax2.set_title("Region Distribution", fontweight="bold")

        # Control points analysis
        ax3 = fig.add_subplot(gs[1, :2])
        control_points_per_region = []
        region_types = []

        for echo_type in ["surface", "bed"]:
            for region in regions_data.get(echo_type, []):
                control_points_per_region.append(len(region.get("control_points", [])))
                region_types.append(echo_type.capitalize())

        if control_points_per_region:
            colors = ["skyblue" if t == "Surface" else "salmon" for t in region_types]
            bars = ax3.bar(
                range(len(control_points_per_region)),
                control_points_per_region,
                color=colors,
                alpha=0.8,
            )
            ax3.set_xlabel("Region Index")
            ax3.set_ylabel("Control Points Count")
            ax3.set_title("Control Points per Region", fontweight="bold")

            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, control_points_per_region)):
                height = bar.get_height()
                ax3.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.1,
                    f"{value}",
                    ha="center",
                    va="bottom",
                )
        else:
            ax3.text(
                0.5,
                0.5,
                "No control points yet",
                ha="center",
                va="center",
                transform=ax3.transAxes,
                fontsize=12,
            )
            ax3.set_title("Control Points per Region", fontweight="bold")

        # Confidence score trends
        ax4 = fig.add_subplot(gs[1, 2:])
        all_confidence_scores = []

        for echo_type in ["surface", "bed"]:
            for region in regions_data.get(echo_type, []):
                if "confidence_scores" in region:
                    valid_scores = [
                        s for s in region["confidence_scores"] if not np.isnan(s)
                    ]
                    all_confidence_scores.extend(valid_scores)

        if all_confidence_scores:
            ax4.hist(
                all_confidence_scores,
                bins=20,
                color="lightgreen",
                alpha=0.7,
                edgecolor="black",
            )
            ax4.axvline(
                np.mean(all_confidence_scores),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(all_confidence_scores):.3f}",
            )
            ax4.set_xlabel("Confidence Score")
            ax4.set_ylabel("Frequency")
            ax4.set_title("Confidence Score Distribution", fontweight="bold")
            ax4.legend()
        else:
            ax4.text(
                0.5,
                0.5,
                "No confidence scores available",
                ha="center",
                va="center",
                transform=ax4.transAxes,
                fontsize=12,
            )
            ax4.set_title("Confidence Score Distribution", fontweight="bold")

        # Session timeline
        ax5 = fig.add_subplot(gs[2, :])
        session_info = session_summary["session_info"]

        # Create timeline visualization
        timeline_text = f"""
        Session: {session_info["name"]}
        Image: {session_info["image"]}
        Created: {session_info["created"][:19].replace("T", " ")}
        Last Modified: {session_info["last_modified"][:19].replace("T", " ")}
        Version: {session_info["version"]}
        Total Regions: {region_summary.get("total_regions", 0)}
        Total Control Points: {region_summary.get("total_control_points", 0)}
        Processing Events: {session_summary.get("processing_events", 0)}
        User Notes: {session_summary.get("user_notes", 0)}
        """

        ax5.text(
            0.05,
            0.95,
            timeline_text,
            transform=ax5.transAxes,
            fontfamily="monospace",
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
        )
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.axis("off")
        ax5.set_title("Session Information", fontweight="bold")

        plt.suptitle(
            "Quality Assessment Dashboard", fontsize=16, fontweight="bold", y=0.98
        )
        return fig

    def create_realtime_feedback_overlay(
        self,
        ax: plt.Axes,
        current_template: Optional[Dict] = None,
        active_region: Optional[Dict] = None,
        cursor_position: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Create real-time feedback overlay for active picking.

        Args:
            ax: Matplotlib axes to overlay on
            current_template: Current template characteristics
            active_region: Currently active region
            cursor_position: Current cursor position
        """
        # Clear previous overlays
        for collection in ax.collections:
            if hasattr(collection, "_realtime_feedback"):
                collection.remove()

        # Add active region highlighting
        if active_region:
            x_start, x_end = active_region["bounds"]
            rect = patches.Rectangle(
                (x_start, 0),
                x_end - x_start,
                ax.get_ylim()[1],
                linewidth=3,
                edgecolor="red",
                facecolor="red",
                alpha=0.1,
                linestyle="--",
            )
            rect._realtime_feedback = True
            ax.add_patch(rect)

        # Add cursor feedback
        if cursor_position and current_template:
            x, y = cursor_position

            # Template matching window
            window_size = current_template.get("window_size", (20, 10))
            window_width, window_height = window_size

            template_rect = patches.Rectangle(
                (x - window_width // 2, y - window_height // 2),
                window_width,
                window_height,
                linewidth=2,
                edgecolor="yellow",
                facecolor="none",
                linestyle="-",
            )
            template_rect._realtime_feedback = True
            ax.add_patch(template_rect)

            # Crosshair
            ax.axhline(y=y, color="yellow", linestyle="-", linewidth=1, alpha=0.8)
            ax.axvline(x=x, color="yellow", linestyle="-", linewidth=1, alpha=0.8)

    def export_visualization_summary(
        self, output_dir: Path, session_name: str, figures: Dict[str, plt.Figure]
    ) -> bool:
        """
        Export all visualizations to summary document.

        Args:
            output_dir: Output directory path
            session_name: Session name for filenames
            figures: Dictionary of figure names and objects

        Returns:
            bool: Success status
        """
        try:
            viz_dir = output_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)

            # Save individual figures
            for fig_name, figure in figures.items():
                output_path = viz_dir / f"{session_name}_{fig_name}.png"
                figure.savefig(output_path, dpi=300, bbox_inches="tight")
                print(f"INFO: Saved visualization: {output_path}")

            # Create summary HTML report
            html_report = self._generate_html_summary(session_name, figures.keys())
            html_path = viz_dir / f"{session_name}_visualization_summary.html"

            with open(html_path, "w") as f:
                f.write(html_report)

            print(f"INFO: Visualization summary exported to {viz_dir}")
            return True

        except Exception as e:
            print(f"ERROR: Failed to export visualization summary: {e}")
            return False

    def _enhance_image_for_display(self, image: np.ndarray) -> np.ndarray:
        """Enhance image for better display quality."""
        # Apply CLAHE for better contrast
        if image.dtype != np.uint8:
            img_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(
                np.uint8
            )
        else:
            img_uint8 = image.copy()

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_uint8)

        return enhanced

    def _add_region_overlays(
        self, ax: plt.Axes, regions_data: Dict, image_width: int
    ) -> None:
        """Add region boundary overlays to plot."""
        colors = self.viz_params["region_colors"]

        for echo_type in ["surface", "bed"]:
            color = colors.get(echo_type, "white")

            for region in regions_data.get(echo_type, []):
                x_start, x_end = region["bounds"]
                status = region.get("status", "active")

                # Choose color based on status
                if status == "completed":
                    edge_color = colors["completed"]
                    alpha = 0.6
                elif status == "active":
                    edge_color = colors["active"]
                    alpha = 0.8
                else:
                    edge_color = color
                    alpha = 0.4

                # Add region boundary
                rect = patches.Rectangle(
                    (x_start, 0),
                    x_end - x_start,
                    ax.get_ylim()[1],
                    linewidth=2,
                    edgecolor=edge_color,
                    facecolor=edge_color,
                    alpha=self.viz_params["alpha_values"]["region_overlay"],
                    linestyle="-" if status == "active" else "--",
                )
                ax.add_patch(rect)

                # Add region label
                label_y = ax.get_ylim()[1] * 0.05
                ax.text(
                    x_start + 10,
                    label_y,
                    f"{echo_type.capitalize()} Region",
                    color=edge_color,
                    fontweight="bold",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

    def _add_pick_overlays(
        self, ax: plt.Axes, regions_data: Dict, processor_ref: Optional[Any] = None
    ) -> None:
        """Add manual and automatic pick overlays - Updated for layer architecture."""
        colors = self.viz_params["pick_colors"]

        # If processor has layer data, use that
        if processor_ref and hasattr(processor_ref, "layers"):
            for layer_name in ["surface", "bed"]:
                layer_data = self._extract_layer_data(processor_ref, layer_name)
                picks = layer_data["picks"]
                intensity = layer_data["intensity"]

                if picks is not None and np.any(np.isfinite(picks)):
                    valid_mask = np.isfinite(picks)
                    x_coords = np.arange(len(picks))[valid_mask]
                    y_coords = picks[valid_mask]

                    # Color picks based on whether they're manual or auto
                    color = (
                        colors["manual"]
                        if layer_name == processor_ref.active_layer
                        else colors["automatic"]
                    )

                    ax.scatter(
                        x_coords,
                        y_coords,
                        c=color,
                        s=30,
                        marker="o",
                        alpha=self.viz_params["alpha_values"]["pick_points"],
                        edgecolors="black",
                        linewidths=0.5,
                        label=f"{layer_name.title()} Picks",
                    )
        else:
            # Fallback to original regions_data approach
            for echo_type in ["surface", "bed"]:
                for region in regions_data.get(echo_type, []):
                    # Manual picks
                    manual_picks = region.get("control_points", [])
                    if manual_picks:
                        manual_x = [p["x"] for p in manual_picks]
                        manual_y = [p["y"] for p in manual_picks]
                        ax.scatter(
                            manual_x,
                            manual_y,
                            c=colors["manual"],
                            s=50,
                            marker="o",
                            alpha=self.viz_params["alpha_values"]["pick_points"],
                            edgecolors="black",
                            linewidths=1,
                            label="Manual Picks",
                        )

                    # Automatic picks with confidence coloring
                    auto_detections = region.get("auto_detections", [])
                    confidence_scores = region.get("confidence_scores", [])

                    if auto_detections and len(auto_detections) > 0:
                        x_start, x_end = region["bounds"]
                        auto_x = []
                        auto_y = []
                        confidence_colors = []

                        for i, detection in enumerate(auto_detections):
                            if not np.isnan(detection):
                                x_pos = x_start + i
                                auto_x.append(x_pos)
                                auto_y.append(detection)

                                # Color based on confidence
                                conf_score = (
                                    confidence_scores[i]
                                    if i < len(confidence_scores)
                                    else 0.5
                                )
                                if conf_score > 0.8:
                                    confidence_colors.append(colors["high_confidence"])
                                elif conf_score > 0.5:
                                    confidence_colors.append(colors["automatic"])
                                else:
                                    confidence_colors.append(colors["low_confidence"])

                        if auto_x:
                            ax.scatter(
                                auto_x,
                                auto_y,
                                c=confidence_colors,
                                s=20,
                                marker="s",
                                alpha=self.viz_params["alpha_values"]["pick_points"],
                                edgecolors="black",
                                linewidths=0.5,
                                label="Automatic Picks",
                            )

    def _add_confidence_visualization(self, ax: plt.Axes, regions_data: Dict) -> None:
        """Add confidence score visualization as line plots."""
        for echo_type in ["surface", "bed"]:
            for region in regions_data.get(echo_type, []):
                confidence_scores = region.get("confidence_scores", [])

                if confidence_scores and len(confidence_scores) > 0:
                    x_start, x_end = region["bounds"]
                    x_positions = np.arange(x_start, x_start + len(confidence_scores))

                    # Create confidence line (scaled for visibility)
                    y_offset = ax.get_ylim()[1] * 0.1  # 10% from top
                    confidence_line = (
                        np.array(confidence_scores) * 50 + y_offset
                    )  # Scale for visibility

                    # Plot confidence as line
                    valid_mask = ~np.isnan(confidence_line)
                    if np.any(valid_mask):
                        ax.plot(
                            x_positions[valid_mask],
                            confidence_line[valid_mask],
                            color="white",
                            linewidth=2,
                            alpha=0.8,
                            label=f"{echo_type.capitalize()} Confidence",
                        )

    def _add_picking_legend(self, ax: plt.Axes) -> None:
        """Add comprehensive legend for picking display."""
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=8,
                label="Manual Picks",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="lime",
                markersize=6,
                label="High Confidence Auto",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="yellow",
                markersize=6,
                label="Medium Confidence Auto",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="orange",
                markersize=6,
                label="Low Confidence Auto",
            ),
            patches.Patch(color="cyan", alpha=0.3, label="Surface Regions"),
            patches.Patch(color="orange", alpha=0.3, label="Bed Regions"),
            patches.Patch(color="red", alpha=0.3, label="Active Region"),
        ]

        ax.legend(
            handles=legend_elements,
            loc="lower right",
            bbox_to_anchor=(0.98, 0.02),
            framealpha=0.9,
        )

    def _add_session_info_panel(self, fig: plt.Figure, session_metadata: Dict) -> None:
        """Add session information panel to figure."""
        info_text = f"""
        Session: {session_metadata.get("session_name", "N/A")}
        Image: {session_metadata.get("image_filename", "N/A")}
        Status: {session_metadata.get("status", "N/A")}
        Regions: {session_metadata.get("total_regions", 0)}
        """

        fig.text(
            0.02,
            0.98,
            info_text.strip(),
            transform=fig.transFigure,
            fontfamily="monospace",
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )

    def _generate_html_summary(self, session_name: str, figure_names: List[str]) -> str:
        """Generate HTML summary report for visualizations."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Visualization Summary - {session_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                .figure {{ margin: 20px 0; text-align: center; }}
                .figure img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                .timestamp {{ color: #888; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <h1>Region-Based Semi-Automatic Picker - Visualization Summary</h1>
            <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h2>Session: {session_name}</h2>
        """

        for fig_name in figure_names:
            html_content += f"""
            <div class="figure">
                <h3>{fig_name.replace("_", " ").title()}</h3>
                <img src="{session_name}_{fig_name}.png" alt="{fig_name}">
            </div>
            """

        html_content += """
        </body>
        </html>
        """

        return html_content
