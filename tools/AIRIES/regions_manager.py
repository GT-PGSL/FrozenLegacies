# regions_manager.py

import numpy as np
import json
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from datetime import datetime


class RegionsManager:
    """
    Manages regions for the semi-automatic echo picker.
    Handles region creation, modification, and template management.
    Compatible with the existing semi_auto_processor architecture.
    """

    def __init__(self, image_shape=None):
        """
        Initialize RegionsManager.

        Args:
            image_shape: Optional tuple (height, width) of radar image
        """
        self.regions = {"surface": [], "bed": []}
        self.active_region_id = None
        self.region_counter = 0
        self.image_shape = image_shape
        self.radar_data = None

        # Configuration parameters
        self.min_region_width = 50
        self.default_search_height = 100
        self.template_window_size = (10, 20)  # (height, width)
        self.detection_confidence_threshold = 0.6

    def initialize_radar_data(self, radar_data: np.ndarray):
        """Initialize with radar data for template matching."""
        self.radar_data = radar_data
        self.image_shape = radar_data.shape
        print(
            f"INFO: RegionsManager initialized with radar data shape: {self.image_shape}"
        )

    def create_region(
        self, echo_type: str, x_start: int, x_end: int, y_center: Optional[int] = None
    ) -> str:
        """
        Create a new region for picking.

        Args:
            echo_type: 'surface' or 'bed'
            x_start: Start X coordinate
            x_end: End X coordinate
            y_center: Optional Y center for search window

        Returns:
            str: Region ID
        """
        # Ensure proper ordering
        if x_start > x_end:
            x_start, x_end = x_end, x_start

        # Validate region bounds
        if self.image_shape:
            x_start = max(0, x_start)
            x_end = min(self.image_shape[1] - 1, x_end)

        if x_end - x_start < self.min_region_width:
            print(
                f"WARNING: Region too narrow: {x_end - x_start} < {self.min_region_width}"
            )

        # Define search window
        if y_center is None and self.image_shape:
            if echo_type == "surface":
                y_center = self.image_shape[0] // 4  # Upper portion for surface
            else:
                y_center = 3 * self.image_shape[0] // 4  # Lower portion for bed

        y_min = 0
        y_max = self.image_shape[0] - 1 if self.image_shape else 1000

        if y_center is not None:
            y_min = max(0, y_center - self.default_search_height // 2)
            y_max = min(
                self.image_shape[0] - 1 if self.image_shape else 1000,
                y_center + self.default_search_height // 2,
            )

        # Create region ID
        region_id = f"{echo_type}_{self.region_counter}"
        self.region_counter += 1

        region = {
            "id": region_id,
            "echo_type": echo_type,
            "bounds": (x_start, x_end),
            "y_bounds": (y_min, y_max),
            "control_points": [],
            "auto_detections": [],
            "confidence_scores": [],
            "status": "active",
            "template": None,
            "created_timestamp": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
        }

        if echo_type in self.regions:
            self.regions[echo_type].append(region)
            print(
                f"INFO: Created {echo_type} region '{region_id}' from x={x_start} to x={x_end}, y={y_min} to {y_max}"
            )

        return region_id

    def add_control_point(self, region_id: str, point: Dict[str, Any]) -> bool:
        """Add a control point to a region, preventing duplicates."""
        region = self.get_region(region_id)
        if region:
            # Check for existing point at same x-coordinate
            x_coord = int(point.get("x", 0))
            for existing_point in region["control_points"]:
                if abs(existing_point.get("x", 0) - x_coord) < 5:  # 5-pixel tolerance
                    print(
                        f"INFO: Control point at x={x_coord} already exists, skipping duplicate"
                    )
                    return False

            # Add timestamp if not present
            if "timestamp" not in point:
                point["timestamp"] = datetime.now().isoformat()

            region["control_points"].append(point)
            region["last_modified"] = datetime.now().isoformat()
            region["status"] = "active"

            # Sort control points by x-coordinate
            region["control_points"].sort(key=lambda p: p.get("x", 0))

            print(
                f"INFO: Added control point at ({point.get('x')}, {point.get('y')}) to region {region_id}"
            )
            return True

        print(f"WARNING: Could not find region {region_id} to add control point")
        return False

    def get_region(self, region_id: str) -> Optional[Dict]:
        """
        Get region by ID.

        Args:
            region_id: Region identifier

        Returns:
            Optional[Dict]: Region data or None if not found
        """
        for echo_type in self.regions:
            for region in self.regions[echo_type]:
                if region["id"] == region_id:
                    return region
        return None

    def get_regions_by_type(self, echo_type: str) -> List[Dict]:
        """
        Get all regions of a specific echo type.

        Args:
            echo_type: 'surface' or 'bed'

        Returns:
            List[Dict]: List of region dictionaries
        """
        return self.regions.get(echo_type, [])

    def get_all_region_ids(self) -> List[str]:
        """
        Get all region IDs.

        Returns:
            List[str]: List of all region IDs
        """
        ids = []
        for echo_type in self.regions:
            for region in self.regions[echo_type]:
                ids.append(region["id"])
        return ids

    def delete_region(self, region_id: str) -> bool:
        """
        Delete a region by ID.

        Args:
            region_id: Region identifier

        Returns:
            bool: Success status
        """
        for echo_type in self.regions:
            for i, region in enumerate(self.regions[echo_type]):
                if region["id"] == region_id:
                    del self.regions[echo_type][i]
                    print(f"INFO: Deleted region {region_id}")
                    return True

        print(f"WARNING: Could not find region {region_id} to delete")
        return False

    def remove_control_point(
        self, region_id: str, x: int, y: int, tolerance: int = 10
    ) -> bool:
        """
        Remove the nearest control point within tolerance.

        Args:
            region_id: Region identifier
            x, y: Target coordinates
            tolerance: Maximum distance for point removal

        Returns:
            bool: Success status
        """
        region = self.get_region(region_id)
        if not region:
            return False

        # Find nearest point
        min_dist = float("inf")
        nearest_point = None

        for point in region["control_points"]:
            dist = np.sqrt((point.get("x", 0) - x) ** 2 + (point.get("y", 0) - y) ** 2)
            if dist < min_dist and dist <= tolerance:
                min_dist = dist
                nearest_point = point

        if nearest_point:
            region["control_points"].remove(nearest_point)
            region["last_modified"] = datetime.now().isoformat()
            print(
                f"INFO: Removed control point ({nearest_point.get('x')}, {nearest_point.get('y')}) from region {region_id}"
            )
            return True

        return False

    def split_region(self, region_id: str, split_x: int) -> Tuple[str, str]:
        """
        Split a region into two at the specified x-coordinate.

        Args:
            region_id: Region to split
            split_x: X-coordinate for split point

        Returns:
            Tuple[str, str]: IDs of new left and right regions
        """
        region = self.get_region(region_id)
        if not region:
            raise ValueError(f"Region {region_id} not found")

        x_start, x_end = region["bounds"]
        if not (x_start < split_x < x_end):
            raise ValueError(f"Split point {split_x} not within region bounds")

        # Create left region
        left_id = self.create_region(
            region["echo_type"],
            x_start,
            split_x,
            (region["y_bounds"][0] + region["y_bounds"][1]) // 2,
        )

        # Create right region
        right_id = self.create_region(
            region["echo_type"],
            split_x,
            x_end,
            (region["y_bounds"][0] + region["y_bounds"][1]) // 2,
        )

        # Distribute control points
        left_region = self.get_region(left_id)
        right_region = self.get_region(right_id)

        for point in region["control_points"]:
            if point.get("x", 0) <= split_x:
                left_region["control_points"].append(point)
            else:
                right_region["control_points"].append(point)

        # Delete original region
        self.delete_region(region_id)

        print(
            f"INFO: Split region {region_id} at x={split_x} into {left_id} and {right_id}"
        )
        return left_id, right_id

    def merge_regions(self, region_id1: str, region_id2: str) -> str:
        """
        Merge two adjacent regions of the same echo type.

        Args:
            region_id1: First region ID
            region_id2: Second region ID

        Returns:
            str: ID of merged region
        """
        region1 = self.get_region(region_id1)
        region2 = self.get_region(region_id2)

        if not region1 or not region2:
            raise ValueError("One or both regions not found")

        if region1["echo_type"] != region2["echo_type"]:
            raise ValueError("Cannot merge regions of different echo types")

        # Determine bounds
        x_start = min(region1["bounds"][0], region2["bounds"][0])
        x_end = max(region1["bounds"][1], region2["bounds"][1])
        y_min = min(region1["y_bounds"][0], region2["y_bounds"][0])
        y_max = max(region1["y_bounds"][1], region2["y_bounds"][1])

        # Create merged region
        merged_id = self.create_region(x_start, x_end, region1["echo_type"])
        merged_region = self.get_region(merged_id)
        merged_region["y_bounds"] = (y_min, y_max)

        # Combine control points
        all_points = region1["control_points"] + region2["control_points"]
        merged_region["control_points"] = sorted(
            all_points, key=lambda p: p.get("x", 0)
        )

        # Delete original regions
        self.delete_region(region_id1)
        self.delete_region(region_id2)

        print(f"INFO: Merged regions {region_id1} and {region_id2} into {merged_id}")
        return merged_id

    def update_region_detections(
        self, region_id: str, detections: np.ndarray, confidence_scores: np.ndarray
    ) -> bool:
        """
        Update automatic detections for a region.

        Args:
            region_id: Region identifier
            detections: Array of detection Y-coordinates
            confidence_scores: Array of confidence scores

        Returns:
            bool: Success status
        """
        region = self.get_region(region_id)
        if region:
            region["auto_detections"] = (
                detections.tolist()
                if isinstance(detections, np.ndarray)
                else detections
            )
            region["confidence_scores"] = (
                confidence_scores.tolist()
                if isinstance(confidence_scores, np.ndarray)
                else confidence_scores
            )
            region["status"] = (
                "completed" if np.any(np.isfinite(detections)) else "active"
            )
            region["last_modified"] = datetime.now().isoformat()
            return True

        return False

    def analyze_echo_characteristics(
        self, region_id: str, radar_data: np.ndarray
    ) -> bool:
        """
        Analyze echo characteristics from control points in a region.

        Args:
            region_id: Region identifier
            radar_data: 2D radar image data

        Returns:
            bool: Success status
        """
        region = self.get_region(region_id)
        if not region:
            return False

        control_points = region.get("control_points", [])
        if len(control_points) < 2:
            print(
                f"WARNING: Need at least 2 control points for analysis in region {region_id}"
            )
            return False

        brightness_values = []
        contrast_values = []
        gradient_values = []
        templates = []

        window_h, window_w = self.template_window_size

        for point in control_points:
            x, y = point.get("x", 0), point.get("y", 0)

            # Extract local window around control point
            y_start = max(0, y - window_h // 2)
            y_end = min(radar_data.shape[0], y + window_h // 2)
            x_start = max(0, x - window_w // 2)
            x_end = min(radar_data.shape[1], x + window_w // 2)

            local_window = radar_data[y_start:y_end, x_start:x_end]

            if local_window.size > 0:
                # Calculate characteristics
                brightness_values.append(np.mean(local_window))
                contrast_values.append(np.std(local_window))
                gradient_values.append(np.mean(np.abs(np.gradient(local_window))))
                templates.append(local_window)

        if brightness_values:
            # Create echo characteristics
            region["template"] = {
                "brightness_mean": float(np.mean(brightness_values)),
                "brightness_std": float(np.std(brightness_values)),
                "contrast_mean": float(np.mean(contrast_values)),
                "contrast_std": float(np.std(contrast_values)),
                "gradient_strength": float(np.mean(gradient_values)),
                "confidence_threshold": self.detection_confidence_threshold,
                "window_size": self.template_window_size,
                "num_samples": len(brightness_values),
            }

            print(f"INFO: Analyzed echo characteristics for region {region_id}")
            print(
                f"  Brightness: {region['template']['brightness_mean']:.1f} ± {region['template']['brightness_std']:.1f}"
            )
            print(
                f"  Contrast: {region['template']['contrast_mean']:.1f} ± {region['template']['contrast_std']:.1f}"
            )

            return True

        return False

    def create_region_mask(self, echo_type: str) -> np.ndarray:
        """
        Create a boolean mask for all regions of specified type.

        Args:
            echo_type: 'surface' or 'bed'

        Returns:
            np.ndarray: Boolean mask array
        """
        if not self.image_shape:
            return np.array([])

        mask = np.zeros(self.image_shape[1], dtype=bool)

        for region in self.regions.get(echo_type, []):
            x_start, x_end = region["bounds"]
            mask[x_start : x_end + 1] = True

        return mask

    def get_control_points_array(self, echo_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all control points for specified echo type as arrays.

        Args:
            echo_type: 'surface' or 'bed'

        Returns:
            Tuple[np.ndarray, np.ndarray]: X and Y coordinate arrays
        """
        x_points = []
        y_points = []

        for region in self.regions.get(echo_type, []):
            for point in region.get("control_points", []):
                x_points.append(point.get("x", 0))
                y_points.append(point.get("y", 0))

        return np.array(x_points), np.array(y_points)

    def get_region_at_position(self, x: int, echo_type: str) -> Optional[str]:
        """
        Find region containing the specified x-coordinate.

        Args:
            x: X-coordinate
            echo_type: 'surface' or 'bed'

        Returns:
            Optional[str]: Region ID or None if not found
        """
        for region in self.regions.get(echo_type, []):
            x_start, x_end = region["bounds"]
            if x_start <= x <= x_end:
                return region["id"]

        return None

    def get_region_coverage(self, echo_type: str) -> float:
        """
        Get percentage of image width covered by regions of specified type.

        Args:
            echo_type: 'surface' or 'bed'

        Returns:
            float: Coverage percentage
        """
        if not self.image_shape:
            return 0.0

        mask = self.create_region_mask(echo_type)
        coverage = np.sum(mask) / len(mask) if len(mask) > 0 else 0.0
        return coverage * 100.0

    def get_region_summary(self) -> Dict[str, Any]:
        """
        Get summary of all regions.

        Returns:
            Dict[str, Any]: Region summary statistics
        """
        surface_regions = self.regions.get("surface", [])
        bed_regions = self.regions.get("bed", [])

        total_control_points = sum(
            len(r.get("control_points", [])) for r in surface_regions + bed_regions
        )

        summary = {
            "total_regions": len(surface_regions) + len(bed_regions),
            "surface_regions": len(surface_regions),
            "bed_regions": len(bed_regions),
            "total_control_points": total_control_points,
            "active_regions": sum(
                1 for r in surface_regions + bed_regions if r.get("status") == "active"
            ),
            "completed_regions": sum(
                1
                for r in surface_regions + bed_regions
                if r.get("status") == "completed"
            ),
            "surface_coverage": self.get_region_coverage("surface"),
            "bed_coverage": self.get_region_coverage("bed"),
        }

        return summary

    def export_regions_data(self) -> Dict[str, List[Dict]]:
        """
        Export all regions data for saving.

        Returns:
            Dict[str, List[Dict]]: Complete regions data
        """
        return {
            "surface": self.regions.get("surface", []),
            "bed": self.regions.get("bed", []),
            "metadata": {
                "region_counter": self.region_counter,
                "image_shape": self.image_shape,
                "active_region_id": self.active_region_id,
            },
        }

    def load_region(self, region_data: Dict):
        """
        Load region from saved data.

        Args:
            region_data: Region dictionary from saved session
        """
        echo_type = region_data.get("echo_type", "surface")
        if echo_type in self.regions:
            self.regions[echo_type].append(region_data)

            # Update region counter to avoid ID conflicts
            if "id" in region_data:
                try:
                    region_num = int(region_data["id"].split("_")[-1])
                    self.region_counter = max(self.region_counter, region_num + 1)
                except (ValueError, IndexError):
                    pass

    def save_to_file(self, filepath: str):
        """Save regions to JSON file."""

        def make_json_serializable(obj):
            """Convert numpy types to JSON-serializable format."""
            try:
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(
                    obj, (np.integer, np.int64, np.int32, np.int16, np.int8)
                ):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                    return float(obj)
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [make_json_serializable(v) for v in obj]
                elif hasattr(obj, "isoformat"):  # datetime objects
                    return obj.isoformat()
                else:
                    # Test if object is JSON serializable
                    json.dumps(obj)
                    return obj
            except (TypeError, ValueError):
                # Return string representation for non-serializable objects
                return str(obj)

        data = self.export_regions_data()
        serializable_data = make_json_serializable(data)

        with open(filepath, "w") as f:
            json.dump(serializable_data, f, indent=2)

        print(f"INFO: Saved regions to {filepath}")

    def load_from_file(self, filepath: str):
        """Load regions from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        # Load metadata
        metadata = data.get("metadata", {})
        self.region_counter = metadata.get("region_counter", 0)
        self.image_shape = (
            tuple(metadata["image_shape"]) if metadata.get("image_shape") else None
        )
        self.active_region_id = metadata.get("active_region_id")

        # Load regions
        self.regions = {"surface": data.get("surface", []), "bed": data.get("bed", [])}

        print(f"INFO: Loaded regions from {filepath}")

    def clear_all_regions(self):
        """Clear all regions and reset counter."""
        self.regions = {"surface": [], "bed": []}
        self.region_counter = 0
        self.active_region_id = None
        print("INFO: Cleared all regions")
