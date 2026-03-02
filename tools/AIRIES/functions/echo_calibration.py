import numpy as np
import cv2


def analyze_echo_points(image, surface_points, bed_points):
    """
    Analyze user-selected points to determine optimal detection parameters.

    Args:
        image: Full radar image
        surface_points: List of (x, y) tuples for surface echo points
        bed_points: List of (x, y) tuples for bed echo points

    Returns:
        dict: Optimized parameters for surface and bed detection
    """
    # Analyze surface echo characteristics
    surface_params = _analyze_echo_points_type(image, surface_points, "surface")

    # Analyze bed echo characteristics
    bed_params = _analyze_echo_points_type(image, bed_points, "bed")

    return {"surface_detection": surface_params, "bed_detection": bed_params}


def _analyze_echo_points_type(image, points, echo_type):
    """
    Analyze characteristics of selected echo points with enhanced slope awareness.

    Args:
        image: Full radar image
        points: List of (x, y) tuples for echo points
        echo_type: "surface" or "bed"

    Returns:
        dict: Optimized parameters for the echo type
    """
    if not points:
        return {}

    # Sort points by x-coordinate for slope analysis
    sorted_points = sorted(points, key=lambda p: p[0])

    all_intensities = []
    all_gradients = []
    polarity_votes = []
    local_maxima_strengths = []

    # Define analysis window around each point
    window_size = 15  # 15x15 pixel window around each point
    half_window = window_size // 2

    for x, y in points:
        # Extract local region around point
        y_start = max(0, y - half_window)
        y_end = min(image.shape[0], y + half_window + 1)
        x_start = max(0, x - half_window)
        x_end = min(image.shape[1], x + half_window + 1)

        local_region = image[y_start:y_end, x_start:x_end]

        if local_region.size == 0:
            continue

        # Analyze intensity characteristics
        intensities = local_region.flatten()
        all_intensities.extend(intensities)

        # Analyze gradient characteristics
        grad_y = np.abs(np.gradient(local_region, axis=0))
        all_gradients.extend(grad_y.flatten())

        # Determine polarity (bright vs dark echoes)
        center_intensity = (
            image[y, x]
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]
            else np.mean(local_region)
        )
        background_mean = np.mean(local_region)
        polarity_votes.append(
            "bright" if center_intensity > background_mean else "dark"
        )

        # Analyze local maxima strength
        center_val = (
            local_region[local_region.shape[0] // 2, local_region.shape[1] // 2]
            if local_region.shape[0] > 0 and local_region.shape[1] > 0
            else 0
        )
        local_max_strength = center_val - np.mean(local_region)
        local_maxima_strengths.append(abs(local_max_strength))

    # Calculate slope characteristics for more sophisticated parameter adjustment
    if len(sorted_points) >= 3:
        # Calculate slope characteristics
        x_coords = [p[0] for p in sorted_points]
        y_coords = [p[1] for p in sorted_points]

        # Fit line to estimate average slope
        slope, intercept = np.polyfit(x_coords, y_coords, 1)
        slope_angle = np.arctan(slope) * 180 / np.pi

        # Calculate curvature if we have enough points
        curvature = 0
        if len(sorted_points) >= 6:
            try:
                # Use quadratic fit to estimate curvature
                poly_coeffs = np.polyfit(x_coords, y_coords, 2)
                curvature = abs(poly_coeffs[0])  # Second derivative approximation
            except (np.linalg.LinAlgError, ValueError):
                curvature = 0

        # Enhanced slope-specific parameter determination
        if abs(slope_angle) > 20:  # Very steep slope
            if echo_type == "surface":
                return {
                    "search_start_offset_px": 60,  # Reduced from 80
                    "search_depth_px": 250,  # Increased from 200
                    "peak_prominence": 15,  # Reduced from 25 for sensitivity
                    "max_slope_angle": 35,  # Increased from 30
                    "search_depth_expansion_factor": 3.0,  # Increased from 2.0
                    "enhancement_clahe_clip": 7.0,  # Increased contrast
                    "use_geometric_compensation": True,
                    "adaptive_prominence_scaling": True,
                }
            else:  # bed
                return {
                    "search_start_offset_from_surface_px": 100,  # Reduced from 150
                    "search_end_offset_from_z_boundary_px": 60,  # Increased from 30
                    "peak_prominence": 15,  # Reduced from 30 for higher sensitivity
                    "max_slope_angle": 35,  # Increased from 30
                    "search_depth_expansion_factor": 3.0,  # Increased from 2.0
                    "enhancement_clahe_clip": 8.0,  # Increased contrast
                    "use_geometric_compensation": True,
                    "adaptive_prominence_scaling": True,
                }
        elif abs(slope_angle) > 12:  # Steep slope
            if echo_type == "surface":
                return {
                    "search_start_offset_px": 80,
                    "search_depth_px": 200,
                    "peak_prominence": 20,  # Reduced from 35
                    "max_slope_angle": 25,
                    "search_depth_expansion_factor": 2.0,
                    "enhancement_clahe_clip": 6.0,
                    "use_geometric_compensation": True,
                    "adaptive_prominence_scaling": True,
                }
            else:  # bed
                return {
                    "search_start_offset_from_surface_px": 120,  # Reduced from 200
                    "search_end_offset_from_z_boundary_px": 50,  # Increased from 25
                    "peak_prominence": 25,  # Reduced from 45
                    "max_slope_angle": 25,
                    "search_depth_expansion_factor": 2.0,
                    "enhancement_clahe_clip": 7.0,
                    "use_geometric_compensation": True,
                    "adaptive_prominence_scaling": True,
                }
        elif abs(slope_angle) > 6:  # Moderate slope
            if echo_type == "surface":
                return {
                    "search_start_offset_px": 90,
                    "search_depth_px": 180,
                    "peak_prominence": 25,  # Reduced from 45
                    "max_slope_angle": 20,
                    "search_depth_expansion_factor": 1.5,
                    "enhancement_clahe_clip": 5.0,
                    "use_geometric_compensation": False,
                    "adaptive_prominence_scaling": True,
                }
            else:  # bed
                return {
                    "search_start_offset_from_surface_px": 180,  # Reduced from 250
                    "search_end_offset_from_z_boundary_px": 35,  # Increased from 20
                    "peak_prominence": 35,  # Reduced from 60
                    "max_slope_angle": 20,
                    "search_depth_expansion_factor": 1.5,
                    "enhancement_clahe_clip": 6.0,
                    "use_geometric_compensation": False,
                    "adaptive_prominence_scaling": True,
                }
        else:  # Gentle slope
            if echo_type == "surface":
                return {
                    "search_start_offset_px": 100,
                    "search_depth_px": 150,
                    "peak_prominence": 35,  # Reduced from 45
                    "max_slope_angle": 15,
                    "search_depth_expansion_factor": 1.2,
                    "enhancement_clahe_clip": 4.0,
                    "use_geometric_compensation": False,
                    "adaptive_prominence_scaling": False,
                }
            else:  # bed
                return {
                    "search_start_offset_from_surface_px": 200,  # Reduced from 250
                    "search_end_offset_from_z_boundary_px": 30,  # Increased from 20
                    "peak_prominence": 45,  # Reduced from 60
                    "max_slope_angle": 15,
                    "search_depth_expansion_factor": 1.2,
                    "enhancement_clahe_clip": 5.0,
                    "use_geometric_compensation": False,
                    "adaptive_prominence_scaling": False,
                }

    # Fallback parameters based on echo characteristics only
    if all_intensities and local_maxima_strengths:
        intensity_std = np.std(all_intensities)
        gradient_std = np.std(all_gradients)
        avg_peak_strength = np.mean(local_maxima_strengths)

        # Determine echo polarity by majority vote
        echo_polarity = (
            max(set(polarity_votes), key=polarity_votes.count)
            if polarity_votes
            else "dark"
        )

        # Calculate prominence based on actual peak strength observed
        prominence = max(10, min(60, avg_peak_strength * 0.8))

        # Calculate CLAHE parameters based on contrast needs
        clahe_clip = max(3.0, min(8.0, intensity_std / 15))

        # Calculate search parameters based on echo type
        if echo_type == "surface":
            search_offset = 50
            search_depth = 150
            min_distance = 10
        else:  # bed
            search_offset = 150
            search_depth = 300
            min_distance = 20

        return {
            "peak_prominence": prominence,
            "echo_polarity": echo_polarity,
            "enhancement_clahe_clip": clahe_clip,
            "search_start_offset_px"
            if echo_type == "surface"
            else "search_start_offset_from_surface_px": search_offset,
            "search_depth_px"
            if echo_type == "surface"
            else "search_end_offset_from_z_boundary_px": 40,
            "peak_min_distance": min_distance,
            "enhancement_blur_ksize": [3, 3] if gradient_std > 30 else [5, 5],
            "adaptive_min_window": 15 if echo_type == "surface" else 25,
            "adaptive_max_window": 101 if echo_type == "surface" else 201,
            "max_slope_angle": 15,
            "search_depth_expansion_factor": 1.2,
            "use_geometric_compensation": False,
            "adaptive_prominence_scaling": False,
        }

    return {}


def analyze_slope_characteristics(image, bed_points):
    """
    Analyze slope characteristics from user-selected bed points for enhanced parameter tuning.

    Args:
        image: Full radar image
        bed_points: List of (x, y) tuples for bed echo points

    Returns:
        dict: Slope-specific parameters
    """
    if len(bed_points) < 3:
        return {}

    # Sort points by x-coordinate
    sorted_points = sorted(bed_points, key=lambda p: p[0])

    # Calculate slope characteristics
    x_coords = [p[0] for p in sorted_points]
    y_coords = [p[1] for p in sorted_points]

    # Fit line to estimate average slope
    slope, intercept = np.polyfit(x_coords, y_coords, 1)

    # Calculate slope angle
    slope_angle = np.arctan(slope) * 180 / np.pi

    # Calculate variability metrics
    slope_variability = np.std(y_coords)

    # Determine optimal parameters based on slope analysis
    if abs(slope_angle) > 20:  # Very steep slope
        return {
            "peak_prominence": 15,
            "search_depth_expansion": 3.0,
            "enhancement_clahe_clip": 8.0,
            "use_multi_pass_detection": True,
            "geometric_compensation": True,
            "adaptive_prominence": True,
        }
    elif abs(slope_angle) > 12:  # Steep slope
        return {
            "peak_prominence": 20,
            "search_depth_expansion": 2.0,
            "enhancement_clahe_clip": 7.0,
            "use_multi_pass_detection": True,
            "geometric_compensation": True,
            "adaptive_prominence": True,
        }
    elif abs(slope_angle) > 6:  # Moderate slope
        return {
            "peak_prominence": 30,
            "search_depth_expansion": 1.5,
            "enhancement_clahe_clip": 6.0,
            "use_multi_pass_detection": True,
            "geometric_compensation": False,
            "adaptive_prominence": True,
        }
    else:  # Gentle slope
        return {
            "peak_prominence": 45,
            "search_depth_expansion": 1.2,
            "enhancement_clahe_clip": 5.0,
            "use_multi_pass_detection": False,
            "geometric_compensation": False,
            "adaptive_prominence": False,
        }
