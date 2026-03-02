# zscope/functions/echo_tracing.py

import numpy as np
import cv2
from scipy.signal import find_peaks
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr


def enhance_image(image, clahe_clip=2.0, clahe_tile=(8, 8), blur_ksize=(5, 5)):
    """Enhance image contrast and reduce noise for better echo tracing."""
    if image.dtype != np.uint8:
        img_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img_uint8 = image.copy()

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    enhanced = clahe.apply(img_uint8)

    if blur_ksize and blur_ksize[0] > 0 and blur_ksize[1] > 0:
        enhanced = cv2.GaussianBlur(enhanced, blur_ksize, 0)

    return enhanced


def extend_boundaries(
    data,
    extension_size=100,
    dampen_factor=0.9,
    trend_points=10,
    use_reflect_padding=True,
):
    """Enhanced boundary extension using trend-based extrapolation."""
    if len(data) < 3:
        return data, 0

    # Find valid indices
    valid_indices = np.where(np.isfinite(data))[0]
    if len(valid_indices) < 3:
        return data, 0

    # Get edge values
    left_valid = valid_indices[0]
    right_valid = valid_indices[-1]

    # Create extended array
    extended = np.full(len(data) + 2 * extension_size, np.nan)

    # Copy original data to center
    extended[extension_size : extension_size + len(data)] = data

    # Mirror left boundary with dampening
    if left_valid > 0:
        left_values = data[left_valid : left_valid + extension_size]
        for i in range(min(extension_size, len(left_values))):
            mirror_idx = extension_size - i - 1
            if mirror_idx >= 0 and i < len(left_values):
                # Apply dampening factor (reduces oscillation)
                dampen = dampen_factor ** (i + 1)
                delta = (left_values[i] - data[left_valid]) * dampen
                extended[mirror_idx] = data[left_valid] - delta

    # For right edge, use reflection padding if requested
    if right_valid < len(data) - 1:
        if use_reflect_padding:
            # Use reflection padding for right edge
            right_reflection_size = min(extension_size, right_valid)
            for i in range(right_reflection_size):
                mirror_idx = extension_size + len(data) + i
                reflect_idx = right_valid - i
                if 0 <= mirror_idx < len(extended) and 0 <= reflect_idx < len(data):
                    extended[mirror_idx] = data[reflect_idx]
        else:
            # Use dampened mirroring for right edge (original method)
            right_values = data[
                max(0, right_valid - extension_size + 1) : right_valid + 1
            ]
            for i in range(min(extension_size, len(right_values))):
                mirror_idx = extension_size + len(data) + i
                if mirror_idx < len(extended) and i < len(right_values):
                    # Apply dampening factor
                    dampen = dampen_factor ** (i + 1)
                    delta = (right_values[-i - 1] - data[right_valid]) * dampen
                    extended[mirror_idx] = data[right_valid] + delta

    return extended, extension_size


def bilateral_filter_1d(
    signal, diameter=5, sigma_color=10.0, sigma_space=2.0, iterations=3
):
    """Apply bilateral filter to 1D signal with multiple iterations for better noise reduction."""
    if diameter % 2 == 0:
        diameter += 1

    half_d = diameter // 2
    working_signal = signal.copy()

    # Handle NaN values first
    valid_indices = np.where(np.isfinite(working_signal))[0]
    if len(valid_indices) < 3:
        return signal

    # Interpolate NaNs for processing
    if len(valid_indices) < len(working_signal):
        x_valid = valid_indices
        y_valid = working_signal[valid_indices]
        x_all = np.arange(len(working_signal))
        if len(valid_indices) >= 2:
            interp_func = interp1d(
                x_valid,
                y_valid,
                kind="linear",
                bounds_error=False,
                fill_value=(y_valid[0], y_valid[-1]),
            )
            working_signal = interp_func(x_all)

    # Apply multiple iterations with decreasing sigma_color
    for iteration in range(iterations):
        filtered = np.zeros_like(working_signal)
        length = len(working_signal)

        # Reduce sigma_color in subsequent iterations to preserve details
        current_sigma_color = sigma_color * (0.8**iteration)

        for i in range(length):
            w_sum = 0.0
            val_sum = 0.0

            for j in range(max(0, i - half_d), min(length, i + half_d + 1)):
                spatial_dist = abs(i - j)
                color_dist = abs(working_signal[i] - working_signal[j])

                w = np.exp(-(spatial_dist**2) / (2 * sigma_space**2)) * np.exp(
                    -(color_dist**2) / (2 * current_sigma_color**2)
                )

                w_sum += w
                val_sum += w * working_signal[j]

            filtered[i] = val_sum / w_sum if w_sum > 0 else working_signal[i]

        working_signal = filtered

    return working_signal


def enforce_trace_continuity(trace, max_jump_pixels=20, window_size=7):
    """Enforce continuity constraints to reduce noise and prevent unrealistic jumps."""
    if len(trace) < window_size * 2:
        return trace

    cleaned_trace = trace.copy()

    for i in range(window_size, len(trace) - window_size):
        if np.isfinite(trace[i]):
            # Get local neighborhood (excluding current point)
            left_window = trace[i - window_size : i]
            right_window = trace[i + 1 : i + window_size + 1]
            neighborhood = np.concatenate([left_window, right_window])

            valid_neighbors = neighborhood[np.isfinite(neighborhood)]

            if len(valid_neighbors) >= 3:
                # Use median of neighbors as reference
                median_neighbor = np.median(valid_neighbors)

                # If current point deviates too much, replace with interpolated value
                if abs(trace[i] - median_neighbor) > max_jump_pixels:
                    # Linear interpolation between nearest valid neighbors
                    left_valid = None
                    right_valid = None

                    # Find nearest valid points
                    for j in range(1, window_size + 1):
                        if (
                            left_valid is None
                            and i - j >= 0
                            and np.isfinite(trace[i - j])
                        ):
                            left_valid = (i - j, trace[i - j])
                        if (
                            right_valid is None
                            and i + j < len(trace)
                            and np.isfinite(trace[i + j])
                        ):
                            right_valid = (i + j, trace[i + j])
                        if left_valid is not None and right_valid is not None:
                            break

                    # Interpolate between valid neighbors
                    if left_valid is not None and right_valid is not None:
                        weight = (i - left_valid[0]) / (right_valid[0] - left_valid[0])
                        cleaned_trace[i] = left_valid[1] + weight * (
                            right_valid[1] - left_valid[1]
                        )
                    elif left_valid is not None:
                        cleaned_trace[i] = left_valid[1]
                    elif right_valid is not None:
                        cleaned_trace[i] = right_valid[1]

    return cleaned_trace


def apply_geometric_spreading_compensation(profile, slope_angle):
    """Apply geometric spreading compensation for sloping interfaces."""
    if abs(slope_angle) < 5:
        return profile

    # Calculate geometric spreading loss
    cos_angle = np.cos(np.radians(abs(slope_angle)))
    compensation_factor = 1 / cos_angle

    # Apply compensation (amplify signal for steep slopes)
    compensation_factor = min(compensation_factor, 2.0)
    compensated = profile * compensation_factor

    return compensated


def calculate_slope_aware_prominence(surface_trace, base_prominence, x_col):
    """Calculate prominence based on local slope characteristics."""
    if x_col < 5 or x_col >= len(surface_trace) - 5:
        return base_prominence

    # Calculate local slope over a 10-pixel window
    local_surface = surface_trace[x_col - 5 : x_col + 5]
    valid_local = local_surface[np.isfinite(local_surface)]

    if len(valid_local) >= 3:
        local_slope = np.polyfit(np.arange(len(valid_local)), valid_local, 1)[0]
        slope_angle = abs(np.arctan(local_slope) * 180 / np.pi)

        # Reduce prominence for steeper slopes
        if slope_angle > 15:
            return base_prominence * 0.4  # Very aggressive
        elif slope_angle > 8:
            return base_prominence * 0.6  # Moderate
        else:
            return base_prominence * 0.8  # Conservative

    return base_prominence


def calculate_enhanced_search_window(surface_y, x_col, surface_trace, config_params):
    """Calculate enhanced search window based on local topography."""
    base_offset = config_params.get("search_start_offset_from_surface_px", 100)

    # Calculate local slope context
    window_size = 20
    start_idx = max(0, x_col - window_size // 2)
    end_idx = min(len(surface_trace), x_col + window_size // 2)

    local_surface = surface_trace[start_idx:end_idx]
    valid_local = local_surface[np.isfinite(local_surface)]

    if len(valid_local) >= 3:
        # Fit polynomial to local surface
        x_local = np.arange(len(valid_local))
        try:
            # Use quadratic fit to capture curvature
            if len(valid_local) >= 6:
                poly_coeffs = np.polyfit(x_local, valid_local, 2)
                curvature = abs(poly_coeffs[0])  # Second derivative
                slope = abs(poly_coeffs[1])  # First derivative
            else:
                poly_coeffs = np.polyfit(x_local, valid_local, 1)
                slope = abs(poly_coeffs[0])
                curvature = 0

            # Expand search based on slope and curvature
            slope_factor = min(3.0, slope / 10)  # Normalize slope
            curvature_factor = min(2.0, curvature * 100)  # Normalize curvature

            # Calculate adaptive offsets
            adaptive_offset = base_offset * (1 + slope_factor + curvature_factor)
            adaptive_depth = 400 * (1 + slope_factor * 0.5)  # Expand search depth

            return int(adaptive_offset), int(adaptive_depth)

        except (np.linalg.LinAlgError, ValueError):
            pass

    return base_offset, 300  # Fallback values


def multi_scale_peak_detection(profile, prominence_range=(10, 30), scales=3):
    """Detect peaks at multiple scales and combine results."""
    if len(profile) < 3:
        return np.array([])

    all_peaks = []
    prominences = np.linspace(prominence_range[0], prominence_range[1], scales)

    for prom in prominences:
        peaks, _ = find_peaks(profile, prominence=prom)
        all_peaks.extend(peaks)

    # Count occurrences of each peak across scales
    if len(all_peaks) == 0:
        return np.array([])

    unique_peaks, counts = np.unique(all_peaks, return_counts=True)

    # Return peaks that appear in multiple scales
    multi_scale_peaks = unique_peaks[counts > 1]

    # If no peaks appear in multiple scales, return peaks from middle scale
    if len(multi_scale_peaks) == 0 and len(all_peaks) > 0:
        mid_scale = scales // 2
        mid_prom = prominences[mid_scale]
        multi_scale_peaks, _ = find_peaks(profile, prominence=mid_prom)

    return multi_scale_peaks


def adaptive_smooth_trace(
    trace,
    min_window=11,
    max_window=101,
    polyorder=3,
    interp_kind="linear",
    max_gradient=0.5,
    max_deviation=5,
    use_bilateral=False,
    bilateral_diameter=25,
    bilateral_sigma_color_factor=2.0,
    bilateral_sigma_space_factor=0.5,
    use_edge_constraints=False,
    left_width_fraction=0.02,
    right_width_fraction=0.04,
    left_strength=0.7,
    right_strength=0.9,
    use_reflect_padding=True,
    use_right_edge_window=False,
    right_edge_window_fraction=0.03,
    use_edge_emphasis=False,
    edge_emphasis_fraction=0.05,
    edge_emphasis_factor=1.5,
):
    """Adaptive smoothing of 1D trace using variable window sizes with enhanced noise reduction."""

    # Extend boundaries to reduce edge effects
    extended_trace, extension_size = extend_boundaries(
        trace,
        extension_size=max(100, int(len(trace) * 0.05)),
        dampen_factor=0.9,
        trend_points=10,
        use_reflect_padding=use_reflect_padding,
    )

    # Process the extended trace
    x_coords_ext = np.arange(len(extended_trace))
    valid_indices_ext = np.where(np.isfinite(extended_trace))[0]

    if len(valid_indices_ext) < 2:
        return trace

    # Interpolate missing values
    interpolated_ext = np.copy(extended_trace)
    if len(valid_indices_ext) > 0:
        fill_val_start = extended_trace[valid_indices_ext[0]]
        fill_val_end = extended_trace[valid_indices_ext[-1]]

        f_interp = interp1d(
            x_coords_ext[valid_indices_ext],
            extended_trace[valid_indices_ext],
            kind=interp_kind,
            bounds_error=False,
            fill_value=(fill_val_start, fill_val_end),
        )
        interpolated_ext = f_interp(x_coords_ext)

    # Extract the result from extended trace
    result = interpolated_ext[extension_size : extension_size + len(trace)]

    # Apply continuity enforcement to reduce noise
    if len(result) > 20:  # Only apply if trace is long enough
        result = enforce_trace_continuity(
            result,
            max_jump_pixels=max_deviation * 2,  # Use config parameter
            window_size=min(15, len(result) // 10),
        )

    # Apply enhanced bilateral filtering with multiple iterations
    if use_bilateral:
        sigma_color = np.nanstd(result) * bilateral_sigma_color_factor
        sigma_space = min_window * bilateral_sigma_space_factor

        # Enhanced bilateral filter with multiple iterations
        result = bilateral_filter_1d(
            result,
            diameter=bilateral_diameter,
            sigma_color=sigma_color,
            sigma_space=sigma_space,
            iterations=3,  # Multiple iterations for better noise reduction
        )
    return result

def extract_echo_template_window(radar_data, x, y, window_size=(20, 10)):
    """
    Extract local window around an echo point for template analysis.

    Args:
        radar_data: 2D radar image data
        x, y: Center coordinates for window extraction
        window_size: (width, height) of extraction window

    Returns:
        dict: Template window data and characteristics
    """
    height, width = radar_data.shape
    window_width, window_height = window_size

    # Calculate window bounds
    y_start = max(0, y - window_height // 2)
    y_end = min(height, y + window_height // 2)
    x_start = max(0, x - window_width // 2)
    x_end = min(width, x + window_width // 2)

    # Extract window
    window = radar_data[y_start:y_end, x_start:x_end]

    if window.size == 0:
        return None

    # Calculate characteristics
    characteristics = {
        "pattern": window.copy(),
        "intensity_mean": np.mean(window),
        "intensity_std": np.std(window),
        "intensity_min": np.min(window),
        "intensity_max": np.max(window),
        "gradient_vertical": np.mean(np.abs(np.gradient(window, axis=0))),
        "gradient_horizontal": np.mean(np.abs(np.gradient(window, axis=1))),
        "center_intensity": window[window.shape[0] // 2, window.shape[1] // 2]
        if window.size > 0
        else 0,
        "window_bounds": (y_start, y_end, x_start, x_end),
        "actual_size": window.shape,
    }

    return characteristics


def create_composite_template(manual_picks, radar_data, window_size=(20, 10)):
    """
    Create composite template from multiple manual picks.

    Args:
        manual_picks: List of manual pick coordinates
        radar_data: 2D radar image data
        window_size: Template window size

    Returns:
        dict: Composite template characteristics
    """
    if not manual_picks or len(manual_picks) < 1:
        return None

    templates = []

    for pick in manual_picks:
        x, y = int(pick["x"]), int(pick["y"])
        template = extract_echo_template_window(radar_data, x, y, window_size)
        if template:
            templates.append(template)

    if not templates:
        return None

    # Calculate composite statistics
    composite = {
        "intensity_mean": np.mean([t["intensity_mean"] for t in templates]),
        "intensity_std": np.mean([t["intensity_std"] for t in templates]),
        "intensity_range": (
            np.mean([t["intensity_min"] for t in templates]),
            np.mean([t["intensity_max"] for t in templates]),
        ),
        "gradient_vertical": np.mean([t["gradient_vertical"] for t in templates]),
        "gradient_horizontal": np.mean([t["gradient_horizontal"] for t in templates]),
        "center_intensity": np.mean([t["center_intensity"] for t in templates]),
        "window_size": window_size,
        "num_samples": len(templates),
        "intensity_tolerance": np.std([t["intensity_mean"] for t in templates]),
        "gradient_tolerance": np.std([t["gradient_vertical"] for t in templates]),
    }

    return composite


def calculate_template_match_score(radar_data, x, y, template, window_size=(20, 10)):
    """
    Calculate similarity score between location and template.

    Args:
        radar_data: 2D radar image data
        x, y: Location to test
        template: Template characteristics dictionary
        window_size: Window size for comparison

    Returns:
        float: Similarity score (0-1)
    """
    if template is None:
        return 0.0

    # Extract window at test location
    test_window = extract_echo_template_window(radar_data, x, y, window_size)
    if test_window is None:
        return 0.0

    # Calculate individual similarity metrics
    scores = []

    # 1. Intensity similarity
    intensity_diff = abs(test_window["intensity_mean"] - template["intensity_mean"])
    intensity_tolerance = max(template.get("intensity_tolerance", 10), 5)
    intensity_score = np.exp(-intensity_diff / intensity_tolerance)
    scores.append(("intensity", intensity_score, 0.3))

    # 2. Center pixel intensity (most important for echo detection)
    center_diff = abs(test_window["center_intensity"] - template["center_intensity"])
    center_score = np.exp(-center_diff / 20)
    scores.append(("center", center_score, 0.4))

    # 3. Gradient similarity
    gradient_diff = abs(
        test_window["gradient_vertical"] - template["gradient_vertical"]
    )
    gradient_tolerance = max(template.get("gradient_tolerance", 5), 2)
    gradient_score = np.exp(-gradient_diff / gradient_tolerance)
    scores.append(("gradient", gradient_score, 0.2))

    # 4. Intensity range consistency
    test_range = test_window["intensity_max"] - test_window["intensity_min"]
    template_range = template["intensity_range"][1] - template["intensity_range"][0]
    range_diff = abs(test_range - template_range)
    range_score = np.exp(-range_diff / 30)
    scores.append(("range", range_score, 0.1))

    # Calculate weighted average
    total_score = sum(score * weight for _, score, weight in scores)

    return min(1.0, max(0.0, total_score))


def generate_pchip_seed_curve(manual_picks, x_start, x_end):
    """
    Generate PCHIP interpolated seed curve from manual picks.

    Args:
        manual_picks: List of manual pick coordinates
        x_start, x_end: X-coordinate range for interpolation

    Returns:
        numpy.ndarray: Seed curve Y-coordinates
    """
    if not manual_picks or len(manual_picks) < 2:
        return None

    # Sort picks by x coordinate
    sorted_picks = sorted(manual_picks, key=lambda p: p["x"])

    x_coords = np.array([p["x"] for p in sorted_picks])
    y_coords = np.array([p["y"] for p in sorted_picks])

    # Remove duplicate x coordinates (keep first occurrence)
    unique_indices = np.unique(x_coords, return_index=True)[1]
    x_coords = x_coords[unique_indices]
    y_coords = y_coords[unique_indices]

    if len(x_coords) < 2:
        return None

    try:
        # Create PCHIP interpolator (shape-preserving)
        interpolator = PchipInterpolator(x_coords, y_coords)

        # Generate full range
        x_range = np.arange(x_start, x_end + 1)
        seed_curve = np.full(len(x_range), np.nan)

        # Only interpolate within bounds of manual picks
        min_x, max_x = x_coords[0], x_coords[-1]
        valid_range = (x_range >= min_x) & (x_range <= max_x)

        if np.any(valid_range):
            seed_curve[valid_range] = interpolator(x_range[valid_range])

        return seed_curve

    except Exception as e:
        print(f"WARNING: PCHIP interpolation failed: {e}")
        return None


def region_based_template_detection(
    radar_data, region_bounds, manual_picks, echo_type="surface", template_params=None
):
    """
    Perform template-based echo detection within a user-defined region.

    Args:
        radar_data: 2D radar image data
        region_bounds: (x_start, x_end) tuple defining region
        manual_picks: List of manual control points
        echo_type: 'surface' or 'bed'
        template_params: Detection parameters

    Returns:
        dict: Detection results with confidence scores
    """
    if template_params is None:
        template_params = {
            "window_size": (20, 10),
            "confidence_threshold": 0.6,
            "max_search_range": 30,
            "use_enhanced_search": True,
        }

    x_start, x_end = region_bounds
    print(
        f"INFO: Starting region-based template detection from x={x_start} to x={x_end}"
    )

    if not manual_picks or len(manual_picks) < 2:
        print("WARNING: Need at least 2 manual picks for template detection")
        return {
            "detections": np.full(x_end - x_start + 1, np.nan),
            "confidence_scores": np.zeros(x_end - x_start + 1),
            "success": False,
        }

    # Step 1: Create template from manual picks
    template = create_composite_template(
        manual_picks, radar_data, template_params["window_size"]
    )

    if template is None:
        print("ERROR: Failed to create template from manual picks")
        return {
            "detections": np.full(x_end - x_start + 1, np.nan),
            "confidence_scores": np.zeros(x_end - x_start + 1),
            "success": False,
        }

    # Step 2: Generate seed curve using PCHIP
    seed_curve = generate_pchip_seed_curve(manual_picks, x_start, x_end)

    if seed_curve is None:
        print("ERROR: Failed to generate seed curve")
        return {
            "detections": np.full(x_end - x_start + 1, np.nan),
            "confidence_scores": np.zeros(x_end - x_start + 1),
            "success": False,
        }

    # Step 3: Template matching across region
    detections = []
    confidence_scores = []
    search_range = template_params["max_search_range"]

    height, width = radar_data.shape

    for i, x in enumerate(range(x_start, x_end + 1)):
        if i >= len(seed_curve) or np.isnan(seed_curve[i]):
            detections.append(np.nan)
            confidence_scores.append(0.0)
            continue

        seed_y = int(np.clip(seed_curve[i], 0, height - 1))

        # Define search window
        if template_params["use_enhanced_search"]:
            # Use adaptive search based on local characteristics
            y_search_start = max(0, seed_y - search_range)
            y_search_end = min(height, seed_y + search_range)
        else:
            # Fixed search window
            y_search_start = max(0, seed_y - search_range // 2)
            y_search_end = min(height, seed_y + search_range // 2)

        best_y = seed_y
        best_score = 0.0

        # Search for best template match in vertical window
        for y in range(y_search_start, y_search_end):
            score = calculate_template_match_score(
                radar_data, x, y, template, template_params["window_size"]
            )

            if score > best_score:
                best_score = score
                best_y = y

        # Apply confidence threshold
        if best_score >= template_params["confidence_threshold"]:
            detections.append(best_y)
        else:
            detections.append(np.nan)

        confidence_scores.append(best_score)

    detections = np.array(detections)
    confidence_scores = np.array(confidence_scores)

    # Step 4: Apply region-specific smoothing
    smoothed_detections = apply_region_smoothing(
        detections, confidence_scores, echo_type
    )

    print(
        f"INFO: Template detection complete. Found {np.sum(np.isfinite(smoothed_detections))} valid detections"
    )

    return {
        "detections": smoothed_detections,
        "confidence_scores": confidence_scores,
        "template_stats": template,
        "success": True,
    }


def apply_region_smoothing(detections, confidence_scores, echo_type="surface"):
    """
    Apply region-specific smoothing to detection results.

    Args:
        detections: Array of detection Y-coordinates
        confidence_scores: Confidence scores for each detection
        echo_type: 'surface' or 'bed'

    Returns:
        numpy.ndarray: Smoothed detections
    """
    if len(detections) == 0:
        return detections

    # Apply confidence-weighted smoothing
    valid_mask = np.isfinite(detections) & (confidence_scores > 0.3)

    if np.sum(valid_mask) < 3:
        return detections  # Not enough points for smoothing

    # Use adaptive smoothing parameters based on echo type
    if echo_type == "surface":
        smoothing_params = {
            "bilateral_diameter": 15,
            "sigma_color": 8.0,
            "sigma_space": 3.0,
            "max_jump": 10,
        }
    else:  # bed
        smoothing_params = {
            "bilateral_diameter": 25,
            "sigma_color": 12.0,
            "sigma_space": 4.0,
            "max_jump": 15,
        }

    # Apply bilateral filtering with confidence weighting
    smoothed = bilateral_filter_1d(
        detections,
        diameter=smoothing_params["bilateral_diameter"],
        sigma_color=smoothing_params["sigma_color"],
        sigma_space=smoothing_params["sigma_space"],
        iterations=2,
    )

    # Apply continuity constraints
    smoothed = enforce_trace_continuity(
        smoothed, max_jump_pixels=smoothing_params["max_jump"], window_size=7
    )

    return smoothed


def update_template_with_new_pick(
    existing_template, new_pick, radar_data, blend_factor=0.3
):
    """
    Update existing template with new manual pick.

    Args:
        existing_template: Current template characteristics
        new_pick: New manual pick coordinates
        radar_data: 2D radar image data
        blend_factor: Blending weight for new pick (0-1)

    Returns:
        dict: Updated template characteristics
    """
    if existing_template is None:
        return None

    # Extract characteristics from new pick
    x, y = int(new_pick["x"]), int(new_pick["y"])
    new_template = extract_echo_template_window(
        radar_data, x, y, existing_template["window_size"]
    )

    if new_template is None:
        return existing_template

    # Blend characteristics
    updated_template = existing_template.copy()

    # Update with weighted average
    updated_template["intensity_mean"] = (1 - blend_factor) * existing_template[
        "intensity_mean"
    ] + blend_factor * new_template["intensity_mean"]

    updated_template["center_intensity"] = (1 - blend_factor) * existing_template[
        "center_intensity"
    ] + blend_factor * new_template["center_intensity"]

    updated_template["gradient_vertical"] = (1 - blend_factor) * existing_template[
        "gradient_vertical"
    ] + blend_factor * new_template["gradient_vertical"]

    # Update sample count
    updated_template["num_samples"] += 1

    return updated_template


def region_constrained_surface_detection(
    image, tx_pulse_y, region_mask, manual_picks=None, config=None
):
    """
    Region-constrained surface echo detection.

    Args:
        image: Radar image data
        tx_pulse_y: Transmitter pulse Y position
        region_mask: Boolean mask defining valid regions
        manual_picks: List of manual control points
        config: Detection configuration

    Returns:
        numpy.ndarray: Surface detections
    """
    if config is None:
        config = {}

    print("INFO: Starting region-constrained surface detection")

    # If manual picks are provided, use template-based detection
    if manual_picks and len(manual_picks) >= 2:
        # Find region bounds from mask
        valid_x = np.where(region_mask)[0]
        if len(valid_x) == 0:
            return np.full(image.shape[1], np.nan)

        x_start, x_end = valid_x[0], valid_x[-1]

        # Use template-based detection
        result = region_based_template_detection(
            image,
            (x_start, x_end),
            manual_picks,
            echo_type="surface",
            template_params=config.get("template_params", {}),
        )

        if result["success"]:
            # Create full-width array
            full_detections = np.full(image.shape[1], np.nan)
            full_detections[x_start : x_end + 1] = result["detections"]
            return full_detections

    # Fallback to traditional detection with region masking
    traditional_detections = detect_surface_echo(image, tx_pulse_y, config)

    # Apply region mask
    traditional_detections[~region_mask] = np.nan

    return traditional_detections


def region_constrained_bed_detection(
    image, surface_picks, z_boundary_y, region_mask, manual_picks=None, config=None
):
    """
    Region-constrained bed echo detection.

    Args:
        image: Radar image data
        surface_picks: Surface echo coordinates
        z_boundary_y: Z-boundary position
        region_mask: Boolean mask defining valid regions
        manual_picks: List of manual control points
        config: Detection configuration

    Returns:
        numpy.ndarray: Bed detections
    """
    if config is None:
        config = {}

    print("INFO: Starting region-constrained bed detection")

    # If manual picks are provided, use template-based detection
    if manual_picks and len(manual_picks) >= 2:
        # Find region bounds from mask
        valid_x = np.where(region_mask)[0]
        if len(valid_x) == 0:
            return np.full(image.shape[1], np.nan)

        x_start, x_end = valid_x[0], valid_x[-1]

        # Use template-based detection
        result = region_based_template_detection(
            image,
            (x_start, x_end),
            manual_picks,
            echo_type="bed",
            template_params=config.get("template_params", {}),
        )

        if result["success"]:
            # Create full-width array
            full_detections = np.full(image.shape[1], np.nan)
            full_detections[x_start : x_end + 1] = result["detections"]
            return full_detections

    # Fallback to traditional detection with region masking
    traditional_detections = detect_bed_echo(image, surface_picks, z_boundary_y, config)

    # Apply region mask
    traditional_detections[~region_mask] = np.nan

    return traditional_detections


# [Keep all the existing functions unchanged for backward compatibility]
# detect_surface_echo and detect_bed_echo remain as they were for legacy support


def detect_surface_echo(image, tx_pulse_y, config):
    """Detect surface echo starting just below Tx pulse."""
    print("INFO: detect_surface_echo called.")
    crop_height, crop_width = image.shape
    cfg_search_start_offset = config.get("search_start_offset_px", 20)
    cfg_search_depth = config.get("search_depth_px", image.shape[0] // 3)

    y_start = tx_pulse_y + cfg_search_start_offset
    y_end = y_start + cfg_search_depth
    y_start = max(0, y_start)
    y_end = min(image.shape[0], y_end)

    if y_start >= y_end:
        print(f"WARNING: Invalid search window [{y_start}, {y_end}]. Returning NaNs.")
        return np.full(crop_width, np.nan)

    enhanced = enhance_image(
        image,
        clahe_clip=config.get("enhancement_clahe_clip", 2.0),
        clahe_tile=tuple(config.get("enhancement_clahe_tile", (8, 8))),
        blur_ksize=tuple(config.get("enhancement_blur_ksize", (3, 3))),
    )

    raw_trace = np.full(crop_width, np.nan)
    polarity = config.get("echo_polarity", "bright")

    # Get multi-scale peak detection parameters
    use_multi_scale = config.get("use_multi_scale", True)
    prominence_min = config.get("prominence_min", 15)
    prominence_max = config.get("prominence_max", 35)
    scales = config.get("scales", 3)

    for x_col in range(crop_width):
        col_profile_data = enhanced[y_start:y_end, x_col]

        if col_profile_data.size == 0:
            continue

        if polarity == "dark":
            profile_to_search = 255 - col_profile_data
        else:
            profile_to_search = col_profile_data

        if use_multi_scale:
            peaks = multi_scale_peak_detection(
                profile_to_search,
                prominence_range=(prominence_min, prominence_max),
                scales=scales,
            )
        else:
            prominence = config.get("peak_prominence", 20)
            peaks, _ = find_peaks(profile_to_search, prominence=prominence)

        if len(peaks) > 0:
            raw_trace[x_col] = y_start + peaks[0]

    # Apply adaptive smoothing
    smoothed_trace = adaptive_smooth_trace(
        raw_trace,
        min_window=config.get("adaptive_min_window", 11),
        max_window=config.get("adaptive_max_window", 101),
        polyorder=config.get("adaptive_polyorder", 3),
        interp_kind=config.get("adaptive_interp_kind", "linear"),
        max_gradient=config.get("max_gradient", 0.5),
        max_deviation=config.get("max_deviation", 5),
        use_bilateral=config.get("use_bilateral", True),
        bilateral_diameter=config.get("bilateral_diameter", 25),
        bilateral_sigma_color_factor=config.get("bilateral_sigma_color_factor", 2.0),
        bilateral_sigma_space_factor=config.get("bilateral_sigma_space_factor", 0.5),
    )

    print("INFO: Ice surface echo detection attempt complete.")
    return smoothed_trace


def detect_bed_echo(
    image_data_region,
    surface_y_coords_relative,
    z_boundary_y_relative,
    config_params=None,
):
    """Enhanced bed echo detection with slope-aware multi-pass detection."""
    if config_params is None:
        config_params = {}

    print("INFO: detect_bed_echo called with slope-aware enhancements.")
    crop_height, crop_width = image_data_region.shape
    raw_bed_y_relative = np.full(crop_width, np.nan)

    # Enhanced image processing
    enh_clip_bed = config_params.get("enhancement_clahe_clip", 6.0)
    enh_tile_bed = tuple(config_params.get("enhancement_clahe_tile", [4, 4]))
    enh_blur_bed = tuple(config_params.get("enhancement_blur_ksize", [5, 5]))

    enhanced_image_for_bed = enhance_image(
        image_data_region, enh_clip_bed, enh_tile_bed, enh_blur_bed
    )

    # Multi-scale peak detection parameters
    use_multi_scale = config_params.get("use_multi_scale", True)
    prominence_min = config_params.get("prominence_min", 15)
    prominence_max = config_params.get("prominence_max", 50)
    scales = config_params.get("scales", 4)

    cfg_polarity = config_params.get("echo_polarity", "bright")
    cfg_search_dir = config_params.get("search_direction", "bottom_up")

    # Process each column
    for x_col in range(crop_width):
        if (
            surface_y_coords_relative is None
            or x_col >= len(surface_y_coords_relative)
            or np.isnan(surface_y_coords_relative[x_col])
        ):
            continue

        # Enhanced search window calculation
        adaptive_offset, adaptive_depth = calculate_enhanced_search_window(
            surface_y_coords_relative[x_col],
            x_col,
            surface_y_coords_relative,
            config_params,
        )

        search_y_start_col = int(surface_y_coords_relative[x_col] + adaptive_offset)
        search_y_end_col = min(crop_height, search_y_start_col + adaptive_depth)

        # Ensure search window is within bounds
        search_y_start_col = max(0, search_y_start_col)
        search_y_end_col = min(crop_height, search_y_end_col)

        if search_y_start_col >= search_y_end_col:
            continue

        column_profile_data_for_bed = enhanced_image_for_bed[
            search_y_start_col:search_y_end_col, x_col
        ]

        if column_profile_data_for_bed.size == 0:
            continue

        # Prepare profile for peak detection
        if cfg_polarity == "dark":
            profile_to_search_for_peaks = 255 - column_profile_data_for_bed
        else:
            profile_to_search_for_peaks = column_profile_data_for_bed

        # Apply geometric spreading compensation
        if x_col > 0 and x_col < len(surface_y_coords_relative) - 1:
            prev_surface = surface_y_coords_relative[x_col - 1]
            next_surface = surface_y_coords_relative[x_col + 1]
            if np.isfinite(prev_surface) and np.isfinite(next_surface):
                local_slope = (next_surface - prev_surface) / 2.0
                slope_angle = np.arctan(local_slope) * 180 / np.pi

                # Apply geometric compensation
                profile_to_search_for_peaks = apply_geometric_spreading_compensation(
                    profile_to_search_for_peaks, slope_angle
                )

        # Multi-pass detection with slope-aware prominence
        passes = [
            {"base_prominence": 60, "description": "Conservative pass"},
            {"base_prominence": 35, "description": "Moderate pass"},
            {"base_prominence": 20, "description": "Aggressive pass"},
            {"base_prominence": 12, "description": "Very aggressive pass"},
        ]

        bed_candidates = []
        for pass_config in passes:
            # Calculate slope-aware prominence for this column
            adaptive_prominence = calculate_slope_aware_prominence(
                surface_y_coords_relative, pass_config["base_prominence"], x_col
            )

            if use_multi_scale:
                temp_peaks = multi_scale_peak_detection(
                    profile_to_search_for_peaks,
                    prominence_range=(
                        adaptive_prominence * 0.7,
                        adaptive_prominence * 1.3,
                    ),
                    scales=scales,
                )
            else:
                temp_peaks, _ = find_peaks(
                    profile_to_search_for_peaks, prominence=adaptive_prominence
                )

            if len(temp_peaks) > 0:
                if cfg_search_dir == "top_down":
                    chosen_peak = temp_peaks[0]
                elif cfg_search_dir == "bottom_up":
                    chosen_peak = temp_peaks[-1]
                else:
                    chosen_peak = temp_peaks[0]

                bed_candidates.append(search_y_start_col + chosen_peak)

        # Combine results using median for robustness
        if bed_candidates:
            raw_bed_y_relative[x_col] = np.median(bed_candidates)

    # Apply enhanced adaptive smoothing
    smoothed_bed_y_relative = adaptive_smooth_trace(
        raw_bed_y_relative,
        min_window=config_params.get("adaptive_min_window", 15),
        max_window=config_params.get("adaptive_max_window", 201),
        polyorder=config_params.get("adaptive_polyorder", 3),
        interp_kind=config_params.get("adaptive_interp_kind", "linear"),
        max_gradient=config_params.get("max_gradient", 0.4),
        max_deviation=config_params.get("max_deviation", 5),
        use_bilateral=True,
        bilateral_diameter=config_params.get("bilateral_diameter", 35),
        bilateral_sigma_color_factor=config_params.get(
            "bilateral_sigma_color_factor", 2.5
        ),
        bilateral_sigma_space_factor=config_params.get(
            "bilateral_sigma_space_factor", 0.5
        ),
        use_edge_constraints=True,
        left_width_fraction=config_params.get("left_width_fraction", 0.02),
        right_width_fraction=config_params.get("right_width_fraction", 0.04),
        left_strength=config_params.get("left_strength", 0.7),
        right_strength=config_params.get("right_strength", 0.9),
        use_reflect_padding=True,
        use_right_edge_window=True,
        right_edge_window_fraction=config_params.get(
            "right_edge_window_fraction", 0.03
        ),
        use_edge_emphasis=True,
        edge_emphasis_fraction=config_params.get("edge_emphasis_fraction", 0.05),
        edge_emphasis_factor=config_params.get("edge_emphasis_factor", 1.5),
    )

    print("INFO: Enhanced bed echo detection complete.")
    return smoothed_bed_y_relative
