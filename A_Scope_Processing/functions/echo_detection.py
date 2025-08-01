import numpy as np
from scipy.signal import find_peaks
import os
import sys

# Add the parent directory to sys.path to allow imports from sibling modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


def detect_double_transmitter_pulse(
    signal_x_clean, signal_y_clean, power_vals, time_vals, config
):
    """
    Enhanced detection algorithm using signal characteristics analysis.
    """
    if time_vals is None or power_vals is None:
        return {"is_double_pulse": False, "recommended_tx_idx": None, "confidence": 0.0}

    # Focus on early time window (first 4 microseconds for TX detection)
    early_time_mask = time_vals <= 4.0
    early_times = time_vals[early_time_mask]
    early_powers = power_vals[early_time_mask]

    if len(early_times) < 10:  # Not enough data
        return {"is_double_pulse": False, "recommended_tx_idx": None, "confidence": 0.0}

    # Find peaks in early region using more sophisticated criteria
    from scipy.signal import find_peaks

    # Use dynamic threshold based on signal characteristics
    noise_floor = (
        np.median(power_vals[-50:]) if len(power_vals) >= 50 else np.median(power_vals)
    )
    signal_range = np.max(early_powers) - noise_floor

    # Look for significant peaks (at least 15 dB above noise floor)
    peak_threshold = noise_floor + max(15.0, signal_range * 0.3)

    peaks, properties = find_peaks(
        early_powers,
        height=peak_threshold,
        distance=max(
            1, int(0.3 / (time_vals[1] - time_vals[0]))
        ),  # Min 0.3μs separation
        prominence=max(3.0, signal_range * 0.1),
        width=2,  # Minimum width to avoid noise spikes
    )

    result = {
        "is_double_pulse": False,
        "tx_peaks": [],
        "recommended_tx_idx": None,
        "confidence": 0.0,
        "all_early_peaks": peaks,
    }

    # Initialize peak_powers here to avoid UnboundLocalError
    if len(peaks) == 0:
        print("WARNING: No TX peaks found in early time window")
        return result

    # Now we know peaks has elements, so we can safely use it
    peak_times = early_times[peaks]
    peak_powers = early_powers[peaks]

    if len(peaks) >= 2:
        # Analyze peak characteristics for double pulse identification
        # Check for double pulse pattern: two strong peaks close in time
        for i in range(len(peaks) - 1):
            time_sep = peak_times[i + 1] - peak_times[i]
            power_diff = abs(peak_powers[i + 1] - peak_powers[i])
            avg_power = (peak_powers[i] + peak_powers[i + 1]) / 2

            # Double pulse criteria: close in time, similar power levels
            if (
                0.1 <= time_sep <= 3.5  # Reasonable separation
                and power_diff <= 8.0  # Similar power levels
                and avg_power >= peak_threshold
            ):  # Both are significant
                result["is_double_pulse"] = True
                result["tx_peaks"] = [peaks[i], peaks[i + 1]]
                result["confidence"] = min(
                    1.0, (8.0 - power_diff) / 8.0 * (3.5 - time_sep) / 3.5
                )

                # Use the later peak as primary TX reference (more conservative)
                early_idx = peaks[i + 1]
                # Convert back to full signal index
                result["recommended_tx_idx"] = np.where(
                    time_vals <= early_times[early_idx]
                )[0][-1]
                break

    if not result["is_double_pulse"] and len(peaks) >= 1:
        # Single TX pulse case
        strongest_peak_idx = peaks[np.argmax(peak_powers)]
        result["recommended_tx_idx"] = np.where(
            time_vals <= early_times[strongest_peak_idx]
        )[0][-1]
        result["confidence"] = 0.5

    return result


def detect_surface_echo_adaptive(power_vals, time_vals, tx_analysis, config):
    """
    Enhanced surface detection using signal energy analysis and context.
    """
    if time_vals is None or power_vals is None:
        return None

    # Determine search window based on TX analysis
    if tx_analysis.get("is_double_pulse", False):
        # For double TX, start search after the last TX component + safety margin
        tx_end_time = time_vals[tx_analysis["recommended_tx_idx"]] + 1.0
        print(
            f"INFO: Double TX detected - starting surface search at {tx_end_time:.2f}μs"
        )
    else:
        # Standard case
        tx_time = (
            time_vals[tx_analysis.get("recommended_tx_idx", 0)]
            if tx_analysis.get("recommended_tx_idx")
            else 0
        )
        tx_end_time = tx_time + 0.5

    # Define surface search window (typically 1-8 μs after TX)
    surface_start_time = max(tx_end_time, 1.0)
    surface_end_time = min(8.0, time_vals.max() * 0.4)

    surface_mask = (time_vals >= surface_start_time) & (time_vals <= surface_end_time)
    if not np.any(surface_mask):
        return None

    surface_times = time_vals[surface_mask]
    surface_powers = power_vals[surface_mask]
    surface_indices = np.where(surface_mask)[0]

    # Enhanced surface detection using multiple criteria
    noise_floor = np.median(power_vals[-50:])  # Estimate noise from end
    signal_std = np.std(power_vals[-50:])

    # Method 1: Look for highest power peak (surface usually strongest after TX)
    max_power_idx = np.argmax(surface_powers)
    max_power_candidate = surface_indices[max_power_idx]

    # Method 2: Look for first significant peak above threshold
    surface_threshold = noise_floor + max(10.0, 4 * signal_std)

    from scipy.signal import find_peaks

    peaks, properties = find_peaks(
        surface_powers,
        height=surface_threshold,
        distance=int(0.2 / (time_vals[1] - time_vals[0])),  # Min 0.2μs separation
        prominence=max(2.0, 2 * signal_std),
        width=1,
    )

    candidates = []

    # Add maximum power candidate
    if surface_powers[max_power_idx] >= surface_threshold:
        candidates.append(
            {
                "idx": max_power_candidate,
                "power": surface_powers[max_power_idx],
                "time": surface_times[max_power_idx],
                "score": surface_powers[max_power_idx] - noise_floor,
            }
        )

    # Add peak candidates
    for peak_idx in peaks:
        global_idx = surface_indices[peak_idx]
        candidates.append(
            {
                "idx": global_idx,
                "power": surface_powers[peak_idx],
                "time": surface_times[peak_idx],
                "score": surface_powers[peak_idx] - noise_floor,
            }
        )

    if not candidates:
        print("WARNING: No surface candidates found above threshold")
        return None

    # Remove duplicates and sort by score (power above noise)
    unique_candidates = []
    for candidate in candidates:
        if not any(
            abs(candidate["time"] - uc["time"]) < 0.1 for uc in unique_candidates
        ):
            unique_candidates.append(candidate)

    unique_candidates.sort(key=lambda x: x["score"], reverse=True)

    # Select best candidate (highest power above noise in surface window)
    best_surface = unique_candidates[0]

    print(
        f"INFO: Surface detected at {best_surface['time']:.2f}μs, "
        f"{best_surface['power']:.1f}dB (score: {best_surface['score']:.1f})"
    )

    return best_surface["idx"]


def find_tx_pulse(signal_x, signal_y, config):
    """Finds the first major positive power peak (minimum y-pixel value)
    near the start of the trace.
    """
    if signal_x is None or len(signal_x) < 5:  # Need some points to search
        print("Warning: Signal too short to find Tx pulse.")
        return None, None

    processing_params = config.get("processing_params", {})
    n = max(
        10, int(processing_params.get("tx_search_frac", 0.20) * len(signal_y))
    )  # Search window size
    search_y = signal_y[:n]
    search_x = signal_x[:n]

    if len(search_y) == 0:
        print("Warning: Tx search window empty.")
        return None, None

    # --- Find Peaks in the NEGATIVE signal to find minima (positive power peaks) ---
    prominence_threshold = max(
        5, np.std(search_y) * processing_params.get("tx_prominence_std_factor", 0.7)
    )  # Min prominence needed

    try:
        peaks, properties = find_peaks(
            -search_y,  # Find peaks in the inverted signal
            prominence=prominence_threshold,
            distance=3,  # Ensure it's not just noise next to another small peak
        )
    except Exception as e:
        print(f"Error during Tx peak finding: {e}")
        peaks = np.array([])

    if len(peaks) == 0:
        # Fallback 1: If no prominent peaks, find the absolute minimum y in the window
        idx = np.argmin(search_y) if len(search_y) > 0 else 0
        # Add a check: is this minimum significantly lower than its neighbors?
        is_significant_min = False
        prom_check = prominence_threshold / 2.0
        if 1 <= idx < len(search_y) - 1:
            if (
                search_y[idx] < search_y[idx - 1] - prom_check
                and search_y[idx] < search_y[idx + 1] - prom_check
            ):
                is_significant_min = True
        elif idx == 0 and len(search_y) > 1:
            if search_y[idx] < search_y[idx + 1] - prom_check:
                is_significant_min = True
        elif idx == len(search_y) - 1 and len(search_y) > 1:
            if search_y[idx] < search_y[idx - 1] - prom_check:
                is_significant_min = True
        elif len(search_y) == 1:  # Only one point
            is_significant_min = True

        if is_significant_min:
            print(
                "Warning: No prominent Tx peak found using find_peaks, using significant minimum in first part."
            )
            tx_idx_in_clean = idx
        else:
            print(
                "Warning: No prominent Tx peak or significant minimum found. Using index 0 as fallback (uncertain)."
            )
            tx_idx_in_clean = 0  # Highly uncertain fallback

    else:
        # Select the first prominent peak found in the negative signal
        tx_idx_in_clean = peaks[0]  # Fixed: use peaks[0] instead of peaks
        print(
            f"Tx peak found at index {tx_idx_in_clean} using inverted signal peak finding."
        )

    # Ensure index is within bounds of the original cleaned signal
    if tx_idx_in_clean >= len(signal_x):
        print(
            f"Warning: Calculated Tx index {tx_idx_in_clean} out of bounds ({len(signal_x)}). Clamping."
        )
        tx_idx_in_clean = len(signal_x) - 1 if len(signal_x) > 0 else 0

    if tx_idx_in_clean < 0:
        tx_idx_in_clean = 0  # Should not happen, but ensure non-negative

    tx_pulse_col = (
        signal_x[tx_idx_in_clean] if len(signal_x) > 0 else 0
    )  # Get the corresponding x-coordinate (column)

    return tx_pulse_col, tx_idx_in_clean


def detect_surface_echo(power_vals, tx_idx_in_clean, config):
    """Finds the first major peak significantly above noise floor after Tx pulse."""
    processing_params = config.get("processing_params", {})

    if tx_idx_in_clean is None or tx_idx_in_clean >= len(
        power_vals
    ) - processing_params.get("surface_search_start_offset_px", 20):
        print(
            "Warning: Cannot search for surface echo (Tx index invalid or too close to end)."
        )
        return None

    # --- Robust Noise Floor Estimation ---
    noise_window_frac = processing_params.get("surface_noise_window_frac", 0.05)
    noise_end_idx = max(0, tx_idx_in_clean - 2)  # End just before Tx starts rising
    noise_start_idx = max(0, noise_end_idx - int(len(power_vals) * noise_window_frac))
    noise_region = power_vals[noise_start_idx:noise_end_idx]

    if len(noise_region) < 5:  # If pre-Tx region is too short
        fallback_noise_end = min(
            len(power_vals),
            max(5, int(len(power_vals) * noise_window_frac)),
        )
        noise_region = power_vals[:fallback_noise_end]
        print(
            f"Warning: Pre-Tx noise region short ({len(power_vals[noise_start_idx:noise_end_idx])} samples). Using fallback region[:{fallback_noise_end}]."
        )

    if len(noise_region) == 0:  # If even fallback fails (very short signal)
        noise_mean = np.min(power_vals) if len(power_vals) > 0 else 0
        noise_std = np.std(power_vals) * 0.1 if len(power_vals) > 0 else 1
        print("Warning: Could not estimate noise floor reliably.")
    else:
        noise_mean = np.mean(noise_region)
        noise_std = np.std(noise_region)
        # Prevent near-zero std dev in flat noise regions
        noise_std = max(
            noise_std, 0.5
        )  # Ensure std dev is at least 0.5 dB for thresholding

    print(
        f"Noise estimated from indices {noise_start_idx}-{noise_end_idx}: Mean={noise_mean:.2f} dB, Std={noise_std:.2f} dB"
    )

    # --- Search for Surface Peak ---
    search_start_offset_px = processing_params.get("surface_search_start_offset_px", 20)
    search_start = tx_idx_in_clean + search_start_offset_px
    if search_start >= len(power_vals):
        print("Warning: Surface echo search start index out of bounds.")
        return None
    search_region = power_vals[search_start:]

    if search_region.size == 0:
        print("Warning: Surface echo search region is empty.")
        return None

    # Define thresholds based on noise floor
    height_noise_std = processing_params.get("surface_peak_height_noise_std", 3.0)
    prominence_noise_std = processing_params.get(
        "surface_peak_prominence_noise_std", 1.5
    )
    min_distance = processing_params.get("surface_peak_distance_px", 10)

    height_threshold = noise_mean + height_noise_std * noise_std
    prominence_threshold = noise_std * prominence_noise_std

    print(
        f"Surface echo search: StartIdx={search_start}, Height>{height_threshold:.2f}, Prominence>{prominence_threshold:.2f}, Distance>{min_distance}"
    )

    try:
        peaks, properties = find_peaks(
            search_region,
            height=height_threshold,
            prominence=prominence_threshold,
            distance=min_distance,  # Ensure it's a distinct peak after Tx decay
        )
    except Exception as e:
        print(f"Error during surface peak finding: {e}")
        peaks = np.array([])

    if len(peaks) == 0:
        print("Warning: No surface echo found matching criteria.")
        return None

    # Return the index of the first detected peak relative to the original power_vals array
    surf_idx = search_start + peaks[0]  # Fixed: use peaks[0] instead of peaks
    print(
        f"Surface echo found at index {surf_idx} (Power: {power_vals[surf_idx]:.2f} dB)"
    )
    return surf_idx


def detect_bed_echo(power_vals, time_vals, surf_idx_in_clean, px_per_us, config):
    """
    Finds the bed echo using sustained decay, min time, dynamic height threshold,
    and peak width validation. Includes a fallback using a relative threshold
    and selecting the most prominent peak if the primary method fails.
    """
    processing_params = config.get("processing_params", {})

    if (
        surf_idx_in_clean is None
        or px_per_us is None
        or px_per_us <= 0
        or time_vals is None
        or len(time_vals) != len(power_vals)
    ):
        print(
            "Warning: Cannot search for bed echo (Missing surface, time, px_per_us, or mismatched lengths)."
        )
        return None
    if surf_idx_in_clean >= len(power_vals):
        print("Warning: Surface index out of bounds for bed search.")
        return None

    surface_peak_power = power_vals[surf_idx_in_clean]
    surface_peak_time = time_vals[surf_idx_in_clean]
    print(
        f"Surface peak: Time={surface_peak_time:.2f} µs, Power={surface_peak_power:.2f} dB"
    )

    us_to_px = lambda us: int(np.round(us * px_per_us)) if px_per_us > 0 else 0

    # --- Find Point of SUSTAINED Significant Decay (dB based) ---
    decay_search_start_time = surface_peak_time + processing_params.get(
        "bed_decay_start_offset_us", 0.2
    )
    decay_search_start_idx_candidates = np.where(time_vals >= decay_search_start_time)[
        0
    ]
    if len(decay_search_start_idx_candidates) == 0:
        return None  # Cannot find start index
    decay_search_start_idx = decay_search_start_idx_candidates[
        0
    ]  # Fixed: use [0] to get first index
    if decay_search_start_idx >= len(power_vals):
        return None  # Start index out of bounds

    decay_threshold_db = surface_peak_power - processing_params.get(
        "bed_decay_db_drop", 0.5
    )
    sustain_px = max(1, us_to_px(processing_params.get("bed_decay_sustain_us", 0.2)))
    print(
        f"Searching for SUSTAINED decay < {decay_threshold_db:.2f} dB for {sustain_px} samples after index {decay_search_start_idx}"
    )

    decay_confirmed_idx = None
    for i in range(decay_search_start_idx, len(power_vals) - sustain_px + 1):
        if np.all(power_vals[i : i + sustain_px] < decay_threshold_db):
            decay_confirmed_idx = i
            break
    if decay_confirmed_idx is None:
        print("Warning: No sustained decay found.")
        return None
    decay_confirmed_time = time_vals[decay_confirmed_idx]
    print(
        f"Sustained decay confirmed starting at index {decay_confirmed_idx} (Time {decay_confirmed_time:.2f} µs)"
    )

    # --- Define Bed Search Start and Minimum Time ---
    bed_search_start_time = decay_confirmed_time + processing_params.get(
        "bed_search_start_offset_us", 0.3
    )
    min_bed_time = surface_peak_time + processing_params.get(
        "bed_min_time_after_surface_us", 0.2
    )
    actual_bed_search_start_time = max(bed_search_start_time, min_bed_time)
    bed_search_start_idx_candidates = np.where(
        time_vals >= actual_bed_search_start_time
    )[0]
    if len(bed_search_start_idx_candidates) == 0:
        print(
            f"Warning: No index found >= bed start time {actual_bed_search_start_time:.2f} µs."
        )
        return None
    bed_search_start_idx = bed_search_start_idx_candidates[
        0
    ]  # Fixed: use [0] to get first index
    if bed_search_start_idx >= len(power_vals):
        print(
            f"Warning: Bed search start index ({bed_search_start_idx}) out of bounds."
        )
        return None
    bed_search_region_power = power_vals[bed_search_start_idx:]
    print(
        f"Bed search starts at index {bed_search_start_idx} (Time {actual_bed_search_start_time:.2f} µs)"
    )
    if len(bed_search_region_power) < 5:
        print("Warning: Bed echo search region too short.")
        return None

    # --- Calculate Dynamic Bed Height Threshold ---
    est_ice_travel_time = max(0.1, actual_bed_search_start_time - surface_peak_time)
    time_ratio = (
        (surface_peak_time + est_ice_travel_time) / surface_peak_time
        if surface_peak_time > 0
        else 1
    )
    geometric_loss_db = 20 * np.log10(time_ratio) if time_ratio > 0 else 0
    dynamic_threshold = (
        surface_peak_power
        - geometric_loss_db
        - processing_params.get("bed_geometric_loss_margin_db", 10)
    )
    bed_height_threshold = max(
        dynamic_threshold, processing_params.get("bed_min_power_db", -40)
    )
    print(
        f"Dynamic Bed Threshold: {bed_height_threshold:.2f} dB (est loss {geometric_loss_db:.2f} dB)"
    )

    # --- Attempt 1: Find Bed Echo Peaks with Dynamic Threshold & Width Check ---
    bed_prominence_threshold = processing_params.get("bed_peak_prominence_db", 2.0)
    min_distance_px = max(
        1, us_to_px(processing_params.get("bed_peak_distance_us", 0.1))
    )
    min_width_samples = max(
        1, us_to_px(processing_params.get("bed_min_peak_width_us", 0.08))
    )
    print(
        f"Attempt 1: Height>{bed_height_threshold:.2f}, Prom>{bed_prominence_threshold:.2f} dB, Dist>{min_distance_px} px, MinWidth>{min_width_samples} samples"
    )

    try:
        bed_peaks_indices_in_region, properties = find_peaks(
            bed_search_region_power,
            height=bed_height_threshold,
            prominence=bed_prominence_threshold,
            distance=min_distance_px,
            width=(min_width_samples, None),
        )
    except Exception as e:
        print(f"Error during primary bed peak finding: {e}")
        bed_peaks_indices_in_region = np.array([])
        properties = {}

    bed_idx = None
    if len(bed_peaks_indices_in_region) > 0:
        # Success with primary method! Take the first peak.
        first_bed_peak_idx_in_region = bed_peaks_indices_in_region[
            0
        ]  # Fixed: use [0] to get first index
        bed_idx = bed_search_start_idx + first_bed_peak_idx_in_region
        peak_width_samples = (
            properties.get("widths", [-1])[0] if properties else -1
        )  # Fixed: use [0] for first width
        print(
            f"Bed echo found (Primary) at index {bed_idx} (Time: {time_vals[bed_idx]:.2f} µs, Width: {peak_width_samples:.1f} samples)"
        )

    # --- Attempt 2: Fallback using Relative Threshold & Max Prominence ---
    if bed_idx is None:  # If primary method failed
        print("Primary bed detection failed. Trying fallback with relative threshold.")
        relative_threshold = surface_peak_power - processing_params.get(
            "bed_relative_fallback_db_drop", 25
        )
        prominence_fallback = bed_prominence_threshold
        print(
            f"Attempt 2 (Fallback): Height>{relative_threshold:.2f}, Prom>{prominence_fallback:.2f} dB, Dist>{min_distance_px} px"
        )

        try:
            # Find peaks using relative threshold, don't require width here
            fallback_peaks_indices, fallback_properties = find_peaks(
                bed_search_region_power,
                height=relative_threshold,
                prominence=prominence_fallback,
                distance=min_distance_px,
            )
        except Exception as e:
            print(f"Error during fallback bed peak finding: {e}")
            fallback_peaks_indices = np.array([])
            fallback_properties = {}

        if len(fallback_peaks_indices) > 0:
            # Select the MOST PROMINENT peak among those meeting the fallback criteria
            prominences = fallback_properties.get("prominences", None)
            if prominences is not None and len(prominences) == len(
                fallback_peaks_indices
            ):
                most_prominent_idx_in_fallback = np.argmax(prominences)
                selected_peak_idx_in_region = fallback_peaks_indices[
                    most_prominent_idx_in_fallback
                ]
                bed_idx = bed_search_start_idx + selected_peak_idx_in_region
                print(
                    f"Bed echo found (Fallback - Max Prominence) at index {bed_idx} (Time: {time_vals[bed_idx]:.2f} µs, Prom: {prominences[most_prominent_idx_in_fallback]:.2f} dB)"
                )
            else:
                # If prominence info failed, just take the first peak found by fallback
                print(
                    "Warning: Could not get prominence for fallback peaks. Taking first fallback peak."
                )
                selected_peak_idx_in_region = fallback_peaks_indices[
                    0
                ]  # Fixed: use [0] to get first index
                bed_idx = bed_search_start_idx + selected_peak_idx_in_region

    return bed_idx
