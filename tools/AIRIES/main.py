# zscope_processor/main.py

import argparse
import sys
from pathlib import Path
import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import datetime

# --- Matplotlib Backend Configuration ---
try:
    matplotlib.use("Qt5Agg")
    print("INFO: Using Matplotlib backend: Qt5Agg")
except ImportError:
    print("WARNING: Qt5Agg backend for Matplotlib not found or failed to load.")
    try:
        matplotlib.use("TkAgg")
        print("INFO: Using Matplotlib backend: TkAgg")
    except ImportError:
        print("WARNING: TkAgg backend for Matplotlib not found or failed to load.")
        print(
            "INFO: Matplotlib will use its default backend. Interactive features may not work if headless."
        )

from functions.image_utils import load_and_preprocess_image
from functions.interactive_tools import ClickSelector
from zscope_processor import ZScopeProcessor


def export_database(processor, output_dir, nav_df=None):
    """
    Export complete database with layer information in both CSV and NPZ formats.

    Args:
        processor: ZScopeProcessor instance
        output_dir: Output directory path
        nav_df: Navigation DataFrame (optional)

    Returns:
        bool: Success status
    """
    print("\n" + "=" * 60)
    print("INFO: Starting enhanced database export (CSV + NPZ)...")

    try:
        # Get CBD tick positions from processor if available
        cbd_tick_positions = getattr(processor, "calculated_ticks", None)
        if cbd_tick_positions is None:
            print("INFO: No CBD tick positions found - coordinates will be NaN")
        else:
            print(f"INFO: Found {len(cbd_tick_positions)} CBD tick positions")

        # First, get the existing enhanced CSV with correct coordinates
        if hasattr(processor, "image_np") and processor.image_np is not None:
            width = processor.image_np.shape[1]  # Image width in pixels
            print(f"INFO: Using image width: {width} pixels")
        else:
            # Fallback: get width from detected picks if image not available
            if (
                hasattr(processor, "detected_surface_y_abs")
                and processor.detected_surface_y_abs is not None
            ):
                width = len(processor.detected_surface_y_abs)
                print(f"INFO: Using pick array width: {width} pixels")
            else:
                print("ERROR: Cannot determine image width from processor")
                return False

        # Extract surface and bed data from processor
        surface_picks = getattr(processor, "detected_surface_y_abs", None)
        bed_picks = getattr(processor, "detected_bed_y_abs", None)

        if surface_picks is None or bed_picks is None:
            print("ERROR: No surface/bed pick data available")
            return False

        # Extract intensities from radar image
        surface_intensity = np.full(width, np.nan)
        bed_intensity = np.full(width, np.nan)

        if hasattr(processor, "image_np") and processor.image_np is not None:
            radar_data = processor.image_np

            # Extract surface intensities
            for i in range(width):
                if np.isfinite(surface_picks[i]):
                    row = int(round(surface_picks[i]))
                    if 0 <= row < radar_data.shape[0]:
                        surface_intensity[i] = radar_data[row, i]

            # Extract bed intensities
            for i in range(width):
                if np.isfinite(bed_picks[i]):
                    row = int(round(bed_picks[i]))
                    if 0 <= row < radar_data.shape[0]:
                        bed_intensity[i] = radar_data[row, i]

            print("INFO: Extracted intensities from radar image for both layers")
        else:
            print("INFO: No radar image available for intensity extraction")

        # Convert surface and bed picks to time units (μs)
        surface_time_us = np.full(width, np.nan)
        bed_time_us = np.full(width, np.nan)

        if hasattr(processor, "pixels_per_microsecond") and hasattr(
            processor, "transmitter_pulse_y_abs"
        ):
            # Convert to time relative to transmitter pulse
            if processor.pixels_per_microsecond > 0:
                surface_time_us = (
                    (surface_picks - processor.transmitter_pulse_y_abs)
                    / processor.pixels_per_microsecond
                    / 2.0
                )
                bed_time_us = (
                    (bed_picks - processor.transmitter_pulse_y_abs)
                    / processor.pixels_per_microsecond
                    / 2.0
                )
        else:
            # Fallback: use pixel coordinates
            surface_time_us = surface_picks.copy()
            bed_time_us = bed_picks.copy()

        # Calculate ice thickness (in meters)
        ice_thickness = np.full(width, np.nan)
        valid = np.isfinite(surface_time_us) & np.isfinite(bed_time_us)
        if np.any(valid):
            # Time difference in microseconds (two-way travel time)
            two_way_time_us = bed_time_us[valid] - surface_time_us[valid]

            if hasattr(processor, "physics_constants"):
                # Get physics constants
                c0 = processor.physics_constants.get(
                    "speed_of_light_vacuum_mps", 299792458
                )  # m/s
                epsilon_r = processor.physics_constants.get(
                    "ice_relative_permittivity_real", 3.17
                )
                firn_corr = processor.physics_constants.get(
                    "firn_correction_meters", 0.0
                )

                # Convert microseconds to seconds
                two_way_time_s = two_way_time_us * 1e-6

                # Calculate ice thickness using correct formula
                # Ice thickness = (c * t) / (2 * sqrt(epsilon_r)) - firn_correction
                # where t is two-way travel time, c is speed of light in vacuum
                ice_thickness[valid] = (c0 * two_way_time_s) / (
                    2 * np.sqrt(epsilon_r)
                ) - firn_corr

                print(
                    f"INFO: Calculated ice thickness using physics constants (c0={c0}, εr={epsilon_r})"
                )
            else:
                # Fallback: use a simple approximation if physics constants not available
                # Approximate speed in ice: ~168 m/μs (c0/sqrt(3.17)/2 ≈ 168e6 μm/s = 168 m/μs)
                ice_speed_m_per_us = 168.0
                ice_thickness[valid] = two_way_time_us * ice_speed_m_per_us / 2.0
                print(
                    "INFO: Used approximate ice thickness calculation (no physics constants)"
                )

        # --- CBD MAPPING ---
        if (
            nav_df is not None
            and cbd_tick_positions is not None
            and len(cbd_tick_positions) > 0
        ):
            # Extract expected CBD range from filename
            import re

            cbd_match = re.search(r"C(\d+)_(\d+)", processor.base_filename)
            if cbd_match:
                cbd_start = int(cbd_match.group(1))
                cbd_end = int(cbd_match.group(2))

                # Create CBD sequence: left-to-right on image (descending)
                expected_cbd_sequence = list(range(cbd_end, cbd_start - 1, -1))

                print(
                    f"INFO: Expected CBD sequence: {expected_cbd_sequence[0]} to {expected_cbd_sequence[-1]}"
                )

                # Initialize CBD array
                cbd_array = np.full(width, np.nan)

                # Map CBD numbers to spatial positions
                if len(cbd_tick_positions) == len(expected_cbd_sequence):
                    # Direct mapping: each tick position gets its corresponding CBD number
                    for i, (tick_pos, cbd_num) in enumerate(
                        zip(cbd_tick_positions, expected_cbd_sequence)
                    ):
                        if 0 <= int(tick_pos) < width:
                            cbd_array[int(tick_pos)] = cbd_num

                    # Interpolate CBD values between tick marks
                    tick_positions_int = [
                        int(pos) for pos in cbd_tick_positions if 0 <= int(pos) < width
                    ]
                    if len(tick_positions_int) > 1:
                        # Interpolate CBD values across the full width
                        cbd_interp = np.interp(
                            np.arange(width),
                            tick_positions_int,
                            expected_cbd_sequence[: len(tick_positions_int)],
                        )
                        cbd_array = cbd_interp
                else:
                    print(
                        f"WARNING: Mismatch between tick positions ({len(cbd_tick_positions)}) and expected CBDs ({len(expected_cbd_sequence)})"
                    )
                    # Fallback: just mark the tick positions
                    for i, tick_pos in enumerate(cbd_tick_positions):
                        if (
                            i < len(expected_cbd_sequence)
                            and 0 <= int(tick_pos) < width
                        ):
                            cbd_array[int(tick_pos)] = expected_cbd_sequence[i]

                print(
                    f"INFO: Assigned CBD values from {np.nanmin(cbd_array):.0f} to {np.nanmax(cbd_array):.0f}"
                )
                print(f"INFO: CBD coverage: {np.sum(~np.isnan(cbd_array))} pixels")

                # Use the corrected CBD array instead of enhanced_df['CBD']
                corrected_cbd = cbd_array
            else:
                print("WARNING: Could not extract CBD range from filename")
                corrected_cbd = np.full(width, np.nan)
        else:
            print("INFO: No navigation data or CBD tick positions available")
            corrected_cbd = np.full(width, np.nan)

        # Calculate transmitter pulse data
        transmitter_depth_us = np.full(width, np.nan)
        transmitter_intensity = np.full(width, np.nan)
        transmitter_y_pixel = np.full(width, np.nan)

        if (
            hasattr(processor, "transmitter_pulse_intensities")
            and processor.transmitter_pulse_intensities is not None
        ):
            transmitter_intensity = processor.transmitter_pulse_intensities
            transmitter_depth_us = np.zeros(width)  # Transmitter is at time zero
            transmitter_y_pixel = np.full(width, processor.transmitter_pulse_y_abs)
            print(
                f"INFO: Including transmitter pulse data - intensity range: {np.min(transmitter_intensity):.1f} - {np.max(transmitter_intensity):.1f}"
            )
        else:
            print("WARNING: No transmitter pulse data available for database export")

        # Create the new database DataFrame with corrected CBD mapping
        database_df = pd.DataFrame(
            {
                "X (pixel)": np.arange(width),
                "Latitude": np.full(width, np.nan),
                "Longitude": np.full(width, np.nan),
                "CBD": corrected_cbd,  # Use corrected CBD mapping
                "Surface Depth (μs)": surface_time_us,
                "Surface Intensity": surface_intensity,
                "Bed Depth (μs)": bed_time_us,
                "Bed Intensity": bed_intensity,
                "Ice Thickness (m)": ice_thickness,
                "Transmitter Depth (μs)": transmitter_depth_us,
                "Transmitter Intensity": transmitter_intensity,
                "Transmitter Y_pixel": transmitter_y_pixel,
            }
        )

        # Export CSV
        csv_path = Path(output_dir) / f"{processor.base_filename}_database.csv"
        database_df.to_csv(csv_path, index=False, float_format="%.6f", na_rep="NaN")
        print(f"INFO: Exported CSV database to {csv_path}")

        # Export NPZ
        npz_path = Path(output_dir) / f"{processor.base_filename}_database.npz"

        # Prepare metadata
        meta_info = {
            "export_timestamp": str(datetime.datetime.now()),
            "filename": processor.base_filename,
            "column_order": [
                "X (pixel)",
                "Latitude",
                "Longitude",
                "CBD",
                "Surface Depth (μs)",
                "Surface Intensity",
                "Bed Depth (μs)",
                "Bed Intensity",
                "Ice Thickness (m)",
                "Transmitter Depth (μs)",
                "Transmitter Intensity",
                "Transmitter Y_pixel",
            ],
            "total_surface_picks": int(np.sum(np.isfinite(surface_time_us))),
            "total_bed_picks": int(np.sum(np.isfinite(bed_time_us))),
            "total_transmitter_data": int(np.sum(np.isfinite(transmitter_intensity))),
            "transmitter_y_pixel": float(processor.transmitter_pulse_y_abs)
            if hasattr(processor, "transmitter_pulse_y_abs")
            and processor.transmitter_pulse_y_abs is not None
            else np.nan,
            "coordinate_coverage": int(np.sum(~pd.isna(database_df["Latitude"]))),
            "cbd_markers": int(np.sum(~pd.isna(database_df["CBD"]))),
        }

        np.savez(
            npz_path,
            x_pixel=database_df["X (pixel)"].values,
            latitude=database_df["Latitude"].values,
            longitude=database_df["Longitude"].values,
            cbd=database_df["CBD"].values,
            surface_depth_us=database_df["Surface Depth (μs)"].values,
            surface_intensity=database_df["Surface Intensity"].values,
            bed_depth_us=database_df["Bed Depth (μs)"].values,
            bed_intensity=database_df["Bed Intensity"].values,
            ice_thickness_m=database_df["Ice Thickness (m)"].values,
            transmitter_depth_us=database_df["Transmitter Depth (μs)"].values,
            transmitter_intensity=database_df["Transmitter Intensity"].values,
            transmitter_y_pixel=database_df["Transmitter Y_pixel"].values,
            meta=meta_info,
        )
        print(f"INFO: Exported NPZ database to {npz_path}")

        # Display summary statistics
        coord_coverage = np.sum(~pd.isna(database_df["Latitude"]))
        print(f"INFO: Coordinate coverage: {coord_coverage}/{len(database_df)} pixels")

        valid_thickness = np.sum(~pd.isna(database_df["Ice Thickness (m)"]))
        print(
            f"INFO: Valid ice thickness measurements: {valid_thickness}/{len(database_df)} pixels"
        )

        valid_surface = np.sum(np.isfinite(database_df["Surface Depth (μs)"]))
        print(f"INFO: Valid surface picks: {valid_surface}/{len(database_df)} pixels")

        valid_bed = np.sum(np.isfinite(database_df["Bed Depth (μs)"]))
        print(f"INFO: Valid bed picks: {valid_bed}/{len(database_df)} pixels")

        print("SUCCESS: Database export (CSV + NPZ) completed successfully")
        return True

    except Exception as e:
        print(f"ERROR: Database export failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        print("=" * 60)


def process_flight_batch(
    flight_dir, output_dir, processor, nav_path, approx_x_pip=None
):
    """
    Batch process all .tiff files in a flight directory with enhanced database export.
    For each file, user can select a new calpip or reuse the last.
    """
    # Set batch mode flag to suppress interactive displays
    processor.config["echo_tracing_params"]["batch_mode"] = True

    tiff_files = sorted(glob.glob(str(Path(flight_dir) / "*.tiff")))
    if not tiff_files:
        print(f"ERROR: No .tiff files found in {flight_dir}")
        return

    # Load navigation data once for the entire batch
    nav_df = None
    if nav_path and Path(nav_path).exists():
        try:
            nav_df = pd.read_csv(nav_path)
            print(
                f"INFO: Loaded navigation data with {len(nav_df)} records for batch processing"
            )
        except Exception as e:
            print(f"WARNING: Could not load navigation file for batch: {e}")
            nav_df = None
    else:
        print("WARNING: No navigation file specified for batch processing")

    last_x_pip = approx_x_pip
    successful_exports = 0
    failed_exports = 0

    for idx, tiff_path in enumerate(tiff_files):
        print(f"\n{'=' * 80}")
        print(f"Processing file {idx + 1}/{len(tiff_files)}: {tiff_path}")
        print(f"{'=' * 80}")
        file_name = Path(tiff_path).name

        while True:
            user_input = (
                input(
                    f"Select new calpip for {file_name}? (y = select, n = reuse last, q = quit): "
                )
                .strip()
                .lower()
            )
            if user_input == "q":
                print("Batch processing aborted by user.")
                print(
                    f"BATCH SUMMARY: {successful_exports} successful exports, {failed_exports} failed exports"
                )
                return
            if user_input == "y":
                temp_image = load_and_preprocess_image(
                    tiff_path, processor.config.get("preprocessing_params", {})
                )
                if temp_image is None:
                    print(
                        f"ERROR: Could not load image {tiff_path} for calpip selection."
                    )
                    continue
                selector_title = f"Select calpip for: {file_name}"
                selector = ClickSelector(temp_image, title=selector_title)
                last_x_pip = selector.selected_x
                if last_x_pip is None:
                    print("No calpip selected. Skipping this file.")
                    break
                else:
                    break
            elif user_input == "n":
                if last_x_pip is None:
                    print("No previous calpip available. Please select one.")
                    continue
                print(f"Reusing last calpip X-pixel: {last_x_pip}")
                break
            else:
                print("Invalid input. Please enter 'y', 'n', or 'q'.")

        if last_x_pip is not None:
            # Process the image
            processing_success = processor.process_image(
                tiff_path, output_dir, last_x_pip, nav_df=nav_df, nav_path=nav_path
            )

            if processing_success:
                # Export database (CSV + NPZ)
                export_success = export_database(processor, Path(output_dir), nav_df)

                if export_success:
                    successful_exports += 1
                else:
                    failed_exports += 1
            else:
                print(f"ERROR: Processing failed for {file_name}")
                failed_exports += 1

    print(f"\n{'=' * 80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"Total files processed: {len(tiff_files)}")
    print(f"Successful database exports: {successful_exports}")
    print(f"Failed database exports: {failed_exports}")
    print(
        f"Success rate: {(successful_exports / (successful_exports + failed_exports) * 100):.1f}%"
        if (successful_exports + failed_exports) > 0
        else "N/A"
    )
    print(f"{'=' * 80}")


def run_processing():
    """
    Main function to parse arguments and run the Z-scope processing workflow.
    """
    parser = argparse.ArgumentParser(
        description="Process Z-scope radar film images from raw image to calibrated data display with enhanced database export.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "image_path",
        type=str,
        nargs="?",
        help="Path to the Z-scope image file (e.g., .tif, .png, .jpg).",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        nargs="?",
        default="output",
        help="Directory where all output files (plots, data) will be saved.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.json",
        help="Path to the JSON file containing processing parameters.",
    )
    parser.add_argument(
        "--physics",
        type=str,
        default="config/physical_constants.json",
        help="Path to the JSON file containing physical constants for calibration.",
    )
    parser.add_argument(
        "--non_interactive_pip_x",
        type=int,
        default=None,
        help="Specify the approximate X-coordinate for the calibration pip non-interactively. "
        "If provided, the ClickSelector GUI will be skipped.",
    )
    parser.add_argument(
        "--batch_dir",
        type=str,
        default=None,
        help="If set, process all .tiff files in this directory sequentially (batch mode).",
    )
    parser.add_argument(
        "--nav_file",
        type=str,
        default=None,
        help="Path to merged navigation CSV (e.g., merged_103_nav.csv) for coordinate interpolation.",
    )

    args = parser.parse_args()

    SCRIPT_DIR = Path(__file__).resolve().parent
    output_path_obj = Path(args.output_dir)
    if not output_path_obj.is_absolute():
        final_output_dir = SCRIPT_DIR / output_path_obj
    else:
        final_output_dir = output_path_obj

    final_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"INFO: Output will be saved to: {final_output_dir.resolve()}")

    try:
        processor = ZScopeProcessor(config_path=args.config, physics_path=args.physics)
    except Exception as e:
        print(f"ERROR: Failed to initialize ZScopeProcessor: {e}")
        sys.exit(1)

    # --- Batch Mode Logic ---
    if args.batch_dir is not None and args.nav_file is not None:
        print(f"\nINFO: Starting batch processing for directory: {args.batch_dir}")
        print(
            f"INFO: Enhanced database export (CSV + NPZ) will be performed for each image"
        )
        approx_x_pip_selected = args.non_interactive_pip_x

        if approx_x_pip_selected is None:
            tiff_files = sorted(glob.glob(str(Path(args.batch_dir) / "*.tiff")))
            if not tiff_files:
                print(f"ERROR: No .tiff files found in {args.batch_dir}")
                sys.exit(1)

            first_image_path = tiff_files[0]
            print(
                "\nINFO: Preparing for interactive calibration pip selection (batch mode, first file)..."
            )
            temp_image_for_selector = load_and_preprocess_image(
                first_image_path, processor.config.get("preprocessing_params", {})
            )
            if temp_image_for_selector is None:
                print(
                    f"ERROR: Failed to load image '{first_image_path}' for pip selection. Exiting."
                )
                sys.exit(1)

            print(
                "INFO: Please click on the approximate vertical location of the calibration pip ticks in the displayed image."
            )
            file_name = Path(first_image_path).name
            selector_title = f"Select calpip for: {file_name}"
            selector = ClickSelector(temp_image_for_selector, title=selector_title)
            approx_x_pip_selected = selector.selected_x

            if approx_x_pip_selected is None:
                print(
                    "ERROR: No location selected for calibration pip via ClickSelector. Exiting."
                )
                sys.exit(1)

            print(
                f"INFO: User selected approximate X-coordinate for calibration pip: {approx_x_pip_selected}"
            )
        else:
            print(
                f"INFO: Using non-interactive X-coordinate for calibration pip: {approx_x_pip_selected}"
            )

        process_flight_batch(
            args.batch_dir,
            str(final_output_dir.resolve()),
            processor,
            args.nav_file,
            approx_x_pip_selected,
        )
        print("\nINFO: Batch processing completed.")
        sys.exit(0)

    # --- Single Image Mode ---
    if args.image_path is None:
        print("ERROR: No image_path provided and not in batch mode. Exiting.")
        sys.exit(1)

    print(f"\nINFO: Single image processing mode")
    print(f"INFO: Enhanced database export (CSV + NPZ) will be performed automatically")

    approx_x_pip_selected = args.non_interactive_pip_x
    if approx_x_pip_selected is None:
        print("\nINFO: Preparing for interactive calibration pip selection...")
        temp_image_for_selector = load_and_preprocess_image(
            args.image_path, processor.config.get("preprocessing_params", {})
        )
        if temp_image_for_selector is None:
            print(
                f"ERROR: Failed to load image '{args.image_path}' for pip selection. Exiting."
            )
            sys.exit(1)

        print(
            "INFO: Please click on the approximate vertical location of the calibration pip ticks in the displayed image."
        )
        selector_title = processor.config.get("click_selector_params", {}).get(
            "title", "Click on the calibration pip column"
        )
        selector = ClickSelector(temp_image_for_selector, title=selector_title)
        approx_x_pip_selected = selector.selected_x

        if approx_x_pip_selected is None:
            print(
                "ERROR: No location selected for calibration pip via ClickSelector. Exiting."
            )
            sys.exit(1)

        print(
            f"INFO: User selected approximate X-coordinate for calibration pip: {approx_x_pip_selected}"
        )
    else:
        print(
            f"INFO: Using non-interactive X-coordinate for calibration pip: {approx_x_pip_selected}"
        )

    print(f"\nINFO: Starting main processing for image: {args.image_path}")

    # Load navigation data BEFORE processing for enhanced x-axis labels
    nav_df = None
    if args.nav_file and Path(args.nav_file).exists():
        try:
            nav_df = pd.read_csv(args.nav_file)
            print(
                f"INFO: Pre-loaded navigation data with {len(nav_df)} records for enhanced visualization"
            )
        except Exception as e:
            print(f"WARNING: Could not pre-load navigation file: {e}")
            nav_df = None

    processing_successful = processor.process_image(
        args.image_path,
        str(final_output_dir.resolve()),
        approx_x_pip_selected,
        nav_df=nav_df,
        nav_path=args.nav_file,
    )

    if not processing_successful:
        print("ERROR: Z-scope image processing failed. Check logs for details.")
        sys.exit(1)

    print("\nINFO: Core processing completed successfully.")

    # Load navigation data for database export
    nav_df = None
    if args.nav_file and Path(args.nav_file).exists():
        try:
            nav_df = pd.read_csv(args.nav_file)
            print(f"INFO: Loaded navigation data with {len(nav_df)} records")
        except Exception as e:
            print(f"WARNING: Could not load navigation file: {e}")
            nav_df = None
    else:
        print("INFO: No navigation file specified - coordinates will be NaN")

    # Export database (CSV + NPZ)
    export_success = export_database(processor, final_output_dir, nav_df)

    if not export_success:
        print(
            "WARNING: Database export failed, but basic processing completed successfully"
        )

    # Continue with existing plotting functionality
    if processor.calibrated_fig and processor.calibrated_ax:
        print(
            "\nINFO: Plotting automatically detected echoes on the calibrated Z-scope image..."
        )

        if (
            processor.image_np is not None
            and processor.data_top_abs is not None
            and processor.data_bottom_abs is not None
            and processor.data_top_abs < processor.data_bottom_abs
        ):
            num_cols = processor.image_np[
                processor.data_top_abs : processor.data_bottom_abs, :
            ].shape[1]
            x_plot_coords = np.arange(num_cols)

            echo_plot_config = processor.config.get("echo_tracing_params", {})
            surface_plot_params = echo_plot_config.get("surface_detection", {})
            bed_plot_params = echo_plot_config.get("bed_detection", {})

            if processor.detected_surface_y_abs is not None and np.any(
                np.isfinite(processor.detected_surface_y_abs)
            ):
                surface_y_cropped = (
                    processor.detected_surface_y_abs - processor.data_top_abs
                )
                valid_indices = np.isfinite(surface_y_cropped)
                if np.any(valid_indices):
                    processor.calibrated_ax.plot(
                        x_plot_coords[valid_indices],
                        surface_y_cropped[valid_indices],
                        color=surface_plot_params.get("plot_color", "cyan"),
                        linestyle=surface_plot_params.get("plot_linestyle", "-"),
                        linewidth=1.5,
                        label="Auto Surface Echo",
                    )
                    print("INFO: Plotted automatically detected surface echo.")
                else:
                    print("INFO: No valid automatic surface echo trace to plot.")

            if processor.detected_bed_y_abs is not None and np.any(
                np.isfinite(processor.detected_bed_y_abs)
            ):
                bed_y_cropped = processor.detected_bed_y_abs - processor.data_top_abs
                valid_indices = np.isfinite(bed_y_cropped)
                if np.any(valid_indices):
                    processor.calibrated_ax.plot(
                        x_plot_coords[valid_indices],
                        bed_y_cropped[valid_indices],
                        color=bed_plot_params.get("plot_color", "lime"),
                        linestyle=bed_plot_params.get("plot_linestyle", "-"),
                        linewidth=1.5,
                        label="Auto Bed Echo",
                    )
                    print("INFO: Plotted automatically detected bed echo.")
                else:
                    print("INFO: No valid automatic bed echo trace to plot.")

            time_vis_params = processor.config.get(
                "time_calibration_visualization_params", {}
            )
            processor.calibrated_ax.legend(
                loc=time_vis_params.get("legend_location", "upper right"),
                fontsize="small",
            )

            auto_echo_plot_filename = f"{processor.base_filename}_auto_echoes.png"
            auto_echo_plot_path = final_output_dir / auto_echo_plot_filename

            output_params_config = processor.config.get("output_params", {})
            save_dpi = output_params_config.get(
                "annotated_figure_save_dpi",
                output_params_config.get("figure_save_dpi", 300),
            )

            try:
                processor.calibrated_fig.savefig(
                    auto_echo_plot_path, dpi=save_dpi, bbox_inches="tight"
                )
                print(
                    f"INFO: Plot with auto-detected echoes saved to: {auto_echo_plot_path}"
                )
            except Exception as e:
                print(f"ERROR: Could not save plot with auto-detected echoes: {e}")
        else:
            print(
                "WARNING: Cannot plot echoes because prerequisite image data is missing from processor."
            )
    elif processing_successful:
        print(
            "WARNING: Core processing completed, but calibrated plot figure/axes are not available for final display."
        )

    print("\n--- Z-scope Processing Script Finished ---")
    print(f"INFO: Output files saved to: {final_output_dir.resolve()}")
    print(f"INFO: Enhanced Database CSV: {processor.base_filename}_database.csv")
    print(f"INFO: Enhanced Database NPZ: {processor.base_filename}_database.npz")
    sys.exit(0)


if __name__ == "__main__":
    run_processing()
