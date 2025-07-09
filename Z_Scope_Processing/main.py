# zscope_processor/main.py

import argparse
import sys
from pathlib import Path
import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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


def export_enhanced_csv_for_image(processor, output_dir, nav_df=None):
    """
    Export enhanced CSV for a single processed image.

    Args:
        processor: ZScopeProcessor instance
        output_dir: Output directory path
        nav_df: Navigation DataFrame (optional)
    """
    print("\n" + "=" * 60)
    print("INFO: Starting enhanced 7-column CSV export...")

    # Get CBD tick positions from processor if available
    cbd_tick_positions = getattr(processor, "calculated_ticks", None)
    if cbd_tick_positions is None:
        print(
            "INFO: No CBD tick positions found - enhanced CSV will have coordinates as NaN"
        )
    else:
        print(f"INFO: Found {len(cbd_tick_positions)} CBD tick positions")

    # Export enhanced CSV
    try:
        enhanced_df = processor.export_enhanced_csv_with_coordinates(
            str(output_dir), nav_df=nav_df, cbd_tick_xs=cbd_tick_positions
        )

        if enhanced_df is not None:
            print("SUCCESS: Enhanced CSV export completed successfully")
            print(f"INFO: Enhanced CSV contains {len(enhanced_df)} data points")

            # Display summary statistics
            coord_coverage = np.sum(~pd.isna(enhanced_df["Latitude"]))
            print(
                f"INFO: Coordinate coverage: {coord_coverage}/{len(enhanced_df)} pixels"
            )

            valid_thickness = np.sum(~pd.isna(enhanced_df["Ice Thickness (m)"]))
            print(
                f"INFO: Valid ice thickness measurements: {valid_thickness}/{len(enhanced_df)} pixels"
            )

            return True
        else:
            print("WARNING: Enhanced CSV export returned None")
            return False

    except Exception as e:
        print(f"ERROR: Enhanced CSV export failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        print("=" * 60)


def process_flight_batch(
    flight_dir, output_dir, processor, nav_path, approx_x_pip=None
):
    """
    Batch process all .tiff files in a flight directory with enhanced CSV export.
    For each file, user can select a new calpip or reuse the last.
    """
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
                # Export enhanced CSV
                export_success = export_enhanced_csv_for_image(
                    processor, Path(output_dir), nav_df
                )

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
    print(f"Successful enhanced CSV exports: {successful_exports}")
    print(f"Failed enhanced CSV exports: {failed_exports}")
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
        description="Process Z-scope radar film images from raw image to calibrated data display with enhanced CSV export.",
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
        print(f"INFO: Enhanced CSV export will be performed for each image")
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
    print(f"INFO: Enhanced CSV export will be performed automatically")

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
    processing_successful = processor.process_image(
        args.image_path, str(final_output_dir.resolve()), approx_x_pip_selected
    )

    if not processing_successful:
        print("ERROR: Z-scope image processing failed. Check logs for details.")
        sys.exit(1)

    print("\nINFO: Core processing completed successfully.")

    # Load navigation data for enhanced CSV export
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

    # Export enhanced CSV
    export_success = export_enhanced_csv_for_image(processor, final_output_dir, nav_df)

    if not export_success:
        print(
            "WARNING: Enhanced CSV export failed, but basic processing completed successfully"
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

            auto_echo_plot_filename = (
                f"{processor.base_filename}_time_calibrated_auto_echoes.png"
            )
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

        print("\nINFO: Displaying final plot (close window to exit script).")
        plt.show()
    elif processing_successful:
        print(
            "WARNING: Core processing completed, but calibrated plot figure/axes are not available for final display."
        )

    print("\n--- Z-scope Processing Script Finished ---")
    print(f"INFO: Output files saved to: {final_output_dir.resolve()}")
    print(f"INFO: Ice Thickness CSV: {processor.base_filename}_thickness.csv")
    sys.exit(0)


if __name__ == "__main__":
    run_processing()
