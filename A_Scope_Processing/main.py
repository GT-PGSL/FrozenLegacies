#!/usr/bin/env python3

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Get the directory of the current script (main.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the current directory to Python path for imports
sys.path.insert(0, current_dir)

from ascope_processor import AScope


def main():
    """
    Main function to process A-scope radar data with enhanced functionality.
    Features:
    - Ice thickness calculation using standard ice velocity (168 m/μs)
    - Automatic CSV and NPZ export with frame-by-frame data
    - CBD assignment from filename parsing
    - Enhanced error handling and validation
    """
    parser = argparse.ArgumentParser(
        description="Process A-scope radar data with enhanced ice thickness calculation and export.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input F103-C0467_0479.tiff
  python main.py --input F103-C0467_0479.tiff --output ./results
  python main.py --input F103-C0467_0479.tiff --config custom_config.json --output ./results

Output files:
  - Frame verification: {filename}_frame_verification.png
  - Individual frame plots: {filename}_frame{XX}_picked.png  
  - Grid QA plots: {filename}_frame{XX}_grid_QA.png
  - CSV database: {filename}_pick.csv
  - NPZ database: {filename}_pick.npz

CSV columns:
  Frame, CBD, Surface_Time_us, Bed_Time_us, Ice_Thickness_m, Surface_Power_dB, Bed_Power_dB
        """,
    )

    # Required arguments
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input A-scope TIFF image file (e.g., F103-C0467_0479.tiff)",
    )

    # Optional arguments
    parser.add_argument(
        "--output", default=None, help="Path to output directory (default: ./output/)"
    )

    # Set default config path relative to main.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, "config", "default_config.json")

    parser.add_argument(
        "--config",
        default=default_config,
        help=f"Path to configuration file (default: {default_config})",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for additional outputs and verbose logging",
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    if not input_path.suffix.lower() in [".tiff", ".tif"]:
        print(f"WARNING: Input file doesn't appear to be a TIFF: {input_path}")
        print("Proceeding anyway...")

    # Validate config file
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Configuration file not found: {config_path}")
        sys.exit(1)

    # Set up output directory
    if args.output:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"INFO: Using output directory: {output_path.resolve()}")
    else:
        output_path = None
        print("INFO: Using default output directory from config")

    # Display processing information
    print("=" * 60)
    print("A-SCOPE RADAR DATA PROCESSOR")
    print("Enhanced with Ice Thickness Calculation & Data Export")
    print("=" * 60)
    print(f"Input file: {input_path.resolve()}")
    print(f"Config file: {config_path.resolve()}")
    if output_path:
        print(f"Output directory: {output_path.resolve()}")
    print(f"Debug mode: {'Enabled' if args.debug else 'Disabled'}")

    # Extract filename info for validation
    filename = input_path.stem
    import re

    cbd_match = re.search(r"C(\d+)_(\d+)", filename)
    if cbd_match:
        cbd_start = int(cbd_match.group(1))
        cbd_end = int(cbd_match.group(2))
        expected_frames = abs(cbd_end - cbd_start) + 1
        print(
            f"Expected CBD range: {cbd_start} to {cbd_end} ({expected_frames} frames)"
        )
    else:
        print("WARNING: Could not extract CBD range from filename")
        print("Expected filename format: FXX-CXXXX_XXXX.tiff")

    print("=" * 60)

    try:
        # Create processor instance
        print("INFO: Initializing A-scope processor...")
        processor = AScope(str(config_path.resolve()))

        # Set debug mode if requested
        if args.debug:
            processor.set_debug_mode(True)
            print("INFO: Debug mode enabled - additional outputs will be generated")

        # Set output directory if specified
        if output_path:
            processor.set_output_directory(str(output_path.resolve()))

        # Process the image
        print(f"INFO: Starting processing of {input_path.name}...")
        processor.process_image(
            str(input_path.resolve()),
            str(output_path.resolve()) if output_path else None,
        )

        # Display completion summary
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        # Show output files
        output_dir = processor.output_dir
        base_name = filename

        expected_outputs = [
            f"{base_name}_frame_verification.png",
            f"{base_name}_pick.csv",
            f"{base_name}_pick.npz",
        ]

        print("Generated output files:")
        for output_file in expected_outputs:
            output_file_path = Path(output_dir) / output_file
            if output_file_path.exists():
                print(f"  ✅ {output_file}")
            else:
                print(f"  ❌ {output_file} (not found)")

        # Show frame-specific outputs
        if hasattr(processor, "frame_results") and processor.frame_results:
            frame_count = len(processor.frame_results)
            print(f"\nFrame-specific outputs ({frame_count} frames):")
            for i in range(1, frame_count + 1):
                frame_picked = f"{base_name}_frame{i:02d}_picked.png"
                frame_grid = f"{base_name}_frame{i:02d}_grid_QA.png"

                frame_picked_path = Path(output_dir) / frame_picked
                frame_grid_path = Path(output_dir) / frame_grid

                if frame_picked_path.exists():
                    print(f"  ✅ {frame_picked}")
                if frame_grid_path.exists():
                    print(f"  ✅ {frame_grid}")

        print(f"\nAll outputs saved to: {output_dir}")

        # Processing statistics
        if hasattr(processor, "frame_results") and processor.frame_results:
            results = processor.frame_results
            total_frames = len(results)
            valid_surface = sum(
                1 for r in results if not pd.isna(r.get("Surface_Time_us", np.nan))
            )
            valid_bed = sum(
                1 for r in results if not pd.isna(r.get("Bed_Time_us", np.nan))
            )
            valid_thickness = sum(
                1 for r in results if not pd.isna(r.get("Ice_Thickness_m", np.nan))
            )

            print(f"\nProcessing Statistics:")
            print(f"  Total frames processed: {total_frames}")
            print(f"  Frames with surface picks: {valid_surface}")
            print(f"  Frames with bed picks: {valid_bed}")
            print(f"  Frames with ice thickness: {valid_thickness}")

            if valid_thickness > 0:
                try:
                    df = pd.DataFrame(results)
                    thickness_values = df["Ice_Thickness_m"].dropna()
                    if len(thickness_values) > 0:
                        print(
                            f"  Ice thickness range: {thickness_values.min():.1f} - {thickness_values.max():.1f} m"
                        )
                        print(
                            f"  Average ice thickness: {thickness_values.mean():.1f} m"
                        )
                except Exception as e:
                    print(f"  Note: Could not calculate thickness statistics: {e}")
                    # Fallback calculation without pandas
                    thickness_list = [
                        r.get("Ice_Thickness_m", np.nan)
                        for r in results
                        if not pd.isna(r.get("Ice_Thickness_m", np.nan))
                    ]
                    if thickness_list:
                        print(
                            f"  Ice thickness range: {min(thickness_list):.1f} - {max(thickness_list):.1f} m"
                        )
                        print(
                            f"  Average ice thickness: {np.mean(thickness_list):.1f} m"
                        )

        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nINFO: Processing interrupted by user")
        sys.exit(1)

    except FileNotFoundError as e:
        print(f"ERROR: File not found - {e}")
        sys.exit(1)

    except ValueError as e:
        print(f"ERROR: Invalid input - {e}")
        sys.exit(1)

    except Exception as e:
        print(f"ERROR: Processing failed - {e}")
        if args.debug:
            import traceback

            print("\nFull error traceback:")
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
