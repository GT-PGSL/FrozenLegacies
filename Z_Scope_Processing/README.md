<p align="left">
  <img src="docs/logo-echo-explore-combine-wbg.png" alt="Z_Scope_Processor Logo" height="120">
  <span style="font-size:2em; vertical-align: middle;">
</p>

# Z-Scope Processor

**Z-Scope Processor** is a Python package for loading, analyzing, interpreting, and labeling historical Z-scope radar sounding images (echograms) collected in the 1970s SPRI/NSF/TUD airborne radar surveys across Antarctica.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Running from VSCode](#running-from-vscode)
  - [Command Line Usage](#command-line-usage)
- [Configuration](#configuration)
- [Output](#output)
  
---

## Features

- Load and preprocess historical Z-scope radar echograms
- Detect film artifact boundaries and transmitter pulse
- Interactive or automated calibration pip detection
- Automatic surface and bed echo tracing
- Time-calibrated visualization and data export
- Save all identified ice surface/bed depths and travel times at each pixel
- Modular functions for artifact detection, calibration, feature detection, and visualization


---

## Requirements

- Python 3.11 or 3.12 recommended
- [VSCode](https://code.visualstudio.com/) (for development)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- Other dependencies as specified in `requirements.txt` 

---

## Installation

1. Clone the repository:
```
git clone https://github.com/GT-PGSL/FrozenLegacies.git
cd FrozenLegacies/Z_Scope_Processing
```

2. (Optional but recommended) Create and activate a virtual environment:
```
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

---

## 📖 Usage

### Single Image Processing

Process a single Z-scope image:

  ```
  python main.py <image_path> <output_dir> --nav_file <navigation_file>
  ```

  Replace `<image_path>` with your radar image file (e.g., `.tiff`, `.png`, `.jpg`), `<output_dir>` with the desired output directory, and `<navigation_file>` with the flight track navigation file (e.g., `.csv`). 
  
  Example:

  ```
  python main.py data/echogram_001.tiff output/ --nav_file 001_nav.csv
  ```

### Batch Processing

Process multiple images in a directory:
```
python main.py --batch_dir <input dir> --nav_file <navigation_file> <output_dir>
```
Replace `--batch_dir` with the folder directory containing multiple .tiff files for batch processing. 

---

## Configuration

- **Processing parameters**: `config/default_config.json`
- **Physical constants**: `config/physical_constants.json`

You can customize detection thresholds, physical constants, and output settings by editing these files.

---
## 🔧 Core Processing Pipeline

### 1. Image Preprocessing
- 16-bit to 8-bit TIFF conversion with percentile normalization
- CLAHE (Contrast Limited Adaptive Histogram Equalization) enhancement
- Film artifact and sprocket hole detection and removal

### 2. Feature Detection
- **Transmitter Pulse Detection**: Locates radar pulse reference point for time calibration
- **Calibration Pip Detection**: Finds vertical tick marks for time-to-depth conversion
- **Enhanced CBD Tick Detection**: Interactive selection with local image recognition refinement

### 3. Echo Tracing
- **Automated Surface Detection**: Ice surface interface identification
- **Automated Bed Detection**: Ice bed interface identification
- **Quality Validation**: Automatic validation of detected echoes

### 4. Data Export
- **Time Calibration**: Accurate one-way travel time calculations
- **Coordinate Interpolation**: Full-resolution lat/lon interpolation using navigation data
- **Physical Unit Conversion**: Ice thickness calculations in meters using proper electromagnetic wave propagation

--- 
## 📊 Output Files

For each processed image, the system generates:
   
1. **Ice Thickness CSV**: `{filename}_thickness.csv`
  - X (pixel): Horizontal pixel index
  - Latitude: Interpolated Bingham coordinates
  - Longitude: Interpolated Bingham coordinates  
  - CBD: Control Block Distance (where available)
  - Surface Depth (μs): One-way travel time to ice surface
  - Bed Depth (μs): One-way travel time to ice bed
  - Ice Thickness (m): Calculated thickness in meters
   
2. **Visualization Plots**:
   - `{filename}_picked.png`: Main calibrated echogram with CBD labels
   - `{filename}_time_calibrated_auto_echoes.png`: Plot with automatically detected echoes
   - `{filename}_enhanced_local_refinement_validation.png`: CBD tick detection validation

3. **Debug Output**: Detailed processing logs and intermediate results

--- 

## 📁 Project Structure

Recommended project structure to run this Python Package: 

Z_Scope_Processing/
├── main.py # Main processing script
├── zscope_processor.py # Core processing engine
├── functions/
│ ├── visualization_utils.py # Enhanced CBD detection & visualization
│ ├── image_utils.py # Image loading and preprocessing
│ ├── artifact_detection.py # Film boundary detection
│ ├── feature_detection.py # Pulse and pip detection
│ ├── calibration_utils.py # Time calibration functions
│ ├── echo_tracing.py # Surface/bed echo detection
│ └── interactive_tools.py # GUI selection tools
├── config/
│ ├── default_config.json # Processing parameters
│ └── physical_constants.json # Physical constants
└── README.md

---

*Part of the FrozenLegacies project - preserving and analyzing historical Antarctic radar data for climate science research.*

