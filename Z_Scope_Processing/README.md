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

## Usage

### Running from VSCode

1. **Open the Project**  
Launch VSCode and open the `Z_Scope_Processing` folder.

2. **Select Python Interpreter**  
Press `Cmd+Shift+P`, type `Python: Select Interpreter`, and choose your Python 3.11/3.12 environment.

3. **Run `main.py`**  
- Open `main.py` in the editor.
- Click the green “Run Python File” play button at the top right,  
  **OR**  
- Open the integrated terminal (`Terminal > New Terminal`) and run:

  ```
  python main.py <image_path> <output_dir>
  ```

  Replace `<image_path>` with your radar image file (e.g., `.tiff`, `.png`, `.jpg`) and `<output_dir>` with the desired output directory.  
  Example:

  ```
  python main.py data/echogram_001.tiff output/
  ```

4. **Interactive Calibration**  
If you do not specify `--non_interactive_pip_x`, you will be prompted to click on the calibration pip in the displayed image.

5. **Debugging**  
- To run in debug mode, click the "Run and Debug" icon in the sidebar and select "Python File".

### Command Line Usage

You can also run the processor directly from the terminal:
```
python main.py <image_path> <output_dir> [options]
```
**Options:**
- `--config <path>`: Path to processing configuration JSON (default: `config/default_config.json`)
- `--physics <path>`: Path to physical constants JSON (default: `config/physical_constants.json`)
- `--non_interactive_pip_x <int>`: Approximate X-coordinate for calibration pip (skips GUI)

---

## Configuration

- **Processing parameters**: `config/default_config.json`
- **Physical constants**: `config/physical_constants.json`

You can customize detection thresholds, physical constants, and output settings by editing these files.

---

## Output

- **Plots**: Time-calibrated echogram images with detected surface and bed echoes
- **Data Export**:  
  All identified ice surface and bed depths (meters), one-way travel times (microseconds), and ice thickness at each pixel are saved as `.csv` or `.txt` files in the output directory.

---

