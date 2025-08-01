# A-Scope Processor

**Automated detection and analysis of A-scope radar data from TIFF images.**

This package provides tools for processing A-scope radar data, including detection of signal traces, reference lines, transmitter pulses, surface echoes, and bed echoes. Ideal for researchers and engineers working with radar data from the Ross Ice Shelf and similar environments.

---

## 🚀 Quick Start

Get started with the A-scope processor in just a few steps:

### 1. Clone the Repository

`git clone https://github.com/tarzona/FrozenLegacies.git`

`cd FrozenLegacies/A_Scope_Processing`


### 2. Install Dependencies

`pip install -r docs/requirements.txt`

*(If you don’t have `pip` installed, see [Python’s official guide](https://pip.pypa.io/en/stable/installation/).)*


### 3. Run the Processor

Use the command line interface to process your A-scope TIFF images:
#### Process a specific image

`python main.py --input path/to/your/image.tiff`

#### Reprocess selected frame with manual picker 
`python main.py --input path/to/your/image.tiff --interactive frame_num`  

- Example usage for re-running frame 04 for AScope file F103-C0467_0479.tiff: 
`python main.py --input data/103/F103-C0467_0479.tiff --interactive 4`


---

## ⚙️ Configuration

- **Default configuration:** `config/default_config.json`

Override defaults by providing a custom configuration file with the `--config` option.

---

## 📂 Output

Processed results are saved to the output directory specified in the configuration (default: `output/`).

---

## 📚 Additional Resources

- **Sample Data:** [Download sample TIFF images here](#)
- **Documentation:** [Detailed documentation](#)
- **Contributing:** See [CONTRIBUTING.md](CONTRIBUTING.md)
- **Citation:** [How to cite this software](#)

---

## ❓ Need Help?

Open an [issue](https://github.com/tarzona/FrozenLegacies/issues) or contact the project maintainers.




