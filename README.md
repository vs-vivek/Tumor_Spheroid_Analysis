# Tumor Spheroid Analysis Pipeline

A comprehensive computational pipeline for automated cell segmentation and morphometric analysis of tumor spheroids using deep learning and ellipse fitting.

![Pipeline Overview](https://img.shields.io/badge/Python-3.8+-blue.svg)
![MATLAB](https://img.shields.io/badge/MATLAB-R2020b+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

This repository provides a complete workflow for analyzing cellular morphology in 3D tumor spheroid images. The pipeline combines state-of-the-art deep learning segmentation (CellSAM) with quantitative morphometric analysis to study spatial variations in cell size and shape within spheroids.

### Key Features

- **Automated Batch Processing**: Segment hundreds of multiplex TIFF images with resume capability
- **Deep Learning Segmentation**: Leverages CellSAM for accurate nuclear and whole-cell segmentation
- **Interactive Quality Control**: Manual refinement tool to remove artifacts and over-segmented regions
- **Radial Morphometry**: Quantify cell area and aspect ratio as a function of distance from spheroid center
- **Physical Unit Conversion**: Automatic conversion from pixels to microns with customizable scaling
- **Publication-Ready Outputs**: Generate high-quality visualizations and exportable data tables

## Pipeline Workflow

```
Input TIFF Images (3-channel: nuclear, whole-cell, auxiliary)
         ↓
   [1] CellSAM Batch Segmentation (Python)
         ↓
   Segmentation Masks (.npy + visualizations)
         ↓
   [2] Interactive Mask Refinement (MATLAB)
         ↓
   Cleaned Masks (updated_mask.npy)
         ↓
   [3] Radial Morphometric Analysis (MATLAB)
         ↓
   Quantitative Results (plots + data tables)
```

## Quick Start

### Prerequisites

**Python 3.8+** with:
- CellSAM
- SimpleITK
- NumPy
- Matplotlib
- tqdm

**MATLAB R2020b+** with:
- Image Processing Toolbox
- Computer Vision Toolbox
- [npy-matlab](https://github.com/kwikteam/npy-matlab)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vs-vivek/Tumor_Spheroid_Analysis.git
   cd Tumor_Spheroid_Analysis
   ```

2. **Install Python dependencies:**
   ```bash
   pip install git+https://github.com/vanvalenlab/cellSAM.git
   pip install SimpleITK numpy matplotlib tqdm
   ```

3. **Install MATLAB dependencies:**
   - Download and add [npy-matlab](https://github.com/kwikteam/npy-matlab) to your MATLAB path

### Basic Usage

**Step 1: Batch Segmentation**
```python
# Edit CellSAM_pipeline.py with your paths
input_dir = "path/to/tiff/images"
output_dir = "path/to/output"

# Run the pipeline
python CellSAM_pipeline.py
```

**Step 2: Interactive Refinement**
```matlab
% In MATLAB, copy mask.npy to working directory
Interactive_mask_editor
% Click to delete incorrect cells, right-click when done
```

**Step 3: Morphometric Analysis**
```matlab
% Copy updated_mask.npy to working directory
Area_AR_Analysis
% Click to select spheroid center
```

## Repository Structure

```
Tumor_Spheroid_Analysis/
│
├── Image analysis codes/
│   ├── CellSAM_pipeline.py          # Batch segmentation script
│   ├── Interactive_mask_editor.m    # Manual mask refinement tool
│   ├── Area_AR_Analysis.m           # Radial morphometry analysis
│   └── Instructions.md              # Detailed usage guide
│
└── README.md                         # This file
```

## Input Requirements

### Image Format
- **File type**: TIFF (.tif or .tiff)
- **Image shape**: (3, H, W)
  - Channel 0: Nuclear signal (e.g., DAPI, Hoechst)
  - Channel 1: Whole-cell signal (e.g., membrane marker)
  - Channel 2: Auxiliary channel (ignored by pipeline)

### Recommended Imaging Parameters
- **Resolution**: ≥ 10× magnification
- **Bit depth**: 8-bit or 16-bit
- **Format**: Uncompressed or losslessly compressed TIFF

## Outputs

### Per-Image Outputs (Step 1)
- `mask.npy` - Integer-labeled segmentation mask
- `nuclear.png` - Nuclear channel visualization
- `whole_cell.png` - Whole-cell channel visualization
- `mask.png` - Colorized segmentation overlay

### Analysis Outputs (Step 3)
- **Radial profile plots**: Mean cell area and aspect ratio vs. distance from center
- **Data tables**: Binned statistics with physical units (pixels and microns)
- **Visualization**: Ellipse overlays on segmentation masks

## Customization

### Adjusting Segmentation Sensitivity
```python
# In CellSAM_pipeline.py
bbox_threshold=0.3           # Lower = more sensitive (0.2-0.5)
low_contrast_enhancement=True  # Enable for dim images
gauge_cell_size=True          # Auto-detect cell size
```

### Setting Physical Scale
```matlab
% In Area_AR_Analysis.m
real_radius_um = 208;  % Measured spheroid radius in microns
```

### Modifying Radial Bins
```matlab
% In Area_AR_Analysis.m
numBins = 10;  % Increase for finer spatial resolution
```

## Troubleshooting

See the [detailed instructions](Image%20analysis%20codes/Instructions.md) for comprehensive troubleshooting guidance.

**Common issues:**
- **Memory errors**: Reduce batch size or process smaller image sets
- **Poor segmentation**: Adjust `bbox_threshold` or enable contrast enhancement
- **Scale conversion**: Verify physical measurements using ImageJ or similar tools

## Documentation

For detailed step-by-step instructions, parameter descriptions, and troubleshooting, see:
- **[Instructions.md](Image%20analysis%20codes/Instructions.md)** - Complete usage guide

## Citation

If you use this pipeline in your research, please cite:

**CellSAM:**
```
Van Valen Lab. (2023). CellSAM: Segment Anything for Microscopy.
https://github.com/vanvalenlab/cellSAM
```

**This pipeline:**
```
Sharma, V. (2026). Tumor Spheroid Analysis Pipeline.
https://github.com/vs-vivek/Tumor_Spheroid_Analysis
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests for:
- Bug fixes
- Documentation improvements
- New features
- Performance optimizations

## License

This project is available under the MIT License. See LICENSE file for details.

## Acknowledgments

- **CellSAM**: Van Valen Lab for the segmentation framework
- **npy-matlab**: kwikteam for NumPy-MATLAB integration

## Contact

**Author**: Vivek Sharma  
**Repository**: [vs-vivek/Tumor_Spheroid_Analysis](https://github.com/vs-vivek/Tumor_Spheroid_Analysis)

For questions or issues, please open an issue on GitHub.

---

**Last Updated**: January 2026  
**Pipeline Version**: 1.0  
**Tested With**: Python 3.8-3.11, MATLAB R2020b-R2024a
