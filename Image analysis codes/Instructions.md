In order to use these codes follow the following instructions.
# Cell Segmentation and Ellipse Analysis Pipeline

This repository contains a comprehensive pipeline for segmenting cellular images and analyzing cell morphology in spheroids. The workflow combines Python-based deep learning segmentation with MATLAB-based morphometric analysis.

## Overview

This pipeline enables:
- Automated cell segmentation using CellSAM
- Interactive mask refinement for quality control
- Radial analysis of cell area and aspect ratio in spheroids
- Export of results with physical unit conversions

## Installation

### Python Dependencies

1. **Install CellSAM from GitHub:**
   ```bash
   pip install git+https://github.com/vanvalenlab/cellSAM.git
   ```

2. **Install other required Python packages:**
   ```bash
   pip install SimpleITK Pillow numpy opencv-python matplotlib pandas torch pathlib
   ```

   Or install with conda:
   ```bash
   conda install -c conda-forge simpleitk pillow numpy opencv matplotlib pandas pytorch
   ```

### MATLAB Dependencies

1. **Install npy-matlab for reading/writing NumPy arrays in MATLAB:**
   - Download from: https://github.com/kwikteam/npy-matlab
   - Add the downloaded folder to your MATLAB path:
     ```matlab
     addpath('/path/to/npy-matlab')
     savepath
     ```

2. **Required MATLAB Toolboxes:**
   - Image Processing Toolbox
   - Computer Vision Toolbox

## Workflow

### Step 1: Initial Segmentation with CellSAM

Use `CellSAM_pipeline.py` to perform initial cell segmentation:

```python
# Update these paths in the script
image_path = "path/to/your/image.tif"  # or .png, .jpg
output_dir = "path/to/output/directory"

# Run the pipeline
process_single_image(image_path, output_dir)
```

**What this step does:**
- Loads and preprocesses your image
- Performs deep learning-based cell segmentation
- Generates preliminary analysis including:
  - Cell area and aspect ratio measurements
  - Distance-based binning analysis
  - Visualization outputs (SVG files)
  - Saves `mask.npy` for further refinement

**Outputs:**
- `mask.npy` - Segmentation mask for interactive editing
- `mask.tif` - TIFF version of the mask
- `cell_data.csv` - Initial cell measurements
- `colored_contours.svg` - Ellipse overlays
- `ellipse_overlay.svg` - Ellipses on original image
- `visualization.svg` - Three-panel comparison

### Step 2: Interactive Mask Refinement

Use `Interactive_mask_editor.m` to manually refine the segmentation:

```matlab
% Place mask.npy in your MATLAB working directory
% Run the interactive editor
Interactive_mask_editor
```

**How to use:**
1. The script will display the segmented mask
2. Click inside any incorrectly segmented cell to delete it
3. Right-click or press Enter when finished
4. The refined mask is saved as `updated_mask.npy`

**When to use this step:**
- Remove spuriously segmented regions
- Delete masks that capture debris or artifacts
- Clean up over-segmented cells
- Ensure only valid cells are included in analysis

### Step 3: Detailed Morphometric Analysis

Use `Area_AR_Analysism.m` for comprehensive analysis of cell sizes and aspect ratio:

```matlab
% Place updated_mask.npy in your MATLAB working directory
% Run the analysis script
ellipse_area_AR_analysism
```

**Interactive steps:**
1. **Select spheroid center:** Click once on the image to define the center point
2. **Review results:** The script generates multiple plots and exports data

**Key parameters to adjust:**

```matlab
% Number of radial bins (adjust based on spheroid size and cell layers)
numBins = 10;  % Increase for more layers, decrease for fewer layers

% Physical scale conversion
real_radius_um = 208;  % Update with your spheroid's actual radius in microns

**Outputs:**
- `AR_Area_Data_Best_*.csv` - Binned statistics with physical units
- Multiple plots showing:
  - Mean ellipse area vs. radial position
  - Mean aspect ratio vs. radial position
  - Both in pixel and micron units


### Setting Physical Scale

Update the physical scale based on your imaging setup:

```matlab
% Option 1: If you know the spheroid radius
real_radius_um = 208;  % Your measured radius in microns

% Option 2: If you know pixel size
pixel_size_um = 0.325;  % microns per pixel
% Then modify the conversion line to:
% pixel_to_um = pixel_size_um;
```

### Memory Management

For large images, the Python pipeline includes memory management features:

```python
# Adjust batch size for label processing (reduce if memory issues)
batch_size = 100  # Process 100 labels at a time

# GPU memory is automatically cleared between operations
```

## Troubleshooting

### Common Issues

1. **"readNPY not found"**
   - Ensure npy-matlab is installed and added to MATLAB path

2. **Memory errors in Python**
   - Reduce batch_size in the pipeline
   - Use smaller images or crop regions of interest

3. **Poor segmentation quality**
   - Adjust bbox_threshold in segment_cellular_image (try 0.2-0.5)
   - Use the interactive editor to clean up results

4. **Scale conversion issues**
   - Verify your physical measurements
   - Check that pixel-to-micron conversion is appropriate for your imaging setup

### Output File Descriptions

| File | Description |
|------|-------------|
| `mask.npy` | Initial segmentation mask from CellSAM |
| `updated_mask.npy` | Refined mask after interactive editing |
| `cell_data.csv` | Individual cell measurements |
| `AR_Area_Data_*.csv` | Binned radial statistics |
| `*.svg` | Vector graphics for publications |


## Support

For issues related to:
- **CellSAM segmentation:** Check the [CellSAM GitHub repository](https://github.com/vanvalenlab/cellSAM)
- **This pipeline:** Open an issue in this repository
- **MATLAB analysis:** Ensure all required toolboxes are installed

---

**Note:** This pipeline has been tested with Python 3.8+ and MATLAB R2020b+. Compatibility with other versions may vary.
