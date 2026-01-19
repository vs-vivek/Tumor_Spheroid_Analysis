In order to use these codes follow the following instructions.
# Cell Segmentation and Ellipse Analysis Pipeline

This repository contains a comprehensive pipeline for segmenting cellular images and analyzing cell morphology in spheroids. The workflow combines Python-based deep learning segmentation with MATLAB-based analysis.

## Overview

This pipeline enables:
- Automated batch cell segmentation using CellSAM
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
   pip install SimpleITK numpy matplotlib tqdm
   ```

   Or install with conda:
   ```bash
   conda install -c conda-forge simpleitk numpy matplotlib tqdm
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

### Step 1: Batch Segmentation with CellSAM

Use `CellSAM_pipeline.py` to perform batch cell segmentation on all TIFF images in a directory:

```python
# Update these paths in the __main__ section of the script
input_dir = "path/to/your/tiff/images"
output_dir = "path/to/output/directory"

# Run the batch pipeline
success, fail = batch_segment_tifs_resume(
    input_dir=input_dir,
    output_dir=output_dir,
    bbox_threshold=0.3,
    use_wsi=False,
    low_contrast_enhancement=False,
    gauge_cell_size=False,
    display_first=False
)
```

**Input Requirements:**
- **Image format:** TIFF files (.tif or .tiff)
- **Expected shape:** (3, H, W)
  - Channel 0: Nuclear signal
  - Channel 1: Whole-cell signal
  - Channel 2: Sytox / auxiliary channel (ignored)

**Key Parameters:**
- `bbox_threshold`: Bounding box confidence threshold (0.2-0.5, default 0.3)
- `use_wsi`: Enable whole-slide-image tiling (disabled by default due to known issues)
- `low_contrast_enhancement`: Enable contrast enhancement preprocessing
- `gauge_cell_size`: Enable automatic cell size estimation
- `display_first`: Show visualization for first processed image (useful for debugging)

**Resume Capability:**
The pipeline automatically skips images that have already been processed (where `mask.npy` exists in the output subdirectory). This allows you to safely re-run the script if processing is interrupted.

**Outputs (per image):**
For each input image, a subdirectory is created containing:
- `mask.npy` - Integer-labeled segmentation mask for further analysis
- `nuclear.png` - Nuclear channel visualization
- `whole_cell.png` - Whole-cell channel visualization
- `mask.png` - Colorized segmentation mask

### Step 2: Interactive Mask Refinement

Use `Interactive_mask_editor.m` to manually refine the segmentation:

```matlab
% Copy mask.npy from your desired image output folder to MATLAB working directory
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

**Note:** You need to run this step separately for each image you want to analyze. Copy the `mask.npy` from the specific image's output folder to your MATLAB working directory before running the editor.

### Step 3: Detailed Morphometric Analysis

Use `Area_AR_Analysis.m` for comprehensive analysis of cell sizes and aspect ratio:

```matlab
% Copy updated_mask.npy from Step 2 to your MATLAB working directory
% Run the analysis script
Area_AR_Analysis
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
```

**Outputs:**
- Radial profile plots showing:
  - Mean ellipse area vs. radial position (in microns)
  - Mean aspect ratio vs. radial position (in microns)
  - Error bars representing standard error of the mean (SEM)

## Configuration

### Setting Physical Scale

Update the physical scale based on your imaging setup in `Area_AR_Analysis.m`:

```matlab
% Option 1: If you know the spheroid radius
real_radius_um = 208;  % Your measured radius in microns

% Option 2: If you know pixel size
pixel_size_um = 0.325;  % microns per pixel
% Then modify the conversion line to:
% pixel_to_um = pixel_size_um;
```

### Adjusting Segmentation Quality

In `CellSAM_pipeline.py`, you can adjust these parameters:

```python
bbox_threshold=0.3           # Lower (0.2) = more sensitive, Higher (0.5) = more specific
low_contrast_enhancement=True  # Enable for low-contrast images
gauge_cell_size=True          # Enable for automatic cell size detection
```

### Memory Management

For large datasets, the Python pipeline includes automatic memory management:
- Garbage collection after each image
- Explicit array deletion
- Matplotlib figure cleanup
- Batch processing with progress tracking

## Troubleshooting

### Common Issues

1. **"readNPY not found"**
   - Ensure npy-matlab is installed and added to MATLAB path
   - Verify installation: `which readNPY` in MATLAB should show the path

2. **"Unexpected TIFF shape" error**
   - Verify your TIFF has shape (3, H, W)
   - Check channel order: nuclear (0), whole-cell (1), auxiliary (2)
   - Use ImageJ or Python to inspect: `sitk.GetArrayFromImage(sitk.ReadImage('file.tif')).shape`

3. **Memory errors in Python**
   - Process smaller batches of images at a time
   - Close other applications to free up RAM
   - Set `display_first=False` to reduce memory usage

4. **Poor segmentation quality**
   - Adjust `bbox_threshold` (try values between 0.2-0.5)
   - Enable `low_contrast_enhancement=True` for dim images
   - Use the interactive editor to clean up results

5. **Scale conversion issues**
   - Verify your physical measurements using ImageJ or similar
   - Check that spheroid radius measurement is accurate
   - Ensure pixel-to-micron conversion matches your microscope settings

6. **"sem_AR not defined" error in MATLAB**
   - This appears to be a typo in line 106 of Area_AR_Analysis.m
   - Should be `semAR` (consistent with line 56)

### Output File Descriptions

| File | Description |
|------|-------------|
| `mask.npy` | Initial segmentation mask from CellSAM |
| `updated_mask.npy` | Refined mask after interactive editing |
| `nuclear.png` | Nuclear channel visualization |
| `whole_cell.png` | Whole-cell channel visualization |
| `mask.png` | Colorized segmentation mask |

## Typical Workflow Example

1. **Prepare your data:**
   - Organize all TIFF images in a single input directory
   - Verify TIFF format (3, H, W) with correct channel order

2. **Run batch segmentation:**
   ```bash
   python CellSAM_pipeline.py
   ```
   - Monitor progress and check for any failed images
   - Review output folders to verify segmentation quality

3. **For each spheroid of interest:**
   - Copy `mask.npy` to MATLAB working directory
   - Run `Interactive_mask_editor.m` to clean up segmentation
   - Run `Area_AR_Analysis.m` to perform radial analysis
   - Update `real_radius_um` with measured spheroid radius
   - Click to select center point
   - Save/export generated plots and data

4. **Analyze results:**
   - Compare radial profiles across conditions
   - Use exported data for statistical analysis

## Support

For issues related to:
- **CellSAM segmentation:** Check the [CellSAM GitHub repository](https://github.com/vanvalenlab/cellSAM)
- **This pipeline:** Open an issue in this repository
- **MATLAB analysis:** Ensure all required toolboxes are installed

---

**Note:** This pipeline has been tested with Python 3.8+ and MATLAB R2020b+. Compatibility with other versions may vary.