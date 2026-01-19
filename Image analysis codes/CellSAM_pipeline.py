"""
Batch Cell Segmentation for Multiplex TIFF Images using CellSAM

This script performs batch segmentation of multiplexed TIFF images containing
nuclear and whole-cell channels using the CellSAM pipeline. It is designed for
large experimental datasets and supports resume-safe execution by skipping
images that have already been processed.

Expected TIFF format:
    Shape: (3, H, W)
    Channel 0: Nuclear signal
    Channel 1: Whole-cell signal
    Channel 2: Sytox / auxiliary channel (ignored)

For each input image, the script generates:
    - mask.npy        : integer-labeled segmentation mask
    - nuclear.png     : nuclear channel visualization
    - whole_cell.png  : whole-cell channel visualization
    - mask.png        : colorized segmentation mask

Author: Vivek Sharma
Intended use: Reproducible analysis for publication
"""

# =========================
# Standard library imports
# =========================
from pathlib import Path
import gc

# =========================
# Third-party imports
# =========================
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# =========================
# Project-specific imports
# =========================
from cellSAM import cellsam_pipeline


# ---------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------
def load_multiplex_tif(path: Path):
    """
    Load a multiplex TIFF image and extract nuclear and whole-cell channels.

    Parameters
    ----------
    path : Path
        Path to a TIFF image of shape (3, H, W).

    Returns
    -------
    seg : np.ndarray
        Array of shape (H, W, 3) formatted for CellSAM input:
        [blank, nuclear, whole-cell].
    nuclei : np.ndarray
        Nuclear channel, shape (H, W).
    whole_cell : np.ndarray
        Whole-cell channel, shape (H, W).

    Raises
    ------
    ValueError
        If the TIFF does not have the expected shape.
    """
    img_sitk = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img_sitk)  # expected shape: (3, H, W)

    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"Unexpected TIFF shape {arr.shape} for {path}")

    nuclei = arr[0, ...]
    whole_cell = arr[1, ...]

    # CellSAM expects a 3-channel image; channel 0 is unused
    H, W = nuclei.shape
    seg = np.zeros((H, W, 3), dtype=nuclei.dtype)
    seg[..., 1] = nuclei
    seg[..., 2] = whole_cell

    # Explicit cleanup for large images
    del img_sitk, arr
    gc.collect()

    return seg, nuclei, whole_cell


def save_pngs(nuclei, whole_cell, mask, subdir: Path):
    """
    Save visualization PNGs for nuclear, whole-cell, and segmentation mask.

    Parameters
    ----------
    nuclei : np.ndarray
        Nuclear channel image.
    whole_cell : np.ndarray
        Whole-cell channel image.
    mask : np.ndarray
        Integer-labeled segmentation mask.
    subdir : Path
        Output directory for saved images.
    """
    subdir = Path(subdir)

    # Save grayscale channels
    plt.imsave(subdir / "nuclear.png", nuclei, cmap="gray")
    plt.imsave(subdir / "whole_cell.png", whole_cell, cmap="gray")

    # Save colorized segmentation mask
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mask, cmap="tab20")
    ax.axis("off")
    fig.savefig(subdir / "mask.png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ---------------------------------------------------------------------
# Batch segmentation driver
# ---------------------------------------------------------------------
def batch_segment_tifs_resume(
    input_dir,
    output_dir,
    bbox_threshold=0.3,
    use_wsi=False,
    low_contrast_enhancement=False,
    gauge_cell_size=False,
    display_first=False,
):
    """
    Batch segment all TIFF images in a directory with resume support.

    Images are skipped if their output subdirectory already contains
    a saved segmentation mask (mask.npy).

    Parameters
    ----------
    input_dir : str or Path
        Directory containing input .tif or .tiff files.
    output_dir : str or Path
        Directory where per-image output folders are created.
    bbox_threshold : float, optional
        Bounding box confidence threshold for CellSAM.
    use_wsi : bool, optional
        Enable whole-slide-image (WSI) tiling. Disabled by default due to
        known tiling issues.
    low_contrast_enhancement : bool, optional
        Enable contrast enhancement prior to segmentation.
    gauge_cell_size : bool, optional
        Enable automatic cell size estimation.
    display_first : bool, optional
        Display segmentation results for the first processed image.

    Returns
    -------
    successes : list of str
        Filenames successfully processed in this run.
    failures : list of tuple
        (filename, error_message) for failed images.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(
        list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
    )
    print(f"Found {len(tif_files)} TIFF files total.")

    # Identify files that still require processing
    remaining = []
    for path in tif_files:
        subdir = output_dir / path.stem.replace(" ", "_")
        if (subdir / "mask.npy").exists():
            print(f"Skipping {path.name} (already processed)")
        else:
            remaining.append(path)

    print(f"\nRemaining to process: {len(remaining)}")

    successes = []
    failures = []

    for idx, path in enumerate(
        tqdm(remaining, desc="Segmenting remaining TIFFs"), start=1
    ):
        print(f"\nProcessing {idx}/{len(remaining)}: {path.name}")

        try:
            # Load and format input image
            seg, nuclei, whole_cell = load_multiplex_tif(path)

            # Create output directory
            subdir = output_dir / path.stem.replace(" ", "_")
            subdir.mkdir(parents=True, exist_ok=True)

            # Run CellSAM segmentation
            mask = cellsam_pipeline(
                seg,
                use_wsi=use_wsi,
                bbox_threshold=bbox_threshold,
                low_contrast_enhancement=low_contrast_enhancement,
                gauge_cell_size=gauge_cell_size,
            )

            # Save results
            np.save(subdir / "mask.npy", mask)
            save_pngs(nuclei, whole_cell, mask, subdir)

            successes.append(path.name)
            print(f"✓ Outputs saved to: {subdir}")

            # Optional visualization for debugging
            if display_first and idx == 1:
                plt.figure(figsize=(15, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(nuclei, cmap="gray")
                plt.title("Nuclear")
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.imshow(whole_cell, cmap="gray")
                plt.title("Whole-cell")
                plt.axis("off")

                plt.subplot(1, 3, 3)
                plt.imshow(mask, cmap="tab20")
                plt.title("Segmentation Mask")
                plt.axis("off")

                plt.tight_layout()
                plt.show()

            # Aggressive memory cleanup for large datasets
            del seg, nuclei, whole_cell, mask
            gc.collect()
            plt.close("all")

        except Exception as e:
            print(f"✗ Failed: {path.name}")
            print("  Error:", e)
            failures.append((path.name, str(e)))
            gc.collect()
            plt.close("all")

    print("\n======================")
    print("Batch segmentation complete")
    print("======================")
    print(f"Successful (this run): {len(successes)}")
    print(f"Failed (this run): {len(failures)}")

    return successes, failures


# ---------------------------------------------------------------------
# Script entry point (example usage)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    input_dir = (
        "..../Day11"
    )

    output_dir = (
        "..../Day11_Analysis_BBOX0.3"
    )

    success, fail = batch_segment_tifs_resume(
        input_dir=input_dir,
        output_dir=output_dir,
        bbox_threshold=0.3,
        use_wsi=False,                 # avoids known WSI tiling issues
        low_contrast_enhancement=False,
        gauge_cell_size=False,
        display_first=False,           # disabled to conserve memory
    )
