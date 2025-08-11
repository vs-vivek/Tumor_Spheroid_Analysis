import SimpleITK as sitk
from cellSAM import segment_cellular_image
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import pickle
import torch
import os
import gc
from pathlib import Path

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clear_memory():
    """Clear memory and garbage collect"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_preprocess_image(image_path):
    ext = Path(image_path).suffix.lower()
    if ext in [".tif", ".tiff"]:
        img = sitk.ReadImage(str(image_path))
        arr = sitk.GetArrayFromImage(img)
        img_channel = arr[1]
        del arr, img  # Clear intermediate variables
    elif ext in [".png", ".jpg", ".jpeg"]:
        img_pil = Image.open(image_path).convert("L")
        img_channel = np.array(img_pil)
        img_pil.close()  # Close PIL image
    else:
        raise ValueError(f"Unsupported format: {ext}")

    # normalize → uint8
    norm = (img_channel - img_channel.min()) / (img_channel.max() - img_channel.min() + 1e-8) * 255
    norm = norm.astype(np.uint8)
    del img_channel  # Clear original channel
    
    # to RGB
    result = np.stack([norm]*3, axis=-1)
    del norm  # Clear normalized array
    clear_memory()  # Clear GPU memory if used
    return result

def analyze_mask(mask, image_np, output_path):
    elong_thresh = 1.8
    col_long   = "#ed00ff"
    col_norm   = "#fbfdfb"

    if mask is None:
        print("No mask!")
        return pd.DataFrame()

    labels = np.unique(mask)
    labels = labels[labels != 0]

    data = {
        "Label": [], 
        "Area": [], 
        "Distance": [], 
        "Aspect_Ratio": [], 
        "Contour": []
    }
    cy0, cx0 = mask.shape[0]//2, mask.shape[1]//2

    # This will be our pixel overlay for the 3-panel view
    overlay_pix = image_np.copy()

    # Collect ellipse parameters
    ellipse_params = []

    # Process labels in smaller batches to manage memory
    batch_size = 100  # Process 100 labels at a time
    for batch_start in range(0, len(labels), batch_size):
        batch_end = min(batch_start + batch_size, len(labels))
        batch_labels = labels[batch_start:batch_end]
        
        for i, lab in enumerate(batch_labels):
            global_i = batch_start + i
            m = (mask == lab).astype(np.uint8)
            area = int(m.sum())
            M = cv2.moments(m)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            dist = np.hypot(cy - cy0, cx - cx0)

            ctrs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not ctrs:
                continue
            cnt = ctrs[0]

            # skip tiny contours
            if cnt.shape[0] < 5:
                continue

            # fit ellipse
            (ex, ey), (ew, eh), ang = cv2.fitEllipse(cnt)

            # skip any degenerate ellipse with zero width or height
            if ew == 0 or eh == 0:
                print(f"  [label {lab}] ellipse has zero axis (ew={ew}, eh={eh}), skipping")
                continue

            ar = max(ew, eh) / min(ew, eh)

            # record & draw on pixel overlay
            color_rgb = (237,0,255) if ar > elong_thresh else (251,253,251)
            cv2.ellipse(overlay_pix, ((ex,ey),(ew,eh),ang), color_rgb, 2)

            # save values
            data["Label"].append(global_i+1)
            data["Area"].append(area)
            data["Distance"].append(dist)
            data["Aspect_Ratio"].append(ar)
            data["Contour"].append(cnt)

            # store for SVGs
            ellipse_params.append(((ex, ey), (ew, eh), ang, ar))
        
        # Clear memory after each batch
        clear_memory()

    # ---- 1) colored_contours.svg: ellipse outlines on TRANSPARENT ----
    fig_cont, ax_cont = plt.subplots()
    fig_cont.set_size_inches(mask.shape[1]/100, mask.shape[0]/100)
    ax_cont.axis("off")
    # transparent background
    fig_cont.patch.set_alpha(0.0)
    ax_cont.set_xlim(0, mask.shape[1])
    ax_cont.set_ylim(mask.shape[0], 0)
    for (center, axes, angle, ar) in ellipse_params:
        w, h = axes
        color = col_long if ar>elong_thresh else col_norm
        e = mpatches.Ellipse(center, width=w, height=h,
                             angle=angle,
                             edgecolor=color,
                             facecolor="none",
                             linewidth=3)
        ax_cont.add_patch(e)
    plt.savefig(output_path/"colored_contours.svg",
                format="svg",
                bbox_inches="tight",
                pad_inches=0,
                transparent=True)
    plt.close(fig_cont)
    clear_memory()

    # ---- 2) ellipse_overlay.svg: ellipses ON TOP OF ORIGINAL IMAGE ----
    fig_el, ax_el = plt.subplots()
    fig_el.set_size_inches(mask.shape[1]/100, mask.shape[0]/100)
    ax_el.imshow(image_np)
    ax_el.axis("off")
    for (center, axes, angle, ar) in ellipse_params:
        w, h = axes
        color = col_long if ar>elong_thresh else col_norm
        e = mpatches.Ellipse(center, width=w, height=h,
                             angle=angle,
                             edgecolor=color,
                             facecolor="none",
                             linewidth=1.2)
        ax_el.add_patch(e)
    ax_el.set_xlim(0, mask.shape[1])
    ax_el.set_ylim(mask.shape[0], 0)
    plt.savefig(output_path/"ellipse_overlay.svg",
                format="svg",
                bbox_inches="tight",
                pad_inches=0,
                transparent=True)
    plt.close(fig_el)
    clear_memory()

    # ---- 3) 3-panel visualization: Original / Mask / Ellipses on Pix ----
    fig3, axs = plt.subplots(1,3,figsize=(18,6))
    axs[0].imshow(image_np);        axs[0].set_title("Original");     axs[0].axis("off")
    axs[1].imshow(mask, cmap="gray");axs[1].set_title("Mask");         axs[1].axis("off")
    axs[2].imshow(overlay_pix);      axs[2].set_title("With Ellipses");axs[2].axis("off")
    plt.tight_layout()
    plt.savefig(output_path/"visualization.svg", format="svg")
    plt.close(fig3)
    clear_memory()

    # ---- 4) save the DataFrame of ellipse‐fit cells ----
    df = pd.DataFrame(data)
    df_clean = df[df["Aspect_Ratio"].notna()]
    with open(output_path/"contour_data.pkl", "wb") as f:
        pickle.dump(df_clean, f)
    
    # ---- 5) save the mask for interactive editing ----
    np.save(output_path/"mask.npy", mask)

    # Clear large variables
    del overlay_pix, ellipse_params
    clear_memory()

    return df_clean

def plot_binned_analysis(df, output_path):
    if df.empty:
        print("No data to plot.")
        return
    d = df["Distance"].values
    a = df["Area"].values
    r = df["Aspect_Ratio"].values
    n = 9
    bins = np.linspace(0, d.max(), n+1)
    idx = np.digitize(d, bins)
    ma, sa, mr, sr, cen = [], [], [], [], []
    for i in range(1, len(bins)):
        m = (idx==i)
        if not m.any(): continue
        cnt = m.sum()
        ma.append(a[m].mean()); sa.append(a[m].std()/np.sqrt(cnt))
        mr.append(r[m].mean()); sr.append(r[m].std()/np.sqrt(cnt))
        cen.append((bins[i-1]+bins[i])/2)
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6))
    ax1.errorbar(cen, ma, yerr=sa, marker='o', linestyle='-', capsize=5)
    ax1.set(xlabel="Distance (px)", ylabel="Mean Area", title="Area vs Distance"); ax1.grid(True)
    ax2.errorbar(cen, mr, yerr=sr, marker='o', linestyle='-', capsize=5)
    ax2.set(xlabel="Distance (px)", ylabel="Mean AR", title="Aspect Ratio vs Distance"); ax2.grid(True)
    plt.tight_layout(); plt.show()
    clear_memory()

def process_single_image(image_path, output_dir):
    try:
        image_file = Path(image_path)
        out_root = Path(output_dir)
        sub = out_root / image_file.stem
        sub.mkdir(parents=True, exist_ok=True)

        print(f"Loading image: {image_file.name}")
        img = load_and_preprocess_image(image_file)
        
        print(f"Running segmentation (this may take a while)...")
        mask, _, _ = segment_cellular_image(img,
                                            device=str(get_device()),
                                            normalize=True,
                                            bbox_threshold=0.3,
                                            postprocess=False)
        
        # Clear GPU memory after segmentation
        clear_memory()
        
        # Save the mask immediately after segmentation for interactive editing
        np.save(sub / "mask.npy", mask)
        print(f"Saved mask to {sub / 'mask.npy'}")
        
        # Also save as TIFF for compatibility
        import SimpleITK as sitk
        mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint16))
        sitk.WriteImage(mask_sitk, str(sub / "mask.tif"))
        print(f"Saved mask to {sub / 'mask.tif'}")
        del mask_sitk
        
        print(f"Analyzing {len(np.unique(mask))-1} detected objects...")
        df = analyze_mask(mask, img, sub)
        df.to_csv(sub/"cell_data.csv", index=False)
        plot_binned_analysis(df, sub)

        df["Image"] = image_file.name
        combined = out_root/"combined_cell_data.csv"
        if combined.exists():
            old = pd.read_csv(combined)
            df = pd.concat([old, df], ignore_index=True)
        df.to_csv(combined, index=False)
        print(f"Saved combined data to {combined}")
        
        # Final cleanup
        del img, mask, df
        clear_memory()
        print("Processing complete!")
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        clear_memory()  # Clear memory even on error
        raise


# Run for one image with memory management
image_path = "......put your image path here......"
output_dir = ".......put your output directory here......"

# Adjust threshold based on your system's available RAM and segmentation quality
process_single_image(image_path, output_dir)
