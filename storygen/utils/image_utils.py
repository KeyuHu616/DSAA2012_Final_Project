"""
Image Utility Functions
Helpers for image processing and visualization
"""

from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional
import os
import numpy as np


def remove_white_borders(img: Image.Image, threshold: int = 200) -> Image.Image:
    """
    Remove white, gray or light-colored borders from an image.
    ENHANCED: More aggressive border detection with edge-aware processing.
    
    Args:
        img: PIL Image to process
        threshold: Pixel value threshold for "border" (0-255)
                   Higher values = more aggressive (white borders use ~240)
                   SDXL gray borders ~180-220
    
    Returns:
        Image with borders removed (resized to fill frame if needed)
    """
    img_array = np.array(img)
    
    # Handle RGBA images
    if img_array.shape[2] == 4:
        alpha = img_array[:, :, 3]
        # Check for transparent edges
        if alpha[0, 0] == 0 or alpha[0, -1] == 0 or alpha[-1, 0] == 0 or alpha[-1, -1] == 0:
            img_array = img_array[:, :, :3]
    
    h, w = img_array.shape[:2]
    orig_size = (w, h)
    
    # === STEP 1: Aggressive border detection ===
    # Check all four corners - if they're all light colored, there's likely a border
    corners = [
        img_array[:10, :10],      # top-left
        img_array[:10, -10:],     # top-right
        img_array[-10:, :10],     # bottom-left
        img_array[-10:, -10:],    # bottom-right
    ]
    
    corner_avg = [np.mean(corner) for corner in corners]
    avg_corner = np.mean(corner_avg)
    avg_content = np.mean(img_array[20:h-20, 20:w-20])
    
    # If corners are much lighter than content, we have borders
    has_border = avg_corner > avg_content + 30
    
    if not has_border:
        # Also check with threshold
        light_pixels = np.all(img_array[:, :, :3] >= threshold, axis=2)
        border_ratio = np.sum(light_pixels) / (h * w)
        has_border = border_ratio > 0.3  # More than 30% light pixels = border
    
    if not has_border:
        return img
    
    # === STEP 2: Find actual content bounds ===
    # Try multiple thresholds to find the edge
    best_bounds = None
    
    for thresh in [240, 220, 200, 180, 160]:
        is_border = np.all(img_array[:, :, :3] >= thresh, axis=2)
        rows = np.any(~is_border, axis=1)
        cols = np.any(~is_border, axis=0)
        
        if np.any(rows) and np.any(cols):
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            content_h = rmax - rmin
            content_w = cmax - cmin
            
            # Check if we found significant content (at least 50% of original)
            if content_h > h * 0.5 and content_w > w * 0.5:
                best_bounds = (rmin, rmax, cmin, cmax)
                break
    
    if best_bounds is None:
        # Fallback: find center of mass of non-light pixels
        is_light = np.all(img_array[:, :, :3] >= 150, axis=2)
        y_coords, x_coords = np.where(~is_light)
        if len(y_coords) > 100:
            rmin, rmax = y_coords.min(), y_coords.max()
            cmin, cmax = x_coords.min(), x_coords.max()
            best_bounds = (rmin, rmax, cmin, cmax)
        else:
            return img
    
    rmin, rmax, cmin, cmax = best_bounds
    
    # === STEP 3: Crop with margin ===
    margin = 10
    rmin = max(0, rmin - margin)
    rmax = min(h, rmax + margin)
    cmin = max(0, cmin - margin)
    cmax = min(w, cmax + margin)
    
    cropped = img.crop((cmin, rmin, cmax, rmax))
    
    # === STEP 4: If border was significant, resize to fill original size ===
    cropped_w, cropped_h = cropped.size
    
    # If we cropped more than 15%, resize back to fill the frame
    if cropped_w < w * 0.85 or cropped_h < h * 0.85:
        # Resize cropped content to fill the original frame
        resized = cropped.resize((w, h), Image.LANCZOS)
        return resized
    
    return cropped


def create_storyboard(
    images: List[Image.Image],
    labels: List[str] = None,
    layout: str = "horizontal",
    image_size: Tuple[int, int] = (512, 512),
    spacing: int = 20,
    background_color: Tuple[int, int, int] = (40, 40, 40),
    label_height: int = 60
) -> Image.Image:
    """
    Create a storyboard from a list of images

    Args:
        images: List of PIL Images
        labels: Optional list of labels for each image
        layout: "horizontal" or "vertical"
        image_size: Target size for each image (width, height)
        spacing: Space between images
        background_color: RGB tuple for background
        label_height: Height for label area

    Returns:
        Combined storyboard image
    """
    if not images:
        return Image.new('RGB', (100, 100), color=background_color)

    # Resize images
    resized = [img.resize(image_size, Image.LANCZOS) for img in images]

    if layout == "horizontal":
        total_width = sum(img.width for img in resized) + spacing * (len(resized) - 1)
        total_height = image_size[1] + (label_height if labels else 0)
        storyboard = Image.new('RGB', (total_width, total_height), color=background_color)

        x_offset = 0
        for i, img in enumerate(resized):
            storyboard.paste(img, (x_offset, 0))

            if labels and i < len(labels):
                draw = ImageDraw.Draw(storyboard)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
                except:
                    font = ImageFont.load_default()

                draw.text((x_offset + 10, image_size[1] + 10), labels[i], fill=(200, 200, 200), font=font)

            x_offset += img.width + spacing

    else:  # vertical
        total_width = image_size[0] + 200  # Extra space for labels
        total_height = sum(img.height for img in resized) + spacing * (len(resized) - 1)
        total_height += label_height if labels else 0

        storyboard = Image.new('RGB', (total_width, total_height), color=background_color)

        y_offset = 0
        for i, img in enumerate(resized):
            storyboard.paste(img, (0, y_offset))

            if labels and i < len(labels):
                draw = ImageDraw.Draw(storyboard)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
                except:
                    font = ImageFont.load_default()

                draw.text((10, y_offset + img.height + 10), labels[i], fill=(200, 200, 200), font=font)

            y_offset += img.height + spacing

    return storyboard


def save_images(
    images: List[Image.Image],
    output_dir: str,
    prefix: str = "frame",
    img_format: str = "PNG"
) -> List[str]:
    """
    Save a list of images to disk

    Args:
        images: List of PIL Images
        output_dir: Directory to save images
        prefix: Filename prefix
        img_format: Image format (PNG, JPEG, etc.)

    Returns:
        List of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    saved_paths = []

    for i, img in enumerate(images, 1):
        filename = f"{prefix}_{i:02d}.{img_format.lower()}"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath, format=img_format)
        saved_paths.append(filepath)

    return saved_paths


def create_comparison_grid(
    rows: List[List[Image.Image]],
    labels: List[List[str]] = None,
    cell_size: Tuple[int, int] = (256, 256),
    padding: int = 10
) -> Image.Image:
    """
    Create a grid comparison image

    Args:
        rows: List of rows, each containing list of images
        labels: Optional labels for each cell
        cell_size: Size of each cell
        padding: Padding between cells

    Returns:
        Grid comparison image
    """
    if not rows:
        return Image.new('RGB', (100, 100), color=(0, 0, 0))

    num_cols = max(len(row) for row in rows)
    num_rows = len(rows)

    grid_width = num_cols * cell_size[0] + (num_cols + 1) * padding
    grid_height = num_rows * cell_size[1] + (num_rows + 1) * padding

    grid = Image.new('RGB', (grid_width, grid_height), color=(50, 50, 50))

    for row_idx, row in enumerate(rows):
        for col_idx, img in enumerate(row):
            x = padding + col_idx * (cell_size[0] + padding)
            y = padding + row_idx * (cell_size[1] + padding)

            # Resize and paste
            resized = img.resize(cell_size, Image.LANCZOS)
            grid.paste(resized, (x, y))

            # Add label if provided
            if labels and row_idx < len(labels) and col_idx < len(labels[row_idx]):
                draw = ImageDraw.Draw(grid)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
                except:
                    font = ImageFont.load_default()

                draw.text((x + 5, y + 5), labels[row_idx][col_idx], fill=(255, 255, 0), font=font)

    return grid
