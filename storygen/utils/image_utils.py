"""
Image Utility Functions
Helpers for image processing and visualization
"""

from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional
import os


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
