"""
img_utils.py

Utility functions for image processing.
"""

from PIL import Image, ImageChops
from typing import Tuple

from typeguard import typechecked


@typechecked
def get_fg_bbox(
    img: Image.Image,
    min_margin: int = 0,
    fuzz: float = 0.0,
) -> Tuple[int, int, int, int]:
    """
    Computes the bounding box (x_min, y_min, x_max, y_max)
    of the foreground in the image.

    Args:
        img: An image to compute foreground bounding box.
        min_margin: The minimum margin to add to the bounding box.
        fuzz: The percentage of fuzziness to add to the bounding box.
            If the background is not uniform, this helps eliminating 
            noisy pixels in the background during the bounding box computation.
    """
    if img.mode == "RGBA":
        img = img.convert("RGB")

    # detect bbox
    bg = Image.new(img.mode, img.size, img.getpixel((0,0)))
    diff = ImageChops.difference(img, bg)
    offset = int(round(float(fuzz)/100.0*255.0))
    diff = ImageChops.add(diff, diff, 2.0, -offset)
    bbox = diff.getbbox()
    if bbox is None:
        return (0, 0, img.size[0], img.size[1])

    # add margin
    bx_min = max(bbox[0] - min_margin, 0)
    by_min = max(bbox[1] - min_margin, 0)
    bx_max = min(bbox[2] + min_margin, img.size[0])
    by_max = min(bbox[3] + min_margin, img.size[1])
    bbox_margin = (bx_min, by_min, bx_max, by_max)
    
    return bbox_margin

@typechecked
def crop_img(
    img: Image.Image,
    bbox: Tuple[int, int, int, int],
) -> Image.Image:
    """
    Crops a region in the given image specified by the bounding box.
    """
    cropped_img = img.crop(bbox)
    return cropped_img

@typechecked
def crop_fg(
    img: Image.Image,
    min_margin: int = 0,
    fuzz: float = 0.0,
) -> Image.Image:
    """
    Crops the foreground in the given image.
    """
    bbox = get_fg_bbox(img, min_margin, fuzz)
    cropped_img = crop_img(img, bbox)
    return cropped_img

# ==========================================================================================================

# ==========================================================================================================
# Resizing functions
@typechecked
def resize_height(
    img: Image.Image,
    new_height: int,
) -> Image.Image:
    """Resizes the image to the given height preserving the aspect ratio"""
    width, height = img.size
    aspect_ratio = width / height
    new_width = int(new_height * aspect_ratio)
    resized_img = img.resize((new_width, new_height), Image.BICUBIC)
    return resized_img

@typechecked
def resize_width(
    img: Image.Image,
    new_width: int,
) -> Image.Image:
    """Resizes the image to the given width preserving the aspect ratio"""
    width, height = img.size
    aspect_ratio = width / height
    new_height = int(new_width / aspect_ratio)
    resized_img = img.resize((new_width, new_height), Image.BICUBIC)
    return resized_img
# ==========================================================================================================

# ==========================================================================================================
# Centering function
@typechecked
def resize_height_and_center_img(
    img: Image.Image,
    new_width: int,
    new_height: int,
) -> Image.Image:
    """
    Resizes the image to the given height preserving the aspect ratio
    and centers the image by padding it.
    """
    # resize the image
    resized_img = resize_height(img, new_height)
    resized_width, resized_height = resized_img.size

    # pad the image to match the width
    bg_color: Tuple[int, int, int] = resized_img.getpixel((0, 0))  # NOTE: Assumption: the background is uniform
    diff_width = new_width - resized_width
    diff_height = new_height - resized_height
    pad_left = diff_width // 2
    pad_top = diff_height // 2
    padded_img = Image.new(resized_img.mode, (new_width, new_height), bg_color)
    padded_img.paste(resized_img, (pad_left, pad_top))

    return padded_img

@typechecked
def resize_width_and_center_img(
    img: Image.Image,
    new_width: int,
    new_height: int,
) -> Image.Image:
    """
    Resizes the image to the given width preserving the aspect ratio
    and centers the image by padding it.
    """
    # resize the image
    resized_img = resize_width(img, new_width)
    resized_width, resized_height = resized_img.size

    # pad the image to match the height
    bg_color: Tuple[int, int, int] = resized_img.getpixel((0, 0))  # NOTE: Assumption: the background is uniform
    diff_width = new_width - resized_width
    diff_height = new_height - resized_height
    pad_left = diff_width // 2
    pad_top = diff_height // 2
    padded_img = Image.new(resized_img.mode, (new_width, new_height), bg_color)
    padded_img.paste(resized_img, (pad_left, pad_top))

    return padded_img
# ==========================================================================================================
