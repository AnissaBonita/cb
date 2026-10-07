#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# The file source is from the [ESRGAN](https://github.com/xinntao/ESRGAN) project 
# forked by authors [joeyballentine](https://github.com/joeyballentine/ESRGAN) and [BlueAmulet](https://github.com/BlueAmulet/ESRGAN).

import gc

import numpy as np
import torch


def bgr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    # flip image channels
    # https://github.com/pytorch/pytorch/issues/229
    out: torch.Tensor = image.flip(-3)
    # out: torch.Tensor = image[[2, 1, 0], :, :] #RGB to BGR #may be faster
    return out


def rgb_to_bgr(image: torch.Tensor) -> torch.Tensor:
    # same operation as bgr_to_rgb(), flip image channels
    return bgr_to_rgb(image)


def bgra_to_rgba(image: torch.Tensor) -> torch.Tensor:
    out: torch.Tensor = image[[2, 1, 0, 3], :, :]
    return out


def rgba_to_bgra(image: torch.Tensor) -> torch.Tensor:
    # same operation as bgra_to_rgba(), flip image channels
    return bgra_to_rgba(image)


def auto_split_upscale(
    lr_img: np.ndarray,
    upscale_function,
    scale: int = 4,
    overlap: int = 32,
    # A heuristic to proactively split tiles that are too large, avoiding a CUDA error.
    # The default (2048*2048) is a conservative value for moderate VRAM (e.g., 8-12GB).
    # Adjust this based on your GPU and model's memory footprint.
    max_tile_pixels: int = 4194304,  # Default: 2048 * 2048 pixels
    # Internal parameters for recursion state. Do not set these manually.
    known_max_depth: int = None,
    current_depth: int = 1,
    current_tile: int = 1,  # Tracks the current tile being processed
    total_tiles: int = 1,  # Total number of tiles at this depth level
):
    # --- Step 0: Handle CPU-only environment ---
    # The entire splitting logic is designed to overcome GPU VRAM limitations.
    # If no CUDA-enabled GPU is present, this logic is unnecessary and adds overhead.
    # Therefore, we process the image in one go on the CPU.
    if not torch.cuda.is_available():
        # Note: This assumes the image fits into system RAM, which is usually the case.
        result, _ = upscale_function(lr_img, scale)
        # The conceptual depth is 1 since no splitting was performed.
        return result, 1

    """
    Automatically splits an image into tiles for upscaling to avoid CUDA out-of-memory errors.
    It uses a combination of a pixel-count heuristic and reactive error handling to find the
    optimal processing depth, then applies this depth to all subsequent tiles.
    """
    input_h, input_w, input_c = lr_img.shape
    
    # --- Step 1: Decide if we should ATTEMPT to upscale or MUST split ---
    # We must split if:
    # A) The tile is too large based on our heuristic, and we don't have a known working depth yet.
    # B) We have a known working depth from a sibling tile, but we haven't recursed deep enough to reach it yet.
    must_split = (known_max_depth is None and (input_h * input_w) > max_tile_pixels) or \
                 (known_max_depth is not None and current_depth < known_max_depth)

    if not must_split:
        # If we are not forced to split, let's try to upscale the current tile.
        try:
            print(f"auto_split_upscale depth: {current_depth}", end=" ", flush=True)
            result, _ = upscale_function(lr_img, scale)
            # SUCCESS! The upscale worked at this depth.
            print(f"progress: {current_tile}/{total_tiles}")
            # Return the result and the current depth, which is now the "known_max_depth".
            return result, current_depth
        except RuntimeError as e:
            # Check to see if its actually the CUDA out of memory error
            if "CUDA" in str(e):
                # OOM ERROR. Our heuristic was too optimistic. This depth is not viable.
                print("RuntimeError: CUDA out of memory...")
                # Clean up VRAM and proceed to the splitting logic below.
                torch.cuda.empty_cache()
                gc.collect()
            else:
                # A different runtime error occurred, so we should not suppress it.
                raise RuntimeError(e)
        # If an OOM error occurred, flow continues to the splitting section.
        
    # --- Step 2: If we reached here, we MUST split the image ---

    # Safety break to prevent infinite recursion if something goes wrong.
    if current_depth > 10:
        raise RuntimeError("Maximum recursion depth exceeded. Check max_tile_pixels or model requirements.")

    # Prepare parameters for the next level of recursion.
    next_depth = current_depth + 1
    new_total_tiles = total_tiles * 4
    base_tile_for_next_level = (current_tile - 1) * 4
    
    # Announce the split only when it's happening.
    print(f"Splitting tile at depth {current_depth} into 4 tiles for depth {next_depth}.")

    # Split the image into 4 quadrants with overlap.
    top_left      = lr_img[: input_h // 2 + overlap, : input_w // 2 + overlap, :]
    top_right     = lr_img[: input_h // 2 + overlap, input_w // 2 - overlap :, :]
    bottom_left   = lr_img[input_h // 2 - overlap :, : input_w // 2 + overlap, :]
    bottom_right  = lr_img[input_h // 2 - overlap :, input_w // 2 - overlap :, :]
    
    # Recursively process each quadrant.
    # Process the first quadrant to discover the safe depth.
    # The first quadrant (top_left) will "discover" the correct processing depth.
    # Pass the current `known_max_depth` down.
    top_left_rlt, discovered_depth = auto_split_upscale(
        top_left, upscale_function, scale=scale, overlap=overlap,
        max_tile_pixels=max_tile_pixels,
        known_max_depth=known_max_depth,
        current_depth=next_depth,
        current_tile=base_tile_for_next_level + 1,
        total_tiles=new_total_tiles,
    )
    # Once the depth is discovered, pass it to the other quadrants to avoid redundant checks.
    top_right_rlt, _ = auto_split_upscale(
        top_right, upscale_function, scale=scale, overlap=overlap,
        max_tile_pixels=max_tile_pixels,
        known_max_depth=discovered_depth,
        current_depth=next_depth,
        current_tile=base_tile_for_next_level + 2,
        total_tiles=new_total_tiles,
    )
    bottom_left_rlt, _ = auto_split_upscale(
        bottom_left, upscale_function, scale=scale, overlap=overlap,
        max_tile_pixels=max_tile_pixels,
        known_max_depth=discovered_depth,
        current_depth=next_depth,
        current_tile=base_tile_for_next_level + 3,
        total_tiles=new_total_tiles,
    )
    bottom_right_rlt, _ = auto_split_upscale(
        bottom_right, upscale_function, scale=scale, overlap=overlap,
        max_tile_pixels=max_tile_pixels,
        known_max_depth=discovered_depth,
        current_depth=next_depth,
        current_tile=base_tile_for_next_level + 4,
        total_tiles=new_total_tiles,
    )
    
    # --- Step 3: Stitch the results back together ---
    # Reassemble the upscaled quadrants into a single image.
    out_h = input_h * scale
    out_w = input_w * scale
    
    # Create an empty output image
    output_img = np.zeros((out_h, out_w, input_c), np.uint8)
    
    # Fill the output image, removing the overlap regions to prevent artifacts
    output_img[: out_h // 2, : out_w // 2, :]   = top_left_rlt[: out_h // 2, : out_w // 2, :]
    output_img[: out_h // 2, -out_w // 2 :, :]  = top_right_rlt[: out_h // 2, -out_w // 2 :, :]
    output_img[-out_h // 2 :, : out_w // 2, :]  = bottom_left_rlt[-out_h // 2 :, : out_w // 2, :]
    output_img[-out_h // 2 :, -out_w // 2 :, :] = bottom_right_rlt[-out_h // 2 :, -out_w // 2 :, :]

    return output_img, discovered_depth
