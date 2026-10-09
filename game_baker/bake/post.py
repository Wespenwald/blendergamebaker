import os

import bpy
import numpy as np


def downsample_covered(values, coverage, factor):
    if factor <= 1:
        return values.astype(np.float32, copy=True), coverage.astype(bool, copy=True)
    height, width = coverage.shape
    out_h, out_w = height // factor, width // factor
    reshaped = values[: out_h * factor, : out_w * factor].reshape(
        out_h, factor, out_w, factor, values.shape[2]
    )
    mask = coverage[: out_h * factor, : out_w * factor].reshape(out_h, factor, out_w, factor)
    weights = mask.astype(np.float32)
    denominator = weights.sum(axis=(1, 3))
    numerator = (reshaped * weights[:, :, :, :, None]).sum(axis=(1, 3))
    output = np.zeros((out_h, out_w, values.shape[2]), dtype=np.float32)
    np.divide(numerator, denominator[:, :, None], out=output, where=denominator[:, :, None] > 0)
    return output, denominator > 0


def fill_holes(values, known, target_coverage=None):
    result = values.copy()
    known = known.copy()
    target = np.ones(known.shape, dtype=bool) if target_coverage is None else target_coverage
    height, width = known.shape
    while np.any(target & ~known):
        total = np.zeros_like(result)
        count = np.zeros((height, width), dtype=np.float32)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shifted = np.roll(result, (dy, dx), axis=(0, 1))
            valid = np.roll(known, (dy, dx), axis=(0, 1))
            if dy < 0:
                valid[-1, :] = False
            elif dy > 0:
                valid[0, :] = False
            if dx < 0:
                valid[:, -1] = False
            elif dx > 0:
                valid[:, 0] = False
            total += shifted * valid[:, :, None]
            count += valid
        new = target & (~known) & (count > 0)
        if not new.any():
            break
        result[new] = total[new] / count[new, None]
        known[new] = True
    return result


def pad_image(values, coverage, padding):
    result = values.copy()
    filled = coverage.copy()
    for _ in range(padding):
        total = np.zeros_like(result)
        count = np.zeros(coverage.shape, dtype=np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == dx == 0:
                    continue
                shifted_values = np.roll(result, (dy, dx), axis=(0, 1))
                shifted_mask = np.roll(filled, (dy, dx), axis=(0, 1))
                if dy < 0:
                    shifted_mask[-1, :] = False
                elif dy > 0:
                    shifted_mask[0, :] = False
                if dx < 0:
                    shifted_mask[:, -1] = False
                elif dx > 0:
                    shifted_mask[:, 0] = False
                total += shifted_values * shifted_mask[:, :, None]
                count += shifted_mask
        new = (~filled) & (count > 0)
        if not new.any():
            break
        result[new] = total[new] / count[new, None]
        filled[new] = True
    return result


def image_to_array(values, coverage=None):
    if values.ndim == 2:
        values = np.repeat(values[:, :, None], 3, axis=2)
    if values.shape[2] == 3:
        values = np.concatenate((values, np.ones((*values.shape[:2], 1), dtype=np.float32)), axis=2)
    if coverage is not None:
        values[:, :, 3] = coverage.astype(np.float32)
    return values.astype(np.float32)


def save_image(name, values, path, file_format, png_depth, scene=None):
    height, width, _ = values.shape
    float_buffer = file_format == "OPEN_EXR"
    image = bpy.data.images.new(name, width=width, height=height, alpha=True, float_buffer=float_buffer)
    image.colorspace_settings.name = "Non-Color"
    image.pixels.foreach_set(values.reshape(-1))
    image.file_format = file_format
    image.filepath_raw = path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save()
    if file_format == "PNG" and str(png_depth) == "16":
        render_scene = scene or bpy.context.scene
        image_settings = render_scene.render.image_settings
        previous = (
            image_settings.file_format,
            image_settings.color_mode,
            image_settings.color_depth,
        )
        try:
            image_settings.file_format = "PNG"
            image_settings.color_mode = "RGBA"
            image_settings.color_depth = "16"
            image.save_render(path, scene=render_scene)
        finally:
            (
                image_settings.file_format,
                image_settings.color_mode,
                image_settings.color_depth,
            ) = previous
    return image


def pack_channels(pack, maps, resolution, padding):
    channels = []
    coverage = np.zeros((resolution, resolution), dtype=bool)
    for source in (pack.r_source, pack.g_source, pack.b_source, pack.a_source):
        if source == "WHITE":
            channel = np.ones((resolution, resolution), dtype=np.float32)
        elif source in {"BLACK", "NONE"}:
            channel = np.zeros((resolution, resolution), dtype=np.float32)
        elif source in maps:
            source_map = maps[source]
            channel_index = "RGBA".index(pack.channel)
            channel = source_map[:, :, channel_index]
            coverage |= source_map[:, :, 3] > 0.5
        else:
            channel = np.zeros((resolution, resolution), dtype=np.float32)
        channels.append(channel)
    result = np.stack(channels, axis=2)
    result[:, :, 3] = np.where(coverage, result[:, :, 3], 1.0)
    if padding:
        result = pad_image(result, coverage, padding)
    return result
