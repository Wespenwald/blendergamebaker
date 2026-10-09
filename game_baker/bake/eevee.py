import os
import tempfile

import bpy
import numpy as np

from .common import (
    ATTRIBUTE_NAMES,
    configure_render,
    object_bbox,
    render_pixels,
    set_uv_camera,
)


def _float_color(mesh, name, values):
    attr = mesh.attributes.get(name) or mesh.attributes.new(name, "FLOAT_COLOR", "POINT")
    attr.data.foreach_set("color", np.asarray(values, dtype=np.float32).reshape(-1))
    return attr


def _object_id_color(obj):
    index = sum((i + 1) * byte for i, byte in enumerate(obj.name.encode("utf-8"))) % 997
    hue = (index * 0.618033988749895) % 1.0
    h = hue * 6.0
    i = int(h)
    f = h - i
    p, q, t = 0.28, 0.95 * (1.0 - f * 0.72), 0.95 * (1.0 - (1.0 - f) * 0.72)
    return ((0.95, t, p), (q, 0.95, p), (p, 0.95, t), (p, q, 0.95), (t, p, 0.95), (0.95, p, q))[i % 6]


def _map_colors(obj, flat_mesh, map_item, depsgraph):
    if map_item.map_type == "NORMAL_WORLD":
        source = flat_mesh.attributes[ATTRIBUTE_NAMES["world_normal"]].data
        values = np.empty(len(source) * 3, dtype=np.float32)
        source.foreach_get("vector", values)
        values = values.reshape(-1, 3) * 0.5 + 0.5
    elif map_item.map_type == "NORMAL_OBJECT":
        source = flat_mesh.attributes[ATTRIBUTE_NAMES["object_normal"]].data
        values = np.empty(len(source) * 3, dtype=np.float32)
        source.foreach_get("vector", values)
        values = values.reshape(-1, 3) * 0.5 + 0.5
    elif map_item.map_type == "POSITION":
        source = flat_mesh.attributes[ATTRIBUTE_NAMES["world_position"]].data
        values = np.empty(len(source) * 3, dtype=np.float32)
        source.foreach_get("vector", values)
        values = values.reshape(-1, 3)
        low, high, _ = object_bbox(obj, depsgraph)
        values = np.clip((values - np.asarray(low[:])) / np.maximum(np.asarray((high - low)[:]), 1e-8), 0.0, 1.0)
    elif map_item.map_type == "GRADIENT":
        axis = "XYZ".index(map_item.gradient_axis)
        source = flat_mesh.attributes[ATTRIBUTE_NAMES["world_position"]].data
        positions = np.empty(len(source) * 3, dtype=np.float32)
        source.foreach_get("vector", positions)
        positions = positions.reshape(-1, 3)
        low, high, _ = object_bbox(obj, depsgraph)
        span = max(high[axis] - low[axis], 1e-8)
        scalar = np.clip((positions[:, axis] - low[axis]) / span, 0.0, 1.0)
        values = np.repeat(scalar[:, None], 3, axis=1)
    else:
        if map_item.id_source == "OBJECT":
            values = np.tile(np.asarray(_object_id_color(obj), dtype=np.float32), (len(flat_mesh.vertices), 1))
        elif map_item.id_source == "COLOR_ATTRIBUTE" and ATTRIBUTE_NAMES["color_id"] in flat_mesh.attributes:
            attr = flat_mesh.attributes[ATTRIBUTE_NAMES["color_id"]].data
            values = np.empty(len(attr) * 4, dtype=np.float32)
            attr.foreach_get("color", values)
            values = values.reshape(-1, 4)[:, :3]
        else:
            attr = flat_mesh.attributes[ATTRIBUTE_NAMES["id_color"]].data
            values = np.empty(len(attr) * 4, dtype=np.float32)
            attr.foreach_get("color", values)
            values = values.reshape(-1, 4)[:, :3]
    return np.column_stack((values, np.ones(len(values), dtype=np.float32)))


def render_attribute_map(
    scene, obj, flat_obj, map_item, resolution, supersample, depsgraph, cycles_device="CPU"
):
    mesh = flat_obj.data
    values = _map_colors(obj, mesh, map_item, depsgraph)
    color_attr = _float_color(mesh, "gb_bake_color", values)
    material = bpy.data.materials.new(f"GB_{map_item.map_type}")
    material.use_nodes = True
    material.diffuse_color = (0.5, 0.5, 0.5, 1.0)
    nodes = material.node_tree.nodes
    nodes.clear()
    attribute = nodes.new("ShaderNodeAttribute")
    attribute.attribute_name = color_attr.name
    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(attribute.outputs["Color"], emission.inputs["Color"])
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    mesh.materials.clear()
    mesh.materials.append(material)
    if flat_obj.name not in scene.objects:
        scene.collection.objects.link(flat_obj)
    scene.render.engine = "BLENDER_EEVEE" if map_item.engine == "EEVEE" else "CYCLES"
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = max(1, map_item.cycles_samples)
        scene.cycles.device = cycles_device
    scene.eevee.taa_render_samples = 1
    scene.eevee.taa_samples = 1
    scene.render.filter_size = 0.01
    scene.display.shading.light = "FLAT"
    scene.display.shading.color_type = "MATERIAL"
    output_size = resolution * supersample
    set_uv_camera(scene, output_size)
    configure_render(scene, output_size)
    path = os.path.join(tempfile.gettempdir(), f"gb_{map_item.map_type.lower()}.exr")
    try:
        pixels = render_pixels(scene, path)
        if map_item.map_type in {"NORMAL_WORLD", "NORMAL_OBJECT"}:
            normals = pixels[:, :, :3] * 2.0 - 1.0
            lengths = np.linalg.norm(normals, axis=2, keepdims=True)
            valid = (pixels[:, :, 3] > 0.5) & (lengths[:, :, 0] > 1e-8)
            normalized = np.zeros_like(normals)
            normalized[valid] = normals[valid] / lengths[valid]
            pixels[valid, :3] = normalized[valid] * 0.5 + 0.5
        return pixels
    finally:
        if os.path.exists(path):
            os.remove(path)
        if material.users == 0:
            bpy.data.materials.remove(material)
