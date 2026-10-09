import math
import os
import tempfile

import bpy
import numpy as np
from mathutils import Vector

from .common import configure_render, effective_distance, object_bbox, render_pixels


def _look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def _view_directions(count):
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(count):
        y = 1.0 - (2.0 * (i + 0.5) / count)
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        angle = golden * i
        yield Vector((math.cos(angle) * radius, y, math.sin(angle) * radius))


def _erode_mask(mask, iterations):
    result = mask.copy()
    for _ in range(iterations):
        eroded = np.zeros_like(result)
        eroded[1:-1, 1:-1] = (
            result[1:-1, 1:-1]
            & result[:-2, 1:-1]
            & result[2:, 1:-1]
            & result[1:-1, :-2]
            & result[1:-1, 2:]
        )
        result = eroded
    return result


def _prepare_projection_mesh(obj, depsgraph, uv_name):
    evaluated = obj.evaluated_get(depsgraph)
    evaluated_mesh = evaluated.to_mesh()
    try:
        mesh = evaluated_mesh.copy()
    finally:
        evaluated.to_mesh_clear()
    uv_layer = mesh.uv_layers.get(uv_name) if uv_name else mesh.uv_layers.active
    if uv_layer is None:
        bpy.data.meshes.remove(mesh)
        raise ValueError(f"{obj.name} has no render UV map")
    return mesh, uv_layer.name


def _projection_corner_data(mesh, uv_name, matrix_world):
    uv_layer = mesh.uv_layers[uv_name]
    uv_values = np.empty((len(uv_layer.data), 2), dtype=np.float32)
    uv_layer.data.foreach_get("uv", uv_values.reshape(-1))
    local_normals = np.empty((len(mesh.corner_normals), 3), dtype=np.float32)
    mesh.corner_normals.foreach_get("vector", local_normals.reshape(-1))
    matrix = np.asarray(matrix_world, dtype=np.float64)
    normal_matrix = np.asarray(
        matrix_world.to_3x3().inverted_safe().transposed(), dtype=np.float64
    )
    world_normals = local_normals @ normal_matrix.T
    world_normals /= np.maximum(np.linalg.norm(world_normals, axis=1, keepdims=True), 1e-12)
    return uv_values, world_normals


def project_views(
    scene,
    obj,
    uv_name,
    resolution,
    views,
    setup_value_pass,
    value_channels=1,
    robust_views=False,
    min_facing=0.2,
    facing_power=3,
    silhouette_erosion=1,
):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh, uv_name = _prepare_projection_mesh(obj, depsgraph, uv_name)
    _low, _high, diagonal = object_bbox(obj, depsgraph)
    radius = diagonal * 0.5
    center = (np.asarray(_low[:]) + np.asarray(_high[:])) * 0.5
    temp_obj = bpy.data.objects.new(f"GB_Projection_{obj.name}", mesh)
    temp_obj.matrix_world = obj.matrix_world.copy()
    scene.collection.objects.link(temp_obj)
    camera_data = bpy.data.cameras.new("GB_ProjectionCamera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = diagonal * 1.15
    camera_data.clip_start = max(radius * 0.01, 1e-6)
    camera_data.clip_end = max(radius * 6.0, camera_data.clip_start * 10.0)
    camera = bpy.data.objects.new("GB_ProjectionCamera", camera_data)
    scene.collection.objects.link(camera)
    shading = scene.display.shading
    previous = (
        scene.camera,
        scene.render.engine,
        scene.display.render_aa,
        scene.display.viewport_aa,
        shading.light,
        shading.color_type,
        tuple(shading.single_color),
        shading.show_shadows,
        shading.show_cavity,
        shading.show_object_outline,
    )
    scene.camera = camera
    render_size = max(512, resolution)
    configure_render(scene, render_size)
    scene.display.render_aa = "OFF"
    scene.display.viewport_aa = "OFF"
    scene.render.engine = "BLENDER_WORKBENCH"
    shading.light = "FLAT"
    shading.color_type = "SINGLE"
    shading.single_color = (0.5, 0.5, 0.5)
    shading.show_shadows = False
    shading.show_cavity = False
    shading.show_object_outline = False
    uv_values, world_normals = _projection_corner_data(mesh, uv_name, temp_obj.matrix_world)
    accumulator = np.zeros((value_channels, resolution * resolution), dtype=np.float64)
    weights = np.zeros(resolution * resolution, dtype=np.float64)
    if robust_views:
        view_min = np.full_like(accumulator, np.inf, dtype=np.float32)
        view_max = np.full_like(accumulator, -np.inf, dtype=np.float32)
    attr_name = "gb_projection_uv"
    try:
        for view_index, direction in enumerate(_view_directions(max(1, views))):
            camera.location = center + np.asarray(direction[:], dtype=np.float64) * (2.5 * radius)
            _look_at(camera, center)
            direction_array = np.asarray(direction[:], dtype=np.float64)
            facing = np.maximum(world_normals @ direction_array, 0.0)
            corner_values = np.column_stack(
                (uv_values, facing.astype(np.float32), np.ones(len(facing), dtype=np.float32))
            )
            attr = mesh.color_attributes.get(attr_name)
            if attr:
                mesh.color_attributes.remove(attr)
            attr = mesh.color_attributes.new(
                name=attr_name, type="FLOAT_COLOR", domain="CORNER"
            )
            attr.data.foreach_set("color", corner_values.reshape(-1))
            mesh.color_attributes.active_color = attr

            scene.render.engine = "BLENDER_WORKBENCH"
            shading.light = "FLAT"
            shading.color_type = "VERTEX"
            shading.show_cavity = False
            uv_path = os.path.join(
                tempfile.gettempdir(), f"gb_projection_uv_{id(temp_obj)}_{view_index}.exr"
            )
            try:
                uv_pixels = render_pixels(scene, uv_path)
            finally:
                if os.path.exists(uv_path):
                    os.remove(uv_path)

            setup_value_pass(scene, temp_obj, camera, direction)
            value_path = os.path.join(
                tempfile.gettempdir(), f"gb_projection_value_{id(temp_obj)}_{view_index}.exr"
            )
            try:
                values = render_pixels(scene, value_path)
            finally:
                if os.path.exists(value_path):
                    os.remove(value_path)

            alpha = _erode_mask(
                uv_pixels[:, :, 3] > 0.5, silhouette_erosion
            )
            facing_pixels = uv_pixels[:, :, 2]
            valid = alpha & (facing_pixels >= min_facing)
            u = np.clip(uv_pixels[:, :, 0], 0.0, 0.999999)
            v = np.clip(uv_pixels[:, :, 1], 0.0, 0.999999)
            x = np.floor(u * resolution).astype(np.int32)
            y = np.floor(v * resolution).astype(np.int32)
            flat_indices = y[valid] * resolution + x[valid]
            sample_weights = facing_pixels[valid] ** facing_power
            view_weights = np.bincount(
                flat_indices,
                weights=sample_weights,
                minlength=resolution * resolution,
            )
            samples = values[:, :, :value_channels]
            if robust_views:
                visible = view_weights > 0.0
                weights[visible] += 1.0
                for channel in range(value_channels):
                    view_sum = np.bincount(
                        flat_indices,
                        weights=samples[:, :, channel][valid] * sample_weights,
                        minlength=resolution * resolution,
                    )
                    view_values = view_sum[visible] / view_weights[visible]
                    accumulator[channel, visible] += view_values
                    view_min[channel, visible] = np.minimum(
                        view_min[channel, visible], view_values
                    )
                    view_max[channel, visible] = np.maximum(
                        view_max[channel, visible], view_values
                    )
            else:
                weights += view_weights
                for channel in range(value_channels):
                    accumulator[channel] += np.bincount(
                        flat_indices,
                        weights=samples[:, :, channel][valid] * sample_weights,
                        minlength=resolution * resolution,
                    )
    finally:
        (
            scene.camera,
            scene.render.engine,
            scene.display.render_aa,
            scene.display.viewport_aa,
            shading.light,
            shading.color_type,
            shading.single_color,
            shading.show_shadows,
            shading.show_cavity,
            shading.show_object_outline,
        ) = previous
        bpy.data.objects.remove(camera, do_unlink=True)
        bpy.data.cameras.remove(camera_data)
        bpy.data.objects.remove(temp_obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)

    result = np.full(
        (resolution * resolution, value_channels), 0.5, dtype=np.float32
    )
    covered = weights > 0.0
    if robust_views:
        trimmed_weights = weights.copy()
        trim = weights >= 3.0
        accumulator[:, trim] -= view_min[:, trim] + view_max[:, trim]
        trimmed_weights[trim] -= 2.0
        result[covered] = (
            accumulator[:, covered] / trimmed_weights[covered]
        ).T.astype(np.float32)
    else:
        result[covered] = (accumulator[:, covered] / weights[covered]).T.astype(np.float32)
    return result.reshape(resolution, resolution, value_channels), covered.reshape(
        resolution, resolution
    )


def render_projection_map(scene, obj, map_item, resolution, views, uv_name=""):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    view_layer = scene.view_layers[0]
    previous_override = view_layer.material_override
    ao_material = None
    if map_item.map_type == "AO":
        ao_material = bpy.data.materials.new(f"GB_AOPreview_{obj.name}")
        ao_material.use_nodes = True
        nodes = ao_material.node_tree.nodes
        nodes.clear()
        ao = nodes.new("ShaderNodeAmbientOcclusion")
        ao.samples = map_item.samples
        ao.inputs["Distance"].default_value = effective_distance(
            obj, map_item.distance, 0.2, depsgraph
        )
        ao.only_local = map_item.only_local
        emission = nodes.new("ShaderNodeEmission")
        output = nodes.new("ShaderNodeOutputMaterial")
        ao_material.node_tree.links.new(ao.outputs["AO"], emission.inputs["Color"])
        ao_material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
        view_layer.material_override = ao_material

    def setup_value_pass(value_scene, _temp_obj, _camera, _direction):
        shading = value_scene.display.shading
        if map_item.map_type == "AO":
            value_scene.render.engine = "BLENDER_EEVEE"
            value_scene.eevee.taa_render_samples = 16
            shading.show_cavity = False
            return
        value_scene.render.engine = "BLENDER_WORKBENCH"
        shading.light = "FLAT"
        shading.color_type = "SINGLE"
        shading.single_color = (0.5, 0.5, 0.5)
        shading.show_shadows = False
        shading.show_cavity = True
        shading.cavity_type = (
            "WORLD" if map_item.map_type == "CAVITY" else "SCREEN"
        )
        shading.cavity_ridge_factor = map_item.cavity_ridge_factor
        shading.cavity_valley_factor = map_item.cavity_valley_factor
        shading.show_object_outline = False

    try:
        result, _coverage = project_views(
            scene,
            obj,
            uv_name,
            resolution,
            views,
            setup_value_pass,
            value_channels=1,
            robust_views=map_item.map_type in {"AO", "CAVITY", "CURVATURE"},
            min_facing=(
                0.65 if map_item.map_type in {"CAVITY", "CURVATURE"} else 0.2
            ),
            facing_power=(
                8 if map_item.map_type in {"CAVITY", "CURVATURE"} else 3
            ),
            silhouette_erosion=(
                4 if map_item.map_type in {"CAVITY", "CURVATURE"} else 1
            ),
        )
        result = result[:, :, 0]
        return np.clip(result, 0.0, 1.0) if map_item.map_type == "AO" else result
    finally:
        view_layer.material_override = previous_override
        if ao_material and ao_material.users == 0:
            bpy.data.materials.remove(ao_material)
