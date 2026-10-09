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


def render_projection_map(scene, obj, map_item, resolution, views, uv_name=""):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh, uv_name = _prepare_projection_mesh(obj, depsgraph, uv_name)
    low, high, diagonal = object_bbox(obj, depsgraph)
    center = (low + high) * 0.5
    radius = diagonal * 0.5
    temp_obj = bpy.data.objects.new(f"GB_Projection_{obj.name}", mesh)
    temp_obj.matrix_world = obj.matrix_world.copy()
    scene.collection.objects.link(temp_obj)
    camera_data = bpy.data.cameras.new("GB_ProjectionCamera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = max(diagonal * 1.15, 1.0)
    camera = bpy.data.objects.new("GB_ProjectionCamera", camera_data)
    scene.collection.objects.link(camera)
    previous_camera = scene.camera
    scene.camera = camera
    render_size = max(512, resolution)
    configure_render(scene, render_size)
    scene.render.engine = "BLENDER_WORKBENCH"
    shading = scene.display.shading
    shading.light = "FLAT"
    shading.color_type = "SINGLE"
    shading.single_color = (0.5, 0.5, 0.5)
    shading.show_shadows = False
    shading.show_cavity = True
    shading.cavity_type = "WORLD" if map_item.map_type in {"AO", "CAVITY"} else "SCREEN"
    shading.cavity_ridge_factor = map_item.cavity_ridge_factor
    shading.cavity_valley_factor = map_item.cavity_valley_factor
    shading.show_object_outline = False
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
        ao.inputs["Distance"].default_value = effective_distance(obj, map_item.distance, 0.2, depsgraph)
        ao.only_local = map_item.only_local
        emission = nodes.new("ShaderNodeEmission")
        output = nodes.new("ShaderNodeOutputMaterial")
        ao_material.node_tree.links.new(ao.outputs["AO"], emission.inputs["Color"])
        ao_material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
        view_layer.material_override = ao_material
    accumulator = np.zeros((resolution, resolution), dtype=np.float64)
    weights = np.zeros((resolution, resolution), dtype=np.float64)
    attr_name = "gb_projection_uv"
    for view_index, direction in enumerate(_view_directions(views)):
        camera.location = center + direction * max(2.5 * radius, 1.0)
        _look_at(camera, center)
        corner_values = []
        for poly in mesh.polygons:
            for loop_index in poly.loop_indices:
                uv = mesh.uv_layers[uv_name].data[loop_index].uv
                local_normal = mesh.corner_normals[loop_index].vector
                normal = (temp_obj.matrix_world.to_3x3().inverted_safe().transposed() @ local_normal).normalized()
                facing = max(0.0, normal.dot(direction))
                corner_values.append((uv.x, uv.y, facing, 1.0))
        attr = mesh.color_attributes.get(attr_name)
        if attr:
            mesh.color_attributes.remove(attr)
        attr = mesh.color_attributes.new(name=attr_name, type="FLOAT_COLOR", domain="CORNER")
        attr.data.foreach_set("color", np.asarray(corner_values, dtype=np.float32).reshape(-1))
        mesh.color_attributes.active_color = attr
        scene.render.engine = "BLENDER_WORKBENCH"
        shading.color_type = "VERTEX"
        path = os.path.join(tempfile.gettempdir(), f"gb_projection_uv_{view_index}.exr")
        try:
            uv_pixels = render_pixels(scene, path)
        finally:
            if os.path.exists(path):
                os.remove(path)
        shading.color_type = "SINGLE"
        shading.single_color = (0.5, 0.5, 0.5)
        value_path = os.path.join(tempfile.gettempdir(), f"gb_projection_value_{view_index}.exr")
        try:
            if map_item.map_type == "AO":
                scene.render.engine = "BLENDER_EEVEE"
            values = render_pixels(scene, value_path)
        finally:
            if os.path.exists(value_path):
                os.remove(value_path)
        alpha = uv_pixels[:, :, 3] > 0.5
        uv_data = uv_pixels[:, :, :3]
        facing = uv_data[:, :, 2]
        valid = alpha & (facing >= 0.2)
        u = np.clip(uv_data[:, :, 0], 0.0, 0.999999)
        v = np.clip(uv_data[:, :, 1], 0.0, 0.999999)
        x = np.floor(u * resolution).astype(np.int32)
        y = np.floor(v * resolution).astype(np.int32)
        if map_item.map_type == "AO":
            sample_value = np.clip(values[:, :, 0], 0.0, 1.0)
        else:
            sample_value = np.clip(values[:, :, 0], 0.0, 1.0)
        weight = facing ** 3
        indices = y[valid] * resolution + x[valid]
        accumulator.flat[:] += np.bincount(
            indices, weights=(sample_value[valid] * weight[valid]), minlength=resolution * resolution
        )
        weights.flat[:] += np.bincount(indices, weights=weight[valid], minlength=resolution * resolution)
    result = np.full((resolution, resolution), 0.5, dtype=np.float32)
    covered = weights > 0.0
    result[covered] = (accumulator[covered] / weights[covered]).astype(np.float32)
    if map_item.map_type == "AO":
        result = np.clip(result, 0.0, 1.0)
    view_layer.material_override = previous_override
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.camera = previous_camera
    bpy.data.objects.remove(camera, do_unlink=True)
    bpy.data.cameras.remove(camera_data)
    bpy.data.objects.remove(temp_obj, do_unlink=True)
    bpy.data.meshes.remove(mesh)
    return result
