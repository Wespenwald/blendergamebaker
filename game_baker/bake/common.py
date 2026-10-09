import math

import bpy
import numpy as np
from mathutils import Vector


ATTRIBUTE_NAMES = {
    "world_position": "gb_world_position",
    "world_normal": "gb_world_normal",
    "object_position": "gb_object_position",
    "object_normal": "gb_object_normal",
    "material_index": "gb_material_index",
    "id_color": "gb_id_color",
    "color_id": "gb_color_id",
}


def build_flat_uv_mesh(obj, depsgraph, uv_name=""):
    evaluated = obj.evaluated_get(depsgraph)
    source = evaluated.to_mesh()
    try:
        source.calc_loop_triangles()
        uv_layer = source.uv_layers.get(uv_name) if uv_name else source.uv_layers.active
        if uv_layer is None:
            raise ValueError(f"{obj.name} has no render UV map")
        triangle_count = len(source.loop_triangles)
        triangle_loops = np.empty((triangle_count, 3), dtype=np.int32)
        source.loop_triangles.foreach_get("loops", triangle_loops.reshape(-1))
        triangle_loops = triangle_loops.reshape(-1)
        triangle_materials = np.empty(triangle_count, dtype=np.int32)
        source.loop_triangles.foreach_get("material_index", triangle_materials)
        triangle_polygons = np.empty(triangle_count, dtype=np.int32)
        source.loop_triangles.foreach_get("polygon_index", triangle_polygons)

        loop_vertices = np.empty(len(source.loops), dtype=np.int32)
        source.loops.foreach_get("vertex_index", loop_vertices)
        vertex_indices = loop_vertices[triangle_loops]
        uv_values = np.empty((len(uv_layer.data), 2), dtype=np.float32)
        uv_layer.data.foreach_get("uv", uv_values.reshape(-1))
        uv_coords = uv_values[triangle_loops]
        vertex_positions = np.empty((len(source.vertices), 3), dtype=np.float32)
        source.vertices.foreach_get("co", vertex_positions.reshape(-1))
        object_positions = vertex_positions[vertex_indices]
        local_normals = np.empty((len(source.corner_normals), 3), dtype=np.float32)
        source.corner_normals.foreach_get("vector", local_normals.reshape(-1))
        object_normals = local_normals[triangle_loops]
        object_normals /= np.maximum(
            np.linalg.norm(object_normals, axis=1, keepdims=True), 1e-12
        )

        matrix = np.asarray(evaluated.matrix_world, dtype=np.float64)
        normal_matrix = np.asarray(
            evaluated.matrix_world.to_3x3().inverted_safe().transposed(), dtype=np.float64
        )
        world_positions = object_positions @ matrix[:3, :3].T + matrix[:3, 3]
        world_normals = object_normals @ normal_matrix.T
        world_normals /= np.maximum(np.linalg.norm(world_normals, axis=1, keepdims=True), 1e-12)
        material_ids = np.repeat(triangle_materials, 3)
        polygon_ids = np.repeat(triangle_polygons, 3)
        flat_vertices = np.column_stack((uv_coords, np.zeros(len(uv_coords), dtype=np.float32)))
        faces = np.arange(len(flat_vertices), dtype=np.int32).reshape(-1, 3)
        mesh = bpy.data.meshes.new(f"GB_Flat_{obj.name}")
        mesh.from_pydata(flat_vertices, [], faces)
        mesh.update()
        mesh.polygons.foreach_set("material_index", triangle_materials)
        uv = mesh.uv_layers.new(name="GameBakerUV")
        uv.data.foreach_set("uv", np.asarray(uv_coords, dtype=np.float32).reshape(-1))
        _new_vector_attribute(mesh, ATTRIBUTE_NAMES["world_position"], world_positions)
        _new_vector_attribute(mesh, ATTRIBUTE_NAMES["world_normal"], world_normals)
        _new_vector_attribute(mesh, ATTRIBUTE_NAMES["object_position"], object_positions)
        _new_vector_attribute(mesh, ATTRIBUTE_NAMES["object_normal"], object_normals)
        mat_attr = mesh.attributes.new(ATTRIBUTE_NAMES["material_index"], "FLOAT", "POINT")
        mat_attr.data.foreach_set("value", np.asarray(material_ids, dtype=np.float32))
        palette = np.asarray(
            [_hsv_color(i) for i in range(max(1, len(obj.material_slots)))],
            dtype=np.float32,
        )
        ids = palette[material_ids % len(palette)]
        id_attr = mesh.attributes.new(ATTRIBUTE_NAMES["id_color"], "FLOAT_COLOR", "POINT")
        id_attr.data.foreach_set("color", np.asarray(ids, dtype=np.float32).reshape(-1))
        color_source = source.color_attributes.active_color
        if color_source:
            source_colors = np.empty((len(color_source.data), 4), dtype=np.float32)
            color_source.data.foreach_get("color", source_colors.reshape(-1))
            if color_source.domain == "POINT":
                color_indices = vertex_indices
            elif color_source.domain == "FACE":
                color_indices = polygon_ids
            else:
                color_indices = triangle_loops
            color_ids = source_colors[color_indices]
            color_attr = mesh.attributes.new(ATTRIBUTE_NAMES["color_id"], "FLOAT_COLOR", "POINT")
            color_attr.data.foreach_set("color", np.asarray(color_ids, dtype=np.float32).reshape(-1))
        flat_obj = bpy.data.objects.new(f"GB_Flat_{obj.name}", mesh)
        return flat_obj, mesh
    finally:
        evaluated.to_mesh_clear()


def _new_vector_attribute(mesh, name, values):
    attr = mesh.attributes.new(name, "FLOAT_VECTOR", "POINT")
    attr.data.foreach_set("vector", np.asarray(values, dtype=np.float32).reshape(-1))


def _material_id_colors(material_ids, material_count):
    palette = [_hsv_color(i) for i in range(max(1, material_count))]
    return [palette[int(index) % len(palette)] for index in material_ids]


def _hsv_color(index):
    hue = (index * 0.618033988749895) % 1.0
    return (*_hsv_to_rgb(hue, 0.72, 0.95), 1.0)


def _hsv_to_rgb(h, s, v):
    i = int(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1.0 - s), v * (1.0 - f * s), v * (1.0 - (1.0 - f) * s)
    return ((v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q))[i % 6]


def set_uv_camera(scene, resolution):
    camera_data = bpy.data.cameras.new("GB_UVCamera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 1.0
    camera = bpy.data.objects.new("GB_UVCamera", camera_data)
    camera.location = (0.5, 0.5, 1.0)
    scene.collection.objects.link(camera)
    scene.camera = camera
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    return camera


def configure_render(scene, resolution, format="OPEN_EXR"):
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = format
    scene.render.image_settings.color_mode = "RGBA"
    if format == "OPEN_EXR":
        scene.render.image_settings.color_depth = "32"
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    try:
        scene.view_settings.look = "None"
    except (TypeError, ValueError):
        pass
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.image_settings.color_management = "FOLLOW_SCENE"


def render_pixels(scene, filepath):
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True, scene=scene.name)
    image = bpy.data.images.load(filepath, check_existing=False)
    try:
        pixels = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(pixels)
        return pixels.reshape((image.size[1], image.size[0], 4)).copy()
    finally:
        bpy.data.images.remove(image)


def uv_coverage(obj, resolution, uv_name=""):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    flat_obj, mesh = build_flat_uv_mesh(obj, depsgraph, uv_name)
    scene = bpy.data.scenes.new("GB_Coverage")
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "FLAT"
    scene.display.shading.color_type = "SINGLE"
    scene.display.shading.single_color = (1.0, 1.0, 1.0)
    scene.display.shading.show_shadows = False
    scene.display.shading.show_cavity = False
    scene.display.shading.show_object_outline = False
    scene.display.render_aa = "OFF"
    scene.collection.objects.link(flat_obj)
    set_uv_camera(scene, resolution)
    configure_render(scene, resolution)
    scene.render.image_settings.color_mode = "RGBA"
    import tempfile
    import os
    path = os.path.join(tempfile.gettempdir(), "gb_coverage.exr")
    try:
        rendered = render_pixels(scene, path)
        return rendered[:, :, 3] > 0.5
    finally:
        if os.path.exists(path):
            os.remove(path)
        _remove_scene_contents(scene)
        bpy.data.scenes.remove(scene)


def _remove_scene_contents(scene):
    for obj in list(scene.objects):
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data and data.users == 0:
            if isinstance(data, bpy.types.Mesh):
                bpy.data.meshes.remove(data)
            elif isinstance(data, bpy.types.Camera):
                bpy.data.cameras.remove(data)


def object_bbox(obj, depsgraph=None):
    if depsgraph is None:
        depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    points = [evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box]
    low = Vector(tuple(min(p[i] for p in points) for i in range(3)))
    high = Vector(tuple(max(p[i] for p in points) for i in range(3)))
    return low, high, max((high - low).length, 1e-6)


def effective_distance(obj, value, multiplier, depsgraph=None):
    if value > 0.0:
        return value
    return object_bbox(obj, depsgraph)[2] * multiplier
