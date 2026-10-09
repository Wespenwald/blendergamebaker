import math
import os
import sys
import time
import traceback
from types import SimpleNamespace

import bpy
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import game_baker
import game_baker.ui as game_baker_ui
from game_baker.bake.common import build_flat_uv_mesh
from game_baker.bake.post import edge_aware_median_filter
from game_baker.bake.projection import project_views
from game_baker.ops import _effective_cycles_device

OUTPUT_DIR = os.path.join(REPO_ROOT, "tests", "output")
MAPS = (
    ("AO", "ao", "CYCLES"),
    ("THICKNESS", "thickness", "CYCLES"),
    ("CAVITY", "cavity_wb", "WORKBENCH"),
    ("CURVATURE", "curvature_wb", "WORKBENCH"),
    ("NORMAL_WORLD", "normal_ws", "EEVEE"),
    ("NORMAL_OBJECT", "normal_os", "EEVEE"),
    ("POSITION", "position", "EEVEE"),
    ("GRADIENT", "gradient", "EEVEE"),
    ("ID", "id", "EEVEE"),
    ("CAVITY", "cavity_cycles", "CYCLES"),
    ("CURVATURE", "curvature_cycles", "CYCLES"),
)

GLYPHS = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    " ": ("00000",) * 7,
}


def make_material(name, color):
    material = bpy.data.materials.new(name)
    material.diffuse_color = (*color, 1.0)
    return material


def make_scene_asset():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    first = make_material("Paint_Red", (0.8, 0.08, 0.03))
    second = make_material("Paint_Blue", (0.03, 0.12, 0.8))
    objects = []

    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))
    plane = bpy.context.object
    plane.name = "Ground"
    plane.data.materials.append(first)
    objects.append(plane)

    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.5))
    contact_box = bpy.context.object
    contact_box.name = "ContactBox"
    contact_box.data.materials.append(first)
    objects.append(contact_box)

    bpy.ops.mesh.primitive_cube_add(size=1, location=(3.0, 0, 0.5))
    slab = bpy.context.object
    slab.name = "ThinSlab"
    slab.dimensions = (0.12, 0.8, 0.8)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    slab.data.materials.append(second)
    objects.append(slab)

    bpy.ops.mesh.primitive_cube_add(size=1, location=(5.0, 0, 0.5))
    block = bpy.context.object
    block.name = "ThickBlock"
    block.dimensions = (0.8, 0.8, 0.8)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    block.data.materials.append(first)
    objects.append(block)

    bpy.ops.mesh.primitive_cube_add(size=1, location=(7.0, 0, 0.5))
    beveled = bpy.context.object
    beveled.name = "BeveledCube"
    bevel = beveled.modifiers.new("AppliedBevel", "BEVEL")
    bevel.width = 0.12
    bevel.segments = 2
    bpy.context.view_layer.objects.active = beveled
    bpy.ops.object.modifier_apply(modifier=bevel.name)
    beveled.data.materials.append(second)
    objects.append(beveled)

    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = contact_box
    bpy.ops.object.join()
    asset = bpy.context.object
    asset.name = "TestAsset"
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(island_margin=0.015)
    bpy.ops.object.mode_set(mode="OBJECT")
    return asset


def read_image(path):
    image = bpy.data.images.load(path, check_existing=False)
    try:
        image.colorspace_settings.name = "Non-Color"
        values = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(values)
        return values.reshape((image.size[1], image.size[0], 4)).copy()
    finally:
        bpy.data.images.remove(image)


def sample_face_pixels(image, obj, predicate, radius=3):
    mesh = obj.data
    candidates = []
    for polygon in mesh.polygons:
        world_center = obj.matrix_world @ polygon.center
        world_normal = (obj.matrix_world.to_3x3().inverted_safe().transposed() @ polygon.normal).normalized()
        if predicate(world_center, world_normal, polygon):
            uv_layer = mesh.uv_layers.active
            uv = np.mean([uv_layer.data[index].uv[:] for index in polygon.loop_indices], axis=0)
            candidates.append((polygon.area, uv))
    if not candidates:
        raise AssertionError("No polygon matched the requested surface")
    uv = max(candidates, key=lambda entry: entry[0])[1]
    x = int(np.clip(uv[0] * image.shape[1], radius, image.shape[1] - radius - 1))
    y = int(np.clip(uv[1] * image.shape[0], radius, image.shape[0] - radius - 1))
    patch = image[y - radius : y + radius + 1, x - radius : x + radius + 1]
    covered = patch[patch[:, :, 3] > 0.5]
    if not len(covered):
        raise AssertionError(f"Expected covered texels near UV {uv}")
    return covered


def sample_face(image, obj, predicate, radius=3):
    return np.mean(sample_face_pixels(image, obj, predicate, radius), axis=0)


def sample_flat_polygon_texels(image, obj, predicate, edge_margin=4, return_coords=False):
    height, width = image.shape[:2]
    polygon_ids = np.zeros((height, width), dtype=np.int32)
    uv_layer = obj.data.uv_layers.active
    selected_ids = []

    for polygon in obj.data.polygons:
        world_center = obj.matrix_world @ polygon.center
        world_normal = (
            obj.matrix_world.to_3x3().inverted_safe().transposed() @ polygon.normal
        ).normalized()
        if not predicate(world_center, world_normal, polygon):
            continue
        uvs = np.asarray(
            [uv_layer.data[index].uv[:] for index in polygon.loop_indices],
            dtype=np.float64,
        )
        x0 = max(0, int(np.floor(uvs[:, 0].min() * width)))
        x1 = min(width, int(np.ceil(uvs[:, 0].max() * width)))
        y0 = max(0, int(np.floor(uvs[:, 1].min() * height)))
        y1 = min(height, int(np.ceil(uvs[:, 1].max() * height)))
        if x0 >= x1 or y0 >= y1:
            continue
        x_coords = (np.arange(x0, x1) + 0.5) / width
        y_coords = (np.arange(y0, y1) + 0.5) / height
        xx, yy = np.meshgrid(x_coords, y_coords)
        inside = np.zeros(xx.shape, dtype=bool)
        for index in range(len(uvs)):
            u1, v1 = uvs[index]
            u2, v2 = uvs[(index + 1) % len(uvs)]
            crosses = (v1 > yy) != (v2 > yy)
            boundary = (u2 - u1) * (yy - v1) / (v2 - v1 + 1e-15) + u1
            inside ^= crosses & (xx < boundary)
        polygon_id = polygon.index + 1
        polygon_ids[y0:y1, x0:x1][inside] = polygon_id
        selected_ids.append(polygon_id)

    coverage = image[:, :, 3] > 0.5
    interior = np.zeros((height, width), dtype=bool)
    for polygon_id in selected_ids:
        polygon_mask = polygon_ids == polygon_id
        for _ in range(edge_margin):
            eroded = polygon_mask.copy()
            for dy, dx in (
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1), (0, 1),
                (1, -1), (1, 0), (1, 1),
            ):
                shifted = np.roll(polygon_mask, (dy, dx), axis=(0, 1))
                if dy < 0:
                    shifted[-1, :] = False
                elif dy > 0:
                    shifted[0, :] = False
                if dx < 0:
                    shifted[:, -1] = False
                elif dx > 0:
                    shifted[:, 0] = False
                eroded &= shifted
            polygon_mask = eroded
        interior |= polygon_mask
    mask = interior & coverage
    if not mask.any():
        raise AssertionError("No covered texels inside the selected flat polygon")
    if return_coords:
        return image[mask], np.argwhere(mask)
    return image[mask]


def sample_ground_contact(image, obj, radius=1):
    mesh = obj.data
    ground_candidates = []
    for polygon in mesh.polygons:
        center = obj.matrix_world @ polygon.center
        normal = (obj.matrix_world.to_3x3().inverted_safe().transposed() @ polygon.normal).normalized()
        if abs(center.z) < 0.01 and abs(center.x) < 0.01 and abs(center.y) < 0.01 and normal.z > 0.9:
            ground_candidates.append(polygon)
    if not ground_candidates:
        raise AssertionError("Ground plane polygon was not found")
    polygon = max(ground_candidates, key=lambda entry: entry.area)
    uv_layer = mesh.uv_layers.active
    loop_indices = list(polygon.loop_indices)
    local_xy = np.asarray(
        [mesh.vertices[mesh.loops[index].vertex_index].co[:2] for index in loop_indices],
        dtype=np.float64,
    )
    polygon_uvs = np.asarray([uv_layer.data[index].uv[:] for index in loop_indices], dtype=np.float64)
    design = np.column_stack((local_xy, np.ones(len(local_xy))))
    uv_transform, *_ = np.linalg.lstsq(design, polygon_uvs, rcond=None)
    values = []
    for x, y in ((0.52, 0.0), (-0.52, 0.0), (0.0, 0.52), (0.0, -0.52)):
        uv = np.asarray((x, y, 1.0)) @ uv_transform
        pixel_x = int(np.clip(uv[0] * image.shape[1], radius, image.shape[1] - radius - 1))
        pixel_y = int(np.clip(uv[1] * image.shape[0], radius, image.shape[0] - radius - 1))
        patch = image[
            pixel_y - radius : pixel_y + radius + 1,
            pixel_x - radius : pixel_x + radius + 1,
        ]
        covered = patch[patch[:, :, 3] > 0.5]
        if len(covered):
            values.extend(covered)
    if not values:
        raise AssertionError("No covered texels near the ground contact")
    return np.mean(values, axis=0)


class StubUILayout:
    def __init__(self):
        self.calls = []

    def row(self, *args, **kwargs):
        self.calls.append(("row", args, kwargs))
        return self

    def column(self, *args, **kwargs):
        self.calls.append(("column", args, kwargs))
        return self

    def split(self, *args, **kwargs):
        self.calls.append(("split", args, kwargs))
        return self

    def box(self, *args, **kwargs):
        self.calls.append(("box", args, kwargs))
        return self

    def label(self, *args, **kwargs):
        self.calls.append(("label", args, kwargs))

    def prop(self, *args, **kwargs):
        self.calls.append(("prop", args, kwargs))
        return self

    def prop_search(self, *args, **kwargs):
        self.calls.append(("prop_search", args, kwargs))
        return self

    def separator(self, *args, **kwargs):
        self.calls.append(("separator", args, kwargs))

    def template_list(self, *args, **kwargs):
        self.calls.append(("template_list", args, kwargs))
        return self

    def menu(self, *args, **kwargs):
        self.calls.append(("menu", args, kwargs))
        return self

    def operator(self, *args, **kwargs):
        self.calls.append(("operator", args, kwargs))
        return self


def test_ui_draws(settings, asset):
    for index, map_type in enumerate(game_baker_ui.MAP_SHORT_LABELS):
        game_baker_ui.GAMEBAKER_UL_maps.draw_item(
            None,
            None,
            StubUILayout(),
            settings,
            SimpleNamespace(enabled=True, map_type=map_type, engine="EEVEE"),
            None,
            settings,
            "maps",
            index,
        )
    game_baker_ui.GAMEBAKER_UL_packs.draw_item(
        None,
        None,
        StubUILayout(),
        settings,
        SimpleNamespace(suffix="orm"),
        None,
        settings,
        "packs",
        0,
    )

    context = SimpleNamespace(
        scene=SimpleNamespace(game_baker=settings),
        selected_objects=[asset],
        view_layer=SimpleNamespace(objects=SimpleNamespace(active=asset)),
        preferences=bpy.context.preferences,
    )
    original_uv_map = settings.uv_map
    settings.uv_map = ""
    main_layout = StubUILayout()
    game_baker_ui.GAMEBAKER_PT_main.draw(
        SimpleNamespace(layout=main_layout), context
    )
    assert any(
        name == "label" and kwargs.get("text") == "Using active render UV"
        for name, _args, kwargs in main_layout.calls
    ), "Empty UV-map hint was not drawn"
    settings.uv_map = original_uv_map

    for panel in (
        game_baker_ui.GAMEBAKER_PT_output,
        game_baker_ui.GAMEBAKER_PT_packing,
        game_baker_ui.GAMEBAKER_PT_advanced,
        game_baker_ui.GAMEBAKER_PT_bake,
    ):
        panel.draw(SimpleNamespace(layout=StubUILayout()), context)


def test_edge_aware_median_filter():
    values = np.full((16, 16), 0.5, dtype=np.float32)
    values[4:12, 4:12] = 0.6
    values[7, 7] = 1.0
    coverage = np.ones(values.shape, dtype=bool)
    filtered, edge_region, filtered_region, median_reference = edge_aware_median_filter(
        values, coverage
    )
    assert edge_region.any(), "Median filter did not identify an edge region"
    assert filtered_region[7, 7], "Median filter did not identify an edge spike"
    assert filtered[7, 7] == 0.6, "Median filter did not remove an isolated edge spike"
    assert filtered[2, 2] == values[2, 2], "Median filter changed a flat covered texel"
    before_residual = values[edge_region] - median_reference[edge_region]
    after_residual = filtered[edge_region] - median_reference[edge_region]
    assert np.std(after_residual) < np.std(before_residual), (
        "Median filter did not reduce edge-region residual variation"
    )


def add_projection_texture(asset):
    texture = np.empty((8, 8, 4), dtype=np.float32)
    for y in range(8):
        for x in range(8):
            checker = (x + y) % 2
            texture[y, x] = (
                0.12 + 0.76 * x / 7.0,
                0.12 + 0.76 * y / 7.0,
                0.16 + 0.68 * checker,
                1.0,
            )
    image = bpy.data.images.new(
        "GB_ProjectionTestTexture", width=8, height=8, alpha=True, float_buffer=True
    )
    image.colorspace_settings.name = "Non-Color"
    image.pixels.foreach_set(texture.reshape(-1))
    for material in asset.data.materials:
        material.use_nodes = True
        nodes = material.node_tree.nodes
        texture_node = nodes.new("ShaderNodeTexImage")
        texture_node.image = image
        texture_node.interpolation = "Closest"
        shader = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
        if shader:
            material.node_tree.links.new(texture_node.outputs["Color"], shader.inputs["Base Color"])
        nodes.active = texture_node
        if material.texture_paint_images:
            material.paint_active_slot = 0
    return image, texture


def test_projection_roundtrip(asset, texture):
    projection_scene = bpy.data.scenes.new("GB_ProjectionRoundtrip")

    def setup_value_pass(scene, _temp_obj, _camera, _direction):
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.light = "FLAT"
        scene.display.shading.color_type = "TEXTURE"
        scene.display.shading.show_cavity = False
        scene.display.shading.show_shadows = False

    try:
        values, covered = project_views(
            projection_scene,
            asset,
            asset.data.uv_layers.active.name,
            256,
            4,
            setup_value_pass,
            value_channels=3,
        )
    finally:
        bpy.data.scenes.remove(projection_scene)
    rows, columns = np.indices(covered.shape)
    source_x = np.floor((columns + 0.5) / covered.shape[1] * 8).astype(np.int32) % 8
    source_y = np.floor((rows + 0.5) / covered.shape[0] * 8).astype(np.int32) % 8
    expected = texture[source_y, source_x, :3]
    error = float(np.mean(np.abs(values[covered] - expected[covered])))
    print(f"PROJECTION_ROUNDTRIP_MAE={error:.6f} covered={int(covered.sum())}")
    assert error < 0.02, f"Projection texture roundtrip MAE {error:.6f} >= 0.02"
    return error


def test_projection_performance():
    subdivisions = 226
    vertices = [
        (0.0, x / (subdivisions - 1) - 0.5, y / (subdivisions - 1) - 0.5)
        for y in range(subdivisions)
        for x in range(subdivisions)
    ]
    faces = []
    for y in range(subdivisions - 1):
        for x in range(subdivisions - 1):
            lower_left = y * subdivisions + x
            lower_right = lower_left + 1
            upper_right = lower_left + subdivisions + 1
            upper_left = lower_left + subdivisions
            faces.append((lower_left, lower_right, upper_right, upper_left))
    grid_mesh = bpy.data.meshes.new("GB_PerfGrid")
    grid_mesh.from_pydata(vertices, [], faces)
    grid_mesh.update()
    grid_uv = grid_mesh.uv_layers.new(name="UVMap")
    loop_vertices = np.empty(len(grid_mesh.loops), dtype=np.int32)
    grid_mesh.loops.foreach_get("vertex_index", loop_vertices)
    vertex_uvs = np.asarray(
        [
            (x / (subdivisions - 1), y / (subdivisions - 1))
            for y in range(subdivisions)
            for x in range(subdivisions)
        ],
        dtype=np.float32,
    )
    grid_uv.data.foreach_set("uv", vertex_uvs[loop_vertices].reshape(-1))
    grid_mesh.calc_loop_triangles()
    triangle_count = len(grid_mesh.loop_triangles)
    grid_obj = bpy.data.objects.new("GB_PerfGrid", grid_mesh)
    bpy.context.scene.collection.objects.link(grid_obj)

    build_start = time.perf_counter()
    flat_obj, flat_mesh = build_flat_uv_mesh(
        grid_obj, bpy.context.evaluated_depsgraph_get(), "UVMap"
    )
    build_seconds = time.perf_counter() - build_start
    bpy.data.objects.remove(flat_obj, do_unlink=True)
    bpy.data.meshes.remove(flat_mesh)

    projection_scene = bpy.data.scenes.new("GB_PerfProjection")

    def setup_value_pass(scene, _temp_obj, _camera, _direction):
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.light = "FLAT"
        scene.display.shading.color_type = "SINGLE"
        scene.display.shading.single_color = (0.5, 0.5, 0.5)
        scene.display.shading.show_cavity = False

    projection_start = time.perf_counter()
    try:
        project_views(
            projection_scene,
            grid_obj,
            "UVMap",
            64,
            1,
            setup_value_pass,
            value_channels=1,
        )
    finally:
        projection_seconds = time.perf_counter() - projection_start
        bpy.data.scenes.remove(projection_scene)
        bpy.data.objects.remove(grid_obj, do_unlink=True)
        bpy.data.meshes.remove(grid_mesh)
    total_seconds = build_seconds + projection_seconds
    print(
        f"PROJECTION_PERF triangles={triangle_count} "
        f"flat_mesh={build_seconds:.3f}s one_view={projection_seconds:.3f}s "
        f"total={total_seconds:.3f}s"
    )
    assert total_seconds <= 60.0, f"100k-triangle projection took {total_seconds:.3f}s"


def make_contact_sheet(previews, path):
    tile_w, tile_h, label_h = 256, 256, 24
    cols, rows = 3, math.ceil(len(previews) / 3)
    width, height = cols * tile_w, rows * (tile_h + label_h)
    canvas = np.zeros((height, width, 4), dtype=np.float32)
    canvas[:, :, :3] = 0.08
    canvas[:, :, 3] = 1.0
    for index, (label, image) in enumerate(previews):
        column, row = index % cols, index // cols
        x0, y0 = column * tile_w, row * (tile_h + label_h)
        src = image[:, :, :3]
        canvas[y0 : y0 + tile_h, x0 : x0 + tile_w, :3] = src
        text = label.upper().replace("_", " ")
        cursor = x0 + 6
        for char in text:
            glyph = GLYPHS.get(char, GLYPHS[" "])
            for gy, line in enumerate(glyph):
                for gx, value in enumerate(line):
                    if value == "1" and cursor + gx < x0 + tile_w:
                        canvas[y0 + tile_h + 4 + gy, cursor + gx, :3] = 1.0
            cursor += 6
        canvas[y0 + tile_h : y0 + tile_h + label_h, :, 3] = 1.0
    sheet = bpy.data.images.new("GB_Test_ContactSheet", width=width, height=height, alpha=True)
    sheet.pixels.foreach_set(canvas.reshape(-1))
    sheet.file_format = "PNG"
    sheet.filepath_raw = path
    sheet.save()
    bpy.data.images.remove(sheet)


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    game_baker.register()
    asset = make_scene_asset()
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(10, 0, 0))
    uvless = bpy.context.object
    uvless.name = "UnwrappedAsset"
    while uvless.data.uv_layers:
        uvless.data.uv_layers.remove(uvless.data.uv_layers[0])
    scene = bpy.context.scene
    settings = scene.game_baker
    settings.resolution = "256"
    settings.supersample = "1"
    settings.padding = 4
    settings.output_dir = OUTPUT_DIR + os.sep
    settings.file_format = "PNG"
    settings.png_depth = "8"
    settings.name_pattern = "{object}_{map}"
    settings.projection_views = 12
    assert settings.cycles_device == "GPU", "Cycles GPU is not the default"
    cycles_addon = bpy.context.preferences.addons.get("cycles")
    if not cycles_addon or cycles_addon.preferences.compute_device_type == "NONE":
        assert _effective_cycles_device("GPU") == "CPU", "GPU was not disabled without a device"
    settings.cycles_device = "CPU"
    settings.auto_unwrap = True
    settings.maps.clear()
    item = settings.maps.add()
    item.map_type = "NORMAL_WORLD"
    item.engine = "EEVEE"
    item.suffix = "normal_ws"
    item.cycles_samples = 2
    item.samples = 8
    settings.active_map_index = 0
    test_ui_draws(settings, asset)
    test_edge_aware_median_filter()
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    uvless.select_set(True)
    bpy.context.view_layer.objects.active = uvless
    assert not uvless.data.uv_layers
    unwrap_result = bpy.ops.game_baker.bake()
    assert "FINISHED" in unwrap_result, f"Auto-unwrap bake returned {unwrap_result}"
    assert uvless.data.uv_layers.get("GameBakerUV"), "Auto-unwrap did not create GameBakerUV"
    unwrap_path = os.path.join(OUTPUT_DIR, "UnwrappedAsset_normal_ws.png")
    assert os.path.exists(unwrap_path), f"Missing {unwrap_path}"
    unwrap_values = read_image(unwrap_path)
    unwrap_image = bpy.data.images.get("GB_UnwrappedAsset_normal_ws")
    assert unwrap_image is not None, "Auto-unwrap bake image was not retained"
    unwrap_memory = np.empty(len(unwrap_image.pixels), dtype=np.float32)
    unwrap_image.pixels.foreach_get(unwrap_memory)
    unwrap_memory = unwrap_memory.reshape((256, 256, 4))
    assert np.max(np.abs(unwrap_values - unwrap_memory)) <= (1.0 / 255.0 + 1e-5), (
        "Auto-unwrap PNG did not reload with the saved values"
    )

    _texture_image, texture = add_projection_texture(asset)
    test_projection_roundtrip(asset, texture)

    settings.maps.clear()
    for map_type, suffix, engine in MAPS:
        item = settings.maps.add()
        item.map_type = map_type
        item.engine = engine
        item.suffix = suffix
        item.cycles_samples = 2
        item.samples = 8
        item.cavity_ridge_factor = 1.0
        item.cavity_valley_factor = 1.0
        item.bevel_samples = 16
        item.strength = 4.0
        if map_type == "CURVATURE" and engine == "CYCLES":
            item.bevel_radius = 0.2
        if map_type == "THICKNESS":
            item.distance = 0.2
    settings.active_map_index = 0
    settings.projection_views = 12
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    asset.select_set(True)
    bpy.context.view_layer.objects.active = asset

    scene_count = len(bpy.data.scenes)
    object_count = len(bpy.data.objects)
    mesh_count = len(bpy.data.meshes)
    material_count = len(bpy.data.materials)
    original_images = set(bpy.data.images[:])
    node_counts = {material.name: len(material.node_tree.nodes) if material.use_nodes else 0 for material in bpy.data.materials}
    original_engine = scene.render.engine
    original_camera = scene.camera
    original_resolution = (scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage)
    original_selected = {obj.name for obj in bpy.context.selected_objects}
    original_active = bpy.context.view_layer.objects.active

    result = bpy.ops.game_baker.bake()
    assert "FINISHED" in result, f"Bake operator returned {result}"
    previews = []
    images = {}
    for map_type, suffix, _engine in MAPS:
        path = os.path.join(OUTPUT_DIR, f"TestAsset_{suffix}.png")
        assert os.path.exists(path), f"Missing {path}"
        values = read_image(path)
        assert values.shape[:2] == (256, 256), (map_type, values.shape)
        assert np.isfinite(values).all(), f"{map_type} contains non-finite values"
        assert values.min() >= -0.01 and values.max() <= 1.01, (
            map_type,
            values.min(),
            values.max(),
        )
        output_image = bpy.data.images.get(f"GB_TestAsset_{suffix}")
        assert output_image is not None, f"Missing retained Blender image for {suffix}"
        memory_values = np.empty(len(output_image.pixels), dtype=np.float32)
        output_image.pixels.foreach_get(memory_values)
        memory_values = memory_values.reshape((256, 256, 4))
        assert np.max(np.abs(values - memory_values)) <= (1.0 / 255.0 + 1e-5), (
            f"{suffix} PNG did not reload with the saved values"
        )
        images[suffix] = values
        previews.append((suffix, values))

    ao_top = sample_face(
        images["ao"], asset, lambda center, normal, _poly: center.x < 1.0 and center.z > 0.9 and normal.z > 0.9
    )
    ao_contact = sample_face(
        images["ao"], asset, lambda center, normal, _poly: abs(center.x) < 0.45 and abs(center.y) < 0.45 and center.z < 0.02 and normal.z < -0.9
    )
    assert ao_contact[0] < ao_top[0], f"AO contact {ao_contact[0]} not darker than top {ao_top[0]}"

    thin_value = sample_face(
        images["thickness"], asset, lambda center, normal, _poly: 2.8 < center.x < 3.2 and normal.y > 0.9
    )[0]
    thick_value = sample_face(
        images["thickness"], asset, lambda center, normal, _poly: 4.8 < center.x < 5.2 and normal.y > 0.9
    )[0]
    assert thin_value < thick_value, f"thin thickness {thin_value} not darker than block {thick_value}"

    for cavity_key in ("cavity_wb", "cavity_cycles"):
        flat_patch = sample_face_pixels(
            images[cavity_key],
            asset,
            lambda center, normal, _poly: center.x < 1 and center.z > 0.9 and normal.z > 0.9,
        )
        flat_mean = float(np.mean(flat_patch[:, 0]))
        flat_std = float(np.std(flat_patch[:, 0]))
        crease_cavity = sample_ground_contact(images[cavity_key], asset)[0]
        crease_delta = flat_mean - crease_cavity
        print(
            f"PROJECTION_CALIBRATION {cavity_key} flat_mean={flat_mean:.6f} "
            f"flat_std={flat_std:.6f} crease_delta={crease_delta:.6f}"
        )
        assert abs(flat_mean - 0.5) <= 0.03, (
            f"{cavity_key} flat mean {flat_mean:.6f} is not calibrated to 0.5"
        )
        assert flat_std < 0.03, f"{cavity_key} flat std {flat_std:.6f} is too noisy"
        assert crease_delta > 0.05, (
            f"{cavity_key} crease delta {crease_delta:.6f} is not strong enough"
        )

    for curvature_key in ("curvature_wb", "curvature_cycles"):
        flat_patch = sample_face_pixels(
            images[curvature_key],
            asset,
            lambda center, normal, _poly: center.x < 1 and center.z > 0.9 and normal.z > 0.9,
        )
        flat_mean = float(np.mean(flat_patch[:, 0]))
        flat_std = float(np.std(flat_patch[:, 0]))
        edge_curvature = sample_face(
            images[curvature_key],
            asset,
            lambda center, normal, _poly: 6.4 < center.x < 7.6
            and abs(normal.x) > 0.1
            and abs(normal.z) > 0.1,
        )[0]
        edge_delta = edge_curvature - flat_mean
        print(
            f"PROJECTION_CALIBRATION {curvature_key} flat_mean={flat_mean:.6f} "
            f"flat_std={flat_std:.6f} edge_delta={edge_delta:.6f}"
        )
        assert abs(flat_mean - 0.5) <= 0.03, (
            f"{curvature_key} flat mean {flat_mean:.6f} is not calibrated to 0.5"
        )
        assert flat_std < 0.03, f"{curvature_key} flat std {flat_std:.6f} is too noisy"
        assert edge_delta > 0.05, (
            f"{curvature_key} edge delta {edge_delta:.6f} is not strong enough"
        )

    flat_top_predicate = (
        lambda center, normal, _poly: center.x < 1
        and center.z > 0.9
        and normal.z > 0.9
    )
    for flat_key in ("cavity_wb", "curvature_wb"):
        flat_texels, flat_coords = sample_flat_polygon_texels(
            images[flat_key],
            asset,
            flat_top_predicate,
            edge_margin=8,
            return_coords=True,
        )
        deviations = np.abs(flat_texels[:, 0] - 0.5)
        p99_deviation = float(np.percentile(deviations, 99))
        outlier_coords = flat_coords[deviations >= 0.05]
        print(
            f"PROJECTION_FLAT_TAIL {flat_key} "
            f"p99_abs_deviation={p99_deviation:.6f} texels={len(flat_texels)} "
            f"outliers={len(outlier_coords)} coords={outlier_coords[:24].tolist()}"
        )
        assert p99_deviation < 0.05, (
            f"{flat_key} flat-face p99 deviation {p99_deviation:.6f} >= 0.05"
        )

    normal_top = sample_face(
        images["normal_ws"],
        asset,
        lambda center, normal, _poly: center.x < 1 and center.z > 0.9 and normal.z > 0.9,
    )
    assert np.allclose(normal_top[:3], (0.5, 0.5, 1.0), atol=0.08), normal_top
    assert images["position"][:, :, :3].min() >= 0 and images["position"][:, :, :3].max() <= 1
    id_colors = []
    for material_index in (0, 1):
        id_colors.append(
            sample_face(images["id"], asset, lambda _center, _normal, polygon: polygon.material_index == material_index)[:3]
        )
    assert np.linalg.norm(id_colors[0] - id_colors[1]) > 0.1, id_colors

    for item in settings.maps:
        item.enabled = item.suffix == "normal_ws"
    settings.active_map_index = next(
        index for index, item in enumerate(settings.maps) if item.suffix == "normal_ws"
    )
    rebake_result = bpy.ops.game_baker.bake()
    assert "FINISHED" in rebake_result, f"Repeat bake returned {rebake_result}"
    duplicates = [
        image.name for image in bpy.data.images
        if image.name.startswith("GB_") and ".001" in image.name
    ]
    assert not duplicates, f"Repeat bake created duplicate images: {duplicates}"

    for item in settings.maps:
        item.enabled = False
    preview_map = settings.maps.add()
    preview_map.map_type = "AO"
    preview_map.engine = "EEVEE"
    preview_map.suffix = "ao_preview"
    preview_map.samples = 8
    settings.projection_views = 12
    preview_result = bpy.ops.game_baker.bake()
    assert "FINISHED" in preview_result, f"Eevee AO preview returned {preview_result}"
    preview_path = os.path.join(OUTPUT_DIR, "TestAsset_ao_preview.png")
    assert os.path.exists(preview_path), f"Missing {preview_path}"
    preview_values = read_image(preview_path)
    assert np.isfinite(preview_values).all() and preview_values.min() >= -0.01 and preview_values.max() <= 1.01
    preview_image = bpy.data.images.get("GB_TestAsset_ao_preview")
    assert preview_image is not None
    preview_memory = np.empty(len(preview_image.pixels), dtype=np.float32)
    preview_image.pixels.foreach_get(preview_memory)
    preview_memory = preview_memory.reshape((256, 256, 4))
    assert np.max(np.abs(preview_values - preview_memory)) <= (1.0 / 255.0 + 1e-5), (
        "Eevee AO PNG did not reload with the saved values"
    )
    preview_top = sample_face(
        preview_values,
        asset,
        lambda center, normal, _poly: center.x < 1 and center.z > 0.9 and normal.z > 0.9,
    )[0]
    preview_top_texels = sample_flat_polygon_texels(
        preview_values, asset, flat_top_predicate, edge_margin=8
    )
    preview_flat_std = float(np.std(preview_top_texels[:, 0]))
    print(f"PROJECTION_AO_FLAT preview_std={preview_flat_std:.6f} texels={len(preview_top_texels)}")
    assert preview_flat_std < 0.05, (
        f"Eevee AO-preview flat-face std {preview_flat_std:.6f} >= 0.05"
    )
    preview_contact = sample_ground_contact(preview_values, asset)[0]
    ao_delta = preview_top - preview_contact
    print(f"PROJECTION_CONTRAST ao_preview crease_delta={ao_delta:.6f}")
    assert preview_contact < preview_top, (
        f"Eevee AO contact {preview_contact} not darker than top {preview_top}"
    )
    previews.append(("ao_preview", preview_values))

    test_projection_performance()

    assert len(bpy.data.scenes) == scene_count, "Temporary scenes leaked"
    assert len(bpy.data.objects) == object_count, "Temporary objects leaked"
    assert len(bpy.data.meshes) == mesh_count, "Temporary meshes leaked"
    assert len(bpy.data.materials) == material_count, "Temporary materials leaked"
    result_images = {f"GB_TestAsset_{suffix}" for _map_type, suffix, _engine in MAPS}
    result_images.update({"GB_UnwrappedAsset_normal_ws", "GB_TestAsset_ao_preview"})
    assert len(
        [
            image
            for image in bpy.data.images
            if image not in original_images and image.name not in result_images
        ]
    ) == 0, "Temporary images leaked"
    for material in bpy.data.materials:
        expected = node_counts.get(material.name)
        if expected is not None:
            assert len(material.node_tree.nodes) == expected, f"Material node tree changed: {material.name}"
    assert scene.render.engine == original_engine
    assert scene.camera == original_camera
    assert (scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage) == original_resolution
    assert {obj.name for obj in bpy.context.selected_objects} == original_selected
    assert bpy.context.view_layer.objects.active == original_active

    contact_path = os.path.join(OUTPUT_DIR, "contact_sheet.png")
    previews.append(("auto unwrap ws", unwrap_values))
    make_contact_sheet(previews, contact_path)
    print(f"TEST_PREVIEWS={OUTPUT_DIR}")
    print(f"TEST_CONTACT_SHEET={contact_path}")
    print("ALL GAME BAKER TESTS PASSED")


try:
    run()
except Exception:
    traceback.print_exc()
    raise
