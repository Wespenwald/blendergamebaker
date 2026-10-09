import math
import os
import sys
import traceback

import bpy
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import game_baker

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


def sample_face(image, obj, predicate, radius=3):
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
    return np.mean(covered, axis=0)


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
        item.bevel_samples = 8
        item.strength = 4.0
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
        flat_cavity = sample_face(
            images[cavity_key],
            asset,
            lambda center, normal, _poly: center.x < 1 and center.z > 0.9 and normal.z > 0.9,
        )[0]
        crease_cavity = sample_face(
            images[cavity_key],
            asset,
            lambda center, normal, _poly: abs(center.x) < 0.45
            and abs(center.y) < 0.45
            and center.z < 0.02
            and normal.z < -0.9,
        )[0]
        assert crease_cavity < flat_cavity, (
            f"{cavity_key} crease {crease_cavity} not below flat {flat_cavity}"
        )

    for curvature_key in ("curvature_wb", "curvature_cycles"):
        flat_curvature = sample_face(
            images[curvature_key],
            asset,
            lambda center, normal, _poly: center.x < 1 and center.z > 0.9 and normal.z > 0.9,
        )[0]
        edge_curvature = sample_face(
            images[curvature_key],
            asset,
            lambda center, normal, _poly: 6.4 < center.x < 7.6
            and abs(normal.x) > 0.1
            and abs(normal.z) > 0.1,
        )[0]
        assert edge_curvature > flat_curvature, (
            f"{curvature_key} edge {edge_curvature} not above flat {flat_curvature}"
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
        item.enabled = False
    preview_map = settings.maps.add()
    preview_map.map_type = "AO"
    preview_map.engine = "EEVEE"
    preview_map.suffix = "ao_preview"
    preview_map.samples = 8
    settings.projection_views = 2
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
    preview_contact = sample_face(
        preview_values,
        asset,
        lambda center, normal, _poly: abs(center.x) < 0.45
        and abs(center.y) < 0.45
        and center.z < 0.02
        and normal.z < -0.9,
    )[0]
    assert preview_contact < preview_top, (
        f"Eevee AO contact {preview_contact} not darker than top {preview_top}"
    )
    previews.append(("ao_preview", preview_values))

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
