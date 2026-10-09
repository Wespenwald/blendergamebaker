import bmesh
import math
import os
import time

import bpy
import numpy as np
from bpy.props import EnumProperty, IntProperty

from .bake import common, cycles, eevee, post, projection
from .props import DEFAULT_SUFFIXES, ENGINE_CHOICES, MAPS


def _datablock_ids():
    return {
        "objects": set(bpy.data.objects[:]),
        "meshes": set(bpy.data.meshes[:]),
        "materials": set(bpy.data.materials[:]),
        "images": set(bpy.data.images[:]),
        "cameras": set(bpy.data.cameras[:]),
        "scenes": set(bpy.data.scenes[:]),
    }


def _cleanup_new(before, keep_images=()):
    keep_images = set(keep_images)
    for obj in list(bpy.data.objects):
        if obj not in before["objects"]:
            bpy.data.objects.remove(obj, do_unlink=True)
    for collection, key in (
        (bpy.data.scenes, "scenes"),
        (bpy.data.meshes, "meshes"),
        (bpy.data.materials, "materials"),
        (bpy.data.cameras, "cameras"),
        (bpy.data.images, "images"),
    ):
        for datablock in list(collection):
            if datablock not in before[key] and datablock not in keep_images:
                collection.remove(datablock)


def _map_texture_array(raw, coverage, size, factor):
    if raw.shape[:2] == coverage.shape:
        values = raw.astype(np.float32, copy=True)
        if values.shape[2] == 3:
            values = np.concatenate((values, np.ones((*values.shape[:2], 1), dtype=np.float32)), axis=2)
        values[:, :, 3] = coverage.astype(np.float32)
        return values, coverage
    rgb = raw[:, :, :3]
    rgba = np.concatenate((rgb, np.ones((*rgb.shape[:2], 1), dtype=np.float32)), axis=2)
    reduced, out_mask = post.downsample_covered(rgba, coverage, factor)
    reduced[:, :, 3] = out_mask.astype(np.float32)
    return reduced, out_mask


def _render_scene():
    scene = bpy.data.scenes.new("GB_TempScene")
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    return scene


def _smart_unwrap_object(context, obj):
    view_layer = context.view_layer
    original_selection = list(context.selected_objects)
    original_active = view_layer.objects.active
    original_mode = context.mode
    created_uv = None
    completed = False
    if original_mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for selected in list(context.selected_objects):
        selected.select_set(False)
    obj.select_set(True)
    view_layer.objects.active = obj
    created_uv = obj.data.uv_layers.new(name="GameBakerUV")
    obj.data.uv_layers.active = created_uv
    try:
        bpy.ops.object.mode_set(mode="EDIT")
        bm = bmesh.from_edit_mesh(obj.data)
        selection = (
            [element.select for element in bm.verts],
            [element.select for element in bm.edges],
            [element.select for element in bm.faces],
        )
        bpy.ops.mesh.select_all(action="SELECT")
        result = bpy.ops.uv.smart_project(
            angle_limit=math.radians(66.0),
            island_margin=0.02,
        )
        if "FINISHED" not in result:
            raise RuntimeError("Smart UV Project was cancelled")
        bm = bmesh.from_edit_mesh(obj.data)
        for elements, saved in zip((bm.verts, bm.edges, bm.faces), selection):
            for element, was_selected in zip(elements, saved):
                element.select_set(was_selected)
        bmesh.update_edit_mesh(obj.data)
        completed = True
    finally:
        if obj.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        if not completed and created_uv in obj.data.uv_layers[:]:
            obj.data.uv_layers.remove(created_uv)
        for selected in list(context.selected_objects):
            selected.select_set(False)
        for selected in original_selection:
            if selected.name in view_layer.objects:
                selected.select_set(True)
        view_layer.objects.active = original_active
        if original_mode.startswith("EDIT") and original_active:
            bpy.ops.object.mode_set(mode="EDIT")


def _save_result(settings, obj, map_item, values, scene):
    suffix = map_item.suffix.strip() or DEFAULT_SUFFIXES[map_item.map_type]
    filename = settings.name_pattern.format(object=obj.name, map=suffix)
    extension = {"PNG": ".png", "TARGA": ".tga", "OPEN_EXR": ".exr"}[settings.file_format]
    filepath = os.path.join(bpy.path.abspath(settings.output_dir), filename + extension)
    name = f"GB_{obj.name}_{suffix}"
    return post.save_image(
        name, values, filepath, settings.file_format, settings.png_depth, scene=scene
    )


class GAMEBAKER_OT_add_map(bpy.types.Operator):
    bl_idname = "game_baker.add_map"
    bl_label = "Add Map"
    map_type: EnumProperty(items=MAPS, default="AO")

    def execute(self, context):
        settings = context.scene.game_baker
        item = settings.maps.add()
        item.map_type = self.map_type
        settings.active_map_index = len(settings.maps) - 1
        return {"FINISHED"}


class GAMEBAKER_OT_add_default_maps(bpy.types.Operator):
    bl_idname = "game_baker.add_default_maps"
    bl_label = "Add Default Maps"

    def execute(self, context):
        settings = context.scene.game_baker
        if settings.maps:
            return {"CANCELLED"}
        for map_type in ("AO", "THICKNESS", "CURVATURE", "CAVITY", "NORMAL_WORLD", "POSITION", "ID"):
            item = settings.maps.add()
            item.map_type = map_type
        settings.active_map_index = 0
        return {"FINISHED"}


class GAMEBAKER_OT_remove_map(bpy.types.Operator):
    bl_idname = "game_baker.remove_map"
    bl_label = "Remove Map"

    def execute(self, context):
        settings = context.scene.game_baker
        if settings.maps:
            settings.maps.remove(settings.active_map_index)
            settings.active_map_index = min(settings.active_map_index, len(settings.maps) - 1)
        return {"FINISHED"}


class GAMEBAKER_OT_move_map(bpy.types.Operator):
    bl_idname = "game_baker.move_map"
    bl_label = "Move Map"
    direction: IntProperty(default=1)

    def execute(self, context):
        settings = context.scene.game_baker
        index = settings.active_map_index
        target = index + self.direction
        if 0 <= target < len(settings.maps):
            settings.maps.move(index, target)
            settings.active_map_index = target
        return {"FINISHED"}


class GAMEBAKER_OT_add_pack(bpy.types.Operator):
    bl_idname = "game_baker.add_pack"
    bl_label = "Add Pack"

    def execute(self, context):
        settings = context.scene.game_baker
        item = settings.packs.add()
        item.suffix = "orm"
        item.r_source, item.g_source, item.b_source = "AO", "THICKNESS", "NONE"
        settings.active_pack_index = len(settings.packs) - 1
        return {"FINISHED"}


class GAMEBAKER_OT_remove_pack(bpy.types.Operator):
    bl_idname = "game_baker.remove_pack"
    bl_label = "Remove Pack"

    def execute(self, context):
        settings = context.scene.game_baker
        if settings.packs:
            settings.packs.remove(settings.active_pack_index)
            settings.active_pack_index = min(settings.active_pack_index, len(settings.packs) - 1)
        return {"FINISHED"}


class GAMEBAKER_OT_bake(bpy.types.Operator):
    bl_idname = "game_baker.bake"
    bl_label = "Bake Game Maps"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return bool(context and context.scene and context.scene.game_baker)

    def execute(self, context):
        settings = context.scene.game_baker
        objects = [obj for obj in context.selected_objects if obj.type == "MESH"]
        maps = [item for item in settings.maps if item.enabled]
        if not objects or not maps:
            self.report({"ERROR"}, "Select mesh objects and enable at least one map")
            return {"CANCELLED"}
        resolution = int(settings.resolution)
        supersample = int(settings.supersample)
        total_steps = len(objects) * len(maps)
        step = 0
        timings = []
        kept_images = []
        auto_unwrapped = []
        wm = context.window_manager
        wm.progress_begin(0, total_steps)
        try:
            for obj in objects:
                depsgraph = context.evaluated_depsgraph_get()
                had_uvs = bool(obj.data.uv_layers)
                if not had_uvs and settings.auto_unwrap:
                    _smart_unwrap_object(context, obj)
                    auto_unwrapped.append(obj.name)
                    depsgraph = context.evaluated_depsgraph_get()
                if had_uvs and settings.uv_map and settings.uv_map not in obj.data.uv_layers:
                    uv_name = ""
                else:
                    uv_layer = (
                        obj.data.uv_layers.get(settings.uv_map)
                        if had_uvs and settings.uv_map
                        else obj.data.uv_layers.active
                    )
                    if not uv_layer and obj.data.uv_layers:
                        uv_layer = obj.data.uv_layers[0]
                    uv_name = uv_layer.name if uv_layer else ""
                    if not had_uvs and obj.data.uv_layers:
                        uv_name = obj.data.uv_layers.active.name
                if not uv_name:
                    self.report({"WARNING"}, f"{obj.name} skipped: no active render UV map")
                    step += len(maps)
                    wm.progress_update(step)
                    continue
                before = _datablock_ids()
                scene = _render_scene()
                try:
                    size = resolution * supersample
                    coverage_hi = common.uv_coverage(obj, size, uv_name)
                    if supersample > 1:
                        coverage = coverage_hi.reshape(
                            resolution, supersample, resolution, supersample
                        ).any(axis=(1, 3))
                    else:
                        coverage = coverage_hi
                    raw_maps = {}
                    flat_obj, _flat_mesh = common.build_flat_uv_mesh(obj, depsgraph, uv_name)
                    for map_item in maps:
                        start = time.perf_counter()
                        if map_item.map_type == "AO" and map_item.engine == "EEVEE":
                            raw = projection.render_projection_map(
                                scene, obj, map_item, resolution, settings.projection_views, uv_name
                            )
                            values = post.image_to_array(raw)
                            values[:, :, 3] = coverage.astype(np.float32)
                        elif map_item.map_type == "CURVATURE" and map_item.engine == "CYCLES":
                            bbox_diagonal = common.object_bbox(obj, depsgraph)[2]
                            radius = map_item.bevel_radius or bbox_diagonal * 0.01
                            raw = cycles.bake_map(
                                scene,
                                obj,
                                map_item,
                                resolution,
                                supersample,
                                2.0 * radius,
                                uv_name,
                                objects,
                                settings.cycles_device,
                                radius,
                            )
                            values, mask_out = _map_texture_array(raw, coverage_hi, resolution, supersample)
                            values[:, :, 3] = mask_out.astype(np.float32)
                        elif map_item.map_type in {"AO", "THICKNESS", "CAVITY"} and not (
                            map_item.map_type == "AO" and map_item.engine == "EEVEE"
                        ) and not (map_item.map_type == "CAVITY" and map_item.engine == "WORKBENCH"):
                            distance_factor = {
                                "AO": 0.2,
                                "THICKNESS": 0.5,
                                "CAVITY": 0.02,
                            }[map_item.map_type]
                            distance = common.effective_distance(
                                obj,
                                map_item.distance,
                                distance_factor,
                                depsgraph,
                            )
                            raw = cycles.bake_map(
                                scene,
                                obj,
                                map_item,
                                resolution,
                                supersample,
                                distance,
                                uv_name,
                                objects,
                                settings.cycles_device,
                            )
                            values, mask_out = _map_texture_array(raw, coverage_hi, resolution, supersample)
                            values[:, :, 3] = mask_out.astype(np.float32)
                        elif map_item.map_type in {"CAVITY", "CURVATURE"}:
                            raw = projection.render_projection_map(
                                scene, obj, map_item, resolution, settings.projection_views, uv_name
                            )
                            values = post.image_to_array(raw)
                            values[:, :, 3] = coverage.astype(np.float32)
                        else:
                            pixels = eevee.render_attribute_map(
                                scene, obj, flat_obj, map_item, resolution, supersample, depsgraph
                            )
                            values, mask_out = _map_texture_array(pixels, coverage_hi, resolution, supersample)
                            values[:, :, 3] = mask_out.astype(np.float32)
                        if values.shape[:2] != (resolution, resolution):
                            resized = np.zeros((resolution, resolution, 4), dtype=np.float32)
                            height = min(resolution, values.shape[0])
                            width = min(resolution, values.shape[1])
                            resized[:height, :width] = values[:height, :width]
                            values = resized
                        known = values[:, :, 3] > 0.5
                        rgb = post.fill_holes(values[:, :, :3], known, coverage)
                        values[:, :, :3] = post.pad_image(rgb, coverage, settings.padding)
                        image = _save_result(settings, obj, map_item, values, scene)
                        kept_images.append(image)
                        raw_maps[map_item.map_type] = values
                        elapsed = time.perf_counter() - start
                        timings.append((obj.name, map_item.suffix, map_item.engine, elapsed))
                        print(
                            f"[Game Baker] {obj.name} {map_item.suffix} "
                            f"({map_item.engine}): {elapsed:.2f}s"
                        )
                        step += 1
                        wm.progress_update(step)
                    for pack in settings.packs:
                        packed = post.pack_channels(pack, raw_maps, resolution, settings.padding)
                        filepath = os.path.join(
                            bpy.path.abspath(settings.output_dir),
                            settings.name_pattern.format(object=obj.name, map=pack.suffix) + ".png",
                        )
                        image = post.save_image(
                            f"GB_{obj.name}_{pack.suffix}",
                            packed,
                            filepath,
                            "PNG",
                            settings.png_depth,
                            scene=scene,
                        )
                        kept_images.append(image)
                finally:
                    _cleanup_new(before, kept_images)
        except Exception as exc:
            self.report({"ERROR"}, f"Bake failed: {exc}")
            raise
        finally:
            wm.progress_end()
        info = f"Baked {len(timings)} maps"
        if auto_unwrapped:
            info += f"; auto-unwrapped: {', '.join(auto_unwrapped)}"
        self.report({"INFO"}, info)
        return {"FINISHED"}


CLASSES = (
    GAMEBAKER_OT_add_map,
    GAMEBAKER_OT_add_default_maps,
    GAMEBAKER_OT_remove_map,
    GAMEBAKER_OT_move_map,
    GAMEBAKER_OT_add_pack,
    GAMEBAKER_OT_remove_pack,
    GAMEBAKER_OT_bake,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
