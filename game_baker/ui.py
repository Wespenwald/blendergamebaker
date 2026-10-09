import bpy
from bpy.types import UIList

from .props import MAPS


def _map_label(map_type):
    return next((label for key, label, *_ in MAPS if key == map_type), map_type.title())


class GAMEBAKER_UL_maps(UIList):
    bl_idname = "GAMEBAKER_UL_maps"

    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname, _index):
        row = layout.row(align=True)
        row.prop(item, "enabled", text="")
        icon = next((entry[3] for entry in MAPS if entry[0] == item.map_type), "TEXTURE")
        row.label(text="", icon=icon)
        row.label(text=_map_label(item.map_type))
        row.alignment = "RIGHT"
        row.label(text=item.engine.title())


class GAMEBAKER_UL_packs(UIList):
    bl_idname = "GAMEBAKER_UL_packs"

    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname, _index):
        layout.prop(item, "suffix", text="", emboss=False, icon="IMAGE")


class GAMEBAKER_MT_add_map(bpy.types.Menu):
    bl_idname = "GAMEBAKER_MT_add_map"
    bl_label = "Add Map"

    def draw(self, _context):
        for map_type, label, *_ in MAPS:
            op = self.layout.operator("game_baker.add_map", text=label)
            op.map_type = map_type


class GAMEBAKER_PT_main(bpy.types.Panel):
    bl_label = "Game Baker"
    bl_idname = "GAMEBAKER_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Game Baker"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.game_baker
        selected = [obj for obj in context.selected_objects if obj.type == "MESH"]
        layout.label(text=f"{len(selected)} mesh object{'s' if len(selected) != 1 else ''} selected")
        active = context.view_layer.objects.active
        if not selected:
            layout.label(text="Select one or more mesh objects", icon="INFO")
        elif not settings.auto_unwrap and any(not obj.data.uv_layers for obj in selected):
            layout.label(text="Selected objects need UV maps", icon="ERROR")
        if active and active.type == "MESH":
            row = layout.row()
            row.label(text="UV Map")
            if active.data.uv_layers:
                row.prop_search(settings, "uv_map", active.data, "uv_layers", text="")
            else:
                row.prop(settings, "uv_map", text="")
        layout.separator()
        layout.label(text="Maps")
        row = layout.row()
        row.template_list(
            "GAMEBAKER_UL_maps",
            "",
            settings,
            "maps",
            settings,
            "active_map_index",
            rows=3,
        )
        controls = row.column(align=True)
        controls.menu("GAMEBAKER_MT_add_map", text="", icon="ADD")
        controls.operator("game_baker.remove_map", text="", icon="REMOVE")
        controls.separator()
        op = controls.operator("game_baker.move_map", text="", icon="TRIA_UP")
        op.direction = -1
        op = controls.operator("game_baker.move_map", text="", icon="TRIA_DOWN")
        op.direction = 1
        if not settings.maps:
            layout.operator("game_baker.add_default_maps", icon="ADD")
        elif 0 <= settings.active_map_index < len(settings.maps):
            item = settings.maps[settings.active_map_index]
            box = layout.box()
            box.prop(item, "engine")
            box.prop(item, "suffix")
            if item.map_type in {"AO", "THICKNESS"} or (
                item.map_type == "CAVITY" and item.engine == "CYCLES"
            ):
                box.prop(item, "samples")
                box.prop(item, "cycles_samples")
                box.prop(item, "distance")
                if item.map_type != "CAVITY":
                    box.prop(item, "only_local")
            if item.map_type in {"CAVITY", "CURVATURE"} and item.engine == "WORKBENCH":
                box.prop(item, "cavity_ridge_factor")
                box.prop(item, "cavity_valley_factor")
            if item.map_type == "CURVATURE" and item.engine == "CYCLES":
                box.prop(item, "bevel_radius")
                box.prop(item, "bevel_samples")
                box.prop(item, "strength")
            if item.map_type == "GRADIENT":
                box.prop(item, "gradient_axis")
            if item.map_type == "ID":
                box.prop(item, "id_source")
        layout.separator()
        bake_row = layout.row()
        bake_row.scale_y = 1.6
        bake_row.operator(
            "game_baker.bake",
            text=f"Bake {sum(item.enabled for item in settings.maps)} Maps",
            icon="RENDER_STILL",
        )


class GAMEBAKER_PT_output(bpy.types.Panel):
    bl_label = "Output"
    bl_idname = "GAMEBAKER_PT_output"
    bl_parent_id = "GAMEBAKER_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Game Baker"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        settings = context.scene.game_baker
        layout = self.layout
        layout.prop(settings, "resolution")
        layout.prop(settings, "supersample")
        layout.prop(settings, "padding")
        layout.prop(settings, "file_format")
        if settings.file_format == "PNG":
            layout.prop(settings, "png_depth")
        layout.prop(settings, "output_dir")
        layout.prop(settings, "name_pattern")


class GAMEBAKER_PT_packing(bpy.types.Panel):
    bl_label = "Packing"
    bl_idname = "GAMEBAKER_PT_packing"
    bl_parent_id = "GAMEBAKER_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Game Baker"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        settings = context.scene.game_baker
        layout = self.layout
        row = layout.row()
        row.template_list("GAMEBAKER_UL_packs", "", settings, "packs", settings, "active_pack_index", rows=2)
        controls = row.column(align=True)
        controls.operator("game_baker.add_pack", text="", icon="ADD")
        controls.operator("game_baker.remove_pack", text="", icon="REMOVE")
        if 0 <= settings.active_pack_index < len(settings.packs):
            item = settings.packs[settings.active_pack_index]
            col = layout.column(align=True)
            col.prop(item, "r_source")
            col.prop(item, "g_source")
            col.prop(item, "b_source")
            col.prop(item, "a_source")
            col.prop(item, "channel")


class GAMEBAKER_PT_advanced(bpy.types.Panel):
    bl_label = "Advanced"
    bl_idname = "GAMEBAKER_PT_advanced"
    bl_parent_id = "GAMEBAKER_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Game Baker"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        settings = context.scene.game_baker
        self.layout.prop(settings, "cycles_device")
        self.layout.prop(settings, "projection_views")
        self.layout.prop(settings, "auto_unwrap")


CLASSES = (
    GAMEBAKER_UL_maps,
    GAMEBAKER_UL_packs,
    GAMEBAKER_MT_add_map,
    GAMEBAKER_PT_main,
    GAMEBAKER_PT_output,
    GAMEBAKER_PT_packing,
    GAMEBAKER_PT_advanced,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
