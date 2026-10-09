import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)


MAPS = (
    ("AO", "Ambient Occlusion", "Contact and ambient occlusion", "SHADING_RENDERED", 0),
    ("THICKNESS", "Thickness", "Approximate mesh thickness", "MOD_SOLIDIFY", 1),
    ("CAVITY", "Cavity", "World-space cavity shading", "SHADING_WIRE", 2),
    ("CURVATURE", "Curvature", "Screen-space curvature shading", "MOD_BEVEL", 3),
    ("NORMAL_WORLD", "Normal (World)", "World-space normals", "ORIENTATION_GIMBAL", 4),
    ("NORMAL_OBJECT", "Normal (Object)", "Object-space normals", "ORIENTATION_LOCAL", 5),
    ("POSITION", "Position", "Normalized world position", "EMPTY_AXIS", 6),
    ("GRADIENT", "Gradient", "Height gradient", "COLOR", 7),
    ("ID", "ID", "Material, color attribute, or object ID", "MATERIAL", 8),
)

DEFAULT_SUFFIXES = {
    "AO": "ao",
    "THICKNESS": "thickness",
    "CAVITY": "cavity",
    "CURVATURE": "curvature",
    "NORMAL_WORLD": "normal_ws",
    "NORMAL_OBJECT": "normal_os",
    "POSITION": "position",
    "GRADIENT": "gradient",
    "ID": "id",
}

ENGINE_CHOICES = {
    "AO": (("CYCLES", "Cycles", ""), ("EEVEE", "Eevee (Preview)", "")),
    "THICKNESS": (("CYCLES", "Cycles", ""),),
    "CAVITY": (("WORKBENCH", "Workbench", ""), ("CYCLES", "Cycles", "")),
    "CURVATURE": (("WORKBENCH", "Workbench", ""), ("CYCLES", "Cycles", "")),
    "NORMAL_WORLD": (("EEVEE", "Eevee", ""), ("CYCLES", "Cycles", "")),
    "NORMAL_OBJECT": (("EEVEE", "Eevee", ""), ("CYCLES", "Cycles", "")),
    "POSITION": (("EEVEE", "Eevee", ""), ("CYCLES", "Cycles", "")),
    "GRADIENT": (("EEVEE", "Eevee", ""), ("CYCLES", "Cycles", "")),
    "ID": (("EEVEE", "Eevee", ""), ("CYCLES", "Cycles", "")),
}

ID_SOURCES = (
    ("MATERIAL", "Material", ""),
    ("COLOR_ATTRIBUTE", "Color Attribute", ""),
    ("OBJECT", "Object", ""),
)
GRADIENT_AXES = (("X", "X", ""), ("Y", "Y", ""), ("Z", "Z", ""))
PACK_SOURCES = tuple((key, label, "") for key, label, _, _, _ in MAPS) + (
    ("NONE", "None", ""),
    ("WHITE", "White", ""),
    ("BLACK", "Black", ""),
)
CHANNELS = (("R", "R", ""), ("G", "G", ""), ("B", "B", ""), ("A", "A", ""))


def engine_items(self, _context):
    return ENGINE_CHOICES.get(self.map_type, ENGINE_CHOICES["AO"])


def map_type_changed(self, _context):
    self.engine = ENGINE_CHOICES[self.map_type][0][0]
    self.suffix = DEFAULT_SUFFIXES[self.map_type]


def map_type_items(_self, _context):
    return MAPS


class BakeMap(bpy.types.PropertyGroup):
    enabled: BoolProperty(name="Enabled", default=True)
    map_type: EnumProperty(name="Map", items=map_type_items, update=map_type_changed)
    engine: EnumProperty(name="Engine", items=engine_items)
    suffix: StringProperty(name="Suffix", default="ao")
    samples: IntProperty(name="AO Samples", default=16, min=1, max=4096)
    cycles_samples: IntProperty(name="Cycles Samples", default=16, min=1, max=4096)
    distance: FloatProperty(name="Distance", default=0.0, min=0.0, soft_max=100.0, subtype="DISTANCE")
    only_local: BoolProperty(name="Only Local", default=False)
    cavity_ridge_factor: FloatProperty(name="Ridge", default=1.0, min=0.0, max=2.0)
    cavity_valley_factor: FloatProperty(name="Valley", default=1.0, min=0.0, max=2.0)
    bevel_radius: FloatProperty(name="Bevel Radius", default=0.0, min=0.0, soft_max=1.0, subtype="DISTANCE")
    bevel_samples: IntProperty(name="Bevel Samples", default=8, min=1, max=64)
    strength: FloatProperty(name="Strength", default=4.0, min=0.0, max=32.0)
    gradient_axis: EnumProperty(name="Axis", items=GRADIENT_AXES, default="Z")
    id_source: EnumProperty(name="ID Source", items=ID_SOURCES, default="MATERIAL")


class PackItem(bpy.types.PropertyGroup):
    suffix: StringProperty(name="Suffix", default="orm")
    r_source: EnumProperty(name="R", items=PACK_SOURCES, default="AO")
    g_source: EnumProperty(name="G", items=PACK_SOURCES, default="NONE")
    b_source: EnumProperty(name="B", items=PACK_SOURCES, default="NONE")
    a_source: EnumProperty(name="A", items=PACK_SOURCES, default="NONE")
    channel: EnumProperty(name="Source Channel", items=CHANNELS, default="R")


class GameBakerSettings(bpy.types.PropertyGroup):
    resolution: EnumProperty(
        name="Resolution",
        items=tuple((str(n), f"{n} px", "") for n in (256, 512, 1024, 2048, 4096, 8192)),
        default="2048",
    )
    supersample: EnumProperty(
        name="Supersampling",
        items=(("1", "1x", ""), ("2", "2x", ""), ("4", "4x", "")),
        default="2",
    )
    padding: IntProperty(name="Padding", default=16, min=0, max=256, subtype="PIXEL")
    uv_map: StringProperty(name="UV Map", default="")
    output_dir: StringProperty(name="Output Folder", subtype="DIR_PATH", default="//textures/")
    file_format: EnumProperty(
        name="Format",
        items=(("PNG", "PNG", ""), ("TARGA", "Targa", ""), ("OPEN_EXR", "OpenEXR", "")),
        default="PNG",
    )
    png_depth: EnumProperty(name="PNG Bit Depth", items=(("8", "8-bit", ""), ("16", "16-bit", "")), default="8")
    name_pattern: StringProperty(name="Name Pattern", default="{object}_{map}")
    cycles_device: EnumProperty(
        name="Cycles Device", items=(("CPU", "CPU", ""), ("GPU", "GPU", "")), default="GPU"
    )
    auto_unwrap: BoolProperty(name="Auto-unwrap objects without UVs", default=True)
    projection_views: IntProperty(name="Projection Views", default=42, min=1, max=512)
    maps: CollectionProperty(type=BakeMap)
    active_map_index: IntProperty(default=0)
    packs: CollectionProperty(type=PackItem)
    active_pack_index: IntProperty(default=0)


CLASSES = (BakeMap, PackItem, GameBakerSettings)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.game_baker = PointerProperty(type=GameBakerSettings)


def unregister():
    del bpy.types.Scene.game_baker
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
