bl_info = {
    "name": "Game Baker",
    "author": "Wespenwald",
    "version": (0, 1, 0),
    "blender": (5, 2, 0),
    "location": "View3D > Sidebar > Game Baker",
    "description": "Bake game-ready texture maps",
    "category": "UV",
}

import bpy

from . import ops, props, ui


def register():
    props.register()
    ops.register()
    ui.register()


def unregister():
    ui.unregister()
    ops.unregister()
    props.unregister()
