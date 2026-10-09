import bpy
import numpy as np


def _evaluated_copy(obj, depsgraph):
    evaluated = obj.evaluated_get(depsgraph)
    evaluated_mesh = evaluated.to_mesh()
    try:
        mesh = evaluated_mesh.copy()
    finally:
        evaluated.to_mesh_clear()
    copy = bpy.data.objects.new(f"GB_Bake_{obj.name}", mesh)
    copy.matrix_world = obj.matrix_world.copy()
    copy.data.materials.clear()
    for slot in obj.material_slots:
        copy.data.materials.append(slot.material.copy() if slot.material else None)
    if not copy.data.materials:
        copy.data.materials.append(bpy.data.materials.new(f"GB_Temp_{obj.name}"))
    return copy


def _math(nodes, operation, a, b=None, clamp=False):
    node = nodes.new("ShaderNodeMath")
    node.operation = operation
    node.inputs[0].default_value = a if isinstance(a, (int, float)) else 0.0
    if b is not None:
        node.inputs[1].default_value = b if isinstance(b, (int, float)) else 0.0
    node.use_clamp = clamp
    if not isinstance(a, (int, float)):
        node.id_data.links.new(a, node.inputs[0])
    if b is not None and not isinstance(b, (int, float)):
        node.id_data.links.new(b, node.inputs[1])
    return node


def _inject_materials(obj, image, map_item, distance, bevel_radius):
    for index, material in enumerate(list(obj.data.materials)):
        if material is None:
            material = bpy.data.materials.new(f"GB_Temp_{obj.name}")
            obj.data.materials[index] = material
        material.use_nodes = True
        nodes = material.node_tree.nodes
        for node in nodes:
            if node.type == "OUTPUT_MATERIAL":
                node.is_active_output = False
        emission = nodes.new("ShaderNodeEmission")
        output = nodes.new("ShaderNodeOutputMaterial")
        output.is_active_output = True
        if map_item.map_type in {"AO", "THICKNESS", "CAVITY"}:
            ao = nodes.new("ShaderNodeAmbientOcclusion")
            ao.samples = map_item.samples
            ao.inputs["Distance"].default_value = distance
            ao.only_local = True if map_item.map_type == "CAVITY" else map_item.only_local
            ao.inside = map_item.map_type == "THICKNESS"
            if map_item.map_type == "CAVITY":
                scale = nodes.new("ShaderNodeMixRGB")
                scale.blend_type = "MULTIPLY"
                scale.inputs["Fac"].default_value = 1.0
                scale.inputs["Color2"].default_value = (0.5, 0.5, 0.5, 1.0)
                material.node_tree.links.new(ao.outputs["AO"], scale.inputs["Color1"])
                material.node_tree.links.new(scale.outputs["Color"], emission.inputs["Color"])
            else:
                material.node_tree.links.new(ao.outputs["AO"], emission.inputs["Color"])
        elif map_item.map_type == "CURVATURE":
            bevel = nodes.new("ShaderNodeBevel")
            bevel.inputs["Radius"].default_value = bevel_radius
            bevel.samples = map_item.bevel_samples
            geometry = nodes.new("ShaderNodeNewGeometry")
            dot = nodes.new("ShaderNodeVectorMath")
            dot.operation = "DOT_PRODUCT"
            material.node_tree.links.new(bevel.outputs["Normal"], dot.inputs[0])
            material.node_tree.links.new(geometry.outputs["Normal"], dot.inputs[1])
            one_minus_dot = _math(nodes, "SUBTRACT", 1.0, dot.outputs["Value"], clamp=True)
            magnitude = _math(nodes, "MULTIPLY", one_minus_dot.outputs[0], map_item.strength, clamp=True)
            ao = nodes.new("ShaderNodeAmbientOcclusion")
            ao.samples = map_item.samples
            ao.inputs["Distance"].default_value = 2.0 * bevel_radius
            ao.only_local = True
            ao.inside = False
            one_minus_ao = _math(nodes, "SUBTRACT", 1.0, ao.outputs["AO"], clamp=True)
            concavity = _math(nodes, "MULTIPLY", one_minus_ao.outputs[0], 4.0, clamp=True)
            twice_concavity = _math(nodes, "MULTIPLY", concavity.outputs[0], 2.0)
            sign = _math(nodes, "SUBTRACT", 1.0, twice_concavity.outputs[0])
            signed_magnitude = _math(nodes, "MULTIPLY", magnitude.outputs[0], sign.outputs[0])
            half_offset = _math(nodes, "MULTIPLY", signed_magnitude.outputs[0], 0.5)
            value = _math(nodes, "ADD", half_offset.outputs[0], 0.5)
            material.node_tree.links.new(value.outputs[0], emission.inputs["Color"])
        else:
            raise ValueError(f"Unsupported Cycles map: {map_item.map_type}")
        material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
        image_node = nodes.new("ShaderNodeTexImage")
        image_node.image = image
        image_node.select = True
        nodes.active = image_node


def bake_map(
    scene,
    source_obj,
    map_item,
    resolution,
    supersample,
    distance,
    uv_name="",
    support_objects=(),
    device="CPU",
    bevel_radius=0.0,
):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    temp_obj = _evaluated_copy(source_obj, depsgraph)
    scene.collection.objects.link(temp_obj)
    supports = []
    for support in support_objects:
        if support == source_obj or support.type != "MESH":
            continue
        blocker = _evaluated_copy(support, depsgraph)
        scene.collection.objects.link(blocker)
        supports.append(blocker)
    width = resolution * supersample
    image = bpy.data.images.new(
        f"GB_BakeTarget_{source_obj.name}_{map_item.map_type}",
        width=width,
        height=width,
        alpha=True,
        float_buffer=True,
    )
    image.colorspace_settings.name = "Non-Color"
    _inject_materials(temp_obj, image, map_item, distance, bevel_radius)
    scene.render.engine = "CYCLES"
    scene.cycles.samples = max(1, map_item.cycles_samples)
    scene.cycles.device = "GPU" if device == "GPU" else "CPU"
    scene.render.bake.margin = 0
    scene.render.bake.use_clear = True
    scene.render.bake.target = "IMAGE_TEXTURES"
    scene.render.bake.use_pass_color = True
    scene.render.bake.use_pass_direct = False
    scene.render.bake.use_pass_indirect = False
    temp_obj.select_set(True, view_layer=scene.view_layers[0])
    scene.view_layers[0].objects.active = temp_obj
    try:
        def run_bake():
            with bpy.context.temp_override(
                scene=scene,
                view_layer=scene.view_layers[0],
                object=temp_obj,
                active_object=temp_obj,
                selected_objects=[temp_obj],
                selected_editable_objects=[temp_obj],
            ):
                bpy.ops.object.bake(type="EMIT", margin=0, use_clear=True, uv_layer=uv_name or "")
        try:
            run_bake()
        except RuntimeError:
            if device != "GPU":
                raise
            scene.cycles.device = "CPU"
            image.pixels.foreach_set(np.zeros(len(image.pixels), dtype=np.float32))
            run_bake()
        pixels = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(pixels)
        return pixels.reshape((width, width, 4)).copy()
    finally:
        bpy.data.images.remove(image)
        for temporary in [temp_obj, *supports]:
            data = temporary.data
            materials = list(data.materials) if data else []
            bpy.data.objects.remove(temporary, do_unlink=True)
            if data and data.users == 0:
                bpy.data.meshes.remove(data)
            for material in materials:
                if material and material.users == 0:
                    bpy.data.materials.remove(material)
