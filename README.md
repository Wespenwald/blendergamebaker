# Game Baker

Game Baker is a Blender 5.2 extension for baking game-ready texture maps from
UV-mapped mesh objects. It uses a per-map render engine and keeps baked images
inside the current `.blend` file as well as saving them to disk.

Version 0.1 includes ambient occlusion, thickness, cavity, curvature,
world/object normals, position, height gradient, and ID maps. High-to-low
(cage) baking is not part of this version.

## Engine per map

| Map | Default engine | Notes |
| --- | --- | --- |
| AO | Cycles | Eevee is available as a preview option |
| Thickness | Cycles | AO-node based approximation |
| Cavity | Workbench | World-space cavity; Cycles is also available |
| Curvature | Workbench | Screen-space curvature; Cycles is also available |
| Normal (World), Normal (Object) | Eevee | Cycles is also available |
| Position, Gradient, ID | Eevee | Cycles is also available |

Eevee/Cycles normal, position, gradient, and ID maps render a UV-space flat
mesh with emission materials. Supersampling and padding are applied in the
post-processing step. Cycles curvature uses the Bevel shader node and an
ambient-occlusion concavity term; it does not use Pointiness. Cycles cavity
uses a short-range, local ambient-occlusion pass.

By default, Game Baker Smart-UV unwraps selected mesh objects that have no UV
maps, creating a `GameBakerUV` layer. This is the only bake-time change to
source mesh data. Disable **Auto-unwrap objects without UVs** in Advanced to
skip those objects instead.

## Install

Build a zip from the repository root with Blender 5.2:

```sh
blender --command extension build --source-dir game_baker --output-dir dist
```

In Blender, open **Edit > Preferences > Get Extensions > Install from Disk**,
select the generated `game_baker-0.1.0.zip`, then enable Game Baker.

Select one or more mesh objects with render UV maps. Open the 3D Viewport
sidebar with **N**, choose the **Game Baker** tab, add maps (or use **Add
Default Maps**), configure the output, and click **Bake N Maps**. Output maps
are saved under the chosen folder and retained in the current file as
`GB_<object>_<suffix>` images.

## Tests

Run the procedural regression scene in background mode:

```sh
blender -b --factory-startup --python tests/run_tests.py
```

The test bakes all maps at 256 px, checks both Workbench and Cycles cavity and
curvature, exercises the Eevee AO preview and auto-unwrap, writes PNG previews
and a contact sheet to `tests/output/`, and checks that saved files reload with
the expected values. Generated preview images and the contact sheet are
ignored by git.
