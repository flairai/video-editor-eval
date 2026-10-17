"""Build one of the three camera-task scenes in Blender from Poly Haven assets and render views.

    /Applications/Blender.app/Contents/MacOS/Blender -b -P generators/camera-path/build/scene.py -- \
        --scene apartment --variant base --assets generators/camera-path/sources/assets --views views.json --out renders/ [--samples 64]

Scenes (metres, z up; compass yaw: 0 faces +y "north", 90 faces +x "east"):
- apartment: a living room (y 0-5, cream walls) and a dining room (y 5.15-10.15, sage walls) joined by a
  doorway at x 3.6-4.8 in the wall between them. Variant dining-b repaints the dining room terracotta and
  swaps its table, chairs and lamp.
- village: a cobbled square around a horse statue (0, 0), houses on every side, and an arched passage
  (x 3.5-8.5, y 14-17, open below 5.5 m) on the north side into a street that runs to y 48. Variant
  street-b repaints the street's houses and the building that closes it, and moves its props.
- gallery: one hall (x 0-14, y 0-9, 4.2 m high) with twelve framed paintings on its walls and three
  sculptures down the middle. Variant painting-swap replaces the painting at the middle of the east wall.
Each view in views.json: name, position [x, y, z], yaw, pitch (degrees down), lens (mm on a 36 mm sensor),
width, height. Writes <out>/<name>.png and <out>/cameras.json with each view's camera-to-world matrix in
OpenCV axes (x right, y down, z forward) and its intrinsics.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

A = None  # assets root
MATS: dict = {}


# ---------------------------------------------------------------- helpers

def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, choices=["apartment", "village", "gallery"])
    ap.add_argument("--variant", default="base")
    ap.add_argument("--assets", required=True)
    ap.add_argument("--views", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=64)
    return ap.parse_args(argv)


def tex_material(tid: str, tile: float, tint=(1, 1, 1), rough_scale=1.0) -> bpy.types.Material:
    key = (tid, tile, tuple(tint))
    if key in MATS:
        return MATS[key]
    mat = bpy.data.materials.new("%s_%s" % (tid, len(MATS)))
    mat.use_nodes = True
    nt = mat.node_tree
    n, l = nt.nodes, nt.links
    bsdf = n["Principled BSDF"]
    coord, mapping = n.new("ShaderNodeTexCoord"), n.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1 / tile, 1 / tile, 1 / tile)
    l.new(coord.outputs["Object"], mapping.inputs["Vector"])
    d = Path(A) / "textures" / tid

    def image(name, non_color=False):
        node = n.new("ShaderNodeTexImage")
        node.image = bpy.data.images.load(str(d / ("%s_%s_1k.jpg" % (tid, name))), check_existing=True)
        if non_color:
            node.image.colorspace_settings.name = "Non-Color"
        node.projection = "BOX"
        node.projection_blend = 0.2
        l.new(mapping.outputs["Vector"], node.inputs["Vector"])
        return node
    diff = image("diff")
    mix = n.new("ShaderNodeMix")
    mix.data_type, mix.blend_type = "RGBA", "MULTIPLY"
    mix.inputs["Factor"].default_value = 1.0
    l.new(diff.outputs["Color"], mix.inputs["A"])
    mix.inputs["B"].default_value = (*tint, 1.0)
    l.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    if (d / ("%s_rough_1k.jpg" % tid)).exists():
        rough = image("rough", True)
        l.new(rough.outputs["Color"], bsdf.inputs["Roughness"])
    if (d / ("%s_nor_gl_1k.jpg" % tid)).exists():
        nor = image("nor_gl", True)
        nm = n.new("ShaderNodeNormalMap")
        nm.inputs["Strength"].default_value = 0.8
        l.new(nor.outputs["Color"], nm.inputs["Color"])
        l.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    MATS[key] = mat
    return mat


def flat_material(rgb, rough=0.6, metal=0.0, emit=0.0) -> bpy.types.Material:
    key = ("flat", tuple(rgb), rough, metal, emit)
    if key in MATS:
        return MATS[key]
    mat = bpy.data.materials.new("flat_%d" % len(MATS))
    mat.use_nodes = True
    b = mat.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if emit:
        b.inputs["Emission Color"].default_value = (*rgb, 1)
        b.inputs["Emission Strength"].default_value = emit
    MATS[key] = mat
    return mat


def box(lo, hi, mat, name="box"):
    lo, hi = Vector(lo), Vector(hi)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(lo + hi) / 2)
    o = bpy.context.active_object
    o.name = name
    o.scale = hi - lo
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    o.data.materials.append(mat)
    return o


def wall_x(y0, y1, x0, x1, h, mat, holes=()):
    """A wall spanning x0..x1 between y0 and y1 (thickness), height h, with (xa, xb, za, zb) holes."""
    xs = sorted({x0, x1, *[v for hh in holes for v in hh[:2]]})
    for a, b in zip(xs, xs[1:]):
        mid = (a + b) / 2
        cuts = [hh for hh in holes if hh[0] <= mid <= hh[1]]
        z = 0.0
        for hh in sorted(cuts, key=lambda c: c[2]):
            if hh[2] > z:
                box((a, y0, z), (b, y1, hh[2]), mat)
            z = hh[3]
        if z < h:
            box((a, y0, z), (b, y1, h), mat)


def wall_y(x0, x1, y0, y1, h, mat, holes=()):
    ys = sorted({y0, y1, *[v for hh in holes for v in hh[:2]]})
    for a, b in zip(ys, ys[1:]):
        mid = (a + b) / 2
        cuts = [hh for hh in holes if hh[0] <= mid <= hh[1]]
        z = 0.0
        for hh in sorted(cuts, key=lambda c: c[2]):
            if hh[2] > z:
                box((x0, a, z), (x1, b, hh[2]), mat)
            z = hh[3]
        if z < h:
            box((x0, a, z), (x1, b, h), mat)


def model(aid, loc, face="south", height=None, scale=None, z=None, yaw=None):
    """Import a Poly Haven model, stand it on loc (x, y, z), its front (-y in the file) toward `face`."""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(Path(A) / "models" / aid / ("%s_1k.gltf" % aid)))
    new = [o for o in bpy.data.objects if o not in before]
    root = bpy.data.objects.new(aid, None)
    bpy.context.scene.collection.objects.link(root)
    for o in new:
        if o.parent is None:
            o.parent = root
    bpy.context.view_layer.update()
    meshes = [o for o in new if o.type == "MESH"]
    pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    lo = Vector([min(p[i] for p in pts) for i in range(3)])
    hi = Vector([max(p[i] for p in pts) for i in range(3)])
    s = scale or (height / (hi.z - lo.z) if height else 1.0)
    pivot = bpy.data.objects.new(aid + "_pivot", None)
    bpy.context.scene.collection.objects.link(pivot)
    root.parent = pivot
    root.location = Vector((-(lo.x + hi.x) / 2, -(lo.y + hi.y) / 2, -lo.z))
    angle = yaw if yaw is not None else {"south": 0, "east": 90, "north": 180, "west": -90}[face]
    pivot.rotation_euler = (0, 0, math.radians(angle))
    pivot.scale = (s, s, s)
    pivot.location = Vector((loc[0], loc[1], loc[2] if len(loc) > 2 else 0.0))
    return pivot


def painting(img_path, centre, facing, w, h, frame_rgb=(0.08, 0.06, 0.04)):
    """A framed picture on a wall: `facing` is the direction it looks (east/west/north/south)."""
    mat = bpy.data.materials.new("painting_%s" % Path(img_path).stem)
    mat.use_nodes = True
    n, l = mat.node_tree.nodes, mat.node_tree.links
    t = n.new("ShaderNodeTexImage")
    t.image = bpy.data.images.load(str(img_path), check_existing=True)
    l.new(t.outputs["Color"], n["Principled BSDF"].inputs["Base Color"])
    n["Principled BSDF"].inputs["Roughness"].default_value = 0.45
    bpy.ops.mesh.primitive_plane_add(size=1.0)
    p = bpy.context.active_object
    p.scale = (w, h, 1)
    rot = {"north": (math.radians(90), 0, math.radians(180)), "south": (math.radians(90), 0, 0),
           "east": (math.radians(90), 0, math.radians(90)), "west": (math.radians(90), 0, math.radians(-90))}[facing]
    p.rotation_euler = rot
    off = {"north": (0, 0.03, 0), "south": (0, -0.03, 0), "east": (0.03, 0, 0), "west": (-0.03, 0, 0)}[facing]
    p.location = Vector(centre) + Vector(off)
    p.data.materials.append(mat)
    fm = flat_material(frame_rgb, 0.4)
    t2, b = 0.07, 0.05
    cx, cy, cz = centre
    if facing in ("north", "south"):
        for dx, ww in ((-(w / 2 + t2 / 2), t2), (w / 2 + t2 / 2, t2)):
            box((cx + dx - ww / 2, cy - b, cz - h / 2 - t2), (cx + dx + ww / 2, cy + b, cz + h / 2 + t2), fm)
        for dz in (-(h / 2 + t2 / 2), h / 2 + t2 / 2):
            box((cx - w / 2 - t2, cy - b, cz + dz - t2 / 2), (cx + w / 2 + t2, cy + b, cz + dz + t2 / 2), fm)
    else:
        for dy in (-(w / 2 + t2 / 2), w / 2 + t2 / 2):
            box((cx - b, cy + dy - t2 / 2, cz - h / 2 - t2), (cx + b, cy + dy + t2 / 2, cz + h / 2 + t2), fm)
        for dz in (-(h / 2 + t2 / 2), h / 2 + t2 / 2):
            box((cx - b, cy - w / 2 - t2, cz + dz - t2 / 2), (cx + b, cy + w / 2 + t2, cz + dz + t2 / 2), fm)


def world(hdri: str, strength=1.0, visible=True):
    w = bpy.data.worlds.new("world")
    bpy.context.scene.world = w
    w.use_nodes = True
    n, l = w.node_tree.nodes, w.node_tree.links
    env = n.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(str(Path(A) / "hdris" / ("%s_2k.hdr" % hdri)))
    l.new(env.outputs["Color"], n["Background"].inputs["Color"])
    n["Background"].inputs["Strength"].default_value = strength
    if not visible:
        w.cycles_visibility.camera = False


def sun(direction_deg=(45, 135), strength=3.0, angle=1.0):
    d = bpy.data.lights.new("sun", "SUN")
    d.energy, d.angle = strength, math.radians(angle)
    o = bpy.data.objects.new("sun", d)
    bpy.context.scene.collection.objects.link(o)
    elev, az = map(math.radians, direction_deg)
    o.rotation_euler = (math.pi / 2 - elev, 0, az)


def area(centre, size, power, rgb=(1, 0.97, 0.92)):
    d = bpy.data.lights.new("area", "AREA")
    d.shape, d.size, d.size_y, d.energy, d.color = "RECTANGLE", size[0], size[1], power, rgb
    o = bpy.data.objects.new("area", d)
    bpy.context.scene.collection.objects.link(o)
    o.location = centre


def windows(face, lo, hi, base, h, spacing=3.0, ww=1.2, wh=1.6, door=None):
    """Dark window panes and sills on one facade. face: north/south/east/west; lo..hi along the facade."""
    glass, sill, doorm = flat_material((0.06, 0.08, 0.1), 0.15, 0.2), flat_material((0.85, 0.83, 0.78), 0.6), flat_material((0.25, 0.14, 0.08), 0.5)
    floors = int((h - 1.0) // 3.0)
    n = int((hi - lo) // spacing)
    start = lo + ((hi - lo) - (n - 1) * spacing) / 2 if n > 0 else lo
    for f in range(floors):
        z0 = 1.2 + 3.0 * f
        for k in range(n):
            c = start + k * spacing
            if f == 0 and door is not None and abs(c - door) < spacing / 2:
                z0d = 0.0
                if face in ("north", "south"):
                    box((c - 0.7, base - 0.08, z0d), (c + 0.7, base + 0.08, 2.4), doorm)
                else:
                    box((base - 0.08, c - 0.7, z0d), (base + 0.08, c + 0.7, 2.4), doorm)
                continue
            if face in ("north", "south"):
                box((c - ww / 2, base - 0.06, z0), (c + ww / 2, base + 0.06, z0 + wh), glass)
                box((c - ww / 2 - 0.1, base - 0.15, z0 - 0.12), (c + ww / 2 + 0.1, base + 0.15, z0), sill)
            else:
                box((base - 0.06, c - ww / 2, z0), (base + 0.06, c + ww / 2, z0 + wh), glass)
                box((base - 0.15, c - ww / 2 - 0.1, z0 - 0.12), (base + 0.15, c + ww / 2 + 0.1, z0), sill)


def house(x0, x1, y0, y1, h, mat, roof_mat, faces=(), door=None):
    box((x0, y0, 0), (x1, y1, h), mat, "house")
    box((x0 - 0.2, y0 - 0.2, h), (x1 + 0.2, y1 + 0.2, h + 0.35), roof_mat, "roof")
    for f in faces:
        if f == "south":
            windows("south", x0 + 1, x1 - 1, y0, h, door=door)
        elif f == "north":
            windows("north", x0 + 1, x1 - 1, y1, h, door=door)
        elif f == "east":
            windows("east", y0 + 1, y1 - 1, x1, h, door=door)
        elif f == "west":
            windows("west", y0 + 1, y1 - 1, x0, h, door=door)


# ---------------------------------------------------------------- scenes

def apartment(variant: str):
    world("kloofendal_48d_partly_cloudy", 1.0)
    sun((38, 150), 3.5)
    floor = tex_material("wood_floor", 2.0)
    cream = tex_material("painted_plaster_wall", 2.5, (1.0, 0.93, 0.8))
    dining = (0.86, 0.5, 0.38) if variant == "dining-b" else (0.66, 0.8, 0.62)
    sage = tex_material("painted_plaster_wall", 2.5, dining)
    ceiling = flat_material((0.92, 0.91, 0.89), 0.8)
    H = 2.8
    box((-0.2, -0.2, -0.1), (6.2, 10.35, 0.0), floor, "floor")
    box((-0.2, -0.2, H), (6.2, 10.35, H + 0.1), ceiling, "ceiling")
    wall_x(-0.15, 0.0, -0.15, 6.15, H, cream, [(1.5, 3.5, 0.9, 2.3)])          # south, window
    wall_x(10.15, 10.3, -0.15, 6.15, H, sage, [(1.0, 2.6, 0.9, 2.3)])         # north, window
    wall_y(-0.15, 0.0, 0.0, 5.075, H, cream)
    wall_y(-0.15, 0.0, 5.075, 10.15, H, sage)
    wall_y(6.0, 6.15, 0.0, 5.075, H, cream)
    wall_y(6.0, 6.15, 5.075, 10.15, H, sage)
    wall_x(5.0, 5.075, 0.0, 6.0, H, cream, [(3.6, 4.8, 0.0, 2.2)])            # between the rooms, doorway
    wall_x(5.075, 5.15, 0.0, 6.0, H, sage, [(3.6, 4.8, 0.0, 2.2)])
    trim = flat_material((0.95, 0.94, 0.9), 0.5)
    for x in (3.55, 4.8):
        box((x, 4.98, 0), (x + 0.05, 5.17, 2.25), trim)
    box((3.55, 4.98, 2.2), (4.85, 5.17, 2.27), trim)
    area((2.6, 2.5, H - 0.02), (2.0, 2.0), 260)
    area((3.0, 7.6, H - 0.02), (2.0, 2.0), 260)
    # living room
    model("Sofa_01", (0.6, 2.5), "east", height=0.9)
    model("coffee_table_round_01", (1.9, 2.5), "east", height=0.45)
    model("ArmChair_01", (3.0, 1.0), yaw=135, height=0.95)
    model("Shelf_01", (5.72, 2.2), "west", height=1.9)
    model("GothicCabinet_01", (2.8, 4.7), "south", height=2.0)  # the grandfather clock fails Blender 4.1 glTF import
    model("potted_plant_01", (5.45, 4.5), "south", height=1.2)
    model("side_table_01", (0.5, 4.3), "east", height=0.6)
    model("ceramic_vase_01", (0.5, 4.3, 0.6), "east", height=0.35)
    model("hanging_picture_frame_01", (0.03, 2.5, 1.35), "east", height=0.75)
    model("modern_ceiling_lamp_01", (2.6, 2.5, H - 0.45), "south", height=0.45)
    # dining room
    # the table stands west of the doorway, on the line the drone turns to face
    if variant == "dining-b":
        model("round_wooden_table_01", (2.3, 6.7), "south", height=0.76)
        for (x, y, f) in ((2.3, 5.85, "north"), (1.45, 6.8, "east"), (2.3, 7.6, "south")):
            model("GreenChair_01", (x, y), f, height=0.95)
    else:
        model("dining_table", (2.3, 6.7), "east", height=0.76)
        for (x, y, f) in ((1.8, 5.95, "north"), (2.8, 5.95, "north"), (1.8, 7.45, "south"), (2.8, 7.45, "south")):
            model("dining_chair_02", (x, y), f, height=0.95)
        model("Chandelier_01", (2.3, 6.7, H - 0.75), "south", height=0.75)
        model("ceramic_vase_01", (2.3, 6.7, 0.76), "south", height=0.3)
    model("painted_wooden_cabinet", (0.38, 6.9), "east", height=1.9)
    model("potted_plant_04", (5.5, 9.6), "south", height=1.3)
    model("fancy_picture_frame_01", (5.97, 7.5, 1.25), "west", height=0.9)
    model("wall_clock", (4.6, 10.12, 1.85), "south", height=0.45)


def village(variant: str):
    world("kloofendal_48d_partly_cloudy_puresky", 1.0)
    sun((42, 140), 4.0)
    cob = tex_material("cobblestone_floor_01", 2.0)
    asph = tex_material("asphalt_02", 4.0)
    box((-40, -40, -0.2), (40, 60, 0.0), cob, "ground")
    box((2.6, 17, 0.0), (9.4, 48, 0.02), asph, "street")
    brick = tex_material("brick_wall_02", 2.5)
    brick2 = tex_material("brick_wall_001", 2.5)
    blue = tex_material("blue_plaster_wall", 3.0)
    grey = tex_material("grey_plaster", 3.0, (0.95, 0.92, 0.86))
    beige = tex_material("beige_wall_001", 3.0)
    roof = tex_material("clay_roof_tiles", 2.0)
    yellow = tex_material("painted_plaster_wall", 3.0, (1.0, 0.8, 0.35))
    rose = tex_material("painted_plaster_wall", 3.0, (0.95, 0.6, 0.55))
    green = tex_material("painted_plaster_wall", 3.0, (0.55, 0.8, 0.6))
    # the square's houses
    house(-20, 3.5, 14, 24, 11, brick, roof, ("south",), door=-6)
    house(8.5, 22, 14, 24, 9, blue, roof, ("south",), door=15)
    box((3.5, 14, 5.5), (8.5, 17, 9.4), brick2, "arch")
    box((3.3, 13.8, 9.4), (8.7, 17.2, 9.8), roof, "arch_roof")
    house(-22, 22, -28, -18, 10, beige, roof, ("north",), door=0)
    house(-28, -18, -18, 14, 12, grey, roof, ("east",), door=-2)
    house(18, 28, -18, 14, 8, rose, roof, ("west",), door=4)
    # the street beyond the arch
    street_w, street_e, end = (yellow, green, blue) if variant == "street-b" else (grey, beige, rose)
    house(-6, 2.6, 24, 36, 10, street_w, roof, ("east",), door=30)
    house(-6, 2.6, 36, 48, 12, brick if variant != "street-b" else rose, roof, ("east",), door=42)
    house(9.4, 18, 24, 36, 12, street_e, roof, ("west",), door=30)
    house(9.4, 18, 36, 48, 10, brick2 if variant != "street-b" else yellow, roof, ("west",), door=42)
    house(-6, 18, 48, 56, 15, end, roof, ("south",), door=6)
    # statue on its pedestal
    stone = tex_material("grey_plaster", 2.0, (0.8, 0.78, 0.74))
    box((-1.2, -1.2, 0), (1.2, 1.2, 1.4), stone, "pedestal")
    model("horse_statue_01", (0, 0, 1.4), "east", height=2.8)
    # props
    for x, y in ((-12, -12), (12, -12), (-12, 10), (12.5, 9.5)):
        model("tree_small_02" if (x + y) % 2 == 0 else "jacaranda_tree", (x, y), "south", height=6.5)
    for x, y, f in ((-6, -6, "east"), (6, -6, "west"), (-6, 6, "east")):
        model("street_lamp_01", (x, y), f, height=4.5)
    model("modular_street_seating", (-3.5, -3.0), "north", height=0.85)
    model("modular_street_seating", (3.5, -3.0), "north", height=0.85)
    model("metal_trash_can", (-4.6, -3.2), "north", height=0.9)
    model("fire_hydrant", (9.0, 12.5), "south", height=0.8)
    model("outdoor_table_chair_set_01", (14.5, 0.0), "west", height=0.95)
    model("outdoor_table_chair_set_01", (14.5, -4.0), "west", height=0.95)
    model("planter_box_01", (2.2, 13.0), "south", height=0.8)
    model("planter_box_01", (9.8, 13.0), "south", height=0.8)
    model("utility_box_01", (-16.5, 0.0), "east", height=1.4)
    # the street's own props, which the street-b variant moves
    if variant == "street-b":
        model("tree_small_02", (4.0, 40), "south", height=6.0)
        model("street_lamp_02", (9.0, 30), "west", height=5.0)
    else:
        model("covered_car", (4.2, 38), "south", height=1.5)
        model("street_lamp_02", (3.0, 30), "east", height=5.0)
        model("potted_plant_02", (8.8, 26), "west", height=1.2)


def gallery(variant: str):
    world("kloofendal_48d_partly_cloudy", 0.35, visible=False)
    floor = tex_material("wood_floor_deck", 3.0)
    # each wall its own colour, so opposite sides of the walk cannot be mistaken for each other
    cream = tex_material("painted_plaster_wall", 3.0, (0.95, 0.9, 0.78))
    red = tex_material("painted_plaster_wall", 3.0, (0.62, 0.22, 0.2))
    blue = tex_material("painted_plaster_wall", 3.0, (0.35, 0.45, 0.62))
    sage = tex_material("painted_plaster_wall", 3.0, (0.5, 0.65, 0.5))
    ceiling = flat_material((0.95, 0.95, 0.94), 0.8)
    H = 4.2
    box((-0.2, -0.2, -0.1), (14.2, 9.2, 0), floor, "floor")
    box((-0.2, -0.2, H), (14.2, 9.2, H + 0.1), ceiling, "ceiling")
    wall_x(-0.15, 0.0, -0.15, 14.15, H, cream)     # south
    wall_x(9.0, 9.15, -0.15, 14.15, H, blue)       # north
    wall_y(-0.15, 0.0, 0, 9, H, sage)              # west
    wall_y(14.0, 14.15, 0, 9, H, red)              # east
    for x in (3.5, 7.0, 10.5):
        for y in (2.5, 6.5):
            area((x, y, H - 0.02), (2.2, 1.6), 380, (1.0, 0.98, 0.95))
    pics = sorted((Path(A) / "paintings_cropped").glob("*.jpg"))
    names = [p.stem for p in pics]
    order = ["belfast_sunset", "autumn_field", "drakensberg_solitary_mountain", "industrial_sunset_02", "farm_field",
             "kloofendal_28d_misty", "evening_road_01", "citrus_orchard_road", "aristea_wreck", "hilly_terrain_01"]
    spare = ["venice_sunset", "snowy_park_01"]
    get = lambda n: Path(A) / "paintings_cropped" / ("%s.jpg" % n)
    slots = [((3.5, 0.0, 1.8), "north"), ((7.0, 0.0, 1.8), "north"), ((10.5, 0.0, 1.8), "north"),
             ((10.5, 9.0, 1.8), "south"), ((7.0, 9.0, 1.8), "south"), ((3.5, 9.0, 1.8), "south"),
             ((0.0, 3.0, 1.8), "east"), ((0.0, 6.0, 1.8), "east"),
             ((14.0, 4.5, 1.8), "west"), ((14.0, 1.8, 1.8), "west")]
    for k, (c, f) in enumerate(slots):
        name = order[k]
        if variant == "painting-swap" and k == 8:
            name = spare[0]
        painting(get(name), c, f, 1.8 if f in ("north", "south") else 1.6, 1.2 if f in ("north", "south") else 1.07)
    model("concrete_cat_statue", (13.2, 7.6), "west", height=0.7)
    painting(get(spare[1]), (14.0, 7.6, 1.8), "west", 1.3, 0.87, (0.55, 0.45, 0.2))
    plinth = flat_material((0.96, 0.96, 0.95), 0.7)
    for x, aid, h in ((5.0, "marble_bust_01", 0.7), (9.0, "bronze_whale_statue", 0.5)):
        box((x - 0.35, 4.15, 0), (x + 0.35, 4.85, 1.0), plinth)
        model(aid, (x, 4.5, 1.0), "south", height=h)
    model("gothic_statue", (7.0, 4.5), "south", height=2.0)
    model("painted_wooden_bench", (7.0, 3.3), "south", height=0.5)
    for x, y in ((0.6, 0.6), (13.4, 0.6), (0.6, 8.4)):
        model("potted_plant_02", (x, y), "south", height=1.2)


# ---------------------------------------------------------------- render

def main():
    global A
    args = parse()
    A = args.assets
    bpy.ops.wm.read_factory_settings(use_empty=True)
    {"apartment": apartment, "village": village, "gallery": gallery}[args.scene](args.variant)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True
        scene.cycles.device = "GPU"
    except (TypeError, AttributeError):
        scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 6
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "AgX"
    scene.render.resolution_percentage = 100
    cam_data = bpy.data.cameras.new("camera")
    cam_data.sensor_width, cam_data.sensor_fit = 36.0, "HORIZONTAL"
    cam_data.clip_start, cam_data.clip_end = 0.05, 500
    cam = bpy.data.objects.new("camera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    for v in json.loads(Path(args.views).read_text()):
        yaw, pitch = math.radians(v["yaw"]), math.radians(v["pitch"])
        f = Vector((math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), -math.sin(pitch)))
        cam.location = Vector(v["position"])
        cam.rotation_euler = f.to_track_quat("-Z", "Y").to_euler()
        cam_data.lens = v.get("lens", 24.0)
        scene.render.resolution_x, scene.render.resolution_y = v["width"], v["height"]
        scene.render.filepath = str(out / (v["name"] + ".png"))
        bpy.context.view_layer.update()
        bpy.ops.render.render(write_still=True)
        M = cam.matrix_world.copy()
        cv = M @ Matrix.Diagonal((1, -1, -1, 1))  # Blender camera (x right, y up, looks -z) -> OpenCV
        fx = v["width"] * cam_data.lens / 36.0
        meta[v["name"]] = {"cam_to_world": [[round(x, 6) for x in row] for row in cv], "fx": fx, "fy": fx,
                           "cx": v["width"] / 2, "cy": v["height"] / 2, "width": v["width"], "height": v["height"]}
    prev = json.loads((out / "cameras.json").read_text()) if (out / "cameras.json").exists() else {}
    prev.update(meta)
    (out / "cameras.json").write_text(json.dumps(prev, indent=1))


main()
