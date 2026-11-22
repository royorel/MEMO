"""Blender renderer of the videos' mesh panels.

Runs in the memo-blender env, whose bpy build needs numpy below 2, as

    python -m memo.video.blender_render <jobs.json>

and imports nothing from the memo package. Each job renders one PLY mesh from
one OpenCV camera to an RGBA PNG with a transparent background, for
compositing on the video background.

A clay job shades the mesh in a gray clay material, darkened in crevices by
ambient occlusion within the job's `ao_distance`, under a key and a fill sun
light fixed to the camera. The lights cast no shadows, since the real
scenes' meshes enclose the subject in walls that would shadow it from any
outside light. A clay job that also sets `matcap_out` then renders the
mesh's normal panel from the same camera at `matcap_samples`, without
importing the mesh again. A job with `shading` set to "matcap" renders the
normal panel alone, at its `samples`.
"""
import json
import math
import sys

import bpy  # type: ignore
import mathutils  # type: ignore


# (name, energy, rotation from the camera axes in degrees about x and y)
LIGHTS = [
    ("key", 1.6, (35.0, -40.0)),
    ("fill", 0.5, (-10.0, 50.0)),
]
CLAY_GRAY = 0.7
WORLD_STRENGTH = 0.25

# OpenCV cameras look down +Z with +Y down, Blender's down -Z with +Y up.
_CV_TO_BLENDER = mathutils.Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))


def _configure_cycles_gpu(samples: int) -> None:
    """Enable Cycles + GPU compute (OPTIX preferred over CUDA) + OIDN.

    Logs the chosen device type and the count of enabled GPU devices. Falls
    back to CPU with a warning if no compatible GPU is found.
    """
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    cycles_prefs = bpy.context.preferences.addons["cycles"].preferences

    chosen = None
    for device_type in ("OPTIX", "CUDA"):
        try:
            cycles_prefs.compute_device_type = device_type
            cycles_prefs.get_devices()
            gpu_devices = [d for d in cycles_prefs.devices if d.type == device_type]
            if gpu_devices:
                chosen = device_type
                for d in cycles_prefs.devices:
                    d.use = (d.type == device_type)
                break
        except (TypeError, RuntimeError):
            continue

    if chosen is None:
        print("[WARN] No CUDA / OPTIX GPU detected; Cycles will run on CPU "
              "and will be much slower.")
        scene.cycles.device = "CPU"
    else:
        gpu_count = sum(1 for d in cycles_prefs.devices
                        if d.type == chosen and d.use)
        print(f"[INFO] Cycles GPU enabled: {chosen} x {gpu_count} device(s)")
        scene.cycles.device = "GPU"

    scene.cycles.samples = samples
    scene.cycles.use_denoising = True

    # Disable view-transform tone mapping on every scene (default AgX/Filmic
    # compresses a linear emission of 1.0 to ~0.6-0.8 in display space, which
    # turns the white world background gray). Iterating over bpy.data.scenes
    # avoids cases where bpy.context.scene isn't the scene actually rendered.
    for scn in bpy.data.scenes:
        scn.view_settings.view_transform = "Standard"
        scn.view_settings.look = "None"
        scn.view_settings.exposure = 0.0
        scn.view_settings.gamma = 1.0
        scn.display_settings.display_device = "sRGB"
    try:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
    except (TypeError, AttributeError):
        # Older Blender versions expose the denoiser per view layer.
        for vl in scene.view_layers:
            try:
                vl.cycles.denoiser = "OPENIMAGEDENOISE"
                vl.cycles.use_denoising = True
            except (TypeError, AttributeError):
                pass

    # Use albedo + normal AOV passes as denoising guides. Without these OIDN
    # runs in color-only mode and aggressively smooths fine geometry detail
    # at low sample counts. The AOVs are clean (deterministic per pixel) and
    # let OIDN preserve the underlying surface structure while denoising the
    # noisy radiance.
    try:
        scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
        scene.cycles.denoising_prefilter = "ACCURATE"
    except (TypeError, AttributeError):
        for vl in scene.view_layers:
            try:
                vl.cycles.denoising_store_passes = True
                vl.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
                vl.cycles.denoising_prefilter = "ACCURATE"
            except (TypeError, AttributeError):
                pass

    print(f"[INFO] Cycles samples={samples}, OIDN denoising enabled "
          f"(passes=RGB_ALBEDO_NORMAL, prefilter=ACCURATE)")


def _clear_scene() -> None:
    """Remove all objects, meshes, lights, cameras, and materials."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.lights, bpy.data.cameras,
                  bpy.data.materials):
        for item in list(block):
            block.remove(item)


def _matrix_world_from_opencv(R_c2w, T_w2c) -> mathutils.Matrix:
    """Build a Blender camera matrix_world from the OpenCV (R_c2w, T_w2c).

    OpenCV R_c2w is the camera-to-world rotation. Camera position in world is
    -R_c2w @ T_w2c. Then flip Y/Z axes in the camera local frame to convert to
    Blender's convention.
    """
    R = mathutils.Matrix([list(row) for row in R_c2w])
    T = mathutils.Vector(T_w2c)
    pos_world = -(R @ T)
    M = mathutils.Matrix.Identity(4)
    for i in range(3):
        for j in range(3):
            M[i][j] = R[i][j]
    M[0][3] = pos_world.x
    M[1][3] = pos_world.y
    M[2][3] = pos_world.z
    return M @ _CV_TO_BLENDER


def _clay_material(ao_distance):
    """Clay of albedo CLAY_GRAY, darkened in crevices by ambient occlusion."""
    mat = bpy.data.materials.new(name="Clay")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)
    rgb = nodes.new("ShaderNodeRGB")
    rgb.outputs["Color"].default_value = (CLAY_GRAY, CLAY_GRAY, CLAY_GRAY, 1.0)
    ao = nodes.new("ShaderNodeAmbientOcclusion")
    ao.inputs["Distance"].default_value = float(ao_distance)
    links.new(rgb.outputs["Color"], ao.inputs["Color"])
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.6
    bsdf.inputs["Specular IOR Level"].default_value = 0.2
    links.new(ao.outputs["Color"], bsdf.inputs["Base Color"])
    out = nodes.new("ShaderNodeOutputMaterial")
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _matcap_material():
    """Emission of the view-space normal mapped to RGB as (-n + 1) / 2 in the
    OpenCV camera frame (x right, y down, z forward), matching the nvdiffrast
    matcap of memo/video/panels.py. The camera space of Cycles' Vector
    Transform has y up and z forward, so the map there is
    ((1 - x) / 2, (1 + y) / 2, (1 - z) / 2). A normal facing away from the
    camera (z > 0) is negated first, as in the nvdiffrast matcap, so that
    orientation noise cannot flip a patch to the dark end."""
    mat = bpy.data.materials.new(name="Matcap")
    mat.use_nodes = True
    # Every surface in the matcap pass emits and none scatters, so light
    # sampling is never used. With the default AUTO, Cycles builds a light
    # tree over every triangle, which stalls on multi-million-face meshes.
    mat.cycles.emission_sampling = "NONE"
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)
    geometry = nodes.new("ShaderNodeNewGeometry")
    to_camera = nodes.new("ShaderNodeVectorTransform")
    to_camera.vector_type = "NORMAL"
    to_camera.convert_from = "WORLD"
    to_camera.convert_to = "CAMERA"
    links.new(geometry.outputs["Normal"], to_camera.inputs["Vector"])
    split = nodes.new("ShaderNodeSeparateXYZ")
    links.new(to_camera.outputs["Vector"], split.inputs["Vector"])
    facing = nodes.new("ShaderNodeMath")
    facing.operation = "SIGN"
    links.new(split.outputs["Z"], facing.inputs[0])
    toward = nodes.new("ShaderNodeMath")
    toward.operation = "MULTIPLY"
    toward.inputs[1].default_value = -1.0
    links.new(facing.outputs["Value"], toward.inputs[0])
    flip = nodes.new("ShaderNodeVectorMath")
    flip.operation = "SCALE"
    links.new(to_camera.outputs["Vector"], flip.inputs[0])
    links.new(toward.outputs["Value"], flip.inputs["Scale"])
    remap = nodes.new("ShaderNodeVectorMath")
    remap.operation = "MULTIPLY_ADD"
    remap.inputs[1].default_value = (-0.5, 0.5, -0.5)
    remap.inputs[2].default_value = (0.5, 0.5, 0.5)
    links.new(flip.outputs["Vector"], remap.inputs[0])
    emission = nodes.new("ShaderNodeEmission")
    links.new(remap.outputs["Vector"], emission.inputs["Color"])
    out = nodes.new("ShaderNodeOutputMaterial")
    links.new(emission.outputs["Emission"], out.inputs["Surface"])
    return mat


def _world():
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs["Strength"].default_value = WORLD_STRENGTH


def _camera_and_lights(job):
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.angle_x = float(job["FoVx"])
    cam_data.clip_start = float(job["clip_start"])
    cam_data.clip_end = 1000.0
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.matrix_world = _matrix_world_from_opencv(job["R_c2w"], job["T_w2c"])
    cam_rot = cam.matrix_world.to_3x3()
    for name, energy, (rx, ry) in LIGHTS:
        light = bpy.data.lights.new(name=name, type="SUN")
        light.energy = energy
        light.use_shadow = False
        obj = bpy.data.objects.new(name, light)
        bpy.context.collection.objects.link(obj)
        offset = (mathutils.Matrix.Rotation(math.radians(rx), 3, "X")
                  @ mathutils.Matrix.Rotation(math.radians(ry), 3, "Y"))
        obj.matrix_world = (cam_rot @ offset).to_4x4()


def render(job, samples):
    _clear_scene()
    for block in (bpy.data.objects, bpy.data.lights, bpy.data.cameras, bpy.data.node_groups):
        for item in list(block):
            block.remove(item)
    scene = bpy.context.scene
    scene.render.resolution_x = int(job["width"])
    scene.render.resolution_y = int(job["height"])
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.cycles.samples = int(job.get("samples", samples))
    matcap = job.get("shading") == "matcap"
    # The matcap stores normals as colors, so it skips the sRGB transform, and
    # its noise-free emission needs no denoising, which would blur its edges.
    scene.view_settings.view_transform = "Raw" if matcap else "Standard"
    scene.view_settings.look = "None"
    scene.cycles.use_denoising = not matcap

    bpy.ops.wm.ply_import(filepath=job["ply"])
    obj = bpy.context.selected_objects[0]
    obj.data.shade_smooth()
    obj.data.materials.append(_matcap_material() if matcap else _clay_material(job["ao_distance"]))
    _world()
    _camera_and_lights(job)

    scene.render.filepath = job["out"]
    bpy.ops.render.render(write_still=True)
    print(f"[blender_render] {job['out']}", flush=True)

    if job.get("matcap_out"):
        # The normal panel of the same mesh, rendered as a matcap job would be
        # but without importing it again, which dominates the time of large
        # meshes.
        scene.cycles.samples = int(job["matcap_samples"])
        scene.view_settings.view_transform = "Raw"
        scene.cycles.use_denoising = False
        obj.data.materials.clear()
        obj.data.materials.append(_matcap_material())
        scene.render.filepath = job["matcap_out"]
        bpy.ops.render.render(write_still=True)
        print(f"[blender_render] {job['matcap_out']}", flush=True)


def main():
    with open(sys.argv[1]) as f:
        spec = json.load(f)
    _configure_cycles_gpu(samples=int(spec["samples"]))
    for job in spec["jobs"]:
        render(job, int(spec["samples"]))


if __name__ == "__main__":
    main()
