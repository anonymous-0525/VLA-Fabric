"""Export the authored four-arm Blender scene as a web animation.

blender -b INPUT.blend --python export_scene.py -- OUTPUT.glb
The motion illustrates coordinated actuation; it is not a policy rollout.
"""

import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def smooth(t):
    return t * t * (3 - 2 * t)


def offset(t):
    poses = [(0, (0, -.32, -.46)), (2, (0, -.32, -.46)),
             (5, (0, -.32, 1.12)), (7, (0, 0, 1.12)),
             (10, (0, 0, -.46)), (12, (0, 0, -.46)),
             (15, (0, -.32, -.46))]
    for (a, p), (b, q) in zip(poses, poses[1:]):
        if a <= t <= b:
            return Vector(p).lerp(Vector(q), smooth((t-a)/(b-a)))
    return Vector(poses[0][1])


def main():
    output = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    scene = bpy.context.scene
    scene.render.fps = 30
    scene.frame_start, scene.frame_end = 0, 450
    arms = []
    for letter in "ABCD":
        nodes = [bpy.data.objects[f"ARM_{letter}.{part}"] for part in
                 ("J2_shoulder", "J3_elbow", "J4_wrist")]
        originals = [node.matrix_world.copy() for node in nodes]
        s, e, w = [mat.translation.copy() for mat in originals]
        direction = (w-s).normalized()
        bend = (e-s) - direction * (e-s).dot(direction)
        arms.append((nodes, originals, s, e, w, bend.normalized()))
        for node in nodes:
            node.rotation_mode = "QUATERNION"
    frame = bpy.data.objects["PROP_four_arm_frame"]
    original_frame = frame.location.copy()
    for index in range(0, 451, 3):
        scene.frame_set(index)
        delta = offset(index/30)
        frame.location = original_frame + delta
        frame.keyframe_insert("location", frame=index)
        for nodes, originals, s, e, w, old_bend in arms:
            target = w + delta
            line = target-s
            length = line.length
            direction = line.normalized()
            upper, lower = (e-s).length, (w-e).length
            projection = (upper*upper-lower*lower+length*length)/(2*length)
            height = math.sqrt(max(0, upper*upper-projection*projection))
            bend = old_bend - direction*old_bend.dot(direction)
            elbow = s + direction*projection + bend.normalized()*height
            q1 = (e-s).rotation_difference(elbow-s)
            q2 = (w-e).rotation_difference(target-elbow)
            transforms = [Matrix.Translation(s) @ q1.to_matrix().to_4x4(),
                          Matrix.Translation(elbow) @ q2.to_matrix().to_4x4(),
                          Matrix.Translation(target)]
            for node, matrix in zip(nodes, transforms):
                node.matrix_world = matrix
                bpy.context.view_layer.update()
                node.keyframe_insert("location", frame=index)
                node.keyframe_insert("rotation_quaternion", frame=index)
    # Reduce web draw calls and file size by leaving cameras/lights to Three.js.
    for obj in list(bpy.data.objects):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    scene.frame_set(0)
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(output), export_format="GLB",
                              export_animations=True, export_frame_range=True,
                              export_force_sampling=True, export_extras=False,
                              export_cameras=False, export_lights=False)
    print(f"Exported {output.name}: {output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
