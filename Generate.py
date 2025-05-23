import blenderproc as bproc
import numpy as np
import os
import bpy
import random
import math
import mathutils
import time
import sys
import argparse
from datetime import datetime
from datetime import timedelta
import json

parser = argparse.ArgumentParser()
parser.add_argument('--color', type=str, default="#0f0f13", help='Color of the model')
parser.add_argument('--model_path', type=str, help='Path to the 3D model')
parser.add_argument('--start_time', type=int, help='Script start time in epoch seconds')
parser.add_argument('--initial_count', type=int, help='Initial image count at the start of the script')

args = parser.parse_args()
# color = args.color
# model_path = args.model_path
# start_time = args.start_time
# initial_image_count = args.initial_count
color = "#0f0f13"
model_path = "/home/fsociety/Code/Projects/Bachelor_Thesis/Blender_Files/models/bottle_holder.stl"
start_time = time.time()
initial_image_count = 0
bproc.init()
# Load your scene while preserving settings
scene = bproc.loader.load_blend("Blender_Files/Scene_Main.blend")

bpy.context.scene.cycles.samples = 2048
bpy.context.scene.cycles.use_light_tree = True
bpy.context.scene.cycles.max_bounces = 12        # Maximum total bounces
bpy.context.scene.cycles.diffuse_bounces = 4      # Maximum diffuse bounces
bpy.context.scene.cycles.glossy_bounces = 4       # Maximum glossy bounces
bpy.context.scene.cycles.transmission_bounces = 12  # Maximum transmission bounces
bpy.context.scene.cycles.volume_bounces = 2       # Maximum volume bounces
bpy.context.scene.render.use_simplify = False
bpy.context.scene.cycles.use_spatial_splits = False
bpy.context.scene.cycles.use_persistent_data = False
bpy.context.scene.view_settings.exposure = -3


greenscreen = bpy.data.objects['GreenScreen']
greenscreen = bproc.filter.one_by_attr(scene, "name", "GreenScreen")

category_map = json.load(open("Blender_Files/models/category_map.json"))

for i, item in enumerate(scene):
        # if item.get_name().startswith("Train_"):
        #     if item.get_name().startswith(train_object_name):
        #         item.set_cp("category_id", i + 1)
        #         train_object = item
        #     else:
        #         item.set_cp("category_id", 0)
        #         item.hide()
        #         item.blender_obj.hide_viewport = True
        # # elif item.get_name().startswith("GreenScreen"):
        # #     greenscreen = item
        # #     item.set_cp("category_id", 0)
        # else:
        item.set_cp("category_id", 0)
        item.set_shading_mode("AUTO")

# model_path = "Blender_Files/Honeycomb Cup for HC wall.stl"
train_object = ""
train_object_name = "_".join(model_path.split("/")[-1].split(".")[0].split())
# test_obj = bproc.loader.load_obj(model_path)
bpy.ops.wm.stl_import(filepath=model_path)
imported_objects = bpy.context.selected_objects

print(train_object_name)
for obj in imported_objects:
    obj.name = train_object_name

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='BOUNDS')

    # Get all vertices
    verts = obj.data.vertices
    if not verts:
        raise ValueError(f"Object {obj.name} has no vertices.")

    # Calculate centroid
    centroid = [0, 0, 0]
    for v in verts:
        centroid[0] += v.co.x
        centroid[1] += v.co.y
        centroid[2] += v.co.z
    num_verts = len(verts)
    centroid = [c / num_verts for c in centroid]

    # Shift all vertices so that centroid is at the origin
    for v in verts:
        v.co.x -= centroid[0]
        v.co.y -= centroid[1]
        v.co.z -= centroid[2]

    # Update mesh
    obj.data.update()
    
    train_object = obj
    break

train_object["category_id"] = int([id for id, name in category_map.items() if name==train_object_name][0])
train_object.scale = (0.001, 0.001, 0.001)


radius = 0.3
light1 = ""
light1_offset = 1.0

touch_z = 0.07

def init():
    global light1

    bproc.camera.set_resolution(1280, 960)
    bproc.camera.set_intrinsics_from_blender_params(lens=1.309, lens_unit="FOV")

    bpy.ops.object.light_add(type='AREA', radius=0.2)
    bpy_light1 = bpy.context.object
    light1 = bproc.types.Light(blender_obj=bpy_light1)

    set_camera()
    set_light()

    set_object_random()
    # set_object_color("#FFFFFF", train_object_name)
    # set_object_color("#0f0f13", train_object_name)
    set_object_color(color, train_object_name)

def set_object_color(new_color, object_name):
    obj = bpy.data.objects.get(object_name)
    if obj:
        # Convert the color from hex to RGB format
        new_color = tuple(int(new_color.lstrip("#")[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        
        # Ensure the object has a material, create a new one if not
        if not obj.active_material:
            mat = bpy.data.materials.new(name="TrainMaterial")
            obj.active_material = mat
        else:
            mat = obj.active_material

        # Set the material's diffuse color (base color)
        mat.diffuse_color = (*new_color, 1)  # Adding alpha value of 1 for full opacity

        print(f"Changed color of {obj.name} to {new_color}")

def set_object_random():
    set_object(get_random_pose())

def get_random_pose():
    # theta = random.uniform(0, 2 * math.pi)  # Azimuth angle
    # phi = random.uniform(0, math.pi)        # Polar angle

    # # Calculate Cartesian coordinates
    # x = radius * math.sin(phi) * math.cos(theta)
    # y = radius * math.sin(phi) * math.sin(theta)
    # z = radius * math.cos(phi)

    # direction = mathutils.Vector((0, 0, 0)) - mathutils.Vector((x, y, z))
    # rotation_quaternion = direction.to_track_quat('-Z', 'Y')
    # rotation = rotation_quaternion.to_euler()

    # return [[x, y, z], [rotation.x, rotation.y, rotation.z], direction]
    train_object_dimensions = train_object.dimensions
    max_dim = max(train_object_dimensions.x, train_object_dimensions.y)
    half_width_x = (0.3 - max_dim) / 2  # Half of the width along the x-axis
    half_width_y = (0.3 - max_dim) / 2   # Half of the height along the y-axis
    # Generate random x, y coordinates within the box's range
    x = random.uniform(-half_width_x, half_width_x)
    y = random.uniform(-half_width_y, half_width_y)
    

    roll = random.uniform(0, 2 * math.pi)

    return [[x, y, touch_z], [0, 0, roll]]

def set_camera(pose=[[0, -0.24, 0.4], [0.523599, 0, 0]]):
    pose_matrix = bproc.math.build_transformation_mat(pose[0], pose[1])
    bproc.camera.add_camera_pose(pose_matrix)
    # bproc.camera.set_resolution(640, 480)
    bproc.camera.set_intrinsics_from_blender_params(lens=1.309, lens_unit="FOV")

def set_light(pose=[[0.15, -0.15, 0.4], [0.523599, 0.523599, 0]]):
    global light1

    # light1_position = mathutils.Vector(pose[0]) - light1_offset * pose[2].normalized()
    
    light1.set_location(pose[0])
    light1.set_rotation_euler(pose[1])
    light1.set_energy(6)

def set_object(pose=[[0, 0, 0], [0, 0, 0]]):
    # train_object.set_location(pose[0])
    # train_object.set_rotation_euler(pose[1])
    train_object.location = pose[0]
    train_object.rotation_euler = pose[1]
    return

def set_viewport_to_camera_view():
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            # Set view to camera
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.region_3d.view_perspective = 'CAMERA'
                    break
            break

def place_obj1_on_top_of_obj2(obj1, obj2):
    global touch_z
    # Get world-space bounding box for obj1 and obj2
    bbox1 = [obj1.matrix_world @ mathutils.Vector(corner) for corner in obj1.bound_box]
    bbox2 = [obj2.matrix_world @ mathutils.Vector(corner) for corner in obj2.bound_box]
    
    # Get min and max Z values for each object
    min_z1 = min(v.z for v in bbox1)
    max_z2 = max(v.z for v in bbox2)
    # Calculate the required Z offset to place obj1 on top of obj2
    offset_z = max_z2 - min_z1

    current_pose = obj1.location
    obj1.location = (current_pose.x, current_pose.y, offset_z)
    # Move obj1 in Z direction by the calculated offset
    # obj1.location.z += offset_z
    # bpy.context.view_layer.update()
    touch_z = offset_z
    print(f"Placed {obj1.name} on top of {obj2.name} at Z = {obj1.location.z}")


def point_camera_at_object(camera_obj, target_object, frame=0):
    # Get the current camera location from BlenderProc
    camera_location = bproc.camera.get_camera_pose()[:3, 3]  # Extract the location vector

    # Calculate the center of the target object in world space
    obj_center = sum((target_object.matrix_world @ mathutils.Vector(corner) for corner in target_object.bound_box), mathutils.Vector()) / 8

    # Calculate the rotation required for the camera to look at the object's center
    direction = (obj_center - mathutils.Vector(camera_location)).normalized()
    rotation_quaternion = direction.to_track_quat('-Z', 'Y')
    rotation_euler = rotation_quaternion.to_euler()

    # Build the transformation matrix for the camera pose (fixed location with new rotation)
    camera_pose = bproc.math.build_transformation_mat(camera_location, rotation_euler)

    # Register the calculated camera pose with BlenderProc
    bproc.camera.add_camera_pose(camera_pose, frame=frame)

def render_scene():
    # Render the scene
    bproc.renderer.enable_experimental_features()
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"], default_values={"category_id": None})
    bproc.renderer.set_max_amount_of_samples(2048)
    bproc.renderer.set_denoiser("INTEL")
    bproc.renderer.set_output_format(file_format="JPEG", jpg_quality=200)
    data = bproc.renderer.render()

    # Write the rendering into an hdf5 file
    bproc.writer.write_coco_annotations(os.path.join(f"gen_data/TrainData_{train_object_name}"),
                                        instance_segmaps=data["instance_segmaps"],
                                        instance_attribute_maps=data["instance_attribute_maps"],
                                        colors=data["colors"],
                                        color_file_format="JPEG")
    # bproc.writer.write_hdf5("output/", data)


init()
place_obj1_on_top_of_obj2(train_object, greenscreen.blender_obj)
set_object_random()  # This updates the train_object position and rotation

# Insert keyframe for object's location and rotation
train_object.keyframe_insert(data_path="location", frame=0)
train_object.keyframe_insert(data_path="rotation_euler", frame=0)
bpy.context.view_layer.update()
# Set camera to look at the object and insert keyframe
# point_camera_at_object(bpy.context.scene.camera, train_object, frame=0)

current_time = datetime.now()
elapsed_time = timedelta(seconds=(current_time.timestamp() - start_time))
# count = len(os.listdir(f"gen_data/TrainData_{train_object_name}/images"))
# generated_since_start = count - initial_image_count
# print(f"Current datetime: {current_time}\nCurrent images: {count}\nGenerated since start: {generated_since_start} in {elapsed_time}")
# Render frames for each keyframe after setting up the scene
render_scene()


