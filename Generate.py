import blenderproc as bproc
import numpy as np
import os
import bpy
import random
import math
import mathutils
import time
import sys

bproc.init()
# Load your scene while preserving settings
scene = bproc.loader.load_blend("Blender_Files/Scene_Main_Temp.blend")

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

train_object = ""
train_object_name = "Train_Honeycomb_Wall_Tool"
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

object_path = "Blender_Files/Honeycomb NEW DOUBLE Wall pliers-cutter.stl"
# test_obj = bproc.loader.load_obj(object_path)
bpy.ops.import_mesh.stl(filepath=object_path)
imported_objects = bpy.context.selected_objects

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

train_object["category_id"] = 1
train_object.scale = (0.001, 0.001, 0.001)


radius = 0.3
light1 = ""
light1_offset = 1.0


def init():
    global light1

    bproc.camera.set_resolution(1280, 960)
    bproc.camera.set_intrinsics_from_blender_params(lens=0.959931, lens_unit="FOV")

    bpy.ops.object.light_add(type='AREA', radius=0.3)
    bpy_light1 = bpy.context.object
    light1 = bproc.types.Light(blender_obj=bpy_light1)

    set_camera()
    set_light()

    set_object_random()
    set_object_color("#FFFFFF", train_object_name)


def set_object_color(new_color, object_name):
    obj = bpy.data.objects.get(object_name)

    new_color = tuple(int(new_color.lstrip("#")[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    if obj.active_material:
        # Get the material
        
        mat = obj.active_material
        # Check if the material is the one we're looking for
        if mat.name.startswith("Material"):
            # Ensure the material uses nodes
            if mat.use_nodes:
                # Get the node tree of the material
                nodes = mat.node_tree.nodes
                print(nodes)
                # Find the shader node (it might be called something else, so check)
                shader_node = None
                for node in nodes:

                    if node.name == '3D Print Filament':  # Principled BSDF or other type
                        shader_node = node
                        break
                
                if shader_node:
                    # Set the Base Color of the shader node
                    print(new_color)
                    shader_node.inputs["Color"].default_value = (*new_color, 1)
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
    half_width_x = (0.46 - max_dim) / 2  # Half of the width along the x-axis
    half_width_y = (0.5 - max_dim) / 2   # Half of the height along the y-axis
    # Generate random x, y coordinates within the box's range
    x = random.uniform(-half_width_x, half_width_x)
    y = random.uniform(-half_width_y, half_width_y)
    

    roll = random.uniform(0, 2 * math.pi)

    return [[x, y, 0.07], [0, 0, roll]]

def set_camera(pose=[[0, -0.23, 0.6], [0.33163, 0, 0]]):
    pose_matrix = bproc.math.build_transformation_mat(pose[0], pose[1])
    bproc.camera.add_camera_pose(pose_matrix)
    # bproc.camera.set_resolution(640, 480)
    bproc.camera.set_intrinsics_from_blender_params(lens=0.959931, lens_unit="FOV")

def set_light(pose=[[0, 0, 0.7], [0, 0, 0]]):
    global light1

    # light1_position = mathutils.Vector(pose[0]) - light1_offset * pose[2].normalized()
    
    light1.set_location(pose[0])
    light1.set_rotation_euler(pose[1])
    light1.set_energy(12)

def set_object(pose=[[0, 0, 0], [0, 0, 0]]):
    # train_object.set_location(pose[0])
    # train_object.set_rotation_euler(pose[1])
    train_object.location = pose[0]
    train_object.rotation_euler = pose[1]
    return
# # def set_green_screen(pose):
# #     green_screen_position = -mathutils.Vector(pose[0]).normalized() * radius
# #     greenscreen.set_location(green_screen_position)
# #     greenscreen.set_rotation_euler(pose[1])

# def set_all_random():
#     pose = get_random_pose()

#     set_camera(pose)
#     set_light(pose)

def objects_touching_z(obj1, obj2):
    # Get world-space bounding box for obj1 (train_object) and obj2 (greenscreen)
    bbox1 = [obj1.matrix_world @ mathutils.Vector(corner) for corner in obj1.bound_box]
    bbox2 = [obj2.matrix_world @ mathutils.Vector(corner) for corner in obj2.bound_box]
    
    # Get min and max Z values for each object
    min_z1 = min([v.z for v in bbox1])
    max_z1 = max([v.z for v in bbox1])
    min_z2 = min([v.z for v in bbox2])
    max_z2 = max([v.z for v in bbox2])
    
    # Check if they are touching along the Z-axis
    # Assuming obj1 is above obj2
    touching = min_z1 <= max_z2 and max_z1 >= min_z2
    print(max_z1, min_z1)
    print(max_z2, min_z2)

    return touching

def place_obj1_on_top_of_obj2(obj1, obj2):
    # Get world-space bounding box for obj1 and obj2
    bbox1 = [obj1.matrix_world @ mathutils.Vector(corner) for corner in obj1.bound_box]
    bbox2 = [obj2.matrix_world @ mathutils.Vector(corner) for corner in obj2.bound_box]
    
    # Get min and max Z values for each object
    min_z1 = min(v.z for v in bbox1)
    max_z2 = max(v.z for v in bbox2)
    print(min_z1, max_z2)
    # Calculate the required Z offset to place obj1 on top of obj2
    offset_z = max_z2 - min_z1

    current_pose = obj1.location
    obj1.location = (current_pose.x, current_pose.y, current_pose.z + offset_z)
    # Move obj1 in Z direction by the calculated offset
    # obj1.location.z += offset_z
    # bpy.context.view_layer.update()

    print(f"Placed {obj1.name} on top of {obj2.name}")

def setup_physics_simulation():
    # Set up train object as an active rigid body
    train_object.select_set(True)
    bpy.context.view_layer.objects.active = train_object
    bpy.ops.rigidbody.object_add()
    train_object.rigid_body.type = 'ACTIVE'
    train_object.rigid_body.collision_shape = 'CONVEX_HULL'
    train_object.rigid_body.mass = 1.0  # Set mass

    

    # train_object.rigid_body.restitution = 0.0  # Set bounciness to zero
    # train_object.rigid_body.friction = 0.8  # Increase friction to prevent sliding
    # train_object.rigid_body.linear_damping = 0.5  # Damping to reduce movement after collision
    # train_object.rigid_body.angular_damping = 0.5

    # Set up GreenScreen as a passive rigid body
    if greenscreen is not None:
        greenscreen_obj = bpy.data.objects.get(greenscreen.get_name())
        greenscreen_obj.select_set(True)
        bpy.context.view_layer.objects.active = greenscreen_obj
        bpy.ops.rigidbody.object_add()
        greenscreen_obj.rigid_body.type = 'PASSIVE'
        greenscreen_obj.rigid_body.collision_shape = 'MESH'
        greenscreen_obj.rigid_body.mass = 0.0


def bake_physics_simulation():
    # Set the physics scene properties
    bpy.context.scene.rigidbody_world.time_scale = 1
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 120  # Number of frames to simulate

    # Bake the simulation
    bpy.context.view_layer.objects.active = train_object
    bpy.ops.ptcache.bake_all(bake=True)

    # Move to the last frame to access final position
    final_frame = bpy.context.scene.frame_end
    bpy.context.scene.frame_set(final_frame)

    # Set start and end frames to the last frame for single-frame render
    bpy.context.scene.frame_start = final_frame
    bpy.context.scene.frame_end = final_frame

def point_camera_at_object(camera_obj, target_object):
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
    bproc.camera.add_camera_pose(camera_pose, 0)

def render_scene():
    final_frame = bpy.context.scene.frame_end
    bpy.context.scene.frame_set(final_frame)
    # Render the scene
    bproc.renderer.enable_experimental_features()
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"], default_values={"category_id": None})
    bproc.renderer.set_max_amount_of_samples(2048)
    bproc.renderer.set_denoiser("INTEL")
    bproc.renderer.set_output_format(file_format="JPEG", jpg_quality=200)
    data = bproc.renderer.render()

    # Write the rendering into an hdf5 file
    bproc.writer.write_coco_annotations(os.path.join("examples/part_2", 'coco_data'),
                                        instance_segmaps=data["instance_segmaps"],
                                        instance_attribute_maps=data["instance_attribute_maps"],
                                        colors=data["colors"],
                                        color_file_format="JPEG")
    # bproc.writer.write_hdf5("output/", data)


init()
# setup_physics_simulation()
# set_object_random()
# bake_physics_simulation()


## Switch the viewport to camera view
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':  # Ensure it's the 3D viewport
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.region_3d.view_perspective = 'CAMERA'
                break
bpy.context.view_layer.update()

place_obj1_on_top_of_obj2(train_object, greenscreen.blender_obj)
point_camera_at_object(bpy.context.scene.camera, train_object)

# render_scene()

# try:
#     count = len(os.listdir("examples/part_2/coco_data/images"))
# except:
#     count = 0

# if count < 2000:
#     set_object_random()
#     render_scene()
#     sys.exit(1)
# else:
#     sys.exit(69)


