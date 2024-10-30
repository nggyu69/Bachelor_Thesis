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

# # greenscreen = bpy.data.objects['GreenScreen']
# # greenscreen = bproc.filter.one_by_attr(scene, "name", "GreenScreen")
train_object = ""
train_object_name = "Train_Honeycomb_Wall_Tool"
for i, item in enumerate(scene):
        if item.get_name().startswith("Train_"):
            if item.get_name().startswith(train_object_name):
                item.set_cp("category_id", i + 1)
                train_object = item
            else:
                item.set_cp("category_id", 0)
                item.hide()
                item.blender_obj.hide_viewport = True
        # elif item.get_name().startswith("GreenScreen"):
        #     greenscreen = item
        #     item.set_cp("category_id", 0)
        else:
            item.set_cp("category_id", 0)
        item.set_shading_mode("AUTO")


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
    set_object_color("#FFFFFF", train_object.get_name())

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
    train_object_dimensions = train_object.blender_obj.dimensions
    max_dim = max(train_object_dimensions.x, train_object_dimensions.y)
    half_width_x = (0.46 - max_dim) / 2  # Half of the width along the x-axis
    half_width_y = (0.5 - max_dim) / 2   # Half of the height along the y-axis
    # Generate random x, y coordinates within the box's range
    x = random.uniform(-half_width_x, half_width_x)
    y = random.uniform(-half_width_y, half_width_y)
    

    roll = random.uniform(0, 2 * math.pi)

    return [[x, y, 0], [0, 0, roll]]

def set_camera(pose=[[0, -0.23, 0.6], [0.33163, 0, 0]]):
    pose = bproc.math.build_transformation_mat(pose[0], pose[1])
    bproc.camera.add_camera_pose(pose, 0)
    # bproc.camera.set_resolution(640, 480)
    bproc.camera.set_intrinsics_from_blender_params(lens=0.959931, lens_unit="FOV")

def set_light(pose=[[0, 0, 0.7], [0, 0, 0]]):
    global light1

    # light1_position = mathutils.Vector(pose[0]) - light1_offset * pose[2].normalized()
    
    light1.set_location(pose[0])
    light1.set_rotation_euler(pose[1])
    light1.set_energy(12)

def set_object(pose=[[0, 0, 0], [0, 0, 0]]):
    train_object.set_location(pose[0])
    train_object.set_rotation_euler(pose[1])

# # def set_green_screen(pose):
# #     green_screen_position = -mathutils.Vector(pose[0]).normalized() * radius
# #     greenscreen.set_location(green_screen_position)
# #     greenscreen.set_rotation_euler(pose[1])

# def set_all_random():
#     pose = get_random_pose()

#     set_camera(pose)
#     set_light(pose)


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
    bproc.writer.write_coco_annotations(os.path.join("examples/part_2", 'coco_data'),
                                        instance_segmaps=data["instance_segmaps"],
                                        instance_attribute_maps=data["instance_attribute_maps"],
                                        colors=data["colors"],
                                        color_file_format="JPEG")
    # bproc.writer.write_hdf5("output/", data)
init()

# Switch the viewport to camera view
# for area in bpy.context.screen.areas:
#     if area.type == 'VIEW_3D':  # Ensure it's the 3D viewport
#         for space in area.spaces:
#             if space.type == 'VIEW_3D':
#                 space.region_3d.view_perspective = 'CAMERA'
#                 break
# # bpy.context.view_layer.update()
try:
    count = len(os.listdir("examples/part_2/coco_data/images"))
except:
    count = 0

if count < 2000:
    set_object_random()
    render_scene()
    sys.exit(1)
else:
    sys.exit(69)


