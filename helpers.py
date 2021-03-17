# pylint:disable=assignment-from-no-return
from typing import NewType

import bpy
import bmesh
import pathlib
# from scipy import constants
import math
import numpy as np
import mathutils
import re
import json
from json import JSONEncoder
# Settings
width = 800
height = 800
z_height = 0.25  # Displace modifier strength
tex_res = 1     # Texture resolution (1:1)
mesh_res = 4    # Mesh resolution (8:1)
iscale = 20


def set_mesh_material(current_ob, name='mnet_material'):

    # Change to return data (should be between 0-1)
    scale = 20
    mesh_width = int(width / mesh_res)

    mesh_height = int(height / mesh_res)
    tex_width = int(width / tex_res)
    tex_height = int(height / tex_res)
    size = 2
    aspect_ratio = width / height

    displace_image = bpy.data.images.new(
        "Displace Map", width=tex_width, height=tex_height)

    diffuse_image = bpy.data.images.new(
        "Diffuse Map", width=tex_width, height=tex_height)

    displace_pixels = [None] * tex_width * tex_height
    diffuse_pixels = [None] * tex_width * tex_height

    for x in range(tex_width):
        for y in range(tex_height):
            a = get_data(x, y)
            displace_pixels[(y * tex_width) + x] = [a, a, a, 1.0]

            r, g, b = get_color(x, y)
            diffuse_pixels[(y * tex_width) + x] = [r, g, b, 1.0]

    displace_pixels = [chan for px in displace_pixels for chan in px]
    diffuse_pixels = [chan for px in diffuse_pixels for chan in px]

    displace_image.pixels = displace_pixels
    diffuse_image.pixels = diffuse_pixels

    # Create a displace texture
    displace_map = bpy.data.textures.new('Displace Texture', type='IMAGE')
    displace_map.image = displace_image

    # Create a displace modifier
    displace_mode = current_ob.modifiers.new("Displace", type='DISPLACE')
    displace_mode.texture = displace_map
    displace_mode.strength = z_height

    # Create a material
    material = bpy.data.materials.new(name="Plot Material")
    # Use nodes
    material.use_nodes = True
    # Add Principled BSDF
    bsdf = material.node_tree.nodes["Principled BSDF"]
    # Add an ImageTexture
    diffuse_map = material.node_tree.nodes.new('ShaderNodeTexImage')
    # Set diffuse image
    diffuse_map.image = diffuse_image


def get_data(x, y):
    x -= width / 2
    y -= height / 2

    x /= iscale
    y /= iscale

    r = math.sqrt(x**2 + y**2)
    return (math.sin(r) + 1) / 2

# Change to get color (you can use the ones defined below)


def get_color(x, y):
    a = get_data(x, y)
    return math.lerp(a, (1, 0, 0), (0, 0, 1))


def create_weight_mesh(self, scene, current_collection, depth_scale=0.2):
    x_min = -1
    y_min = -1 + 1 / self.count_N
    # y_max = 1
    y_step = 2 / self.count_N
    x_step = 2 * (1 / (self.count_L - 1))
    master_collection = scene.collection
    bpy.ops.curve.primitive_bezier_curve_add()
    base_weight = scene.active_object
    base_weight.data.name = 'weight_curve'
    bpy.ops.object.editmode_toggle()
    base_weight.data.splines[0].bezier_points[0].select_control_point = False
    base_weight.data.splines[0].bezier_points[0].select_left_handle = False
    base_weight.data.splines[0].bezier_points[0].select_right_handle = False

    base_weight.data.splines[0].bezier_points[1].select_control_point = True
    base_weight.data.splines[0].bezier_points[1].select_left_handle = True
    base_weight.data.splines[0].bezier_points[1].select_right_handle = True

    bpy.ops.curve.extrude_move()
    bpy.ops.object.editmode_toggle()
    base_weight.name = 'weight_object'
    current_collection.objects.link(base_weight)
    master_collection.objects.unlink(base_weight)
    base_weight.location.xyz = [-2.5, 0, 0]
    node_width_offset = 0.25
    b_point_rad = 0.25
    # bpy.ops.object.editmode_toggle()
    base_weight.data.splines[0].bezier_points[0].co.xyz = [-1, 0, 0]
    base_weight.data.splines[0].bezier_points[0].handle_left.xyz = [
        -1 - b_point_rad, 0, 0]
    base_weight.data.splines[0].bezier_points[0].handle_right.xyz = [
        -1 + b_point_rad, 0, 0]

    base_weight.data.splines[0].bezier_points[1].co.xyz = [0, 0, 0]
    base_weight.data.splines[0].bezier_points[1].handle_left.xyz = [
        0 - b_point_rad, 0, 0]
    base_weight.data.splines[0].bezier_points[1].handle_right.xyz = [
        0 + b_point_rad, 0, 0]

    base_weight.data.splines[0].bezier_points[2].co.xyz = [1, 0, 0]
    base_weight.data.splines[0].bezier_points[2].handle_left.xyz = [
        1 - b_point_rad, 0, 0]
    base_weight.data.splines[0].bezier_points[2].handle_right.xyz = [
        1 + b_point_rad, 0, 0]
    # weight_scale = 1/((self.count_L-1))
    # current_weight.scale = [weight_scale, weight_scale, weight_scale]
    base_weight.data.extrude = 0.1
    # bpy.ops.transform.translate(value=(0.605873, 0.325457, -0.142508), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
    #                             mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)

    # current_weight.location.x = 3
    # bpy.ops.object.editmode_toggle()

    material = bpy.data.materials.new(name="weight_material")
    # print(bpy.data.materials)
    # Use nodes
    material.use_nodes = True
    # Add Principled BSDF
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs[0].default_value = (0.8, 0.0405743, 0.119029, 1)

    base_weight.data.materials.append(material)
    count_WL = self.count_L - 1
    for idn_from in range(self.count_N):
        pos_idn_from = np.array([x_min, y_min + y_step * idn_from, 0])
        node_depth_offset_to = (depth_scale / 2) * \
            (-0.5 + 0.5 / self.count_N + idn_from / self.count_N)

        for idn in range(self.count_N):
            # if idn_from < 3:
            #     continue
            x_to = x_min + x_step
            y_to = y_min + y_step * idn
            pos_idn_to = np.array([x_to, y_to, 0])
            pos_mid = (pos_idn_to + pos_idn_from) / 2
            node_depth_offset_from = (depth_scale / 2) * \
                (-0.5 + 0.5 / self.count_N + idn / self.count_N)
            node_depth_offset_mid = (
                node_depth_offset_from + node_depth_offset_to) / 2
            # print(pos_mid)

            # scene.active_object.select_set(False)
            # base_weight.select_set(True)
            prev_ob = scene.active_object
            bpy.ops.object.duplicate()
            # base_weight.data.copy()
            bez_point_y = pos_mid[1] - pos_idn_from[1]

            weight_cp = scene.active_object
            weight_cp.location.xyz = pos_mid

            weight_spline = weight_cp.data.splines[0]

            x_min = -1
            x_max = 1
            # bez_first_y=node_depth_offset-
            weight_spline.bezier_points[0].co.xy = [
                x_min + node_width_offset, -bez_point_y + node_depth_offset_from]
            weight_spline.bezier_points[0].handle_left.xy = [
                x_min - b_point_rad + node_width_offset, -bez_point_y + node_depth_offset_from]
            weight_spline.bezier_points[0].handle_right.xy = [
                x_min + b_point_rad + node_width_offset, -bez_point_y + node_depth_offset_from]

            weight_spline.bezier_points[1].co.xy = [0, node_depth_offset_mid]
            weight_spline.bezier_points[1].handle_left.xy = [
                0 - b_point_rad, node_depth_offset_mid]
            weight_spline.bezier_points[1].handle_right.xy = [
                0 + b_point_rad, node_depth_offset_mid]

            weight_spline.bezier_points[2].co.xy = [
                x_max - node_width_offset, bez_point_y
                + node_depth_offset_to
            ]
            weight_spline.bezier_points[2].handle_left.xy = [
                x_max - b_point_rad - node_width_offset, bez_point_y
                + node_depth_offset_to
            ]
            weight_spline.bezier_points[2].handle_right.xy = [
                x_max + b_point_rad - node_width_offset, bez_point_y
                + node_depth_offset_to
            ]

            # weight_spline.bezier_points[0].select_control_point = True
            # weight_spline.bezier_points[0].select_left_handle = True
            # weight_spline.bezier_points[0].select_right_handle = True

            # weight_spline.bezier_points[1].select_control_point = True
            # weight_spline.bezier_points[1].select_left_handle = True
            # weight_spline.bezier_points[1].select_right_handle = True
            # weight_spline.bezier_points[2].select_control_point = True
            # weight_spline.bezier_points[2].select_left_handle = True
            # weight_spline.bezier_points[2].select_right_handle = True

        # bpy.ops.object.convert(target='MESH', keep_original=True)
        # bpy.ops.object.editmode_toggle()
        # print(str(idn), ' AFTER ',
        #   weight_spline.bezier_points[0].handle_left.xyz)

        # ob_spline.bezier_points[0].co.xyz = pos_idn_from
        # ob_spline.bezier_points[1].co.xyz = pos_idn_to
        # bpy.ops.object.editmode_toggle()

        #  C.active_object.data.splines[0].bezier_points[0].select_control_point
        # ob_spline.bezier_points[1].co.xyz = pos_idn_to
        # bpy.ops.object.transform_apply(
        #     location=True, rotation=True, scale=True)
        # base_weight.select_set(True)

    # base_weight.select_set(True)
    # if idn < 1:
    #     for next_idn in range(self.count_N):
    #         next_y = y + next_idn/self.count_N
    #         next_idn_pos = np.array([next_x, next_y, 0])
    #         midpoint = (base_idn_pos+next_idn_pos)/2
    #         # print(midpoint)
    #         # ob = current_weight.copy()
    #         # bpy.ops.object.duplicate_move()
    #         weight_cp = scene.active_object

    #         # print("CURRENTWEIGHT STUFF")
    #         # print(dir(bpy.ops.object))
    #         # print("CURRENTWEIGHT STUFF")
    #         # ob
    #         # current_collection.objects.link(ob)
    #         weight_cp.location = midpoint
    #         weight_cp.data.splines[0].bezier_points[0].co.x
    #         # current_weight.data.splines[0].bezier_points[0].co.x = next_idn
    #         bpy.ops.object.convert(target='MESH', keep_original=True)

##############################################################################


def create_node_mesh(self, scene, current_collection, depth_scale=0.2):
    max_n = 64
    # print(text_ob)

    y_min = -1 + 1 / self.count_N
    # y_max = 1
    y = 2 / self.count_N
    x_min = -1
    # x_max = 1
    master_collection = scene.collection

    for idl in range(self.count_L):

        x = x_min + 2 * (idl / (self.count_L - 1))
        bpy.ops.mesh.primitive_cube_add(
            size=1, location=(x, y_min, 0), scale=(1, depth_scale, 1.3))
        current_cube = scene.active_object
        current_cube.name = 'nodes.layer.' + str(idl + 1).zfill(2)
        current_collection.objects.link(current_cube)
        master_collection.objects.unlink(current_cube)
        # bpy.ops.object.material_slot_add()

        current_cube_data = current_cube.data
        bm = bmesh.new()
        bm.from_mesh(current_cube_data)
        bm.edges.ensure_lookup_table()
        idx = 0
        #
        # verts = [v for v in bm.verts if len(v.link_edges) < 10.11]
        # for iv in verts:
        # print(len(iv.link_edges))bpy.ops.mesh.bevel(offset=0.02, offset_pct=0, segments=3, affect='EDGES')
        smallest_edges = []
        edge_length = 999
        for curr_edge in bm.edges:
            curr_edge_length = curr_edge.calc_length()
            if curr_edge_length == edge_length:
                smallest_edges.append(curr_edge)
            if curr_edge_length < edge_length:
                smallest_edges.clear()
                edge_length = curr_edge_length
                smallest_edges.append(curr_edge)

        # edges = [bm.edges[0], bm.edges[1]]
        bmesh.ops.bevel(bm, geom=smallest_edges, offset=0.015, profile=0.5,
                        segments=8, affect='EDGES')
        # edge.co.x += 0.1*idx
        bm.to_mesh(current_cube_data)
        bm.free()  # free and prevent further access
        current_cube = scene.active_object

        # #################################SMOOTHING###############
        # bpy.ops.object.editmode_toggle()
        # bpy.ops.mesh.select_all(action='SELECT')
        # bpy.ops.mesh.faces_shade_smooth()
        # bpy.ops.object.editmode_toggle()
        # #################################SMOOTHING###############
        # for f in current_cube_data.polygons:
        #     f.use_smooth = True
        # bpy.ops.object.editmode_toggle()

        #############################MATERIAL######################
        material = bpy.data.materials.new(name="node_material")
        # print(bpy.data.materials)
        # Use nodes
        material.use_nodes = True
        # Add Principled BSDF
        bsdf = material.node_tree.nodes["Principled BSDF"]
        bsdf.inputs[0].default_value = (0.0284887, 0.8, 0.0864939, 1)
        current_cube.data.materials.append(material)

        ##################################ARRAY_MODIFIER###############
        bpy.ops.object.modifier_add(type='ARRAY')
        mod_array = current_cube.modifiers['Array']
        mod_array.count = self.count_N
        mod_array.use_relative_offset = False
        mod_array.use_constant_offset = True
        mod_array.constant_offset_displace = [0, y, 0]
        ##################################ARRAY_MODIFIER###############
        # bpy.ops.object.modifier_add(type='BEVEL')

    # return current_cube


def ensure_collection(scene, collection_name) -> bpy.types.Collection:
    # curr_scene = NewType('curr_scene', bpy.types.Scene.collection)
    # curr_scene.re
    if collection_name in scene.collection.children:
        link_to = scene.collection.children[collection_name]
        for ob in list(bpy.data.objects):
            match_w = re.findall(r'^weight', ob.name)
            match_n = re.findall(r'^node', ob.name)
            if match_w or match_n:
                # link_to.objects.unlink(ob)
                bpy.data.objects.remove(ob)
        for node_group in list(bpy.data.node_groups):
            match_w = re.findall(r'^weight', node_group.name)
            match_n = re.findall(r'^node', node_group.name)
            if match_w or match_n:
                bpy.data.node_groups.remove(node_group)
        for curve in list(bpy.data.curves):
            match_w = re.findall(r'^weight', curve.name)
            match_n = re.findall(r'^node', curve.name)
            if match_w or match_n:
                bpy.data.curves.remove(curve)

        for mat in list(bpy.data.materials):
            match_w = re.findall(r'^weight', mat.name)
            match_n = re.findall(r'^node', mat.name)
            if match_w or match_n:
                bpy.data.materials.remove(mat)

        for mesh in list(bpy.data.meshes):
            match_w = re.findall(r'^weight', mesh.name)
            match_n = re.findall(r'^node', mesh.name)
            if match_w or match_n:
                bpy.data.meshes.remove(mesh)

        bpy.data.collections.remove(link_to)
    link_to = bpy.data.collections.new(collection_name)
    scene.collection.children.link(link_to)
    return link_to


def old_ensure_collection(scene, collection_name) -> bpy.types.Collection:
    # curr_scene = NewType('curr_scene', bpy.types.Scene.collection)
    # curr_scene.re
    if collection_name in scene.collection.children:
        link_to = scene.collection.children[collection_name]
        for ob in list(bpy.data.objects):
            match_w = re.findall(r'^weight', ob.name)
            match_n = re.findall(r'^node', ob.name)
            if match_w or match_n:
                link_to.objects.unlink(ob)
                bpy.data.objects.remove(ob)

        for curve in list(bpy.data.curves):
            match_w = re.findall(r'^weight', curve.name)
            match_n = re.findall(r'^node', curve.name)
            if match_w or match_n:
                bpy.data.curves.remove(curve)

        for mat in list(bpy.data.materials):
            match_w = re.findall(r'^weight', mat.name)
            match_n = re.findall(r'^node', mat.name)
            if match_w or match_n:
                bpy.data.materials.remove(mat)

        for mesh in list(bpy.data.meshes):
            match_w = re.findall(r'^weight', mesh.name)
            match_n = re.findall(r'^node', mesh.name)
            if match_w or match_n:
                bpy.data.meshes.remove(mesh)

        bpy.data.collections.remove(link_to)
    # for current_curve in list(bpy.data.curves):
    #     # print(current)
    #     bpy.data.curves.remove(current_curve)
    # for current_mesh in list(bpy.data.meshes):
    #     bpy.data.meshes.remove(current_mesh)
    # for current_material in list(bpy.data.materials):
    #     bpy.data.materials.remove(current_material)
    # for current_material in list(bpy.data.materials):
    #     bpy.data.materials.remove(current_material)

    # for icoll, curr_collection in enumerate(scene.collection.children):
        # if icoll > 0:
        # print(curr_collection)
        # bpy.data.collections.remove(curr_collection)

    link_to = bpy.data.collections.new(collection_name)
    scene.collection.children.link(link_to)

    return link_to


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        # JSONEncoder.FLOAT_REPR = lambda o: format(o, '.2f')
        if isinstance(obj, np.ndarray):
            out = obj.tolist()

            # out=np.round(out,5)
            for i, iarray in enumerate(out):
                if isinstance(iarray, list):
                    out[i] = np.around(iarray, 5)
                    # out[i] = [f"{num:.5f}" for num in iarray]
                else:
                    out[i] = np.round(iarray, 5)
            # if isinstance(out[0],list)==False:
                # print('hello')
                # out=np.around(out,5)

            return out

            # print(i)
            # print('done')
            # return out
        return JSONEncoder.default(self, obj)


def load_json_model():
    with open('/home/scott/dev/blender/scripts/addons/visual_ml/data/numpyData.json') as f:
        data = json.load(f)
    return data
