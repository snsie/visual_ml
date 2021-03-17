
from typing import NewType

import bpy
from bpy import types

import bmesh
import pathlib
from scipy import constants
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


def create_weight_grid(self, context: bpy.types.Context,
                       current_collection: bpy.types.Collection,
                       json_data: dict):
    master_collection = context.collection
    verts = [
        (0.0, 0.5, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.5, 0.0),
        (0.0, 2.0, 0.0),
        (0.0, 2.5, 0.0),
        (0.0, 3.0, 0.0)]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
    ]
    mesh_name = "weight_grid_ob"
    mesh_data = bpy.data.meshes.new(mesh_name)
    mesh_data.from_pydata(verts, edges, [])
    corrections = mesh_data.validate(
        verbose=True,
        clean_customdata=True)
    bm = bmesh.new()
    bm.from_mesh(mesh_data)

    bm.to_mesh(mesh_data)
    bm.free()
    mesh_obj = bpy.data.objects.new(mesh_data.name, mesh_data)
    current_collection.objects.link(mesh_obj)


def set_weight_anim(self, context: bpy.types.Context,
                    current_collection: bpy.types.Collection,
                    json_data_begin: list,
                    json_data_end: list,
                    ):
    max_n = 64
    y_step = 2 / max_n
    for i in range(max_n):
        weight_cp_name = 'weight_prim_' + str(i + 1).zfill(3)
        ob = bpy.data.objects[weight_cp_name]
        context.scene.frame_set(1)
        ob.location = (0, 0, json_data_begin[i] / 5)
        ob.keyframe_insert(data_path="location")
        context.scene.frame_set(30)

        ob.location = (0, 0, json_data_end[i] / 5)

        ob.keyframe_insert(data_path="location")
        context.scene.frame_set(60)
        ob.location = (0, 0, json_data_begin[i] / 5)
        ob.keyframe_insert(data_path="location")
    context.scene.frame_set(1)
    # base_w.select_set(True)
    # bpy.ops.object.hide_view_set(unselected=False)
    # bpy.ops.object.select_all(action='DESELECT')
    # collection_name = 'asdf'
    # for j in range(2):
    #     curr_coll_name = collection_name + str(j)
    #     bpy.ops.object.duplicate_move()
    #     bpy.ops.transform.translate(value=[0, y_step, 0])
    #     bpy.ops.object.move_to_collection(
    #         collection_index=-1,
    #  is_new=True, new_collection_name=repr(curr_coll_name))


def create_weight_prims(self, context: bpy.types.Context,
                        current_collection: bpy.types.Collection,
                        json_data: dict):
    max_n = 64
    x_min = -1
    y_min = -1 + 1 / max_n
    node_width = 1 / (2 * max_n)
    # y_max = 1
    y_step = 2 / max_n
    master_collection = context.collection
    bpy.ops.curve.primitive_bezier_curve_add()
    weight_prim = context.active_object
    context.object.data.resolution_u = 32

    # ########################################################33
    material = bpy.data.materials.new(name="node_material")
    material.blend_method = 'BLEND'

    # Use nodes
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    # bsdf.inputs[0].default_value = (0.0284887, 0.8, 0.0864939, 1)
    weight_prim.data.materials.append(material)

    weight_prim_name = 'weight_curve_prim'
    weight_prim.name = weight_prim_name
    nt_mat = bpy.data.materials['node_material'].node_tree
    # ######
    curr_node = nt_mat.nodes.new(type="ShaderNodeValToRGB")
    curr_node.location.x = curr_node.location.x - curr_node.width * 1.5
    curr_node.location.y = curr_node.location.y + curr_node.height * 2
    curr_node.color_ramp.elements[0].color = (0.623953, 0.00248372, 0, 1)
    curr_node.color_ramp.elements[1].color = (0.00127905, 1, 0, 1)
    sock_out = curr_node.outputs['Color']
    sock_in = bsdf.inputs['Base Color']
    nt_mat.links.new(sock_out, sock_in)
    # ######
    prev_node = curr_node
    curr_node = nt_mat.nodes.new(type="ShaderNodeMapRange")
    curr_node.location.x = prev_node.location.x - curr_node.width * 1.5
    curr_node.location.y = prev_node.location.y
    curr_node.inputs['From Min'].default_value = -0.05
    curr_node.inputs['From Max'].default_value = 0.05
    sock_out = curr_node.outputs['Result']
    sock_in = prev_node.inputs['Fac']
    nt_mat.links.new(sock_out, sock_in)
    # #######
    prev_node = curr_node
    curr_node = nt_mat.nodes.new(type="ShaderNodeSeparateXYZ")
    curr_node.location.x = prev_node.location.x - curr_node.width * 1.5
    curr_node.location.y = prev_node.location.y
    sock_out = curr_node.outputs['Z']
    sock_in = prev_node.inputs['Value']
    nt_mat.links.new(sock_out, sock_in)
    # #######
    prev_node = curr_node
    curr_node = nt_mat.nodes.new(type="ShaderNodeNewGeometry")
    curr_node.location.x = prev_node.location.x - curr_node.width * 1.5
    curr_node.location.y = prev_node.location.y
    sock_out = curr_node.outputs['Position']
    sock_in = prev_node.inputs['Vector']
    nt_mat.links.new(sock_out, sock_in)

    # ######## NODES FOR ALPHA BELOW
    prev_node = nt_mat.nodes['Map Range']
    curr_node = nt_mat.nodes.new(type="ShaderNodeMath")
    curr_node.location.x = prev_node.location.x
    curr_node.location.y = prev_node.location.y - prev_node.height * 2.5
    curr_node.operation = 'ABSOLUTE'
    prev_node = nt_mat.nodes['Separate XYZ']
    sock_out = prev_node.outputs['Z']
    sock_in = curr_node.inputs['Value']
    nt_mat.links.new(sock_out, sock_in)
    # ########
    prev_node = nt_mat.nodes['ColorRamp']
    curr_node = nt_mat.nodes.new(type="ShaderNodeMapRange")
    curr_node.name = 'Map Range Alpha'
    curr_node.location.x = prev_node.location.x
    curr_node.location.y = prev_node.location.y - prev_node.height * 2.5
    curr_node.inputs['From Min'].default_value = 0
    curr_node.inputs['From Max'].default_value = 0.05
    curr_node.inputs['To Min'].default_value = 0.1

    prev_node = nt_mat.nodes['Math']
    sock_out = prev_node.outputs['Value']
    sock_in = curr_node.inputs['Value']
    nt_mat.links.new(sock_out, sock_in)

    sock_out = curr_node.outputs['Result']
    sock_in = bsdf.inputs['Alpha']
    nt_mat.links.new(sock_out, sock_in)
    # ########################################################33

    bpy.ops.object.mode_set(mode='EDIT')
    weight_prim.data.splines[0].bezier_points[0].select_control_point = False
    weight_prim.data.splines[0].bezier_points[0].select_left_handle = False
    weight_prim.data.splines[0].bezier_points[0].select_right_handle = False

    weight_prim.data.splines[0].bezier_points[1].select_control_point = True
    weight_prim.data.splines[0].bezier_points[1].select_left_handle = True
    weight_prim.data.splines[0].bezier_points[1].select_right_handle = True

    bpy.ops.curve.extrude_move()
    # current_collection.objects.link(weight_prim)
    # master_collection.objects.unlink(weight_prim)
    # weight_prim.location = (-2.5, 0, 0)
    y_space_between_nodes = 1 / (10 * max_n)
    bez_point_rad = 0.5
    # bpy.ops.object.editmode_toggle()
    weight_prim.data.splines[0].bezier_points[0].co = (-1, 0, 0)
    weight_prim.data.splines[0].bezier_points[0].handle_left = (
        -1 - bez_point_rad, 0, 0)
    weight_prim.data.splines[0].bezier_points[0].handle_right = (
        -1 + bez_point_rad, 0, 0)

    weight_prim.data.splines[0].bezier_points[1].co = (0, 0, 0)
    weight_prim.data.splines[0].bezier_points[1].handle_left = (
        0 - bez_point_rad, 0, 0)
    weight_prim.data.splines[0].bezier_points[1].handle_right = (
        0 + bez_point_rad, 0, 0)

    weight_prim.data.splines[0].bezier_points[2].co = (1, 0, 0)
    weight_prim.data.splines[0].bezier_points[2].handle_left = (
        1 - bez_point_rad, 0, 0)
    weight_prim.data.splines[0].bezier_points[2].handle_right = (
        1 + bez_point_rad, 0, 0)
    weight_prim.data.extrude = 0.05 / 4
    bpy.ops.object.mode_set(mode='OBJECT')
    # base_weight_prim = bpy.data.objects[weight_prim_name]
    # print(base_weight_prim)
    count_WL = self.count_L - 1
    x_step = 2
    idn_from = 0
    node_depth_offset = 0
    node_depth_offset_mid = 0
    x_to = x_min + x_step
    pos_idn_from = np.array([x_min, y_min, 0])

    for idn in range(self.max_N):

        y_to = y_min + y_step * idn
        pos_idn_to = np.array([x_to, y_to, 0])
        pos_mid = (pos_idn_to + pos_idn_from) / 2
        # node_depth_offset = (node_width / 2) * \
        # (-0.5 + 0.5 / self.count_N + idn / self.count_N)

        # context.selected_objects=bpy

        # weight_cp = context.active_object
        bpy.ops.object.duplicate()
        weight_cp = context.active_object
        if idn == 0:
            current_collection.objects.link(weight_cp)
            master_collection.objects.unlink(weight_cp)
        weight_cp_name = 'weight_prim_' + str(idn + 1).zfill(3)
        weight_cp.name = weight_cp_name
        bez_point_y = pos_mid[1] - pos_idn_from[1]

        weight_cp.location = pos_mid

        weight_spline = weight_cp.data.splines[0]
        weight_spline.bezier_points[0].co = (
            x_min, -bez_point_y + node_depth_offset, 0)
        weight_spline.bezier_points[0].handle_left = (
            x_min - bez_point_rad,
            -bez_point_y + node_depth_offset, 0)
        weight_spline.bezier_points[0].handle_right = (
            x_min + bez_point_rad,
            -bez_point_y + node_depth_offset, 0)

        weight_spline.bezier_points[1].co = (0, node_depth_offset_mid, 0)
        weight_spline.bezier_points[1].handle_left = (
            0 - bez_point_rad, node_depth_offset_mid, 0)
        weight_spline.bezier_points[1].handle_right = (
            0 + bez_point_rad, node_depth_offset_mid, 0)

        weight_spline.bezier_points[2].co = (
            1, bez_point_y - node_depth_offset, 0)
        weight_spline.bezier_points[2].handle_left = (
            1 - bez_point_rad, bez_point_y - node_depth_offset, 0)
        weight_spline.bezier_points[2].handle_right = (
            1 + bez_point_rad, bez_point_y - node_depth_offset, 0)

    bpy.ops.object.select_all(action='DESELECT')
    base_w = bpy.data.objects[weight_prim_name]
    base_w.select_set(True)
    bpy.ops.object.hide_view_set(unselected=False)
    bpy.ops.object.select_all(action='DESELECT')

    for i in range(self.max_N):
        weight_cp_name = 'weight_prim_' + str(i + 1).zfill(3)
        current_w: bpy.types.Object = \
            current_collection.objects[weight_cp_name]
        current_w.select_set(True)

        # print(context.area.type)
        # current_w = context.active_object
        # context.area.type = '?'
        # bpy.ops.object.mode_set(mode='EDIT')
        # if bpy.ops.object.mode_set.poll():
        # print(current_w)

        # if match_w or match_n:
        # link_to.objects.unlink(ob)
        # bpy.data.objects.remove(ob)
    bpy.ops.object.convert(target='MESH')
    bpy.ops.transform.translate(value=[1, 1 - 1 / max_n, 0])
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    weight_cp_name = 'weight_prim_' + str(1).zfill(3)
    current_w: bpy.types.Object = \
        current_collection.objects[weight_cp_name]
    # current_w.select_set(False)
    # bpy.ops.object.duplicate()
    # bpy.ops.transform.rotate(value=3.14159, orient_axis='X')
    # for j in range(max_n, 0, -1):

    # for j in range(2):
    #     bpy.ops.object.duplicate_move()
    #     bpy.ops.transform.translate(value=[0, y_step, 0])
    #     bpy.ops.object.move_to_collection(collection_index=2)


def add_geonode_linear(self, context: bpy.types.Context,):
    bpy.ops.mesh.primitive_grid_add(
        size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))

    weight_ob = context.active_object
    bpy.ops.node.new_geometry_nodes_modifier()
    geonode_mod = weight_ob.modifiers.active

    # weight_ob.modifiers[0].name = 'asdf'
    geo_ng: bpy.types.GeometryNodeGroup = geonode_mod.node_group
    geo_ng.name = 'weight_geo_nodes'
    geo_ng.links.clear()

    # bpy.ops.object.mode_set(mode='EDIT')
    # links.new(gradColOut, backColIn)
    # curr_node.width
    group_in: bpy.types. \
        GeometryNode = geo_ng.nodes["Group Input"]
    group_out: bpy.types. \
        GeometryNodeAttributeFill = geo_ng.nodes["Group Output"]

    curr_node: bpy.types. \
        GeometryNode = geo_ng.nodes.new(
            type="GeometryNodeAttributeSeparateXYZ")
    # Connect In to Attribute Fill
    curr_in = curr_node.inputs["Geometry"]
    curr_out = group_in.outputs["Geometry"]

    geo_ng.links.new(curr_out, curr_in)

    curr_node.inputs[1].default_value = "position"
    curr_node.inputs[3].default_value = "xin"
    curr_node.inputs[4].default_value = "yin"
    curr_node.inputs[5].default_value = "zin"
    xloc = curr_node.location.x
    # xloc += curr_node.width
    yloc = curr_node.location.y
    # group_out.location.x += xloc
    node_list = {
        "GeometryNodeAttributeMath":
        {"operation": "MULTIPLY", 1: "xin", 3: "zin", 7: "zout"},
        "GeometryNodeAttributeMath":
        {"operation": "ADD", 1: "zout", 3: "yin", 7: "zout"},
        "GeometryNodeAttributeCombineXYZ":
        {"input_type_x": 'ATTRIBUTE', "input_type_y": 'ATTRIBUTE',
         "input_type_z": 'ATTRIBUTE', 1: "xin", 3: "yin", 5: "zout",
         7: "position"}, }
    prev_node = curr_node
    curr_node.select = False
    for i in node_list:

        curr_node = geo_ng.nodes.new(
            type=i)
        xloc += prev_node.width * 1.5
        curr_node.location.x = xloc
        for j in node_list[i]:
            check_input_type = re.findall(r'^input_type', str(j))
            if isinstance(j, str):
                setattr(curr_node, j, node_list[i][j])
            # if j == "operation":
                # setattr(curr_node, j, node_list[i][j])
                # curr_node[j] = node_list[i][j]
            if isinstance(j, int):
                curr_node.inputs[j].default_value = node_list[i][j]

        curr_in = curr_node.inputs["Geometry"]
        curr_out = prev_node.outputs["Geometry"]
        geo_ng.links.new(curr_out, curr_in)

        curr_node.select = False
        prev_node = curr_node

    xloc += prev_node.width * 1.5
    group_out.location.x = xloc
    curr_in = group_out.inputs["Geometry"]
    curr_out = curr_node.outputs["Geometry"]
    geo_ng.links.new(curr_out, curr_in)
    # curr_node: bpy.types. \
    #     GeometryNode = geo_ng.nodes.new(
    #         type="GeometryNodeAttributeSeparateXYZ")
# bpy.ops.node.add_node(type="GeometryNodeAttributeMath", use_transform=True)


def create_weight_mesh_cyl(self, context: bpy.types.Context,
                           current_collection: bpy.types.Collection):
    depth = 64
    master_collection = context.collection
    vertices = 32
    bpy.ops.mesh.primitive_cylinder_add(radius=0.25, vertices=vertices,
                                        depth=depth,
                                        enter_editmode=False, align='WORLD',
                                        location=(0, 0, 0), scale=(1, 1, 1))

    weight_ob: bpy.types.Object = context.active_object
    weight_ob.name = "weight_shape_ob"
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.transform.translate(value=(0, 0, depth / 2), orient_matrix=(
        (1, 0, 0), (0, 1, 0), (0, 0, 1)),)
    bpy.ops.mesh.loopcut_slide(
        MESH_OT_loopcut={"number_cuts": 11, "smoothness": 0,
                         "falloff": 'INVERSE_SQUARE', "object_index": 0,
                         "edge_index": 12,
                         "mesh_select_mode_init": (True, False, False)})

    current_collection.objects.link(weight_ob)
    master_collection.objects.unlink(weight_ob)
    bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.transform.rotate(
        value=1.5708, orient_axis='Y',
        orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        constraint_axis=(False, True, False), mirror=True,
        proportional_edit_falloff='SMOOTH', proportional_size=1,
        use_proportional_connected=False, use_proportional_projected=False)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # bpy.context.area.ui_type = 'GeometryNodeTree'

    bpy.ops.object.modifier_add(type='EDGE_SPLIT')
    bpy.ops.object.modifier_apply(modifier="EdgeSplit")
    bpy.ops.object.shade_smooth()


##############################################################################


def create_node_mesh(self, scene, current_collection, depth_scale=0.2):
    max_n = 64

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

        # ############################MATERIAL######################
        material = bpy.data.materials.new(name="node_material")
        # Use nodes
        material.use_nodes = True
        # Add Principled BSDF
        bsdf = material.node_tree.nodes["Principled BSDF"]
        bsdf.inputs[0].default_value = (0.0284887, 0.8, 0.0864939, 1)
        current_cube.data.materials.append(material)

        # # #################################ARRAY_MODIFIER###############
        # bpy.ops.object.modifier_add(type='ARRAY')
        # mod_array = current_cube.modifiers['Array']
        # mod_array.count = self.count_N
        # mod_array.use_relative_offset = False
        # mod_array.use_constant_offset = True
        # mod_array.constant_offset_displace = [0, y, 0]
        # #################################ARRAY_MODIFIER###############
        # bpy.ops.object.modifier_add(type='BEVEL')

    # return current_cube


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


def ensure_collection(context, collection_name) -> bpy.types.Collection:
    # curr_scene = NewType('curr_scene', bpy.types.Scene.collection)
    # curr_scene.re
    asdf = "weight"
    if collection_name in context.collection.children:
        link_to = context.collection.children[collection_name]
        for ob in list(bpy.data.objects):
            match_w = re.findall(r'^{}'.format(asdf), ob.name)
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
        # for coll in list(context.collection.children):
        # print(coll.name)

    link_to = bpy.data.collections.new(collection_name)
    context.collection.children.link(link_to)

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
                # out=np.around(out,5)

            return out
            # return out
        return JSONEncoder.default(self, obj)


def load_json_model(json_fname: str) -> dict:
    json_path = '/home/scott/dev/blender/scripts/addons/visual_ml/data/'

    with open(json_path + json_fname) as f:
        data = json.load(f)
    return data
