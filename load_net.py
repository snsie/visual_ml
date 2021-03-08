# pylint:disable=wrong-import-position,used-before-assignment,assignment-from-no-return,relative-beyond-top-level,no-value-for-parameter

if "bpy" in locals():
    import importlib
    importlib.reload(helpers)
else:
    from . import helpers


import bpy
from bpy.props import EnumProperty, IntProperty
import bmesh
import pathlib

from scipy import constants
import math
import numpy as np
import mathutils

set_mesh_material = helpers.set_mesh_material
get_data = helpers.get_data
get_color = helpers.get_color
# create_weight_mesh = helpers.create_weight_mesh
# create_node_mesh = helpers.create_node_mesh
ensure_collection = helpers.ensure_collection
load_json_model = helpers.load_json_model


def attach_weights(self, scene, current_collection):
    pass


def create_node_grid(self, scene, current_collection,
                     node_counts, depth_scale=0.15):
    # bpy.ops.mesh.primitive_plane_add(size=1, enter_editmode=False, align='WORLD', location=(
    #     0, 0, 0))
    # ob = bpy.context.active_object.data
    # ob.name = 'node_grid'
    bm = bmesh.new()
    bmesh.ops.create_grid(
        bm, x_segments=node_counts[0] + 1, y_segments=2, size=1)
    me = bpy.data.meshes.new("node_grid_mesh")
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new("node_grid_object", me)
    current_collection.objects.link(obj)
    # Select and make active
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    # bm2.free()
    # bm.from_mesh(ob)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap()
    # bm.verts.ensure_lookup_table()
    # for i, vert in enumerate(bm.verts):
    #     if i % 2 == 0:
    #         print(vert)
    # print(bm.verts)
    # verts = [bm.edges[0]]
    # extruded = bmesh.ops.extrude_edge_only(bm, edges=edge)
    # bm.free()
    bpy.ops.object.mode_set(mode='OBJECT')
    obj = scene.active_object
    obj.instance_type = 'FACES'
    # obj.use_instance_faces_scale = True
    # obj.instance_faces_scale = 1
    obj.show_instancer_for_render = False
    obj.show_instancer_for_viewport = False

    nodes_mesh = bpy.data.objects['nodes_mesh']
    nodes_mesh.parent = obj
    obj = scene.active_object
    nodes_mesh = bpy.data.objects['nodes_mesh']
    nodes_mesh.select_set(True)
    obj.select_set(False)
    bpy.ops.object.hide_view_set(unselected=False)

    # obj.children[0].hide_viewport = True
    # nodes_mesh.hide_viewport = True
    # bpy.ops.mesh.extrude_region_move(MESH_OT_extrude_region={
    #                                  "use_normal_flip": False, "use_dissolve_ortho_edges": False, "mirror": False}, TRANSFORM_OT_translate={"value": (0.174121, 0, 0)})


def create_node(self, scene, current_collection, depth_scale=0.1):
    master_collection = scene.collection

    bpy.ops.mesh.primitive_cube_add(
        size=1, location=(0, 0, 0), scale=(depth_scale, 0.9, 1.3))
    current_cube = scene.active_object
    current_cube.name = 'nodes_mesh'
    current_collection.objects.link(current_cube)
    master_collection.objects.unlink(current_cube)
    # bpy.ops.object.material_slot_add()

    current_cube_data = current_cube.data
    bm = bmesh.new()
    bm.from_mesh(current_cube_data)
    bm.edges.ensure_lookup_table()

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

    ##################################SMOOTHING###############
    # bpy.ops.object.editmode_toggle()
    # bpy.ops.mesh.select_all(action='SELECT')
    # bpy.ops.mesh.faces_shade_smooth()
    # bpy.ops.object.editmode_toggle()
    ##################################SMOOTHING###############
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
    return current_cube


class MESH_OT_load_net(bpy.types.Operator):
    """Load a Neural Network Mesh"""
    bl_idname = "mesh.load_net"
    bl_label = "Neural Network"
    bl_options = {'REGISTER', 'UNDO'}
    model_data = load_json_model()
    model_list = [(k, v) for k, v in model_data.items()]

    @ classmethod
    def poll(cls, context):
        # print(f"Current Context is: {context.area.type}")
        return context.area.type == 'VIEW_3D'

    def execute(self, context):
        collection_name = context.scene.collection_name

        current_collection = ensure_collection(context, collection_name)
        json_data = load_json_model()
        # print(type(json_data))
        # np.shape(json_data)
        # if context.object:
        # print('asdf')
        # .mode == 'EDIT':
        # bpy.ops.object.mode_set(mode='OBJECT')
        depth_scale = 0.2
        nc_layer_1 = json_data['layer_0']['nodeCount']
        node_counts = []
        for i_layer in json_data:
            node_counts.append(json_data[i_layer]['nodeCount'])

        current_cube = create_node(self, context, current_collection)
        create_node_grid(self, context, current_collection, node_counts)
        # text_context = bpy.ops.object.text_add(radius=0.32, enter_editmode=False, align='WORLD', location=(
        #     0.5, 0, 0), rotation=(0, 0, -1.5708), scale=(1, 1, 1))
        # # context.
        # create_node_mesh(self, context, current_collection,
        #  depth_scale=depth_scale)
        # create_weight_mesh(self, context, current_collection,
        #    depth_scale=depth_scale, )
        # master_collection.objects.unlink(current_cube)
        # print(master_collection.objects)
        return {"FINISHED"}
