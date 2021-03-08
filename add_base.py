# pylint:disable=wrong-import-position,used-before-assignment,assignment-from-no-return,relative-beyond-top-level
from typing import NewType

if "bpy" in locals():
    import importlib
    importlib.reload(helpers)
else:
    from . import helpers


import bpy
from bpy.props import IntProperty
from bpy.types import (Scene, Object)
import bmesh
import pathlib

from scipy import constants
import math
import numpy as np
import mathutils

set_mesh_material = helpers.set_mesh_material
get_data = helpers.get_data
get_color = helpers.get_color


def create_base_mesh(self, scene, current_collection, depth_scale=0.2):
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
        current_cube.name = 'base'
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
        material = bpy.data.materials.new(name="base_material")
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


class MESH_OT_add_base(bpy.types.Operator):
    """Construct a Neural Network Mesh"""
    bl_idname = "mesh.add_nets"
    bl_label = "Neural Network"
    bl_options = {'REGISTER', 'UNDO'}

    # count_L: IntProperty(
    #     name="count L", description="Number of layers", default=2, step=1, min=2, max=4)

    # count_N: IntProperty(
    #     name="count N", description="Max number of nodes in each layer", default=8, min=2, step=2, soft_max=32)

    @classmethod
    def poll(cls, context):
        # print(f"Current Context is: {context.area.type}")
        return context.area.type == 'VIEW_3D'

    def execute(self, context):
        collection_name = context.scene.collection_name
        # curr_scene = context.scene
        curr_scene = NewType('curr_scene', bpy.types.Scene)
        print(dir(curr_scene))
        # curr_scene.
        # curr_scene.
        # curr_scene = NewType('curr_scene', bpy.types.Scene)
        # curr_scene
        # if context.object:
        # print('asdf')
        # .mode == 'EDIT':
        # bpy.ops.object.mode_set(mode='OBJECT')
        depth_scale = 0.2
        # text_context = bpy.ops.object.text_add(radius=0.32, enter_editmode=False, align='WORLD', location=(
        #     0.5, 0, 0), rotation=(0, 0, -1.5708), scale=(1, 1, 1))
        # # context.
        create_base_mesh(self, context, current_collection,
                         depth_scale=depth_scale)
        # master_collection.objects.unlink(current_cube)
        # print(master_collection.objects)
        return {"FINISHED"}
