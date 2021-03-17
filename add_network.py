# pylint:disable=wrong-import-position,used-before-assignment,assignment-from-no-return,relative-beyond-top-level
from typing import NewType

if "bpy" in locals():
    import importlib
    importlib.reload(nf)
else:
    from . import network_functions as nf

import bpy
from bpy.props import IntProperty
import bmesh
import pathlib

from scipy import constants
import math
import numpy as np
import mathutils


class MESH_OT_add_network(bpy.types.Operator):
    """Construct a Neural Network Mesh"""
    bl_idname = "mesh.visual_ml_network"
    bl_label = "Network"
    bl_options = {'REGISTER', 'UNDO'}

    count_L: IntProperty(
        name="count L", description="Number of layers",
        default=2, step=1, min=2, max=4)

    count_N: IntProperty(
        name="count N", description="Max number of nodes in each layer",
        default=8, min=2, step=2, soft_max=32)
    max_N: IntProperty(
        name="max N", description="Max number of nodes in each layer",
        default=64)

    @classmethod
    def poll(cls, context):
        # print(f"Current Context is: {context.area.type}")
        return (context.area.type == 'VIEW_3D') or (
            context.area.type == 'NODE_EDITOR')

    def execute(self, context):
        collection_name = context.scene.collection_name
        current_collection = nf.ensure_collection(context, collection_name)
        prim_coll_name = context.scene.prim_coll_name
        prim_coll = nf.ensure_collection(context, prim_coll_name)
        json_fname_begin = 'numpyData_e_0000.json'
        json_fname_end = 'numpyData_e_0100.json'

        json_data_begin = nf.load_json_model(json_fname_begin)
        node_begin_vals = []
        for i in json_data_begin['layer_1']['weight']:
            node_begin_vals.append(i[0])
        json_data_end = nf.load_json_model(json_fname_end)
        node_end_vals = []
        for i in json_data_end['layer_1']['weight']:
            node_end_vals.append(i[0])

        # if context.object:
        # print('asdf')
        # .mode == 'EDIT':
        # bpy.ops.object.mode_set(mode='OBJECT')
        depth_scale = 0.2
        # create_node_mesh(self, context, current_collection,
        #  depth_scale=depth_scale)
        json_data = []
        json_data_begin
        nf.create_weight_prims(
            self, context, prim_coll, json_data)
        nf.set_weight_anim(self, context, prim_coll,
                           node_begin_vals, node_end_vals)
        # nf.create_weight_mesh(self, context, current_collection,)
        # nf.add_geonode_linear(self, context)
        # master_collection.objects.unlink(current_cube)
        # print(master_collection.objects)
        return {"FINISHED"}
