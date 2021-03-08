import bpy
import bmesh
import pathlib
import importlib
from scipy import constants
import math
import numpy as np
import mathutils
from . import helpers

# set_mesh_material = helpers.set_mesh_material
# get_data = helpers.get_data
# get_color = helpers.get_color
# create_weight_mesh = helpers.create_weight_mesh
# create_node_mesh = helpers.create_node_mesh
ensure_collection = helpers.ensure_collection


class OBJECT_OT_remove_nets(bpy.types.Operator):
    bl_idname = "object.remove_nets"
    bl_label = "Remove Nets"

    def execute(self, context):
        collection_name = context.scene.collection_name

        for c in scene.collection.children:
            scene.collection.children.unlink(c)
        return {"FINISHED"}
