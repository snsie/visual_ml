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
create_weight_mesh = helpers.create_weight_mesh
create_node_mesh = helpers.create_node_mesh
ensure_collection = helpers.ensure_collection
load_json_model = helpers.load_json_model


class MESH_OT_add_nets(bpy.types.Operator):
    """Construct a Neural Network Mesh"""
    bl_idname = "mesh.add_nets"
    bl_label = "Neural Network"
    bl_options = {'REGISTER', 'UNDO'}

    count_L: IntProperty(
        name="count L", description="Number of layers", default=2, step=1, min=2, max=4)

    count_N: IntProperty(
        name="count N", description="Max number of nodes in each layer", default=8, min=2, step=2, soft_max=32)

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
        current_collection = ensure_collection(context, collection_name)
        json_data = load_json_model()

        # if context.object:
        # print('asdf')
        # .mode == 'EDIT':
        # bpy.ops.object.mode_set(mode='OBJECT')
        depth_scale = 0.2
        # text_context = bpy.ops.object.text_add(radius=0.32, enter_editmode=False, align='WORLD', location=(
        #     0.5, 0, 0), rotation=(0, 0, -1.5708), scale=(1, 1, 1))
        # # context.
        create_node_mesh(self, context, current_collection,
                         depth_scale=depth_scale)
        create_weight_mesh(self, context, current_collection,
                           depth_scale=depth_scale, )
        # master_collection.objects.unlink(current_cube)
        # print(master_collection.objects)
        return {"FINISHED"}


class IMPORT_SCENE_OT_gltf_reload(bpy.types.Operator):
    """import gltf files"""
    bl_idname = 'import_scene.gltf_reload'
    bl_label = "reload gltf files"

    def execute(self, context):
        ob = context.object
        gltf_fname = ob.gltf_fname
        matrix_world = ob.matrix_world.copy()
        for collection in list(ob.users_collection):
            collection.objects.unlink(ob)
            # Find gltf files

        if ob.users == 0:
            bpy.data.objects.remove(ob)
        del ob

        import_dir = gltf_import_dir(context.scene)
        import_fpath = import_dir / gltf_fname
        bpy.ops.import_scene.gltf(filepath=str(import_fpath))

        for imported_ob in context.selected_objects:
            imported_ob.gltf_fname = import_fpath.name
            # imported_ob.gltf_fname = matrix_world

        return {"FINISHED"}


def gltf_import_dir(scene) -> pathlib.Path:
    abspath = pathlib.Path(bpy.path.abspath(scene.gltf_dir))
    # return pathlib.Path(abspath)
    return abspath


class IMPORT_SCENE_OT_gltf_dir(bpy.types.Operator):
    """import gltf files"""
    bl_idname = 'import_scene.gltf_dir'
    bl_label = "import gltf files"

    def execute(self, context):
        # Find gltf files

        import_dir = gltf_import_dir(context.scene)

        # Import each file and save file name

        for import_fpath in import_dir.glob("*.glb"):
            bpy.ops.import_scene.gltf(filepath=str(import_fpath))
            for imported_ob in context.selected_objects:
                imported_ob.gltf_fname = import_fpath.name

        # self.report(
        #     {'ERROR'}, {f'gltf file not found in {context.scene.gltf_dir}'})
        return {'FINISHED'}
