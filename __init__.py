# <pep8 compliant>

bl_info = {
    "name": "Visual ML",
    "author": "Scott Siegel",
    "version": (0, 0, 1),
    "blender": (2, 93, 0),
    "location": "View3D >  Sidebar > Create Tab",
    "description": "Generate meshes for visual machine learning",
    "doc_url": "",
    "category": "Add Mesh",
}

import sys
import os


# ----------------------------------------------
# Import modules
# ----------------------------------------------

if "bpy" in locals():
    import importlib
    importlib.reload(panel)
    importlib.reload(add_nets)
    importlib.reload(load_net)
else:
    from . import panel
    from . import add_nets
    from . import load_net


import bpy
from bpy.types import (Scene, Object)
from bpy.props import (StringProperty, IntProperty)
# context.area: VIEW_3D


MESH_OT_add_nets = add_nets.MESH_OT_add_nets
IMPORT_SCENE_OT_gltf_dir = add_nets.IMPORT_SCENE_OT_gltf_dir
IMPORT_SCENE_OT_gltf_reload = add_nets.IMPORT_SCENE_OT_gltf_reload

MESH_OT_load_net = load_net.MESH_OT_load_net

VIEW3D_PT_visual_ml = panel.VIEW3D_PT_visual_ml
# MeshNetProperties = panel.MeshNetProperties

classes = (
    MESH_OT_add_nets,
    VIEW3D_PT_visual_ml,
    IMPORT_SCENE_OT_gltf_dir,
    IMPORT_SCENE_OT_gltf_reload,
    MESH_OT_load_net,
)


def mesh_add_menu_draw(self, context):
    self.layout.operator('mesh.add_nets', icon="NETWORK_DRIVE")


def register():
    # des = "Visual ML Collection Name"

    Scene.count_l = IntProperty(
        name="count L", description="Number of layers",
        default=2, step=1, min=2, max=5)

    Scene.count_n = IntProperty(
        name="count N", description="Max number of nodes in each layer",
        default=8, min=2, step=2, max=64,)
    # Scene.MeshNetProperties = MeshNetProperties
    Scene.collection_name = StringProperty(
        name="collection name",
        default="visual_ml")
    Scene.gltf_dir = StringProperty(
        name="gltf dir",
        subtype="DIR_PATH",
        default="/home/scott/dev/blender/scripts/addons/visual_ml/gltf/",)
    Object.gltf_fname = StringProperty(
        name='gltf file',
        # subtype="FILE_NAME",
        default='BoxAnimated.glb',
    )
    bpy.types.VIEW3D_MT_mesh_add.append(mesh_add_menu_draw)
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    del Scene.gltf_dir
    del Object.gltf_fname
    bpy.types.VIEW3D_MT_mesh_add.remove(mesh_add_menu_draw)
    for cls in classes:
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
