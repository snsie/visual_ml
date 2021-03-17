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
    importlib.reload(add_network)
    importlib.reload(add_geonodes)
else:
    from . import panel
    from . import add_nets
    from . import load_net
    from . import add_network
    from . import add_geonodes


import bpy
from bpy.types import (Scene, Object)
from bpy.props import (StringProperty, IntProperty)
import nodeitems_utils
from nodeitems_utils import NodeCategory, NodeItem, NodeItemCustom
from nodeitems_utils import (
    NodeCategory,
    NodeItem,
    NodeItemCustom,
)
# context.area: VIEW_3D

MESH_OT_add_nets = add_nets.MESH_OT_add_nets
IMPORT_SCENE_OT_gltf_dir = add_nets.IMPORT_SCENE_OT_gltf_dir
IMPORT_SCENE_OT_gltf_reload = add_nets.IMPORT_SCENE_OT_gltf_reload

MESH_OT_load_net = load_net.MESH_OT_load_net
MESH_OT_add_network = add_network.MESH_OT_add_network

VIEW3D_PT_visual_ml = panel.VIEW3D_PT_visual_ml
# MeshNetProperties = panel.MeshNetProperties

NODE_ST_MyCustomSocket = add_geonodes.NODE_ST_MyCustomSocket
NODE_NT_GeometryNodeCustom = add_geonodes.NODE_NT_GeometryNodeCustom
node_categories = add_geonodes.node_categories

classes = (
    MESH_OT_add_nets,
    VIEW3D_PT_visual_ml,
    # IMPORT_SCENE_OT_gltf_dir,
    # IMPORT_SCENE_OT_gltf_reload,
    MESH_OT_load_net,
    MESH_OT_add_network,
    NODE_ST_MyCustomSocket,
    NODE_NT_GeometryNodeCustom
)


# our own base class with an appropriate poll function,
# so the categories only show up in our own tree type


# class MyNodeCategory(NodeCategory):
#     @classmethod
#     def poll(cls, context):
#         return context.space_data.tree_type == 'GeometryNodeTree'


# # all categories in a list
# node_categories = [
#     # identifier, label, items list
#     MyNodeCategory('SOMENODES', "Some Nodes", items=[
#         # our basic node
#         NodeItem("CustomNodeType"),
#     ]),
#     MyNodeCategory('OTHERNODES', "Other Nodes", items=[
#         # the node item can have additional settings,
#         # which are applied to new nodes
#         # NB: settings values are stored as string expressions,
#         # for this reason they should be converted to strings using repr()
#         NodeItem("CustomNodeType", label="Node A", settings={
#             "my_string_prop": repr("Lorem ipsum dolor sit amet"),
#             "my_float_prop": repr(1.0),
#         }),
#         NodeItem("CustomNodeType", label="Node B", settings={
#             "my_string_prop": repr("consectetur adipisicing elit"),
#             "my_float_prop": repr(2.0),
#         }),
#     ]),
# ]


def mesh_add_menu_draw(self, context):
    self.layout.operator('mesh.add_nets', icon="NETWORK_DRIVE")


def register():
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
    Scene.prim_coll_name = StringProperty(
        name="prim collection name",
        default="prim_vis_ml")
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
    nodeitems_utils.register_node_categories('CUSTOM_NODES', node_categories)


def unregister():
    del Scene.gltf_dir
    del Object.gltf_fname
    bpy.types.VIEW3D_MT_mesh_add.remove(mesh_add_menu_draw)
    for cls in classes:
        bpy.utils.unregister_class(cls)
    nodeitems_utils.unregister_node_categories('CUSTOM_NODES')


if __name__ == "__main__":
    register()
