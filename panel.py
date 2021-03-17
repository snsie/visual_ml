# pylint:disable=assignment-from-no-return
import bpy
from bpy.props import CollectionProperty


# bpy.utils.register_class(MeshNetProperties)
# Object.WindowPanelGenerator = CollectionProperty(type=MeshNetProperties)


class VIEW3D_PT_visual_ml(bpy.types.Panel):
    bl_category = 'Create'
    bl_label = 'Visual ML'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_context = "objectmode"
    bl_options = {'DEFAULT_CLOSED'}
    # bl_options = {'REGISTER'}
    # count_N: bpy.props.IntProperty(
    #     name="Count N",
    #     default=8, min=2, step=2, soft_max=32,)
    # count_L: bpy.props.IntProperty(
    #     name="Count L",
    #     default=2, min=2, step=1, soft_max=4,)

    def draw(self, context):
        layout = self.layout

        col = layout.column()

        props = col.operator(
            'mesh.add_nets', text="Default Net", icon='NETWORK_DRIVE')
        # props.count_L = 2
        # props.count_N = 8
        # col.prop(slider=True)
        box = col.box()
        box_col = box.column()
        box_col.operator('mesh.visual_ml_network',
                         text="Add Network", icon="NETWORK_DRIVE")
        # custom_props = box_col.operator(
        #     'mesh.add_nets', text="Custom Net", icon='NETWORK_DRIVE')
        # col = row.col()
        # print(dir(context.scene.cu))

        # box_col = box.column()
        # box_col.prop(context.scene, "count_l")
        # box_col.prop(context.scene, "count_n", )
        # custom_props.count_L = context.scene.count_l
        # custom_props.count_N = context.scene.count_n
        col.operator(
            'mesh.load_net', text="Retrieve Net", icon='NETWORK_DRIVE')
        # box_col.prop(self, 'count_N')
        # custom_props.count_L = box_col.prop(self, 'count_L')
        # custom_props.count_N = box_col.prop(self, 'count_N')

        # props.count_L = 3
        # props.count_N = 16
        # col = layout.column(align=True)
        # return super().draw(context)
        # col.prop(context.scene, 'gltf_dir')
        # col.operator('import_scene.gltf_dir',)
        #  text='Default Grid', icon='MONKEY')
        # ###############################################################
        # if context.object:
        #     col.operator("import_scene.gltf_reload")
        #     col.prop(context.object, 'gltf_fname')
        # else:
        #     col.label(text='-no active object-')

        # ###############################################################
        # if context.active_object is None:
        # col.label(text='-no active object-')
        # else:
        # col.prop(context.active_object, 'hide_viewport')
        split = layout.split()


# class MeshNetProperties(PropertyGroup):
#     def get_countL(self):
#         return self.get("count_L", self.default_L)

#     def set_countL(self, value):
#         self.count_L = value

#     def get_countN(self):
#         return self.get("count_N", self.default_N)

#     def set_countN(self, value):
#         self.count_N = value

#     default_L: bpy.props.IntProperty(
#         name="Default L",
#         step=1,
#         default=2
#     )
#     count_L: bpy.props.IntProperty(
#         name="Count L",
#         step=1,
#         get=get_countL,
#         set=set_countL,
#     )

#     default_N: bpy.props.IntProperty(
#         name="Default N",
#         step=1,
#         default=2
#     )
#     count_N: bpy.props.IntProperty(
#         name="Count N",
#         default=8, min=2, step=2, soft_max=32,
#         get=get_countN,
#         set=set_countN,
#     )
