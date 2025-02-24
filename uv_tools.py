# Nikita Akimov
# interplanety@interplanety.org
#
# GitHub
#    https://github.com/Korchy/1d_uv_tools
import math

import bmesh
import bpy
from bmesh.types import BMVert
from bpy.props import BoolProperty, FloatVectorProperty
from bpy.types import Operator, Panel, Scene, WindowManager
from bpy.utils import register_class, unregister_class
from math import ceil, floor, modf, sqrt
from mathutils import Vector, Matrix

bl_info = {
    "name": "1D UV Tools",
    "description": "Tools for working with UV Maps",
    "author": "Nikita Akimov, Paul Kotelevets",
    "version": (1, 1, 0),
    "blender": (2, 79, 0),
    "location": "View3D > Tool panel > 1D > UV Tools",
    "doc_url": "https://github.com/Korchy/1d_uv_tools",
    "tracker_url": "https://github.com/Korchy/1d_uv_tools",
    "category": "All"
}


# MAIN CLASS

class UVTools:

    _round_base = 3

    @classmethod
    def uv_pack_tile(cls, context, pack_to_cursor=True):
        # pack current existed UV Map to tile from 0 to 1
        # current mode
        mode = context.active_object.mode
        if context.active_object.mode == 'EDIT':
            bpy.ops.object.mode_set(mode='OBJECT')
        # switch to vertex selection mode
        bm = bmesh.new()
        bm.from_mesh(context.active_object.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        # process UV points
        uv_layer = cls._active_uv_layer(bm)
        # bm.loops.layers.uv.verify()
        # bm.faces.layers.tex.verify()
        for face in (_face for _face in bm.faces if \
                     _face.select \
                     and (context.scene.tool_settings.use_uv_select_sync  # if sync selection is True - by selected Faces
                         or all((_loop[uv_layer].select for _loop in _face.loops)))):  # else - if all face's uv-loops selected
            # # pack by uv points coordinates
            # for loop in face.loops:
            #     # in this case we can consider loop as face corner (vertex)
            #     if context.scene.tool_settings.use_uv_select_sync or loop[uv_layer].select:
            #         # if sync selection is enabled - by selected faces on object, else - by selection on UV
            #         # print(loop[uv_layer].uv.x)
            #         loop[uv_layer].uv.x = round(loop[uv_layer].uv.x % 1, cls._round_base)
            #         loop[uv_layer].uv.y = round(loop[uv_layer].uv.y % 1, cls._round_base)
            #         # print(loop[uv_layer].uv.x)

            # Paul: pack by uv faces centroid coordinates
            if pack_to_cursor:
                cursor_co = context.area.spaces.active.cursor_location
                for loop in face.loops:
                    loop[uv_layer].uv -= cursor_co
            # get vector to the current uv face centroid
            uv_face_centroid = cls._centroid([loop[uv_layer].uv for loop in face.loops])
            # get vector to the point in which uv face centroid should be placed in 0...1 tile
            # uv_face_centroid_in_0_1 = Vector((round(uv_face_centroid.x % 1, cls._round_base),
            #                                   round(uv_face_centroid.y % 1, cls._round_base)))
            uv_face_centroid_in_0_1 = Vector((uv_face_centroid.x % 1, uv_face_centroid.y % 1))
            # vector for moving from current centroid position to its desired position in 0...1 tile
            diff = uv_face_centroid_in_0_1 - uv_face_centroid
            # use pack_to_cursor
            if pack_to_cursor:
                cursor_co = context.area.spaces.active.cursor_location
                diff += cursor_co
            # move all points by this vector
            for loop in face.loops:
                loop[uv_layer].uv += diff
        # save changed data to mesh
        bm.to_mesh(context.active_object.data)
        bm.free()
        # return mode back
        bpy.ops.object.mode_set(mode=mode)

    @classmethod
    def get_fit_to_tile_points(cls, context, op):
        # save 3 points on the uv for further using them in fit_to_tile function
        #   set context.scene.tool_settings.use_uv_select_sync == False
        #   select 3 points on the uv
        #   then call this function
        mode = context.active_object.mode
        if context.active_object.mode == 'EDIT':
            bpy.ops.object.mode_set(mode='OBJECT')
        # use bmesh object
        bm = bmesh.new()
        bm.from_mesh(context.active_object.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        # process UV points
        uv_layer = cls._active_uv_layer(bm)
        # get 3 selected uv points
        uv_vertices_selected = set()
        # for face in (_face for _face in bm.faces):
        # for face in (_face for _face in bm.faces if \
        #              any((_v.select and _v.hide == False for _v in _face.verts)) # face has at least one selected vertex
        #              and (context.scene.tool_settings.use_uv_select_sync  # if sync selection is True - by selected Faces
        #                  or any((_loop[uv_layer].select for _loop in _face.loops)))):  # else - if all face's uv-loops selected
        for face in (_face for _face in bm.faces if \
                     _face.select # face is selected
                     and (context.scene.tool_settings.use_uv_select_sync  # if sync selection is True - by selected Faces
                         or any((_loop[uv_layer].select for _loop in _face.loops)))):  # else - if all face's uv-loops selected
            # print(face)
            # get coordinates of 3 points
            for loop in face.loops:
                if loop.vert.select and loop[uv_layer].select:
                    # using set because coordinates will be the same for loops from different neighbour faces
                    uv_vertices_selected.add(loop[uv_layer].uv[:])
        uv_points_co = list(Vector(_co) for _co in uv_vertices_selected)  # to list to keep order
        # print('uv_p_co', uv_points_co)
        """
        > Paul:
            Мы делаем просто
            1) определяем какая из этих точек ближе к курсору. Это будет опора вращения
            2) определяем какая из оставшихся точек имеет меньшую дельта У с опорной. Это будет горизонталь
        """
        # anchor point co
        v0_co = min([(_vector, (_vector - context.area.spaces.active.cursor_location).length) \
                     for _vector in uv_points_co], key=lambda _item: _item[1])[0]
        # print(v0_co)
        # v1 - second point co (for horizontal)
        uv_points_co.remove(v0_co)
        # Me - 2 point is closest to the first point
        # v1_co = uv_points_co[0] if (v0_co - uv_points_co[0]).length < (v0_co - uv_points_co[1]).length else uv_points_co[1]
        # Paul
        v1_co = uv_points_co[0] if abs(v0_co.y - uv_points_co[0].y) < abs(v0_co.y - uv_points_co[1].y) else uv_points_co[1]
        # v2 - third point (for scaling)
        uv_points_co.remove(v1_co)
        v2_co = uv_points_co[0]
        if v0_co and v1_co and v2_co:
            # save coordinates to global variables
            context.window_manager.uv_tools_1d_prop_fit_to_tile_v0 = v0_co
            context.window_manager.uv_tools_1d_prop_fit_to_tile_v1 = v1_co
            context.window_manager.uv_tools_1d_prop_fit_to_tile_v2 = v2_co
            op.report(
                type={'INFO'},
                message='Saved coordinates: ' + str(v0_co) + ', ' + str(v1_co) + ', ' + str(v2_co)
            )
        # save changed data to mesh
        # bm.to_mesh(context.active_object.data)
        bm.free()
        # return mode back
        bpy.ops.object.mode_set(mode=mode)

    @classmethod
    def fit_to_tile(cls, context, add_scale=False):
        # works with selection
        # first - save 3 uv points
        #   with context.scene.tool_settings.use_uv_select_sync == False
        #   select 3 points on the uv
        #   call the get_fit_to_tile_points function
        # next - call this function
        # get 3 previously saved points (in get_fit_to_tile_points), then rotate selected faces of the uv around 1 point
        #   to have 2 point horizontally, then translate selected faces of the uv to the 0.0 (by anchor point),
        #   then scale selected faces of the uv by the ratio (length between 1 and 3 points)
        if context.window_manager.uv_tools_1d_prop_fit_to_tile_v0[:] != (0.0, 0.0) and \
                context.window_manager.uv_tools_1d_prop_fit_to_tile_v0[:] != (0.0, 0.0) and \
                context.window_manager.uv_tools_1d_prop_fit_to_tile_v0[:] != (0.0, 0.0):
            mode = context.active_object.mode
            if context.active_object.mode == 'EDIT':
                bpy.ops.object.mode_set(mode='OBJECT')
            # use bmesh object
            bm = bmesh.new()
            bm.from_mesh(context.active_object.data)
            bm.verts.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            # process UV points
            uv_layer = cls._active_uv_layer(bm)
            v0_co = Vector(context.window_manager.uv_tools_1d_prop_fit_to_tile_v0)  # anchor point
            v1_co = Vector(context.window_manager.uv_tools_1d_prop_fit_to_tile_v1)  # next point, used for horizontal
            v2_co = Vector(context.window_manager.uv_tools_1d_prop_fit_to_tile_v2)  # scale point
            # print(v0_co, v1_co, v2_co)
            if v0_co and v1_co and v2_co:
                # get rotation matrix
                matrix_rot = cls._rotation_matrix(v1_co - v0_co, Vector((1.0, 0.0)))
                # process all the uv points
                # translation
                #   first not to use translation there and back in rotations
                for face in (_face for _face in bm.faces if
                             _face.hide == False and all((_loop[uv_layer].select for _loop in _face.loops))):
                    for loop in face.loops:
                        loop[uv_layer].uv += Vector((0.0, 0.0)) - v0_co
                # rotation
                for face in (_face for _face in bm.faces if
                             _face.hide == False and all((_loop[uv_layer].select for _loop in _face.loops))):
                    for loop in face.loops:
                        loop[uv_layer].uv = matrix_rot * loop[uv_layer].uv
                # scale
                if add_scale:
                    # get scale matrix
                    # count scale ratio
                    # scale_ratio = sqrt(2) / (v2_co - v0_co).length    # fit v0-v2 to diagonal of length sqrt(2)
                    # Paul: 1 / length of longest(v0-v1, v1-v2)
                    scale_ratio = 1 / max((v1_co - v0_co).length, (v2_co - v1_co).length)
                    # print('scale', scale_ratio)
                    matrix_scale_x = Matrix.Scale(scale_ratio, 2, (1.0, 0.0))
                    matrix_scale_y = Matrix.Scale(scale_ratio, 2, (0.0, 1.0))
                    matrix_scale = matrix_scale_x * matrix_scale_y
                    for face in (_face for _face in bm.faces if
                                 _face.hide == False and all((_loop[uv_layer].select for _loop in _face.loops))):
                        for loop in face.loops:
                            loop[uv_layer].uv = matrix_scale * loop[uv_layer].uv
            # save changed data to mesh
            bm.to_mesh(context.active_object.data)
            bm.free()
            # return mode back
            bpy.ops.object.mode_set(mode=mode)


    # EXPERIMENTAL

    @classmethod
    def select_uv_cut(cls, context):
        # select faces only needed to be cutted by UV tile (0...1)
        mode = context.active_object.mode
        if context.active_object.mode == 'EDIT':
            bpy.ops.object.mode_set(mode='OBJECT')
        # switch to vertex selection mode
        bm = bmesh.new()
        bm.from_mesh(context.active_object.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        # process UV points
        uv_layer = cls._active_uv_layer(bm)
        # check faces if they should be cutted
        for face in (_face for _face in bm.faces):
            for loop in face.loops:
                print(loop)
                print(loop[uv_layer].uv, loop.link_loop_next[uv_layer].uv)
                loop_vector = loop.link_loop_next[uv_layer].uv - loop[uv_layer].uv
                print(loop_vector)
                # horizontal cuts
                low = min(loop[uv_layer].uv.y, loop.link_loop_next[uv_layer].uv.y)
                low = floor(low) if low < 0 else ceil(low)
                hig = max(loop[uv_layer].uv.y, loop.link_loop_next[uv_layer].uv.y)
                hig = floor(hig) if hig < 0 else ceil(hig)
                print(low, hig, '->', list(range(low, hig + 1)), list(range(low, hig + 1))[1:-1])
                h_cuts = len(list(range(low, hig + 1))[1:-1])
                print(h_cuts)
                # vertical cuts
                left = min(loop[uv_layer].uv.x, loop.link_loop_next[uv_layer].uv.x)
                left = floor(left) if left < 0 else ceil(left)
                right = max(loop[uv_layer].uv.x, loop.link_loop_next[uv_layer].uv.x)
                right = floor(right) if right < 0 else ceil(right)
                print(left, right, '->', list(range(left, right + 1)), list(range(left, right + 1))[1:-1])
                v_cuts = len(list(range(left, right + 1))[1:-1])
                print(v_cuts)
                if v_cuts or h_cuts:
                    face.select = True
        # save changed data to mesh
        bm.to_mesh(context.active_object.data)
        bm.free()
        # return mode back
        bpy.ops.object.mode_set(mode=mode)

    @classmethod
    def uv_cut_tile(cls, context):
        # cut (create additional edges) all faces of current object by its UV tile (0...1)
        # ToDo: EXPERIMENTAL - not work properly (!!!)
        # current mode
        mode = context.active_object.mode
        if context.active_object.mode == 'EDIT':
            bpy.ops.object.mode_set(mode='OBJECT')
        # switch to vertex selection mode
        bm = bmesh.new()
        bm.from_mesh(context.active_object.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        # process UV points
        uv_layer = cls._active_uv_layer(bm)
        # bm.loops.layers.uv.verify()
        # bm.faces.layers.tex.verify()
        for face in (_face for _face in bm.faces if \
                     _face.select \
                     and (context.scene.tool_settings.use_uv_select_sync    # if sync selection is True - by selected Faces
                          or all((_loop[uv_layer].select for _loop in _face.loops)))):  # else - if all face's uv-loops selected
            # print(face.loops)
            # print(dir(face.loops))
            for loop in face.loops:
                # in this case we can consider loop as face corner (vertex)
                # print(loop.edge)
                print(loop)
                # print(dir(loop))
                # print(loop.link_loop_next)
                # print(loop[uv_layer].uv)
                # print(loop.link_loop_next[uv_layer].uv)
                print(loop[uv_layer].uv, loop.link_loop_next[uv_layer].uv)

                loop_vector = loop.link_loop_next[uv_layer].uv - loop[uv_layer].uv
                # print(loop_vector)

                # x_cut_num = abs(modf(round(loop_vector.x, cls._round_base))[1])     # integer part of a number
                # if loop.link_loop_next[uv_layer].uv <= 0 <= loop[uv_layer].uv.x or \
                #         loop.link_loop_next[uv_layer].uv >= 0 >= loop[uv_layer].uv.x:
                #     x_cut_num += 1
                # if x_cut_num > 0:
                #     # cut this edge
                #     new_data = bmesh.ops.bisect_edges(bm, edges=[loop.edge], cuts=x_cut_num)
                #     new_verts = [_item for _item in new_data['geom_split'] if isinstance(_item, BMVert)]
                #     print(new_verts)

                low = min(loop[uv_layer].uv.y, loop.link_loop_next[uv_layer].uv.y)
                low = floor(low) if low < 0 else ceil(low)
                hig = max(loop[uv_layer].uv.y, loop.link_loop_next[uv_layer].uv.y)
                hig = floor(hig) if hig < 0 else ceil(hig)
                print(low, hig, '->', list(range(low, hig + 1)), list(range(low, hig + 1))[1:-1])

                cuts = len(list(range(low, hig + 1))[1:-1])
                if cuts > 0:
                    new_data = bmesh.ops.bisect_edges(bm, edges=[loop.edge], cuts=1)
                    print(new_data)
                    new_verts = [_item for _item in new_data['geom_split'] if isinstance(_item, BMVert)]
                    print(new_verts)

                    for cut in range(low, hig + 1)[1:-1]:
                        # set new vertex to correspondence cut place
                        print(new_verts[cut], new_verts[cut].co)
                        # print(loop.link_loop_next[uv_layer].uv)
                        print('loop_vector.y', loop_vector.y)
                        print('loop[uv_layer].uv.y + cut', abs(loop[uv_layer].uv.y) + cut)
                        y_ratio = loop_vector.y / (abs(loop[uv_layer].uv.y) + cut)
                        print('ratio', y_ratio)

                        df = (loop.link_loop_next.vert.co - loop.vert.co) / y_ratio

                        # new_verts[cut].co.y = loop.vert.co.y + loop.link_loop_next.vert.co.y / y_ratio
                        new_verts[cut].co.y = loop.vert.co.y + df.y

                # y_cut_num = int(abs(modf(round(loop_vector.y, cls._round_base))[1]))    # integer part of a number
                # if loop.link_loop_next[uv_layer].uv.y <= 0 <= loop[uv_layer].uv.y or \
                #         loop.link_loop_next[uv_layer].uv.y >= 0 >= loop[uv_layer].uv.y:
                #     # edge intersects coordinate axis X
                #     y_cut_num += 1
                # print('y_cun_num', y_cut_num)
                # if y_cut_num > 0:
                #     # cut this edge
                #     new_data = bmesh.ops.bisect_edges(bm, edges=[loop.edge], cuts=y_cut_num)
                #     print(new_data)
                #     new_verts = [_item for _item in new_data['geom_split'] if isinstance(_item, BMVert)]
                #     print(new_verts)
                # print(x_cut_num, y_cut_num)

        # save changed data to mesh
        bm.to_mesh(context.active_object.data)
        bm.free()
        # return mode back
        bpy.ops.object.mode_set(mode=mode)

    @staticmethod
    def _active_uv_layer(bm):
        # get active uv-layer (UV Map) from bmesh object
        return bm.loops.layers.uv.active

    @staticmethod
    def _centroid(vertexes):
        x_list = [vertex[0] for vertex in vertexes]
        y_list = [vertex[1] for vertex in vertexes]
        length = len(vertexes)
        x = sum(x_list) / length
        y = sum(y_list) / length
        return Vector((x, y))

    @staticmethod
    def _rotation_matrix(src_vector, dest_vector):
        angle = src_vector.angle(dest_vector)  # rad
        axis = src_vector.cross(dest_vector)
        if axis < 0.0:
            angle = -angle
        return Matrix.Rotation(angle, 2, 'Z')

    @staticmethod
    def ui(layout, context):
        # ui panel
        # UV Pack
        box = layout.box()
        op = box.operator(
            operator='uvtools.uvpack',
            icon='UV_FACESEL'
        )
        box.prop(
            data=context.scene,
            property='uv_tools_1d_prop_pack_to_cursor'
        )
        op.pack_to_cursor = context.scene.uv_tools_1d_prop_pack_to_cursor
        # Fit to Tile
        box = layout.box()
        box.operator(
            operator='uvtools.store_diagonal',
            icon='OUTLINER_DATA_LATTICE'
        )
        op = box.operator(
            operator='uvtools.fit_to_tile',
            icon='FULLSCREEN_ENTER'
        )
        op.add_scale = context.window_manager.uv_tools_1d_prop_fit_to_tile_add_scale
        box.prop(
            data=context.window_manager,
            property='uv_tools_1d_prop_fit_to_tile_add_scale'
        )

        # ToDo: EXPERIMENTAL
        layout.separator()
        box = layout.box()
        box.label(text='Experimental')
        # UV Select Cut Tile
        box.operator(
            operator='uvtools.select_uvcut_tile',
            icon='MOD_UVPROJECT'
        )
        # UV Cut Tile
        box.operator(
            operator='uvtools.uvcut_tile',
            icon='UV_EDGESEL'
        )


# OPERATORS

class UVTools_OT_uvpack(Operator):
    bl_idname = 'uvtools.uvpack'
    bl_label = 'UV Pack'
    bl_options = {'REGISTER', 'UNDO'}

    pack_to_cursor = BoolProperty(
        name='Pack to Cursor',
        default=True
    )

    def execute(self, context):
        UVTools.uv_pack_tile(
            context=context,
            pack_to_cursor=self.pack_to_cursor
        )
        return {'FINISHED'}

class UVTools_OT_store_diagonal(Operator):
    bl_idname = 'uvtools.store_diagonal'
    bl_label = 'Store Key'
    bl_description = '3 vertices selection, with 2d cursor position closest to key base is required'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        UVTools.get_fit_to_tile_points(
            context=context,
            op=self
        )
        return {'FINISHED'}

class UVTools_OT_fit_to_tile(Operator):
    bl_idname = 'uvtools.fit_to_tile'
    bl_label = 'Fit to Tile'
    bl_options = {'REGISTER', 'UNDO'}

    add_scale = BoolProperty(
        name='Apply Scale',
        default=False
    )

    def execute(self, context):
        UVTools.fit_to_tile(
            context=context,
            add_scale=self.add_scale
        )
        return {'FINISHED'}

# Experimental

class UVTools_OT_uvcut_tile(Operator):
    bl_idname = 'uvtools.uvcut_tile'
    bl_label = 'UV Cut Tile'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        UVTools.uv_cut_tile(
            context=context
        )
        return {'FINISHED'}

class UVTools_OT_select_uvcut_tile(Operator):
    bl_idname = 'uvtools.select_uvcut_tile'
    bl_label = 'Select UV Cut Tile'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        UVTools.select_uv_cut(
            context=context
        )
        return {'FINISHED'}


# PANELS

class UVTools_PT_panel(Panel):
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'TOOLS'
    bl_label = 'UV Tools'
    bl_category = '1D'

    def draw(self, context):
        UVTools.ui(
            layout=self.layout,
            context=context
        )


# REGISTER

def register(ui=True):
    Scene.uv_tools_1d_prop_pack_to_cursor = BoolProperty(
        name='Pack to Cursor',
        default=False
    )
    WindowManager.uv_tools_1d_prop_fit_to_tile_v0 = FloatVectorProperty(
        name='Fit to Tile V0',
        default=(0.0, 0.0),
        size=2
    )
    WindowManager.uv_tools_1d_prop_fit_to_tile_v1 = FloatVectorProperty(
        name='Fit to Tile V1',
        default=(0.0, 0.0),
        size=2
    )
    WindowManager.uv_tools_1d_prop_fit_to_tile_v2 = FloatVectorProperty(
        name='Fit to Tile V2',
        default=(0.0, 0.0),
        size=2
    )
    WindowManager.uv_tools_1d_prop_fit_to_tile_add_scale = BoolProperty(
        name='Apply Scale',
        default=False
    )
    register_class(UVTools_OT_uvpack)
    register_class(UVTools_OT_fit_to_tile)
    register_class(UVTools_OT_store_diagonal)
    register_class(UVTools_OT_select_uvcut_tile)
    register_class(UVTools_OT_uvcut_tile)
    if ui:
        register_class(UVTools_PT_panel)


def unregister(ui=True):
    if ui:
        unregister_class(UVTools_PT_panel)
    # butch clean
    unregister_class(UVTools_OT_uvcut_tile)
    unregister_class(UVTools_OT_select_uvcut_tile)
    unregister_class(UVTools_OT_store_diagonal)
    unregister_class(UVTools_OT_fit_to_tile)
    unregister_class(UVTools_OT_uvpack)
    del WindowManager.uv_tools_1d_prop_fit_to_tile_add_scale
    del WindowManager.uv_tools_1d_prop_fit_to_tile_v2
    del WindowManager.uv_tools_1d_prop_fit_to_tile_v1
    del WindowManager.uv_tools_1d_prop_fit_to_tile_v0
    del Scene.uv_tools_1d_prop_pack_to_cursor


if __name__ == "__main__":
    register()
