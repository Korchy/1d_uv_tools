# Nikita Akimov
# interplanety@interplanety.org
#
# GitHub
#    https://github.com/Korchy/1d_uv_tools
from platform import mac_ver

import bmesh
import bpy
from bmesh.types import BMVert
from bpy.props import BoolProperty, FloatProperty, FloatVectorProperty, StringProperty
from bpy.types import Operator, Panel, Scene, WindowManager
from bpy.utils import register_class, unregister_class
from math import ceil, cos, floor, sin, pi
from mathutils import Vector, Matrix

bl_info = {
    "name": "1D UV Tools",
    "description": "Tools for working with UV Maps",
    "author": "Nikita Akimov, Paul Kotelevets",
    "version": (1, 2, 5),
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

    @classmethod
    def texel_scale(cls, context, scale_by_x=True, scale_by_y=True):
        # Texel Scale
        #   return scale of active polygon UV to value before executing Multy Sure Uv
        #   set scale for other selected polygons UVs to the same value
        #   shift uv-points by difference of coordinates of any one uv-point of active face before and after Multy Sure Uv
        # switch to Object mode
        mode = context.active_object.mode
        if context.active_object.mode == 'EDIT':
            bpy.ops.object.mode_set(mode='OBJECT')
        # active UV layer
        uv_layer = cls._active_uv_layer(obj=context.active_object)
        # 1. get source scale lengths by X and Y of the UV by active face
        active_face = context.object.data.polygons[context.object.data.polygons.active]
        # get UV points for active polygon
        active_face_uv_points = cls._uv_points(faces_list=[active_face,], uv_layer=uv_layer)
        # find max/min by X and Y
        min_x_point = min(active_face_uv_points, key=lambda _point: _point.uv.x)
        max_x_point = max(active_face_uv_points, key=lambda _point: _point.uv.x)
        min_y_point = min(active_face_uv_points, key=lambda _point: _point.uv.y)
        max_y_point = max(active_face_uv_points, key=lambda _point: _point.uv.y)
        # get src x and y lengths
        src_x = max_x_point.uv.x - min_x_point.uv.x
        src_y = max_y_point.uv.y - min_y_point.uv.y
        # get src coordinates of one point
        src_co = Vector((active_face_uv_points[0].uv.x, active_face_uv_points[0].uv.y))
        # 2. execute Multi Sure UV
        # change mode back to correctly work of Multy Sure UV, and after its execution switch back to edit mode
        bpy.ops.object.mode_set(mode=mode)
        bpy.ops.uvtools.multy_sureuv()
        if context.active_object.mode == 'EDIT':
            bpy.ops.object.mode_set(mode='OBJECT')
        # 3. get current scale lengths by X and Y of the UV by active face
        # uv_layer, active_face, active_face_uv_points - renew, because all of them were lost after object mode changing
        uv_layer = cls._active_uv_layer(obj=context.active_object)
        active_face = context.object.data.polygons[context.object.data.polygons.active]
        active_face_uv_points = cls._uv_points(faces_list=[active_face, ], uv_layer=uv_layer)
        # find max/min by X and Y
        min_x_point = min(active_face_uv_points, key=lambda _point: _point.uv.x)
        max_x_point = max(active_face_uv_points, key=lambda _point: _point.uv.x)
        min_y_point = min(active_face_uv_points, key=lambda _point: _point.uv.y)
        max_y_point = max(active_face_uv_points, key=lambda _point: _point.uv.y)
        # get current x and y lengths
        current_x = max_x_point.uv.x - min_x_point.uv.x
        current_y = max_y_point.uv.y - min_y_point.uv.y
        # 4. get scale factor by X and Y and move factor
        scale_factor_x = src_x / current_x
        scale_factor_y = src_y / current_y
        # 5. multiply each UV point coordinates by scale factor
        # process uv points only from selected faces
        selected_faces_uv_points = cls._uv_points(
            faces_list=[_face for _face in context.object.data.polygons if _face.select],
            uv_layer=uv_layer
        )
        for uv_point in selected_faces_uv_points:
            # scale
            if scale_by_x:
                uv_point.uv.x = scale_factor_x * uv_point.uv.x
            if scale_by_y:
                uv_point.uv.y = scale_factor_y * uv_point.uv.y
        # 6. shift each UV point coordinates by move factor
        # get current coordinates of one point
        current_co = Vector((active_face_uv_points[0].uv.x, active_face_uv_points[0].uv.y))
        # get move factor
        move_factor = current_co - src_co
        # shift by move factor
        # process uv points only from selected faces
        for uv_point in selected_faces_uv_points:
            # move
            uv_point.uv -= move_factor
        c_co = active_face_uv_points[0].uv
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

    # END EXPERIMENTAL

    @staticmethod
    def show_texture():
        # Multy Sure UV from 1D_Scripts
        obj = bpy.context.active_object
        mesh = obj.data
        is_editmode = (obj.mode == 'EDIT')
        # if in EDIT Mode switch to OBJECT
        if is_editmode:
            bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

        # if no UVtex - create it
        if not mesh.uv_textures:
            uvtex = bpy.ops.mesh.uv_texture_add()
        uvtex = mesh.uv_textures.active
        uvtex.active_render = True

        img = None
        aspect = 1.0
        mat = obj.active_material

        try:
            if mat:
                img = mat.active_texture
                for f in mesh.polygons:
                    if not is_editmode or f.select:
                        uvtex.data[f.index].image = img.image
            else:
                img = None
        except:
            pass

        # Back to EDIT Mode
        if is_editmode:
            bpy.ops.object.mode_set(mode='EDIT', toggle=False)

    @staticmethod
    def box_map(all_scale_def, x_offset_def, y_offset_def, z_offset_def, x_rot_def, y_rot_def, z_rot_def, tex_aspect):
        # Multy Sure UV from 1D_Scripts
        obj = bpy.context.active_object
        mesh = obj.data

        is_editmode = (obj.mode == 'EDIT')

        # if in EDIT Mode switch to OBJECT
        if is_editmode:
            bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

        # if no UVtex - create it
        if not mesh.uv_textures:
            uvtex = bpy.ops.mesh.uv_texture_add()
        uvtex = mesh.uv_textures.active
        # uvtex.active_render = True

        img = None
        aspect = 1.0
        mat = obj.active_material
        try:
            if mat:
                img = mat.active_texture
                aspect = img.image.size[0] / img.image.size[1]
        except:
            pass
        aspect = aspect * tex_aspect

        #
        # Main action
        #
        if all_scale_def:
            sc = 1.0 / all_scale_def
        else:
            sc = 1.0

        sx = 1 * sc
        sy = 1 * sc
        sz = 1 * sc
        ofx = x_offset_def
        ofy = y_offset_def
        ofz = z_offset_def
        rx = x_rot_def / 180 * pi
        ry = y_rot_def / 180 * pi
        rz = z_rot_def / 180 * pi

        crx = cos(rx)
        srx = sin(rx)
        cry = cos(ry)
        sry = sin(ry)
        crz = cos(rz)
        srz = sin(rz)
        ofycrx = ofy * crx
        ofzsrx = ofz * srx

        ofysrx = ofy * srx
        ofzcrx = ofz * crx

        ofxcry = ofx * cry
        ofzsry = ofz * sry

        ofxsry = ofx * sry
        ofzcry = ofz * cry

        ofxcrz = ofx * crz
        ofysrz = ofy * srz

        ofxsrz = ofx * srz
        ofycrz = ofy * crz

        # uvs = mesh.uv_loop_layers[mesh.uv_loop_layers.active_index].data
        uvs = mesh.uv_layers.active.data
        for i, pol in enumerate(mesh.polygons):
            if not is_editmode or mesh.polygons[i].select:
                for j, loop in enumerate(mesh.polygons[i].loop_indices):
                    v_idx = mesh.loops[loop].vertex_index
                    # print('before[%s]:' % v_idx)
                    # print(uvs[loop].uv)
                    n = mesh.polygons[i].normal
                    co = mesh.vertices[v_idx].co
                    x = co.x * sx
                    y = co.y * sy
                    z = co.z * sz
                    if abs(n[0]) > abs(n[1]) and abs(n[0]) > abs(n[2]):
                        # X
                        if n[0] >= 0:
                            uvs[loop].uv[0] = y * crx + z * srx - ofycrx - ofzsrx
                            uvs[loop].uv[1] = -y * aspect * srx + z * aspect * crx + ofysrx - ofzcrx
                        else:
                            uvs[loop].uv[0] = -y * crx + z * srx + ofycrx - ofzsrx
                            uvs[loop].uv[1] = y * aspect * srx + z * aspect * crx - ofysrx - ofzcrx
                    elif abs(n[1]) > abs(n[0]) and abs(n[1]) > abs(n[2]):
                        # Y
                        if n[1] >= 0:
                            uvs[loop].uv[0] = -x * cry + z * sry + ofxcry - ofzsry
                            uvs[loop].uv[1] = x * aspect * sry + z * aspect * cry - ofxsry - ofzcry
                        else:
                            uvs[loop].uv[0] = x * cry + z * sry - ofxcry - ofzsry
                            uvs[loop].uv[1] = -x * aspect * sry + z * aspect * cry + ofxsry - ofzcry
                    else:
                        # Z
                        if n[2] >= 0:
                            uvs[loop].uv[0] = x * crz + y * srz + - ofxcrz - ofysrz
                            uvs[loop].uv[1] = -x * aspect * srz + y * aspect * crz + ofxsrz - ofycrz
                        else:
                            uvs[loop].uv[0] = -y * srz - x * crz + ofxcrz - ofysrz
                            uvs[loop].uv[1] = y * aspect * crz - x * aspect * srz - ofxsrz - ofycrz

        # Back to EDIT Mode
        if is_editmode:
            bpy.ops.object.mode_set(mode='EDIT', toggle=False)

    @staticmethod
    def best_planar_map(all_scale_def, xoffset_def, yoffset_def, zrot_def, tex_aspect):
        # Multy Sure UV from 1D_Scripts
        # Best Planar Mapping
        # global all_scale_def, xoffset_def, yoffset_def, zrot_def, tex_aspect

        obj = bpy.context.active_object
        mesh = obj.data

        is_editmode = (obj.mode == 'EDIT')

        # if in EDIT Mode switch to OBJECT
        if is_editmode:
            bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

        # if no UVtex - create it
        if not mesh.uv_textures:
            uvtex = bpy.ops.mesh.uv_texture_add()
        uvtex = mesh.uv_textures.active
        # uvtex.active_render = True

        img = None
        aspect = 1.0
        mat = obj.active_material
        try:
            if mat:
                img = mat.active_texture
                aspect = img.image.size[0] / img.image.size[1]
        except:
            pass
        aspect = aspect * tex_aspect

        #
        # Main action
        #
        if all_scale_def:
            sc = 1.0 / all_scale_def
        else:
            sc = 1.0

            # Calculate Average Normal
        v = Vector((0, 0, 0))
        cnt = 0
        for f in mesh.polygons:
            if f.select:
                cnt += 1
                v = v + f.normal

        zv = Vector((0, 0, 1))
        q = v.rotation_difference(zv)

        sx = 1 * sc
        sy = 1 * sc
        sz = 1 * sc
        ofx = xoffset_def
        ofy = yoffset_def
        rz = zrot_def / 180 * pi

        cosrz = cos(rz)
        sinrz = sin(rz)

        # uvs = mesh.uv_loop_layers[mesh.uv_loop_layers.active_index].data
        uvs = mesh.uv_layers.active.data
        for i, pol in enumerate(mesh.polygons):
            if not is_editmode or mesh.polygons[i].select:
                for j, loop in enumerate(mesh.polygons[i].loop_indices):
                    v_idx = mesh.loops[loop].vertex_index

                    n = pol.normal
                    co = q * mesh.vertices[v_idx].co
                    x = co.x * sx
                    y = co.y * sy
                    z = co.z * sz
                    uvs[loop].uv[0] = x * cosrz - y * sinrz + xoffset_def
                    uvs[loop].uv[1] = aspect * (- x * sinrz - y * cosrz) + yoffset_def

        # Back to EDIT Mode
        if is_editmode:
            bpy.ops.object.mode_set(mode='EDIT', toggle=False)

    @staticmethod
    def _active_uv_layer(obj):
        # get active uv-layer (UV Map)
        if isinstance(obj, bmesh.types.BMesh):
            # from bmesh object
            return obj.loops.layers.uv.active
        else:
            # from object
            return obj.data.uv_layers.active

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
    def _uv_points(faces_list, uv_layer):
        # get list of UV points for mesh faces from faces_list
        # [uv_point, uv_point, ...]
        return [uv_layer.data[loop_index] for _face in faces_list for loop_index in _face.loop_indices]

    @staticmethod
    def ui(layout, context, area):
        # ui panels
        if area == 'VIEWPORT':
            # Texel Scale
            box = layout.box().column()
            box.label(text='Texel Scale')
            op = box.operator(
                operator='uvtools.texel_scale',
                icon='FORCE_TEXTURE'
            )
            op.scale_by_x = context.window_manager.uv_tools_1d_prop_texel_scale_by_x
            op.scale_by_y = context.window_manager.uv_tools_1d_prop_texel_scale_by_y
            row = box.row()
            row.prop(
                data=context.window_manager,
                property='uv_tools_1d_prop_texel_scale_by_x'
            )
            row.prop(
                data=context.window_manager,
                property='uv_tools_1d_prop_texel_scale_by_y'
            )
            # Multy Sure UV
            box = layout.box().column()
            box.label(text='Sure UV Map')
            row = box.row()
            if context.window_manager.uv_tools_1d_prop_disp_omsureuv:
                row.prop(
                    data=context.window_manager,
                    property='uv_tools_1d_prop_disp_omsureuv',
                    text='',
                    icon='DOWNARROW_HLT'
                )
            else:
                row.prop(
                    data=context.window_manager,
                    property='uv_tools_1d_prop_disp_omsureuv',
                    text='',
                    icon='RIGHTARROW'
                )
            row.operator(
                operator='uvtools.multy_sureuv',
                text='Obj Multy SureUV'
            )
            if context.window_manager.uv_tools_1d_prop_disp_omsureuv:
                box2 = box.box().column()
                layout = box2.column(align=True)
                layout.label('XYZ rotation')
                col2 = layout.column()
                col2.prop(
                    data=context.window_manager,
                    property='uv_tools_1d_prop_omsureuv_rot',
                    text=''
                )
                layout.label('XYZ offset')
                col2 = layout.column()
                col2.prop(
                    data=context.window_manager,
                    property='uv_tools_1d_prop_omsureuv_offset',
                    text=''
                )
            box.label('Press this button first:')
            op = box.operator(
                operator='uvtools.sureuvw_operator',
                text='Show active texture on object')
            op.action = 'showtex'
            op.size=context.window_manager.uv_tools_1d_prop_omsureuv_all_scale_def
            op.rot=context.window_manager.uv_tools_1d_prop_omsureuv_rot
            op.offset=context.window_manager.uv_tools_1d_prop_omsureuv_offset
            box.label('UVW Mapping:')
            op = box.operator(
                operator='uvtools.sureuvw_operator',
                text='UVW Box Map'
            )
            op.action = 'box'
            op.size=context.window_manager.uv_tools_1d_prop_omsureuv_all_scale_def
            op.rot=context.window_manager.uv_tools_1d_prop_omsureuv_rot
            op.offset=context.window_manager.uv_tools_1d_prop_omsureuv_offset
            op = box.operator(
                operator='uvtools.sureuvw_operator',
                text='Best Planar Map'
            )
            op.action = 'bestplanar'
            op.size=context.window_manager.uv_tools_1d_prop_omsureuv_all_scale_def
            op.rot=context.window_manager.uv_tools_1d_prop_omsureuv_rot
            op.offset=context.window_manager.uv_tools_1d_prop_omsureuv_offset
            box.label('1. Make Material With Raster Texture!')
            box.label('2. Set Texture Mapping Coords: UV!')
            box.label('3. Use Addon buttons')
        elif area == 'UV':
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


class UVTools_OT_texel_scale(Operator):
    bl_idname = 'uvtools.texel_scale'
    bl_label = 'Retexel'
    bl_options = {'REGISTER', 'UNDO'}

    scale_by_x = BoolProperty(
        name='Scale by X',
        default=True
    )
    scale_by_y = BoolProperty(
        name='Scale by Y',
        default=True
    )

    def execute(self, context):
        UVTools.texel_scale(
            context=context,
            scale_by_x=self.scale_by_x,
            scale_by_y=self.scale_by_y
        )
        return {'FINISHED'}


class UVTools_PaObjMultySureUV(Operator):

    """
        Multy SureUV from 1D_Scripts
    """

    bl_idname = 'uvtools.multy_sureuv'
    bl_label = 'Obj MMuulty SureUV MM mm ьь ЬЬ'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and \
               context.active_object.type == 'MESH'

    def execute(self, context):
        for ob in bpy.context.selected_objects:
            if ob.type == 'MESH':
                bpy.context.scene.objects.active = ob
                bpy.ops.uvtools.sureuvw_operator(
                    action='box',
                    size=context.window_manager.uv_tools_1d_prop_omsureuv_all_scale_def,
                    rot=context.window_manager.uv_tools_1d_prop_omsureuv_rot,
                    offset=context.window_manager.uv_tools_1d_prop_omsureuv_offset,
                    zrot=0,
                    xoffset=0,
                    yoffset=0,
                    texaspect=1.0
                )

        return {'FINISHED'}

class UVTools_SureUVWOperator(Operator):

    """
        Multy SureUV from 1D_Scripts
    """

    bl_idname = 'uvtools.sureuvw_operator'
    bl_label = 'Sure UVW Map'
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'data'
    bl_options = {'REGISTER', 'UNDO'}

    action = StringProperty(
        default='box'
    )
    size = FloatProperty(
        name='Size',
        default=1.0,
        precision=4
    )
    rot = FloatVectorProperty(
        name='XYZ Rotation'
    )
    offset = FloatVectorProperty(
        name='XYZ offset',
        precision=4
    )
    zrot = FloatProperty(
        name='Z rotation',
        default=0.0
    )
    xoffset = FloatProperty(
        name='X offset',
        default=0.0,
        precision=4
    )
    yoffset = FloatProperty(
        name='Y offset',
        default=0.0,
        precision=4
    )
    texaspect = FloatProperty(
        name='Texture aspect',
        default=1.0,
        precision=4
    )
    flag90 = BoolProperty()
    flag90ccw = BoolProperty()

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH'

    def execute(self, context):
        # print('** execute **')
        # print(self.action)

        all_scale_def = self.size
        tex_aspect = self.texaspect

        x_offset_def = self.offset[0]
        y_offset_def = self.offset[1]
        z_offset_def = self.offset[2]
        x_rot_def = self.rot[0]
        y_rot_def = self.rot[1]
        z_rot_def = self.rot[2]

        xoffset_def = self.xoffset
        yoffset_def = self.yoffset
        zrot_def = self.zrot

        if self.flag90:
            self.zrot += 90
            zrot_def += 90
            self.flag90 = False

        if self.flag90ccw:
            self.zrot += -90
            zrot_def += -90
            self.flag90ccw = False

        if self.action == 'bestplanar':
            UVTools.best_planar_map(
                all_scale_def=self.size,
                xoffset_def=self.offset[0],
                yoffset_def=self.offset[1],
                zrot_def=self.rot[2],
                tex_aspect=self.texaspect
            )
        elif self.action == 'box':
            UVTools.box_map(
                all_scale_def=self.size,
                x_offset_def=self.offset[0],
                y_offset_def=self.offset[1],
                z_offset_def=self.offset[2],
                x_rot_def=self.rot[0],
                y_rot_def=self.rot[1],
                z_rot_def=self.rot[2],
                tex_aspect=self.texaspect
            )
        elif self.action == 'showtex':
            UVTools.show_texture()
        elif self.action == 'doneplanar':
            UVTools.best_planar_map(
                all_scale_def=self.size,
                xoffset_def=self.offset[0],
                yoffset_def=self.offset[1],
                zrot_def=self.rot[2],
                tex_aspect=self.texaspect
            )
        elif self.action == 'donebox':
            UVTools.box_map(
                all_scale_def=self.size,
                x_offset_def=self.offset[0],
                y_offset_def=self.offset[1],
                z_offset_def=self.offset[2],
                x_rot_def=self.rot[0],
                y_rot_def=self.rot[1],
                z_rot_def=self.rot[2],
                tex_aspect=self.texaspect
            )
        # print('finish execute')
        return {'FINISHED'}

    def invoke(self, context, event):
        # print('** invoke **')
        # print(self.action)
        if self.action == 'bestplanar':
            UVTools.best_planar_map(
                all_scale_def=self.size,
                xoffset_def=self.offset[0],
                yoffset_def=self.offset[1],
                zrot_def=self.rot[2],
                tex_aspect=self.texaspect
            )
        elif self.action == 'box':
            UVTools.box_map(
                all_scale_def=self.size,
                x_offset_def=self.offset[0],
                y_offset_def=self.offset[1],
                z_offset_def=self.offset[2],
                x_rot_def=self.rot[0],
                y_rot_def=self.rot[1],
                z_rot_def=self.rot[2],
                tex_aspect=self.texaspect
            )
        elif self.action == 'showtex':
            UVTools.show_texture()
        elif self.action == 'doneplanar':
            UVTools.best_planar_map(
                all_scale_def=self.size,
                xoffset_def=self.offset[0],
                yoffset_def=self.offset[1],
                zrot_def=self.rot[2],
                tex_aspect=self.texaspect
            )
        elif self.action == 'donebox':
            UVTools.box_map(
                all_scale_def=self.size,
                x_offset_def=self.offset[0],
                y_offset_def=self.offset[1],
                z_offset_def=self.offset[2],
                x_rot_def=self.rot[0],
                y_rot_def=self.rot[1],
                z_rot_def=self.rot[2],
                tex_aspect=self.texaspect
            )
        # print('finish invoke')
        return {'FINISHED'}

    def draw(self, context):
        if self.action == 'bestplanar' or self.action == 'rotatecw' or self.action == 'rotateccw':
            self.action = 'bestplanar'
            layout = self.layout
            layout.label("Size - " + self.action)
            layout.prop(self, 'size', text="")
            layout.label("Z rotation")
            col = layout.column()
            col.prop(self, 'zrot', text="")
            row = layout.row()
            row.prop(self, 'flag90ccw', text="-90 (CCW)")
            row.prop(self, 'flag90', text="+90 (CW)")
            layout.label("XY offset")
            col = layout.column()
            col.prop(self, 'xoffset', text="")
            col.prop(self, 'yoffset', text="")

            layout.label("Texture aspect")
            layout.prop(self, 'texaspect', text="")

            # layout.prop(self,'preview_flag', text="Interactive Preview")
            # layout.operator("uvtools.sureuvw_operator",text="Done").action='doneplanar'

        elif self.action == 'box':
            layout = self.layout
            layout.label("Size")
            layout.prop(self, 'size', text="")
            layout.label("XYZ rotation")
            col = layout.column()
            col.prop(self, 'rot', text="")
            layout.label("XYZ offset")
            col = layout.column()
            col.prop(self, 'offset', text="")
            layout.label("Texture squash (optional)")
            layout.label("Always must be 1.0 !!!")
            layout.prop(self, 'texaspect', text="")

            # layout.prop(self,'preview_flag', text="Interactive Preview")
            # layout.operator("uvtools.sureuvw_operator",text="Done").action='donebox'

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
            context=context,
            area='UV'
        )

class UVTools_PT_panel_Viewport(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_label = 'UV Tools'
    bl_category = '1D'

    def draw(self, context):
        UVTools.ui(
            layout=self.layout,
            context=context,
            area='VIEWPORT'
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
    WindowManager.uv_tools_1d_prop_texel_scale_by_x = BoolProperty(
        name='Scale by X',
        default=True
    )
    WindowManager.uv_tools_1d_prop_texel_scale_by_y = BoolProperty(
        name='Scale by Y',
        default=True
    )
    WindowManager.uv_tools_1d_prop_omsureuv_all_scale_def = FloatProperty(
        # Multy SureUV from 1D_Scripts
        name='omsureuv_all_scale_def',
        default=3.0,
        precision=4
    )
    WindowManager.uv_tools_1d_prop_omsureuv_rot = FloatVectorProperty(
        # Multy SureUV from 1D_Scripts
        name='omsureuv_rot',
        precision=2
    )
    WindowManager.uv_tools_1d_prop_omsureuv_offset = FloatVectorProperty(
        # Multy SureUV from 1D_Scripts
        name='omsureuv_offset',
        precision=4
    )
    WindowManager.uv_tools_1d_prop_disp_omsureuv = BoolProperty(
        # Multy SureUV from 1D_Scripts
        name='disp_omsureuv',
        default=False
    )
    register_class(UVTools_OT_uvpack)
    register_class(UVTools_OT_fit_to_tile)
    register_class(UVTools_OT_texel_scale)
    register_class(UVTools_OT_store_diagonal)
    register_class(UVTools_OT_select_uvcut_tile)
    register_class(UVTools_OT_uvcut_tile)
    register_class(UVTools_PaObjMultySureUV)
    register_class(UVTools_SureUVWOperator)
    if ui:
        register_class(UVTools_PT_panel)
        register_class(UVTools_PT_panel_Viewport)


def unregister(ui=True):
    if ui:
        unregister_class(UVTools_PT_panel_Viewport)
        unregister_class(UVTools_PT_panel)
    # butch clean
    unregister_class(UVTools_SureUVWOperator)
    unregister_class(UVTools_PaObjMultySureUV)
    unregister_class(UVTools_OT_uvcut_tile)
    unregister_class(UVTools_OT_select_uvcut_tile)
    unregister_class(UVTools_OT_store_diagonal)
    unregister_class(UVTools_OT_texel_scale)
    unregister_class(UVTools_OT_fit_to_tile)
    unregister_class(UVTools_OT_uvpack)
    del WindowManager.uv_tools_1d_prop_disp_omsureuv
    del WindowManager.uv_tools_1d_prop_omsureuv_all_scale_def
    del WindowManager.uv_tools_1d_prop_omsureuv_rot
    del WindowManager.uv_tools_1d_prop_omsureuv_offset
    del WindowManager.uv_tools_1d_prop_texel_scale_by_y
    del WindowManager.uv_tools_1d_prop_texel_scale_by_x
    del WindowManager.uv_tools_1d_prop_fit_to_tile_add_scale
    del WindowManager.uv_tools_1d_prop_fit_to_tile_v2
    del WindowManager.uv_tools_1d_prop_fit_to_tile_v1
    del WindowManager.uv_tools_1d_prop_fit_to_tile_v0
    del Scene.uv_tools_1d_prop_pack_to_cursor


if __name__ == "__main__":
    register()
