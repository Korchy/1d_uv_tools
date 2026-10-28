# Nikita Akimov
# interplanety@interplanety.org
#
# GitHub
#    https://github.com/Korchy/1d_uv_tools


import bmesh
import bpy
from bmesh.types import BMVert
from bpy.props import BoolProperty, FloatProperty, FloatVectorProperty, StringProperty
from bpy.types import Operator, Panel, Scene, WindowManager
from bpy.utils import register_class, unregister_class
from math import atan2, ceil, cos, degrees, floor, sin, pi
from mathutils import Vector, Matrix
from mathutils.geometry import barycentric_transform

bl_info = {
    "name": "1D UV Tools",
    "description": "Tools for working with UV Maps",
    "author": "Nikita Akimov, Paul Kotelevets",
    "version": (1, 8, 0),
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
    def texel_scale_face(cls, context, op, vertical_threshold=15):
        # Retexel
        #   "active face + selection" version - used with face selection mode in 3D_VIEW + active face + selection
        #   after executing Sure Uv we need to return uv-face of active face to its state before executing
        #       and apply this transform for all uv-points of selected faces
        #   1 calculate source transform (3 vectors of 3 points) on uv-face of active face
        #   2 execute Multy Sure Uv
        #   3 calculate dest transform (3 vectors of 3 points) on uv-face of active face
        #   4 apply transform to all uv-points of selected faces
        print('RETEXEL: face active + selection mode')
        # switch to Object mode
        mode = context.active_object.mode if context.active_object.mode != 'OBJECT' else None
        if mode is not None:
            bpy.ops.object.mode_set(mode='OBJECT')
        # filter selected faces by vertical threshold
        active_face_id = context.object.data.polygons.active
        bm = bmesh.new()
        bm.from_mesh(context.active_object.data)
        bm.faces.ensure_lookup_table()
        bm_active_face = next((_face for _face in bm.faces if _face.index == active_face_id), None)
        # filter - remove collapsed faces and faces by threshold from selection
        active_face_normal_vert_diff = round(degrees(bm_active_face.normal.angle(Vector((0.0, 0.0, 1.0)))), 2)  # 0 - 180
        for face in (_face for _face in bm.faces if _face.select):
            # deselect faces with zero-length normal (faces with zero area, ex: all vertices are in the same location)
            if face.normal.length == 0.0:
                face.select = False
                continue
            # deselect faces by threshold
            face_normal_vert_diff = round(degrees(face.normal.angle(Vector((0.0, 0.0, 1.0)))), 2)
            if  abs(face_normal_vert_diff - active_face_normal_vert_diff) > vertical_threshold:
                face.select = False
        bm.to_mesh(context.active_object.data)
        bm.free()
        # 1. get source 3 uv-point coordinates for transformations
        # active UV layer
        uv_layer = cls._active_uv_layer(obj=context.active_object)
        active_face = context.object.data.polygons[context.object.data.polygons.active]
        # get longest uv-edge on uv-face from active face
        active_face_mesh_loops = [uv_layer.data[loop_index] for loop_index in active_face.loop_indices]
        # get 3 points on active face to calculate transformation
        src_triangle_points = active_face_mesh_loops[:3]
        src_triangle_co = [Vector(_uv_point.uv).to_3d() for _uv_point in src_triangle_points]
        # 2. execute Multi Sure UV
        #   use as function not to switch object/edit mode, not to lose previously calculated data when switching mode
        cls.box_map(
            all_scale_def=context.window_manager.uv_tools_1d_prop_omsureuv_all_scale_def,
            x_offset_def=context.window_manager.uv_tools_1d_prop_omsureuv_offset[0],
            y_offset_def=context.window_manager.uv_tools_1d_prop_omsureuv_offset[1],
            z_offset_def=context.window_manager.uv_tools_1d_prop_omsureuv_offset[2],
            x_rot_def=context.window_manager.uv_tools_1d_prop_omsureuv_rot[0],
            y_rot_def=context.window_manager.uv_tools_1d_prop_omsureuv_rot[1],
            z_rot_def=context.window_manager.uv_tools_1d_prop_omsureuv_rot[2],
            tex_aspect=1.0,
            obj_mode='OBJECT'
        )
        # 3. get 3 uv-point coordinates after Multy Sure Uv
        dest_triangle_co = [Vector(_uv_point.uv).to_3d() for _uv_point in src_triangle_points]
        # points to apply transformation
        selected_faces_uv_points = cls._uv_points(
            faces_list=[_face for _face in context.object.data.polygons if _face.select],
            uv_layer=uv_layer
        )
        # 4. apply transform from active face to all selected faces uv-points
        for uv_point in selected_faces_uv_points:
            wco = uv_point.uv.to_3d()
            wco = barycentric_transform(wco, *dest_triangle_co, *src_triangle_co)
            uv_point.uv = wco.to_2d()
        # output transformation made scale and angle factors to INFO
        src_vector = (src_triangle_co[0] - src_triangle_co[1]).to_2d()      # first vector of src triangle
        dest_vector = (dest_triangle_co[0] - dest_triangle_co[1]).to_2d()   # first vector of dest triangle
        transform_scale = src_vector.length / dest_vector.length
        transform_scale_x = (src_vector.x / dest_vector.x) if dest_vector.x != 0 else 0.0
        transform_scale_y = (src_vector.y / dest_vector.y) if dest_vector.y != 0 else 0.0
        transform_angle = dest_vector.angle(src_vector)
        transform_angle_x = dest_vector.angle(Vector((0.0, 1.0)))
        op.report(
            type={'INFO'},
            message='scale: ' + str(round(transform_scale, 4)) + ', ' \
                    + '(x: ' + str(round(transform_scale_x, 4))  + ', '\
                    + 'y: ' + str(round(transform_scale_y, 4)) + ') ' \
                    + 'angle: ' + str(round(degrees(transform_angle), 2)) + ' ' \
                    + '('  + str(round(degrees(transform_angle_x), 2)) + ')' \
                    + ' deg'
        )
        # return mode back
        if mode is not None:
            bpy.ops.object.mode_set(mode=mode)

    @classmethod
    def texel_scale_face_only_active(cls, context, op):
        # Retexel
        #   "face only active" version - used when only one active face is in 3D_VIEW
        #   check what uv-edge of activ face is more horizontal
        #   get next uv-edge and rotate uv-face so, to make this next edge horizontal
        #   repeat on each click (continuously rotate uv-face on each click)
        print('RETEXEL: face only active mode')
        # switch to Object mode
        mode = context.active_object.mode if context.active_object.mode != 'OBJECT' else None
        if mode is not None:
            bpy.ops.object.mode_set(mode='OBJECT')
        # get data for rotation
        uv_layer = cls._active_uv_layer(obj=context.active_object)
        active_face = context.object.data.polygons[context.object.data.polygons.active]
        # get longest uv-edge on uv-face from active face
        active_face_mesh_loops = [uv_layer.data[loop_index] for loop_index in active_face.loop_indices]
        # active_face_uv_points = cls._uv_points(faces_list=[active_face, ], uv_layer=uv_layer)
        selected_faces_uv_points = cls._uv_points(
            faces_list=[_face for _face in context.object.data.polygons if _face.select],
            uv_layer=uv_layer
        )
        # find left bottom uv-point, we will transform to this point
        #   sort by left (x), and next find the bottom (y)
        mesh_loops_x_sorted = sorted(active_face_mesh_loops, key=lambda _uv_point: round(_uv_point.uv.x, 4))
        lb_point = min(mesh_loops_x_sorted, key=lambda _uv_point: round(_uv_point.uv.y, 4))
        # get 3 points to calculate transformation to
        #   we have 1 left bottom uv-point, this is the center point
        #   1-st and 3-rd uv-point we can get as horizontal and vertical vectors
        dest_triangle_co = [(lb_point.uv + Vector((0.0, 1.0))).to_3d(),
                            lb_point.uv.to_3d(),
                            (lb_point.uv + Vector((1.0, 0.0))).to_3d()]
        # get next point, we will transform from this point
        next_point_idx = active_face_mesh_loops.index(lb_point) + 1
        next_point = active_face_mesh_loops[0] if next_point_idx == len(active_face_mesh_loops) \
            else active_face_mesh_loops[next_point_idx]
        # get next-next point to have 3 point
        next2_point_idx = active_face_mesh_loops.index(next_point) + 1
        next2_point = active_face_mesh_loops[0] if next2_point_idx == len(active_face_mesh_loops) \
            else active_face_mesh_loops[next2_point_idx]
        # get 3 points to calculate transformation from
        #   we have 2 uv-points (next_point and next2_point)
        #   3-rd uv-point we can get as normal to next2_point - next_point with length of 1
        v =next2_point.uv - next_point.uv
        v_normal = Vector((-v.y, v.x))  # CV (Paul works better)
        # v_normal = Vector((v.y, -v.x))    # CCV
        src_triangle_co = [(next_point.uv + v_normal.normalized()).to_3d(),    # length = 1
                           next_point.uv.to_3d(),
                           (next_point.uv + v.normalized()).to_3d()            # length = 1
                           ]
        # 4. apply transform from active face to all selected faces uv-points
        for uv_point in selected_faces_uv_points:
            wco = uv_point.uv.to_3d()
            wco = barycentric_transform(wco, *src_triangle_co, *dest_triangle_co)
            uv_point.uv = wco.to_2d()
        # output transformation made angle factor to INFO
        src_vector = (src_triangle_co[0] - src_triangle_co[1]).to_2d()      # first vector of src triangle
        dest_vector = (dest_triangle_co[0] - dest_triangle_co[1]).to_2d()   # first vector of dest triangle
        transform_angle = dest_vector.angle(src_vector)
        transform_angle_x = dest_vector.angle(Vector((0.0, 1.0)))
        op.report(
            type={'INFO'},
            message='angle: ' + str(round(degrees(transform_angle), 2)) + ' ' \
                    + '('  + str(round(degrees(transform_angle_x), 2)) + ')' \
                    + ' deg'
        )
        # return mode back
        if mode is not None:
            bpy.ops.object.mode_set(mode=mode)

    @classmethod
    def texel_scale_edge(cls, context, op):
        # Retexel
        # "edge mode" - rotate uv-points of selected faces to make horizontally uv-edge of active edge of active face
        #   active edge must be always on active face
        print('RETEXEL: edge active mode')
        # switch to Object mode
        mode = context.active_object.mode if context.active_object.mode != 'OBJECT' else None
        if mode is not None:
            bpy.ops.object.mode_set(mode='OBJECT')
        # get data for rotation
        uv_layer = cls._active_uv_layer(obj=context.active_object)
        active_edge = cls.active_edge(context=context, obj=context.active_object)
        active_edge_vertices_idxs = active_edge.vertices[:]
        face_with_active_edge = context.object.data.polygons[context.object.data.polygons.active]   # active face
        # get uv-points for counting rotation
        edge_mesh_loops = []
        for _i, loop_id in enumerate(face_with_active_edge.loop_indices):
            if context.object.data.vertices[face_with_active_edge.vertices[_i]].index in active_edge_vertices_idxs:
                edge_mesh_loops.append(uv_layer.data[loop_id])
        # all uv-points to rotate
        selected_faces_uv_points = cls._uv_points(
            faces_list=[_face for _face in context.object.data.polygons if _face.select],
            uv_layer=uv_layer
        )
        # get 3 points to calculate transformation
        #   we have 2 uv-points from active edge
        #   3-rd uv-point we can get as normal to active uv-edge with length of active uv-edge
        v = edge_mesh_loops[1].uv - edge_mesh_loops[0].uv
        v_normal = Vector((-v.y, v.x))  # CV (Paul works better)
        # v_normal = Vector((v.y, -v.x))    # CCV
        p0 = edge_mesh_loops[0].uv + v_normal
        src_triangle_co = [p0.to_3d(), edge_mesh_loops[0].uv.to_3d(), edge_mesh_loops[1].uv.to_3d()]
        # get 3 uv-point coordinates for having active uv-edge horizontal (rotate around central point)
        dest_p0 = Vector((0.0, 1.0)) * (p0 - edge_mesh_loops[0].uv).length + edge_mesh_loops[0].uv
        dest_p3 = Vector((1.0, 0.0)) * (edge_mesh_loops[1].uv - edge_mesh_loops[0].uv).length + edge_mesh_loops[0].uv
        dest_triangle_co = [dest_p0.to_3d(), edge_mesh_loops[0].uv.to_3d(), dest_p3.to_3d()]
        # 4. apply transform from active face to all selected faces uv-points
        for uv_point in selected_faces_uv_points:
            wco = uv_point.uv.to_3d()
            wco = barycentric_transform(wco, *src_triangle_co, *dest_triangle_co)
            uv_point.uv = wco.to_2d()
        # output transformation made scale and angle factors to INFO
        src_vector = (src_triangle_co[0] - src_triangle_co[1]).to_2d()      # first vector of src triangle
        dest_vector = (dest_triangle_co[0] - dest_triangle_co[1]).to_2d()   # first vector of dest triangle
        transform_angle = dest_vector.angle(src_vector)
        transform_angle_x = dest_vector.angle(Vector((0.0, 1.0)))
        op.report(
            type={'INFO'},
            message='angle: ' + str(round(degrees(transform_angle), 2)) + ' ' \
                    + '(' + str(round(degrees(transform_angle_x), 2)) + ')' \
                    + ' deg'
        )
        # return mode back
        if mode is not None:
            bpy.ops.object.mode_set(mode=mode)

    @classmethod
    def quad_level(cls, context, pin_level=False):
        # Quad Level
        # rotates selected faces on the UV horizontally by the longest edge
        # switch to Object mode
        mode = context.active_object.mode if context.active_object.mode != 'OBJECT' else None
        if mode is not None:
            bpy.ops.object.mode_set(mode='OBJECT')
        # bm
        bm = bmesh.new()
        bm.from_mesh(context.active_object.data)
        # process UV points
        uv_layer = cls._active_uv_layer(bm)
        # for each selected face
        for face in (_face for _face in bm.faces if _face.select ):
            longest_uv_edge = cls._longest_uv_edge(
                bm_face=face,
                uv_layer=uv_layer,
                bm=bm,
                offset=(1 if len(face.edges) == 3 else 0),
            )
            # rotate uv around first point of the longest uv edge
            cls._rotate_uv_around_point(
                center_of_rotation=longest_uv_edge[0],
                point_to_rotate=longest_uv_edge[1],
                dest_vector=Vector((1.0, 0.0)),
                uv_points=[_loop[uv_layer] for _loop in face.loops]
            )
            # pin longest edge
            if pin_level:
                longest_uv_edge[0].pin_uv = True
                longest_uv_edge[1].pin_uv = True
        # save changed data to mesh
        bm.to_mesh(context.active_object.data)
        bm.free()
        # return mode back
        if mode is not None:
            bpy.ops.object.mode_set(mode=mode)

    @classmethod
    def scan_transform(cls, context, max_value=99.0,
                       select_only=False, process_only_selection=False,
                       check_translation=False, check_rotation=False, check_scale=False,
                       check_threshold = 0.02):
        # Check what faces was changed via Multi Shure UV and moves them up proportionally by max_value
        #   select_only - only select faces, don't move
        #   only_selection - process only selected faces
        # current object
        obj = context.active_object
        # current mode
        mode = obj.mode if obj.mode != 'OBJECT' else None
        if mode is not None:
            bpy.ops.object.mode_set(mode='OBJECT')
        # use bmesh object to split mesh by all edges
        #   can't use bpy.ops.mesh.edge_split() because it requires EDIT mode for execution, and doesn't save selection
        #   after execution and switching to OBJECT mode
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        # split
        bmesh.ops.split_edges(bm, edges=bm.edges)
        # save changed data to mesh
        bm.to_mesh(obj.data)
        bm.free()
        # src uv name
        src_uv_layer_name = cls._active_uv_layer(obj=obj).name
        # create temp copy of active uv
        bpy.ops.mesh.uv_texture_add()
        # copy uv
        copy_uv_layer_name = cls._active_uv_layer(obj=obj).name
        # execute Multi Sure UV on copy uv_layer
        bpy.ops.uvtools.multy_sureuv()
        # use bmesh for proper selection
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        src_uv_layer = bm.loops.layers.uv[src_uv_layer_name]
        copy_uv_layer = bm.loops.layers.uv[copy_uv_layer_name]
        # get changes on faces between src_ and copy_ uv layers
        # proces faces
        faces = [_face for _face in bm.faces if _face.select] if process_only_selection else bm.faces
        changes = []    # [[face, rotation, translation, scale], ...]
        for face in faces:
            # check each face for transforms
            # rotation - compare rotation of the first longest uv-vector on copy uv_layer with the same on src uv_layer
            src_longest_edge, src_edge_idx, src_edge_length = cls._first_longest_uv_edge(
                face=face,
                uv_layer=src_uv_layer,
                bm=bm
            )
            src_longest_vec = src_longest_edge[1].uv - src_longest_edge[0].uv
            copy_uv_edges = cls._uv_edges(face=face, uv_layer=copy_uv_layer, bm=bm)
            copy_same_edge = copy_uv_edges[src_edge_idx]
            copy_same_vec = copy_same_edge[1].uv - copy_same_edge[0].uv
            # get rotation from src_vec to copy_vec in radians CCV
            cross = src_longest_vec.x * copy_same_vec.y - src_longest_vec.y * copy_same_vec.x
            dot = src_longest_vec.dot(copy_same_vec)
            rotation = atan2(cross, dot)    # angle between src_ and copy_ vectors in radians
            if abs(rotation) < check_threshold:
                rotation = 0.0
            rotation %= (2 * pi)    # set rotation angle to diapason 0 - 2pi (0 - 360 deg)
            # scale - compare lengths on this two vectors
            scale = src_edge_length / copy_same_vec.length
            if abs(scale - 1.0) < check_threshold:
                scale = 0.0
            # translation - compare positions of src_ and copy_ of the uv-face centroids
            src_uv_co = [_point.uv for _point in cls._uv_points(faces_list=[face], uv_layer=src_uv_layer, bm=bm)]
            src_centroid = cls._centroid(vertexes=src_uv_co)
            copy_uv_co = [_point.uv for _point in cls._uv_points(faces_list=[face], uv_layer=copy_uv_layer, bm=bm)]
            copy_centroid = cls._centroid(vertexes=copy_uv_co)
            translation = (copy_centroid - src_centroid).length
            if abs(translation) < check_threshold:
                translation = 0.0
            # append to changes
            if rotation != 0.0 or translation != 0.0 or scale != 0.0:
                changes.append([face, rotation, translation, scale])
        # apply changes to the faces
        if changes:
            if select_only:
                cls._deselect_all_bm(bm=bm)
            max_rotation = max(changes, key=lambda _item: _item[1])[1]
            min_rotation = min(changes, key=lambda _item: _item[1])[1]
            max_translation = max(changes, key=lambda _item: _item[2])[2]
            min_translation = min(changes, key=lambda _item: _item[2])[2]
            max_scale = max(changes, key=lambda _item: _item[3])[3]
            min_scale = min(changes, key=lambda _item: _item[3])[3]
            # for each change
            for face, rotation, translation, scale in changes:
                for vertex in face.verts:
                    # rotation
                    if check_rotation and rotation and max_rotation > min_rotation:
                       if select_only:
                            vertex.select = True
                            face.select = True
                       else:
                            vertex.co.z += max_value * (rotation - min_rotation) / (max_rotation - min_rotation)
                    # translation
                    if check_translation and translation and max_translation > min_translation:
                        if select_only:
                            vertex.select = True
                            face.select = True
                        else:
                            vertex.co.z += max_value * (translation - min_translation) / (max_translation - min_translation)
                    # scale
                    if check_scale and scale and max_scale > min_scale:
                        if select_only:
                            vertex.select = True
                            face.select = True
                        else:
                            vertex.co.z += max_value * (scale - min_scale) / (max_scale - min_scale)
        # save changed data to mesh
        bm.to_mesh(obj.data)
        bm.free()
        # remove temporary copy of the uv-map
        obj.data.uv_textures.remove(obj.data.uv_textures[copy_uv_layer_name])
        # return mode back
        if mode is not None:
            bpy.ops.object.mode_set(mode=mode)

    @classmethod
    def store_custom_tile(cls, context, op):
        # get and store custom area for future packing with "Pack to Custom Tile"
        # current object
        obj = context.active_object
        # current mode
        mode = obj.mode if obj.mode != 'OBJECT' else None
        if mode is not None:
            bpy.ops.object.mode_set(mode='OBJECT')
        # uv layer
        uv_layer = cls._active_uv_layer(obj)
        # get selected uv-points
        selected_uv_points = [_uv_point for _uv_point in \
                              cls._uv_points(faces_list=obj.data.polygons,uv_layer=uv_layer) \
                              if _uv_point.select]
        # bbox = {'max_y': -5.6, 'max_x': -11.14, 'min_x': -12.94, 'min_y': -7.16}
        bbox = cls._bbox_uv(uv_points=selected_uv_points)
        # save bbox to scene variables
        context.window_manager.uv_tools_1d_prop_pack_to_c_tile_rt = (bbox['max_x'], bbox['max_y'])
        context.window_manager.uv_tools_1d_prop_pack_to_c_tile_lb = (bbox['min_x'], bbox['min_y'])
        # print to INFO
        op.report(
            type={'INFO'},
            message='Custom Tile:' \
                    + ' x=' + str(abs(bbox['max_x'] - bbox['min_x'])) \
                    + ' y=' + str(abs(bbox['max_y'] - bbox['min_y'])) \
                    + ' [(' + str(bbox['min_x']) + ', ' + str(bbox['min_y']) + '), (' + str(bbox['max_x']) + ', ' + str(bbox['max_y']) + ')]'
        )
        # return mode back
        if mode is not None:
            bpy.ops.object.mode_set(mode=mode)

    @classmethod
    def pack_to_custom_tile(cls, context):
        # Pack to Custom Tile
        # all selected uv_points mast be transformed by stored tile to (0,0, 1,1) tile
        #   them use uv_pack_tile
        #   and then transform back
        # current object
        obj = context.active_object
        # current mode
        mode = obj.mode if obj.mode != 'OBJECT' else None
        if mode is not None:
            bpy.ops.object.mode_set(mode='OBJECT')
        # make first transform for all selected uv-points to ((0, 0),  (1, 1)) tile
        # use bmesh
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        # src triangle - get form stored tile
        src_triangle_co = [
            Vector(context.window_manager.uv_tools_1d_prop_pack_to_c_tile_lb).to_3d(),
            Vector((context.window_manager.uv_tools_1d_prop_pack_to_c_tile_rt[0], context.window_manager.uv_tools_1d_prop_pack_to_c_tile_lb[1], 0.0)),
            Vector(context.window_manager.uv_tools_1d_prop_pack_to_c_tile_rt).to_3d(),
        ]
        # dest triangle - get form ((0, 0),  (1, 1)) tile
        dest_triangle_co = [
            Vector((0.0, 0.0, 0.0)),
            Vector((1.0, 0.0, 0.0)),
            Vector((1.0, 1.0, 0.0))
        ]
        # active uv_layer
        uv_layer = cls._active_uv_layer(bm)
        # for all selected uv-points
        selected_uv_points = cls._uv_points(faces_list=bm.faces, uv_layer=uv_layer, bm=bm)
        # make transform from src_ to dest_
        for uv_point in selected_uv_points:
            wco = uv_point.uv.to_3d()
            wco = barycentric_transform(wco, *src_triangle_co, *dest_triangle_co)
            uv_point.uv = wco.to_2d()
        # save changed data to mesh
        bm.to_mesh(obj.data)
        bm.free()
        # pack using uv_pack_tile
        cls.uv_pack_tile(
            context=context,
            pack_to_cursor=False
        )
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        # active uv_layer
        uv_layer = cls._active_uv_layer(bm)
        # for all selected uv-points
        selected_uv_points = cls._uv_points(faces_list=bm.faces, uv_layer=uv_layer, bm=bm)
        # make back transform
        for uv_point in selected_uv_points:
            wco = uv_point.uv.to_3d()
            wco = barycentric_transform(wco, *dest_triangle_co, *src_triangle_co)
            uv_point.uv = wco.to_2d()
        # save changed data to mesh
        bm.to_mesh(obj.data)
        bm.free()
        # return mode back
        if mode is not None:
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
    def box_map(all_scale_def, x_offset_def, y_offset_def, z_offset_def, x_rot_def, y_rot_def, z_rot_def, tex_aspect,
                obj_mode=None):
        # Multy Sure UV from 1D_Scripts
        #   obj_mode - force setting of the object's mode
        #       None - function was called from Multy Sure UV operator
        #       not None - function was called from another UVTools function, with specified object (edit|object) mode
        obj = bpy.context.active_object
        mesh = obj.data

        # print(all_scale_def, x_offset_def, y_offset_def, z_offset_def, x_rot_def, y_rot_def, z_rot_def, tex_aspect,
        #         obj_mode)

        if obj_mode is not None:
            # new
            is_editmode = (obj_mode == 'EDIT')
        else:
            # original
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
            # if not is_editmode or mesh.polygons[i].select:    # original
            # object mode and come from operator - process all faces
            # edit mode and come from operator - process only selected faces
            # object or edit mode and called as function - process only selected faces
            if ((obj_mode is None) and (not is_editmode or mesh.polygons[i].select)) \
                    or ((obj_mode is not None) and mesh.polygons[i].select):
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
    def _rotation_matrix(src_vector, dest_vector, size=2):
        angle = src_vector.angle(dest_vector)  # rad
        axis = src_vector.cross(dest_vector)
        if axis < 0.0:
            angle = -angle
        return Matrix.Rotation(angle, size, 'Z')

    @staticmethod
    def _translation_matrix(src_vector, dest_vector):
        # create translation matrix from src_vector to dest_vector
        #   src_vector = Vector(x, y)
        #   dest_vector = Vector(x, y)
        translation_vector = dest_vector - src_vector
        return Matrix((
            (1.0, 0.0, translation_vector.x),
            (0.0, 1.0, translation_vector.y),
            (0.0, 0.0, 1.0)
        ))

    @staticmethod
    def _vector3(vector):
        # transform 2d vector Vector(x, y) to 3d vector Vector(x, y, 1.0)
        #   only this type of 3d vector could be used in operations in 2d-space with 3x3 transform matrices
        return Vector((vector.x, vector.y, 1.0))

    @staticmethod
    def _uv_points(faces_list, uv_layer, bm=None):
        # get list of UV points for mesh faces from faces_list
        # [uv_point, uv_point, ...]
        if bm:
            return [_loop[uv_layer] for _face in faces_list for _loop in _face.loops]
        else:
            return [uv_layer.data[loop_index] for _face in faces_list for loop_index in _face.loop_indices]

    @classmethod
    def _uv_edges(cls, face, uv_layer, bm=None):
        # get list of UV edges from uv_points
        uv_points = cls._uv_points(faces_list=[face], uv_layer=uv_layer, bm=bm)
        uv_points.append(uv_points[0])  # last point -> first point edge
        # [[uv_point, uv_point], ...]
        return list(cls._chunks(lst=uv_points, n=2, offset=1))[:-1]

    @staticmethod
    def active_edge(context, obj):
        # get active edge on the mesh
        #   return edge or None
        mode = context.active_object.mode if context.active_object.mode != 'OBJECT' else None
        if mode is not None:
            bpy.ops.object.mode_set(mode='OBJECT')
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.edges.ensure_lookup_table()
        active_edge = None
        if bm.select_history:
            active_edge_bm = next((_edge for _edge in reversed(bm.select_history) \
                                   if isinstance(_edge, bmesh.types.BMEdge)), None)
            active_edge = context.object.data.edges[active_edge_bm.index] if active_edge_bm else None
        bm.free()
        if mode is not None:
            bpy.ops.object.mode_set(mode=mode)
        return active_edge

    @staticmethod
    def selected_faces(context):
        # get selected faces list
        mode = context.active_object.mode if context.active_object.mode != 'OBJECT' else None
        if mode is not None:
            bpy.ops.object.mode_set(mode='OBJECT')
        selected_faces = [_face for _face in context.active_object.data.polygons if _face.select]
        # return mode back
        if mode is not None:
            bpy.ops.object.mode_set(mode=mode)
        return selected_faces

    @staticmethod
    def active_face(context):
        # get active face of the object
        mode = context.active_object.mode if context.active_object.mode != 'OBJECT' else None
        if mode is not None:
            bpy.ops.object.mode_set(mode='OBJECT')
        active_face_index = context.active_object.data.polygons.active
        active_face = context.active_object.data.polygons[active_face_index] if active_face_index != -1 else None
        if mode is not None:
            bpy.ops.object.mode_set(mode=mode)
        return active_face

    @staticmethod
    def _chunks(lst, n, offset=0):
        for i in range(0, len(lst), n - offset):
            yield lst[i:i + n]

    @classmethod
    def _rotate_uv_around_point(cls, center_of_rotation, point_to_rotate, dest_vector, uv_points):
        # rotate point_to_rotate around center_of_rotation on angle got by dest_vector and apply this rotation to
        #       all of uv_points
        #   center_of_rotation = Vector(x, y)
        #   point_to_rotate = Vector(x, y)
        #   dest_vector = Vector(x, y)
        #   uv_points = [point, ...]
        # translation matrix to center of rotation
        matrix_transl = cls._translation_matrix(
            dest_vector=center_of_rotation.uv,
            src_vector=Vector((0.0, 0.0))
        )
        # inverted translation matrix
        matrix_transl_inv = matrix_transl.copy()
        matrix_transl_inv.invert()
        # rotation matrix to rotate point_to_rotate around center_of_rotation
        matrix_rot = cls._rotation_matrix(
            src_vector=center_of_rotation.uv - point_to_rotate.uv,
            dest_vector=dest_vector,
            size=3
        )
        for point in uv_points:
            # move to (0.0, 0.0)
            point.uv = (matrix_transl_inv * cls._vector3(point.uv)).to_2d()
            # rotate
            point.uv = (matrix_rot * cls._vector3(point.uv)).to_2d()
            # move back to (rotation_center.x, rotation_center.y)
            point.uv = (matrix_transl * cls._vector3(point.uv)).to_2d()

    @classmethod
    def _longest_uv_edge(cls, bm_face, uv_layer, bm=None, offset=0):
        # get longest edge on the uv face (uv loop) of the bmesh face
        # [[uv_point, uv_point], ...]
        uv_edges = cls._uv_edges(face=bm_face, uv_layer=uv_layer, bm=bm)
        # sort by edges length
        uv_edges_sorted = sorted(
            uv_edges,
            key=lambda _uv_edge: (_uv_edge[1].uv - _uv_edge[0].uv).length,
            reverse=True
        )
        # return uv points of the longest uv edge
        return uv_edges_sorted[0 + offset]  # [uv_point, uv_point]

    @classmethod
    def _first_longest_uv_edge(cls, face, uv_layer, bm=None, offset=0):
        # get first longest uv-edge on the uv-face (uv loop) of the face
        # [[uv_point, uv_point], ...]
        uv_edges = cls._uv_edges(face=face, uv_layer=uv_layer, bm=bm)
        # sort by edges length
        longest_edge = uv_edges[0]
        edge_length = (longest_edge[1].uv - longest_edge[0].uv).length
        edge_idx = 0
        for _i, edge in enumerate(uv_edges[1:]):
            _edge_length = (edge[1].uv - edge[0].uv).length
            if _edge_length > edge_length:
                longest_edge = edge
                edge_length = _edge_length
                edge_idx = _i + 1
        # return uv points of the longest uv edge
        return longest_edge, edge_idx, edge_length

    @staticmethod
    def _deselect_all_bm(bm):
        # remove all selection in bmesh
        for face in bm.faces:
            face.select = False
        for edge in bm.edges:
            edge.select = False
        for vertex in bm.verts:
            vertex.select = False

    @staticmethod
    def _bbox_uv(uv_points):
        x, y = zip(*(_p.uv for _p in uv_points))
        return {
            'min_x': min(x),
            'min_y': min(y),
            'max_x': max(x),
            'max_y': max(y),
        }

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
            op.rotate_threshold = context.window_manager.uv_tools_1d_prop_retexel_rotate_threshold
            row = box.row()
            row.prop(
                data=context.window_manager,
                property='uv_tools_1d_prop_retexel_rotate_threshold'
            )
            # Scan UV Transform
            box = layout.box()
            op = box.operator(
                operator='uvtools.scan_transform',
                icon='MOD_UVPROJECT'
            )
            op.max_value = context.window_manager.uv_tools_1d_prop_scan_transform_max_value
            op.select_only = context.window_manager.uv_tools_1d_prop_scan_transform_select_only
            op.only_selection = context.window_manager.uv_tools_1d_prop_scan_transform_only_selection
            op.check_translation = context.window_manager.uv_tools_1d_prop_scan_transform_translation
            op.check_rotation = context.window_manager.uv_tools_1d_prop_scan_transform_rotation
            op.check_scale = context.window_manager.uv_tools_1d_prop_scan_transform_scale
            op.threshold = context.window_manager.uv_tools_1d_prop_scan_transform_threshold
            box.prop(
                data=context.window_manager,
                property='uv_tools_1d_prop_scan_transform_max_value'
            )
            box.prop(
                data=context.window_manager,
                property='uv_tools_1d_prop_scan_transform_select_only'
            )
            box.prop(
                data=context.window_manager,
                property='uv_tools_1d_prop_scan_transform_only_selection'
            )
            box.prop(
                data=context.window_manager,
                property='uv_tools_1d_prop_scan_transform_threshold'
            )
            box.prop(
                data=context.window_manager,
                property='uv_tools_1d_prop_scan_transform_translation'
            )
            box.prop(
                data=context.window_manager,
                property='uv_tools_1d_prop_scan_transform_rotation'
            )
            box.prop(
                data=context.window_manager,
                property='uv_tools_1d_prop_scan_transform_scale'
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
            # Pack to Custom Tile
            box.operator(
                operator='uvtools.store_custom_tile',
                icon='MOD_LATTICE'
            )
            box.operator(
                operator='uvtools.pack_to_custom_tile',
                icon='MOD_BEVEL'
            )
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
            # Quad Level
            box = layout.box()
            op = box.operator(
                operator='uvtools.quad_level',
                icon='NLA'
            )
            op.pin_level = context.window_manager.uv_tools_1d_prop_quad_level_pin
            box.prop(
                data=context.window_manager,
                property='uv_tools_1d_prop_quad_level_pin'
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

    rotate_threshold = FloatProperty(
        name='Rotation threshold',
        default=15.0,
        max=90.0,
        min=0.0,
        precision=2
    )

    def execute(self, context):
        if context.tool_settings.mesh_select_mode[:] == (False, True, False):
            # edge selection mode in 3D_VIEW area
            active_edge = UVTools.active_edge(context=context, obj=context.active_object)
            if active_edge:
                UVTools.texel_scale_edge(
                    context=context,
                    op=self
                )
        elif context.tool_settings.mesh_select_mode[:] == (False, False, True):
            # face selection mode in 3D_VIEW area
            selected_faces_idxs = [_face.index for _face in UVTools.selected_faces(context=context)]
            active_face = UVTools.active_face(context=context)
            if active_face:
                # only single active face and no other selection
                # use rotation only if angle between vertical and active face normal less than threshold
                use_rotation = True if round(degrees(Vector((0.0, 0.0, 1.0)).angle(active_face.normal)), 2) \
                                       < self.rotate_threshold else False
                if len(selected_faces_idxs) == 0 or selected_faces_idxs == [active_face.index,]:
                    UVTools.texel_scale_face_only_active(
                        context=context,
                        op=self
                    )
                # active face + selection
                elif len(selected_faces_idxs) > 0:
                    UVTools.texel_scale_face(
                        context=context,
                        op=self,
                        vertical_threshold=self.rotate_threshold
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
        for ob in context.selected_objects:
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

        # all_scale_def = self.size
        # tex_aspect = self.texaspect
        #
        # x_offset_def = self.offset[0]
        # y_offset_def = self.offset[1]
        # z_offset_def = self.offset[2]
        # x_rot_def = self.rot[0]
        # y_rot_def = self.rot[1]
        # z_rot_def = self.rot[2]
        # xoffset_def = self.xoffset
        # yoffset_def = self.yoffset

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

class UVTools_OT_quad_level(Operator):
    bl_idname = 'uvtools.quad_level'
    bl_label = 'Quad Level'
    bl_options = {'REGISTER', 'UNDO'}

    pin_level = BoolProperty(
        # Pin longest edge after rotation
        name='Pin Level',
        default=False
    )

    def execute(self, context):
        UVTools.quad_level(
            context=context,
            pin_level=self.pin_level
        )
        return {'FINISHED'}

class UVTools_OT_scan_transform(Operator):
    bl_idname = 'uvtools.scan_transform'
    bl_label = 'Scan UV Transform'
    bl_options = {'REGISTER', 'UNDO'}

    max_value = FloatProperty(
        # Max value for moving faces
        name='Max Dist',
        default=99.0
    )
    select_only = BoolProperty(
        # Only select, don't move
        name='Select Only',
        default=False
    )
    only_selection = BoolProperty(
        # Process only selected faces
        name='Only Selection',
        default=False
    )
    threshold = FloatProperty(
        # Threshold
        name='Threshold',
        default=0.06,
        min=0.0
    )
    check_translation = BoolProperty(
        # Check translation
        name='Location',
        default=False
    )
    check_rotation = BoolProperty(
        # Check rotation
        name='Rotation',
        default=False
    )
    check_scale = BoolProperty(
        # Check scale
        name='Scale',
        default=False
    )

    def execute(self, context):
        UVTools.scan_transform(
            context=context,
            max_value=self.max_value,
            select_only=self.select_only,
            process_only_selection=self.only_selection,
            check_translation=self.check_translation,
            check_rotation=self.check_rotation,
            check_scale=self.check_scale,
            check_threshold=self.threshold
        )
        return {'FINISHED'}

class UVTools_OT_store_custom_tile(Operator):
    bl_idname = 'uvtools.store_custom_tile'
    bl_label = 'Store Custom Tile'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        UVTools.store_custom_tile(
            context=context,
            op=self
        )
        return {'FINISHED'}

class UVTools_OT_pack_to_custom_tile(Operator):
    bl_idname = 'uvtools.pack_to_custom_tile'
    bl_label = 'Pack to Custom Tile'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        UVTools.pack_to_custom_tile(
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
    WindowManager.uv_tools_1d_prop_retexel_rotate_threshold = FloatProperty(
        name='Rotation threshold',
        default=15.0,
        max=90.0,
        min=0.0,
        precision=2
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
    WindowManager.uv_tools_1d_prop_quad_level_pin = BoolProperty(
        # Pin edge for the Quad Level tool
        name='Pin Level',
        default=False
    )
    WindowManager.uv_tools_1d_prop_scan_transform_max_value = FloatProperty(
        # Max value for moving faces
        name='Max Dist',
        default=99.0
    )
    WindowManager.uv_tools_1d_prop_scan_transform_select_only = BoolProperty(
        # Only select faces, don't move
        name='Select Only',
        default=False
    )
    WindowManager.uv_tools_1d_prop_scan_transform_only_selection = BoolProperty(
        # Process only selected faces
        name='Check Only Selection',
        default=False
    )
    WindowManager.uv_tools_1d_prop_scan_transform_threshold = FloatProperty(
        # Threshold
        name='Threshold',
        default=0.06,
        min=0.0
    )
    WindowManager.uv_tools_1d_prop_scan_transform_translation = BoolProperty(
        # Check translation
        name='Location',
        default=False
    )
    WindowManager.uv_tools_1d_prop_scan_transform_rotation = BoolProperty(
        # Check rotation
        name='Rotation',
        default=False
    )
    WindowManager.uv_tools_1d_prop_scan_transform_scale = BoolProperty(
        # Check scale
        name='Scale',
        default=False
    )
    WindowManager.uv_tools_1d_prop_pack_to_c_tile_lb = FloatVectorProperty(
        name='Left-bottom angle of packed area',
        default=(0.0, 0.0),
        size=2
    )
    WindowManager.uv_tools_1d_prop_pack_to_c_tile_rt = FloatVectorProperty(
        name='Right-top angle of packed area',
        default=(0.0, 0.0),
        size=2
    )
    register_class(UVTools_OT_uvpack)
    register_class(UVTools_OT_fit_to_tile)
    register_class(UVTools_OT_texel_scale)
    register_class(UVTools_OT_store_diagonal)
    register_class(UVTools_OT_select_uvcut_tile)
    register_class(UVTools_OT_uvcut_tile)
    register_class(UVTools_PaObjMultySureUV)
    register_class(UVTools_SureUVWOperator)
    register_class(UVTools_OT_quad_level)
    register_class(UVTools_OT_scan_transform)
    register_class(UVTools_OT_store_custom_tile)
    register_class(UVTools_OT_pack_to_custom_tile)
    if ui:
        register_class(UVTools_PT_panel)
        register_class(UVTools_PT_panel_Viewport)


def unregister(ui=True):
    if ui:
        unregister_class(UVTools_PT_panel_Viewport)
        unregister_class(UVTools_PT_panel)
    # butch clean
    unregister_class(UVTools_OT_pack_to_custom_tile)
    unregister_class(UVTools_OT_store_custom_tile)
    unregister_class(UVTools_OT_scan_transform)
    unregister_class(UVTools_OT_quad_level)
    unregister_class(UVTools_SureUVWOperator)
    unregister_class(UVTools_PaObjMultySureUV)
    unregister_class(UVTools_OT_uvcut_tile)
    unregister_class(UVTools_OT_select_uvcut_tile)
    unregister_class(UVTools_OT_store_diagonal)
    unregister_class(UVTools_OT_texel_scale)
    unregister_class(UVTools_OT_fit_to_tile)
    unregister_class(UVTools_OT_uvpack)
    del WindowManager.uv_tools_1d_prop_pack_to_c_tile_rt
    del WindowManager.uv_tools_1d_prop_pack_to_c_tile_lb
    del WindowManager.uv_tools_1d_prop_scan_transform_scale
    del WindowManager.uv_tools_1d_prop_scan_transform_rotation
    del WindowManager.uv_tools_1d_prop_scan_transform_translation
    del WindowManager.uv_tools_1d_prop_scan_transform_threshold
    del WindowManager.uv_tools_1d_prop_scan_transform_select_only
    del WindowManager.uv_tools_1d_prop_scan_transform_only_selection
    del WindowManager.uv_tools_1d_prop_scan_transform_max_value
    del WindowManager.uv_tools_1d_prop_quad_level_pin
    del WindowManager.uv_tools_1d_prop_disp_omsureuv
    del WindowManager.uv_tools_1d_prop_omsureuv_all_scale_def
    del WindowManager.uv_tools_1d_prop_omsureuv_rot
    del WindowManager.uv_tools_1d_prop_omsureuv_offset
    del WindowManager.uv_tools_1d_prop_retexel_rotate_threshold
    del WindowManager.uv_tools_1d_prop_fit_to_tile_add_scale
    del WindowManager.uv_tools_1d_prop_fit_to_tile_v2
    del WindowManager.uv_tools_1d_prop_fit_to_tile_v1
    del WindowManager.uv_tools_1d_prop_fit_to_tile_v0
    del Scene.uv_tools_1d_prop_pack_to_cursor


if __name__ == "__main__":
    register()
