"""Run in Blender after opening episode14_alignment.blend to install controls."""
import bpy
from bpy.props import StringProperty
from mathutils import Matrix
import json
import math
from pathlib import Path


def inputs():
    return json.loads(bpy.data.texts['alignment_inputs.json'].as_string())


def master(name):
    return bpy.data.objects['EDIT_MOUNT__' + name]


def rigid(matrix):
    values = [v for row in matrix for v in row]
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Nonfinite transform')
    rotation = matrix.to_3x3()
    error = rotation.transposed() @ rotation - Matrix.Identity(3)
    if max(abs(v) for row in error for v in row) > 1e-5 or abs(rotation.determinant()-1) > 1e-5:
        raise ValueError('Only rigid transforms are supported; scale must remain 1')
    if max(abs(matrix[3][i] - (1 if i == 3 else 0)) for i in range(4)) > 1e-7:
        raise ValueError('Invalid homogeneous row')
    return matrix


def export_mounts(path=None):
    document = inputs()
    geometry = document['baseline_geometry']
    for row, tool in zip(geometry['tools'], document['tools']):
        obj = master(tool['name'])
        if obj.mode != 'OBJECT':
            raise ValueError('Exit Edit Mode before exporting')
        if obj.constraints or obj.modifiers:
            raise ValueError('Do not add constraints or modifiers to the editable mounting object')
        if len(obj.data.vertices) != len(tool['vertices_link_m']):
            raise ValueError('Mesh editing is not supported; edit object position and rotation only')
        for vertex, original in zip(obj.data.vertices, tool['vertices_link_m']):
            if max(abs(vertex.co[i]-original[i]) for i in range(3)) > 1e-6:
                raise ValueError('Mesh vertices changed; export would not preserve those edits')
        if document.get('workspace_layout') == 'recorded_scene':
            recorded = next(t for t in document['frames'][0]['tools'] if t['name'] == tool['name'])
            expected_parent = Matrix(recorded['mocap_from_marker'])
            if obj.parent is None or max(abs(obj.parent.matrix_world[r][c]-expected_parent[r][c])
                                         for r in range(4) for c in range(4)) > 1e-5:
                raise ValueError('Recorded marker parent was changed; only edit child mounting objects')
        matrix = rigid(obj.matrix_basis.copy())
        baseline = Matrix(tool['baseline_marker_from_mesh'])
        correction = matrix @ baseline.inverted()
        row['marker_from_mesh'] = [list(r) for r in matrix]
        row['marker_from_collider'] = [list(r) for r in correction @ Matrix(row['marker_from_collider'])]
    geometry['manual_alignment'] = {
        'status': 'manual_candidate_requires_validation',
        'baseline_geometry_sha256': document['baseline_geometry_sha256'],
        'units': 'metres', 'recorded_trajectories_unchanged': True,
        'matrix_convention': document['matrix_convention'],
        'preview_raw_frames': [f['raw'] for f in document['frames']],
        'source_blend': bpy.data.filepath,
    }
    path = Path(path) if path is not None else Path(bpy.data.filepath).with_name('manual_tool_geometry.json')
    # Number repeated exports instead of replacing a previous manual adjustment.
    if path.exists():
        stem = path.stem
        for i in range(1, 10000):
            candidate = path.with_name(f'{stem}_{i:03d}{path.suffix}')
            if not candidate.exists():
                path = candidate
                break
        else:
            raise FileExistsError('No unused export filename')
    path.write_text(json.dumps(geometry, indent=2)+'\n')
    bpy.context.scene['last_export'] = str(path)
    return path


def reset_mounts():
    for tool in inputs()['tools']:
        obj = master(tool['name'])
        obj.matrix_basis = Matrix(tool['baseline_marker_from_mesh'])
    bpy.context.view_layer.update()


class EP14_OT_export(bpy.types.Operator):
    bl_idname = 'ep14.export_mounts'
    bl_label = 'Export candidate JSON'
    def execute(self, context):
        try:
            path = export_mounts()
        except (ValueError, OSError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        self.report({'INFO'}, str(path))
        return {'FINISHED'}


class EP14_OT_reset(bpy.types.Operator):
    bl_idname = 'ep14.reset_mounts'
    bl_label = 'Reset both to baseline'
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        reset_mounts()
        return {'FINISHED'}


class EP14_OT_select(bpy.types.Operator):
    bl_idname = 'ep14.select_mount'
    bl_label = 'Select editable mounting'
    tool: StringProperty()
    def execute(self, context):
        bpy.ops.object.select_all(action='DESELECT')
        obj = master(self.tool)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return {'FINISHED'}


class EP14_PT_alignment(bpy.types.Panel):
    bl_label = 'Episode14 shared mounting'
    bl_idname = 'EP14_PT_alignment'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Episode14'
    def draw(self, context):
        layout = self.layout
        layout.label(text='Metres; Object Mode only. G/R to adjust.')
        layout.label(text='One mounting per tool drives every frame.')
        for tool in inputs()['tools']:
            box = layout.box()
            box.operator('ep14.select_mount', text='Select '+tool['name']).tool = tool['name']
            obj = master(tool['name'])
            box.prop(obj, 'location', text='Position (m)')
            box.prop(obj, 'rotation_euler', text='Rotation')
        layout.operator('ep14.reset_mounts')
        layout.operator('ep14.export_mounts')
        layout.label(text='Save .blend with Ctrl-S. Export does not apply.')


CLASSES = (EP14_OT_export, EP14_OT_reset, EP14_OT_select, EP14_PT_alignment)

def register():
    for cls in CLASSES:
        previous = getattr(bpy.types, cls.__name__, None)
        if previous is not None:
            bpy.utils.unregister_class(previous)
        bpy.utils.register_class(cls)


if __name__ == '__main__':
    register()
