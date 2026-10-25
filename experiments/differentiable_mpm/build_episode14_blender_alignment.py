"""Build the manual mounting workspace using Blender's bundled Python."""
import bpy
import json
from mathutils import Matrix, Quaternion
from pathlib import Path
import runpy
import shutil
import sys

ROOT = Path(sys.argv[sys.argv.index('--')+1]).resolve()
SOURCE = Path(__file__).resolve().parent
DOCUMENT = json.loads((ROOT/'alignment_inputs.json').read_text())
COLORS = [(0.14, 0.48, 0.92, 1), (0.92, 0.40, 0.20, 1)]


def material(name, color):
    value = bpy.data.materials.new(name)
    value.diffuse_color = color
    return value


def empty(name, position, parent=None):
    obj = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(obj)
    obj.location = position
    obj.parent = parent
    obj.empty_display_type = 'ARROWS'
    obj.empty_display_size = .065
    obj.show_in_front = True
    obj.hide_select = True
    return obj


def text(body, position, size=.018):
    data = bpy.data.curves.new(body, 'FONT')
    data.body = body
    data.size = size
    obj = bpy.data.objects.new(body, data)
    bpy.context.collection.objects.link(obj)
    obj.location = position
    obj.hide_select = True
    return obj


def point_cloud(name, points, parent):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(points, [], [])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.parent = parent
    obj.hide_select = True
    group = bpy.data.node_groups.new(name+'_points', 'GeometryNodeTree')
    if hasattr(group, 'interface'):
        group.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
        group.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    else:
        group.inputs.new('NodeSocketGeometry', 'Geometry')
        group.outputs.new('NodeSocketGeometry', 'Geometry')
    nodes, links = group.nodes, group.links
    source = nodes.new('NodeGroupInput')
    output = nodes.new('NodeGroupOutput')
    spheres = nodes.new('GeometryNodeMeshIcoSphere')
    spheres.inputs['Radius'].default_value = .0008
    spheres.inputs['Subdivisions'].default_value = 1
    set_material = nodes.new('GeometryNodeSetMaterial')
    set_material.inputs['Material'].default_value = bpy.data.materials['Observed points']
    instances = nodes.new('GeometryNodeInstanceOnPoints')
    links.new(source.outputs['Geometry'], instances.inputs['Points'])
    links.new(spheres.outputs['Mesh'], set_material.inputs['Geometry'])
    links.new(set_material.outputs['Geometry'], instances.inputs['Instance'])
    links.new(instances.outputs['Instances'], output.inputs['Geometry'])
    modifier = obj.modifiers.new('Visible RGB-D points', 'NODES')
    modifier.node_group = group
    return obj


def image_reference(path, position):
    image = bpy.data.images.load(str(path), check_existing=True)
    image.pack()
    # Empty image references stay visible in solid mode without render setup.
    obj = bpy.data.objects.new(path.stem, None)
    bpy.context.collection.objects.link(obj)
    obj.empty_display_type = 'IMAGE'
    obj.data = image
    obj.empty_display_size = .32
    obj.empty_image_depth = 'FRONT'
    obj.location = position
    obj.hide_select = True
    return obj


def main():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 1
    scene.unit_settings.length_unit = 'METERS'
    material('Observed points', (.76, .96, .40, 1))
    bpy.data.texts.load(str(ROOT/'alignment_inputs.json'))
    controls_path = ROOT/'episode14_blender_controls.py'
    shutil.copyfile(SOURCE/'episode14_blender_controls.py', controls_path)
    bpy.data.texts.load(str(controls_path))
    masters, previews = [], []
    single = len(DOCUMENT['frames']) == 1
    for j, tool in enumerate(DOCUMENT['tools']):
        name = tool['name']
        row_y = -j*.72
        parent = empty('DISPLAY_ONLY__'+name, (0, row_y, 0))
        mesh = bpy.data.meshes.new(name+'_visual_transformed_metres')
        mesh.from_pydata(tool['vertices_link_m'], [], tool['triangles'])
        mesh.update()
        obj = bpy.data.objects.new('EDIT_MOUNT__'+name, mesh)
        bpy.context.collection.objects.link(obj)
        obj.parent = parent
        obj.rotation_mode = 'XYZ'
        obj.matrix_basis = Matrix(tool['baseline_marker_from_mesh'])
        obj.lock_scale = (True, True, True)
        obj.display_type = 'WIRE'
        obj.show_wire = True
        obj.color = COLORS[j]
        obj.data.materials.append(material(name, COLORS[j]))
        masters.append(obj)
        text('EDIT '+name, (-.16, row_y+.15, 0))
        text('G/R: position / rotation', (-.16, row_y-.22, 0), .013)
        text('Marker axes: X red, Y green, Z blue', (-.16, row_y-.25, 0), .011)
        if single:
            frame = DOCUMENT['frames'][0]
            observation = frame['tools'][j]
            point_cloud(f'OBSERVED__{name}__pc{frame["raw"]}', observation['points_marker_m'], parent)
            text(f'pc {frame["raw"]} | {len(observation["points_marker_m"])} measured points',
                 (-.16, row_y-.28, 0), .012)
            continue
        for i, frame in enumerate(DOCUMENT['frames']):
            x = (i+1)*.38
            observation = frame['tools'][j]
            marker = empty(f'MARKER_AXES__{name}__pc{frame["raw"]}', (x, row_y, 0))
            point_cloud(f'OBSERVED__{name}__pc{frame["raw"]}', observation['points_marker_m'], marker)
            preview = bpy.data.objects.new(f'PREVIEW_ONLY__{name}__pc{frame["raw"]}', mesh)
            bpy.context.collection.objects.link(preview)
            preview.parent = marker
            preview.display_type = 'WIRE'
            preview.color = COLORS[j]
            preview.hide_select = True
            constraint = preview.constraints.new('COPY_TRANSFORMS')
            constraint.target = obj
            constraint.target_space = 'LOCAL'
            constraint.owner_space = 'LOCAL'
            previews.append((preview, marker, obj))
            count = len(observation['points_marker_m'])
            text(f'pc {frame["raw"]} | {count} points', (x-.15, row_y-.22, 0), .013)
            image_reference(ROOT/frame['rgb_file'], (x, row_y+.31, 0))
    if single:
        image = image_reference(ROOT/DOCUMENT['frames'][0]['rgb_file'], (.53, -.31, 0))
        image.empty_display_size = .72
        text(f'EPISODE14 | frame {DOCUMENT["frames"][0]["raw"]} | manual tool mounting', (-.16, .30, 0), .022)
        text('Wire: simulator mesh | green: measured points', (-.16, .25, 0), .016)
        text('Original RGB reference', (.23, .03, 0), .016)
        text('Edit the two wire meshes. G/R to move/rotate. Export via Episode14 sidebar.', (-.16, -1.06, 0), .014)
    else:
        text('EPISODE14 | shared manual mounting | wire: simulator STL | green: measured tool points',
             (-.16, .52, 0), .025)
        text('Edit only the two left objects. All frame previews share each tool adjustment. RGB references are packed.',
             (-.16, -.99, 0), .017)
        text('No per-frame fit. No automatic dataset update. Rejected ICP is NOT loaded.', (-.16, -1.03, 0), .017)
    bpy.context.view_layer.update()
    # Check shared local-space constraints numerically, including a temporary edit.
    errors = [max(abs(obj.matrix_world[r][c] - (obj.parent.matrix_world @ obj.matrix_basis)[r][c])
                  for r in range(4) for c in range(4)) for obj in masters]
    for preview, marker, obj in previews:
        actual = preview.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world
        expected = marker.matrix_world @ obj.matrix_basis
        errors.append(max(abs(actual[r][c]-expected[r][c]) for r in range(4) for c in range(4)))
    baseline = masters[0].matrix_basis.copy()
    masters[0].location.x += .01
    bpy.context.view_layer.update()
    for preview, marker, obj in previews:
        actual = preview.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world
        expected = marker.matrix_world @ obj.matrix_basis
        errors.append(max(abs(actual[r][c]-expected[r][c]) for r in range(4) for c in range(4)))
    masters[0].matrix_basis = baseline
    bpy.context.view_layer.update()
    if max(errors) > 1e-5:
        raise ValueError(f'Preview composition failed: {max(errors)}')
    namespace = runpy.run_path(str(controls_path))
    namespace['register']()
    # Baseline export verifies both mesh and collider matrices after float32 Blender conversion.
    roundtrip_path = namespace['export_mounts'](ROOT/'baseline_roundtrip.json')
    roundtrip = json.loads(roundtrip_path.read_text())
    roundtrip_errors = []
    for actual, expected in zip(roundtrip['tools'], DOCUMENT['baseline_geometry']['tools']):
        for key in ('marker_from_mesh', 'marker_from_collider'):
            roundtrip_errors.append(max(abs(actual[key][r][c]-expected[key][r][c]) for r in range(4) for c in range(4)))
    if max(roundtrip_errors) > 1e-5:
        raise ValueError(f'Matrix roundtrip failed: {max(roundtrip_errors)}')
    bpy.ops.object.select_all(action='DESELECT')
    masters[0].select_set(True)
    bpy.context.view_layer.objects.active = masters[0]
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                space = area.spaces.active
                space.shading.type = 'SOLID'
                space.shading.color_type = 'MATERIAL'
                space.overlay.show_floor = False
                space.clip_start = .0001
                space.clip_end = 100
                space.region_3d.view_rotation = Quaternion((1, 0, 0, 0))
                space.region_3d.view_perspective = 'ORTHO'
                space.region_3d.view_location = (.32, -.35, 0) if single else (1.05, -.25, 0)
                space.region_3d.view_distance = 1.7 if single else 2.9
                space.show_region_ui = True
    scene['instructions'] = 'Run episode14_blender_controls.py text, then N sidebar > Episode14. Edit only EDIT_MOUNT objects.'
    scene['baseline_geometry_sha256'] = DOCUMENT['baseline_geometry_sha256']
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'episode14_alignment.blend'))
    (ROOT/'build_validation.json').write_text(json.dumps({'status': 'passed',
        'max_shared_preview_matrix_error': max(errors), 'max_baseline_export_matrix_error': max(roundtrip_errors),
        'mesh_vertices': [len(m.data.vertices) for m in masters],
        'preview_count': len(previews), 'packed_rgb_count': len([im for im in bpy.data.images if im.packed_file]),
        'units': 'metres', 'blender_version': bpy.app.version_string}, indent=2)+'\n')


if __name__ == '__main__':
    main()
