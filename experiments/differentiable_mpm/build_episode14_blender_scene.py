"""Build solid tools over the full recorded RGB-D scene, without display offsets."""
import bpy
import json
from mathutils import Matrix, Vector
from pathlib import Path
import runpy
import shutil
import sys

ROOT = Path(sys.argv[sys.argv.index('--')+1]).resolve()
SOURCE = Path(__file__).resolve().parent
DOC = json.loads((ROOT/'alignment_inputs.json').read_text())
FRAME = DOC['frames'][0]


def material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = color
    mat.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = .65
    return mat


def full_cloud():
    mesh = bpy.data.meshes.new('Recorded RGB-D points, mocap metres')
    mesh.from_pydata(FRAME['points_mocap_m'], [], [])
    mesh.update()
    colors = mesh.color_attributes.new(name='rgb', type='FLOAT_COLOR', domain='POINT')
    # Recorded camera RGB bytes are sRGB; material emission expects linear values.
    def linear(value):
        v = value/255
        return v/12.92 if v <= .04045 else ((v+.055)/1.055)**2.4
    for entry, rgb in zip(colors.data, FRAME['rgb']):
        entry.color = tuple(linear(v) for v in rgb)+(1,)
    obj = bpy.data.objects.new('FULL_RECORDED_RGBD__table_dough_hands_tools', mesh)
    bpy.context.collection.objects.link(obj)
    obj.hide_select = True
    mat = bpy.data.materials.new('Recorded RGB colors')
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    attr = nodes.new('ShaderNodeAttribute')
    attr.attribute_name = 'rgb'
    emission = nodes.new('ShaderNodeEmission')
    output = nodes.new('ShaderNodeOutputMaterial')
    links.new(attr.outputs['Color'], emission.inputs['Color'])
    links.new(emission.outputs[0], output.inputs['Surface'])
    mesh.materials.append(mat)
    group = bpy.data.node_groups.new('Show measured RGB-D points as colored small spheres', 'GeometryNodeTree')
    if hasattr(group, 'interface'):
        group.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
        group.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    else:
        group.inputs.new('NodeSocketGeometry', 'Geometry')
        group.outputs.new('NodeSocketGeometry', 'Geometry')
    nodes, links = group.nodes, group.links
    inp, out = nodes.new('NodeGroupInput'), nodes.new('NodeGroupOutput')
    ico = nodes.new('GeometryNodeMeshIcoSphere')
    ico.inputs['Radius'].default_value = .0015
    ico.inputs['Subdivisions'].default_value = 1
    inst = nodes.new('GeometryNodeInstanceOnPoints')
    real = nodes.new('GeometryNodeRealizeInstances')
    setmat = nodes.new('GeometryNodeSetMaterial')
    setmat.inputs['Material'].default_value = mat
    links.new(inp.outputs['Geometry'], inst.inputs['Points'])
    links.new(ico.outputs['Mesh'], inst.inputs['Instance'])
    links.new(inst.outputs['Instances'], real.inputs['Geometry'])
    links.new(real.outputs['Geometry'], setmat.inputs['Geometry'])
    links.new(setmat.outputs['Geometry'], out.inputs['Geometry'])
    obj.modifiers.new('RGB-D point display', 'NODES').node_group = group
    return obj


def main():
    if DOC.get('workspace_layout') != 'recorded_scene':
        raise ValueError('Expected full original scene inputs')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 1
    scene.unit_settings.length_unit = 'METERS'
    bpy.data.texts.load(str(ROOT/'alignment_inputs.json'))
    controls_path = ROOT/'episode14_blender_controls.py'
    shutil.copyfile(SOURCE/'episode14_blender_controls.py', controls_path)
    bpy.data.texts.load(str(controls_path))
    cloud = full_cloud()
    masters = []
    colors = [(0.08,.28,.8,1), (.85,.24,.06,1)]
    for j, tool in enumerate(DOC['tools']):
        observation = FRAME['tools'][j]
        parent = bpy.data.objects.new('FIXED_RECORDED_MARKER__'+tool['name'], None)
        bpy.context.collection.objects.link(parent)
        parent.matrix_world = Matrix(observation['mocap_from_marker'])
        parent.empty_display_type = 'ARROWS'
        parent.empty_display_size = .04
        parent.hide_select = True
        parent.lock_location = parent.lock_rotation = parent.lock_scale = (True, True, True)
        mesh = bpy.data.meshes.new(tool['name']+'_actual_simulator_mesh_metres')
        mesh.from_pydata(tool['vertices_link_m'], [], tool['triangles'])
        mesh.update()
        mesh.materials.append(material(tool['name'], colors[j]))
        obj = bpy.data.objects.new('EDIT_MOUNT__'+tool['name'], mesh)
        bpy.context.collection.objects.link(obj)
        obj.parent = parent
        obj.rotation_mode = 'XYZ'
        obj.matrix_basis = Matrix(tool['baseline_marker_from_mesh'])
        obj.lock_scale = (True, True, True)
        obj.display_type = 'SOLID'
        obj.color = colors[j]
        masters.append(obj)
    camera_data = bpy.data.cameras.new('Recorded color camera')
    camera = bpy.data.objects.new('FIXED_RECORDED_RGB_CAMERA', camera_data)
    bpy.context.collection.objects.link(camera)
    camera.matrix_world = Matrix(DOC['camera_from_mocap']).inverted() @ Matrix.Diagonal((1,-1,-1,1))
    camera.hide_select = True
    intrinsic = DOC['camera']
    camera_data.sensor_width = 36
    camera_data.sensor_fit = 'HORIZONTAL'
    camera_data.lens = intrinsic['fx']*36/intrinsic['width']
    camera_data.shift_x = (intrinsic['width']/2-intrinsic['cx'])/intrinsic['width']
    camera_data.shift_y = (intrinsic['cy']-intrinsic['height']/2)/intrinsic['width']
    camera_data.clip_start = .005
    camera_data.clip_end = 20
    image = bpy.data.images.load(str(ROOT/FRAME['rgb_file']))
    image.pack()
    background = camera_data.background_images.new()
    background.image = image
    background.alpha = .25
    camera_data.show_background_images = False
    scene.camera = camera
    scene.render.resolution_x = intrinsic['width']
    scene.render.resolution_y = intrinsic['height']
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_y = intrinsic['fx']/intrinsic['fy']
    scene.render.engine = 'BLENDER_EEVEE'
    scene.view_settings.view_transform = 'Standard'
    scene.world.color = (.15,.15,.15)
    light_data = bpy.data.lights.new('Tool illumination', 'AREA')
    light_data.energy = 150
    light_data.size = 2
    light = bpy.data.objects.new('Tool illumination', light_data)
    bpy.context.collection.objects.link(light)
    light.matrix_world = camera.matrix_world.copy()
    light.hide_select = True
    bpy.context.view_layer.update()
    namespace = runpy.run_path(str(controls_path))
    namespace['register']()
    roundtrip_path = namespace['export_mounts'](ROOT/'baseline_roundtrip.json')
    exported = json.loads(roundtrip_path.read_text())
    matrix_errors = []
    composition_errors = []
    for j, (actual, expected) in enumerate(zip(exported['tools'], DOC['baseline_geometry']['tools'])):
        for key in ('marker_from_mesh', 'marker_from_collider'):
            matrix_errors.append(max(abs(actual[key][r][c]-expected[key][r][c]) for r in range(4) for c in range(4)))
        expected_world = Matrix(FRAME['tools'][j]['mocap_from_marker']) @ Matrix(expected['marker_from_mesh'])
        composition_errors.append(max(abs(masters[j].matrix_world[r][c]-expected_world[r][c]) for r in range(4) for c in range(4)))
    assert max(matrix_errors+composition_errors) < 1e-5
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                space = area.spaces.active
                space.shading.type = 'MATERIAL'
                space.shading.use_scene_world = False
                space.overlay.show_floor = False
                space.clip_start = .001
                space.clip_end = 30
                space.region_3d.view_perspective = 'CAMERA'
                space.region_3d.view_camera_zoom = 10
                space.region_3d.view_distance = .8
                space.region_3d.view_location = sum((obj.matrix_world.translation for obj in masters), Vector())/2
                space.show_region_ui = True
    bpy.ops.object.select_all(action='DESELECT')
    masters[0].select_set(True)
    bpy.context.view_layer.objects.active = masters[0]
    scene['instructions'] = 'Original mocap scene. Solid tool meshes over full RGB-D. N sidebar > Episode14. G/R edits only child mounting.'
    scene['reference_raw_frame'] = FRAME['raw']
    scene['full_cloud_points'] = len(FRAME['points_mocap_m'])
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'episode14_alignment.blend'))
    report = {'status': 'passed', 'coordinate_frame': 'mocap', 'units': 'metres', 'raw_frame': FRAME['raw'],
              'point_count': len(FRAME['points_mocap_m']), 'color_filter_applied': False,
              'display_offsets': False, 'solid_meshes': 2, 'packed_rgb_count': 1,
              'max_baseline_export_matrix_error': max(matrix_errors),
              'max_recorded_pose_composition_error': max(composition_errors), 'blender_version': bpy.app.version_string}
    (ROOT/'build_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    scene.render.filepath = str(ROOT/'scene_preview.png')
    bpy.ops.render.render(write_still=True)


if __name__ == '__main__':
    main()
