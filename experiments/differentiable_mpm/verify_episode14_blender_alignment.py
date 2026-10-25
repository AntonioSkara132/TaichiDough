"""Check the saved Blender project's export and reset operations without saving edits."""
import bpy
import json
from mathutils import Matrix
from pathlib import Path
import runpy

root = Path(bpy.data.filepath).parent
namespace = runpy.run_path(str(root/'episode14_blender_controls.py'))
document = namespace['inputs']()
obj = namespace['master'](document['tools'][0]['name'])
baseline = Matrix(document['tools'][0]['baseline_marker_from_mesh'])
delta = Matrix.Translation((.012, -.004, .002)) @ Matrix.Rotation(.1, 4, 'Y')
obj.matrix_basis = delta @ baseline
bpy.context.view_layer.update()
folder = root/'verification'
folder.mkdir(exist_ok=True)
exported_path = namespace['export_mounts'](folder/'perturbed_roundtrip.json')
exported = json.loads(exported_path.read_text())
errors = []
for key in ('marker_from_mesh', 'marker_from_collider'):
    expected = delta @ Matrix(document['baseline_geometry']['tools'][0][key])
    actual = Matrix(exported['tools'][0][key])
    errors.append(max(abs(actual[r][c]-expected[r][c]) for r in range(4) for c in range(4)))
assert max(errors) < 1e-5, errors
obj.scale.x *= 1.02
try:
    namespace['export_mounts'](folder/'INVALID_SCALE.json')
except ValueError:
    pass
else:
    raise AssertionError('Nonrigid scaling was accepted')
namespace['reset_mounts']()
reset_error = max(abs(obj.matrix_basis[r][c]-baseline[r][c]) for r in range(4) for c in range(4))
assert reset_error < 1e-5
original = obj.data.vertices[0].co.copy()
obj.data.vertices[0].co.x += .001
try:
    namespace['export_mounts'](folder/'INVALID_VERTEX_EDIT.json')
except ValueError:
    pass
else:
    raise AssertionError('Vertex editing was accepted')
obj.data.vertices[0].co = original
result = {'status': 'passed', 'reopened_saved_blend': True,
          'perturbed_mesh_and_collider_export_max_error': max(errors),
          'reset_max_error': reset_error, 'nonrigid_scale_rejected': True,
          'vertex_edit_rejected': True, 'saved_blend_unchanged': True}
(folder/'export_validation.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
