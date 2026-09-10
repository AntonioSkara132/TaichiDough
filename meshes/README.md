These binary STL files were copied on 2026-09-08 from
`ros2_ws/src/ur5_gen3_dual_setup/ur5/ur_dual_bringup/meshes/`:

- `ur_spathla.stl`: UR5e visual tool.
- `gen3_spathla.stl`: Kinova visual tool.

Their coordinates are in millimetres. The simulation applies scale 0.001 and
the visual origins in `rf_lab_setup/urdf/rf_lab.urdf.xacro`. The test fixture in
`test/fixtures/rf_lab_tool_visuals.urdf` records those visual definitions.

The bundled files allow standalone rendering without a ROS installation. When
the ROS workspace is sourced, the simulation prefers its installed meshes.

`--tool-collision sdf` voxelizes these meshes for approximate contact. The URDF visual
transforms above are applied before voxelization, so each SDF is in its tool-link frame.
For captured replay, `marker_from_mesh` maps the recorded marker pose into that frame;
it is identity only when those frames already coincide and must not include the visual
origin or rotation. `--tool-collision box` remains the default and instead uses the
separate `marker_from_collider` proxy transform.
