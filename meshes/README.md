These binary STL files were copied on 2026-09-08 from
`ros2_ws/src/ur5_gen3_dual_setup/ur5/ur_dual_bringup/meshes/`:

- `ur_spathla.stl`: UR5e visual tool.
- `gen3_spathla.stl`: Kinova visual tool.

Their coordinates are in millimetres. The simulation applies scale 0.001 and
the visual origins in `rf_lab_setup/urdf/rf_lab.urdf.xacro`. The test fixture in
`test/fixtures/rf_lab_tool_visuals.urdf` records those visual definitions.

The bundled files allow standalone rendering without a ROS installation. When
the ROS workspace is sourced, the simulation prefers its installed meshes.
