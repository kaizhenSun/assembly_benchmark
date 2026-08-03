# Piper + Pika2 Asset Provenance

The robot sources in this directory are copied from the local `agx_arm_sim` checkout used to develop this
integration:

- Source repository: <https://github.com/agilexrobotics/agx_arm_sim>
- Source checkout commit: `f8cd8b147c75d59e14f90fb0646770eefa268ed0`
- Piper arm: `agx_arm_description/agx_arm_urdf/piper/urdf/piper_description.urdf`
- Piper arm meshes: `agx_arm_description/agx_arm_urdf/piper/meshes`
- Pika2 gripper: `agx_arm_description/urdf/pika2_gripper.urdf`
- Pika2 meshes: `agx_arm_description/meshes/pika_gripper_base.dae`, `gripper_left_link.dae`, and
  `gripper_right_link.dae`

The source projects declare the MIT license reproduced in `LICENSE.agx-arm-urdf-MIT.txt`. The reusable USD and
relation-specific fixed-plug USDs under `fixed_plug/` are generated integration assets. Fabrica socket USDs are
stored with their assembly parts under `assets/fabrica/<assembly>/usd/fixed_plug_socket/`. Regenerate them with:

```bash
python scripts/tools/convert_piper_urdf.py --force
python scripts/tools/generate_fabrica_fixedplug_assets.py --assembly beam --overwrite
```

The conversion helpers compose the standalone Piper and Pika2 URDF files directly, so ROS and the `xacro` Python
package are not runtime or regeneration dependencies.
