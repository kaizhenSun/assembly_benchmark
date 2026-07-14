# Camera Sensing

Every assembly scene exposes two cameras:

- `front_left_work_camera` is the fixed RGB overview used by the scripted assembly recorder.
- `head_camera` is an RGB-D and semantic camera attached to the R1 Pro `zed_link` at the midpoint of its physical
  stereo pair.

## R1 Pro Head Camera

The head-camera profile follows the R1 Pro generation represented by the checked-in robot asset:

| Parameter | Value |
| --- | --- |
| Native output | 1920 x 1080 at 30 FPS |
| Published field of view | 118 degrees horizontal x 62 degrees vertical |
| Stereo baseline | 0.120 m |
| Simulated depth range | 0.1 m to 20 m |
| Segmentation output | Integer semantic IDs, one channel |
| Scene key | `head_camera` |
| Parent link | `zed_link` |

Galaxea does not publish a depth range or distortion calibration for this R1 Pro head-camera generation. The simulation
therefore provides ideal RGB-aligned Z depth over the configured 0.1 m to 20 m clipping range. Pixels outside that range
are zero.

The Isaac Lab camera uses a square-pixel pinhole approximation. It matches the published 118-degree horizontal field of
view; at 1920 x 1080 this implies an approximately 86.22-degree vertical field of view. The published 62-degree vertical
value remains hardware metadata rather than an invented distortion model.

## Reading Camera Data

Launch tasks with `--enable_cameras`, then read the device tensors from the scene:

```python
camera = env.unwrapped.scene["head_camera"]
rgb = camera.data.output["rgb"]
depth = camera.data.output["distance_to_image_plane"]
segmentation = camera.data.output["semantic_segmentation"]
id_to_labels = camera.data.info["semantic_segmentation"]["idToLabels"]
intrinsics = camera.data.intrinsic_matrices
```

For `N` environments, RGB has shape `[N, 1080, 1920, 3]` with `torch.uint8` values. Depth has shape
`[N, 1080, 1920, 1]`, uses `torch.float32`, and is expressed in meters along the camera optical Z axis. Camera pose data
is updated with the moving head link. Semantic segmentation has shape `[N, 1080, 1920, 1]` and uses `torch.int32`.

## Semantic Labels

Segmentation uses Replicator `class` semantics. The shared scene labels are `robot`, `ground`, and `lab_table`. Each
assembly part uses its unique scene key as its label. Parts that share the same geometry or category therefore remain
separate, for example `desk_leg1`, `desk_leg2`, `chair_nut1`, and `chair_nut2` receive different semantic IDs.

Semantic integer values are assigned at runtime and must not be hard-coded. Use `idToLabels` to resolve each value to
its stable label. Corresponding logical parts in cloned parallel environments share a semantic class; unlabeled pixels
are background. Instance segmentation is not enabled.

The native RGB-D buffers and render targets are expensive for multiple parallel environments. Use one environment for
camera diagnostics, or override `cfg.scene.head_camera.width` and `height` before environment construction when native
resolution is not required.

## Recording the Head View

The scripted assembly tool records RGB from any named scene camera. Select the head camera explicitly:

```bash
python scripts/tools/run_r1_pro_one_leg_scripted_assembly.py \
  --num_envs 1 --device cuda:0 --enable_cameras \
  --record_camera --camera_name head_camera
```

This records RGB only. Depth and semantic segmentation remain available through the scene sensor data interface.
