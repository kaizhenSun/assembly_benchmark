# Fabrica Beam Asset Provenance

The meshes `mesh/beam_part_{0,1,2,3,6}.obj` are derived from the following Fabrica repository files:

- Repository: <https://github.com/yunshengtian/Fabrica>
- Commit: `215a30fe51e59299588a5b5e417a9cb934fb393e`
- Source paths: `assets/fabrica/beam/{0,1,2,3,6}.obj`
- Source Git blob IDs, in the same order: `3d2ee3952a22b7e5e7d91b1e7141dd589d99b4a4`,
  `d61dfee8d01d807f181ed33e2b0ac08efec53f21`, `9e2f09f1c36305e090819b24d01f26a83682260b`,
  `783f1128d74f3a590eefaf5736cc8e8ff7ca3528`, and `6577f770e0002c8144d6bfb04b1099850701fe05`.

Every vertex coordinate was multiplied by `0.01` to convert the source centimetre geometry to metres. Faces and
the shared assembled coordinate frame were preserved; the five parts were not independently recentered. The URDF
wrappers and generated USD assets are repository-authored integration files.

Fabrica's root repository is distributed under the MIT license reproduced in `LICENSE.Fabrica-MIT.txt`. The
Fabrica learning package carries NVIDIA's BSD-3-Clause notice, reproduced in
`LICENSE.Fabrica-Learning-BSD-3-Clause.txt`, for migrated learning-package material. Fabrica's learning notice
refers to `assets/licenses` for asset-specific terms; the pinned source tree contains no beam-specific license
file. Confirm downstream redistribution requirements before publishing the beam meshes outside this repository.

Generate the free-root SDF USD assets from the packaged URDFs with:

```bash
python scripts/tools/generate_assembly_usd_assets.py --assembly beam --overwrite
```
