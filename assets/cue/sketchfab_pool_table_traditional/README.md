# Sketchfab Cue Visual Asset

This directory contains the localized visual mesh for the cue extracted from the downloaded Sketchfab `Pool Table Traditional` glTF asset.

- `pool_cue_visual_local.obj`: visual mesh centered on the MuJoCo cue body.
- `cue_mat_baseColor.png`: base color texture used by `models/cue_physics.xml`.

The active physics model is defined by primitive geoms in `models/cue_physics.xml`: `cue_tip`, `cue_shaft` and the additional `cue_shaft_*` taper segments. The mesh here is visual-only and has no collision or mass. The active visual mesh is `snooker_cue_shaft_visual.obj`, derived from the original OBJ by `scripts/assets/build_snooker_cue_visual.py`. It preserves the texture coordinates, corrects the offset centreline and inward face winding, removes the old ferrule/tip, and joins the coaxial 9.5 mm ferrule exactly at local x=0.710 m. The original OBJ remains unchanged; both meshes share the source license.

This work is based on [Pool Table Traditional](https://sketchfab.com/3d-models/pool-table-traditional-e0b938c0c2e74eb794a49ebde2543977)
by [fizyman](https://sketchfab.com/fizyman), licensed under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
The cue mesh was extracted and localized for this scene. The original attribution
and license notice is retained in [license.txt](license.txt).
