# Website media and reproducibility

The site uses static GitHub Pages hosting, Three.js, and local video assets.
It does not send visitor input or video data to an inference service. No model
weights, camera identifiers, original workstation paths, or experiment logs
are embedded in the page.

## Real recordings

The four recordings are Frame4, Relay4, Relay3, and Basket3. Each global view
and its wrist views are split from the same camera mosaic. All clips retain
their original temporal scale. No audio track is shipped.

The source title strip is removed. Wrist surroundings are blurred with a
22-pixel Gaussian; the central working region is mildly filtered (2 pixels),
with feathered transitions. Global views are not blurred. This is a spatial
privacy treatment, not a semantic person detector. When replacing a recording,
inspect its full duration and revise the mask if a person or identifiable
detail enters the central region. Never publish the raw wrist source by default.

`scripts/site/prepare_physical_media.py` creates the split MP4s and records the
processing in `static/media/media-processing.json`. Its input directory is a
command-line argument, not a sibling project dependency. The source clips had
a uniform brightness correction before this export; geometry and actions are
not generated or retouched. `transcode_webm.py` adds VP9 for browsers without
H.264 decoding. The hero uses 16-second excerpts, at their original speed.

## Simulation sources

| Task | Video source | Reported quantitative results |
| --- | --- | --- |
| Stack Cube | Archived Eagle Full rollout | Separate pi0.5 Full / Independent evaluation |
| Three-arm frame | Original global-camera demonstration replay | Separate policy evaluation |
| Four-arm frame | pi0.5 Full evaluation rollout | Same task's Full / Independent evaluation |
| Arch Building | Original global-camera demonstration replay | Separate policy evaluation, including low success |

The visible source labels are intentional: demonstration clips are not claimed
to be policy successes. `export_demo_video.py` extracts original camera frames
from an HDF5 trajectory using its timestamps. Videos do not substitute for
aggregate trial results. The site reports the current manuscript's task results;
shared-policy values use their own development evaluation protocol, not a
matched 200-condition comparison with the single-task rows.

## Interactive scene

`static/models/four-arm.glb` is derived from the project-owned Blender scene.
`export_scene.py` bakes joint motion and frame movement into a portable GLB.
`optimize-scene.mjs` shares identical geometry/accessors without changing
the node hierarchy and writes a losslessly compressed `.glb.gz`. Supported
browsers decompress it locally; other browsers load the ordinary GLB.
The sequence and signal pulses are an explanatory illustration, not a recorded
policy execution or a measurement of network traffic. Hover or agent selection
highlights the selected arm and both directions of its communication relations.
Reduced-motion preferences pause the scene and disable background-video motion.

## Build and inspect

```bash
# Node.js 22 or newer
npm ci
npm run build
npm run preview
# In another terminal:
npx playwright install chromium
npm run test:site
```

Browser checks cover desktop/mobile framing, live canvas motion, task/camera
selection, synchronized playback, figure dialogs, and overflow. Screenshots
and test artifacts stay outside the published tree. Third-party browser code
retains its required license notices; project attribution is anonymous.
