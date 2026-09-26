# Project website

GitHub Pages serves this directory from the main branch. The static document
presents the paper in three chapters: interaction discovery, cross-backbone
transfer, and team expansion. Figures and result tables do not depend on
JavaScript; optional interactions are built from ../site-src.

Run npm run build at the repository root after JavaScript changes, then
npm run preview to inspect the website. Node.js 22 or newer is required.
The entry bundle is deliberately small; Three.js and the four-arm scene
load separately after the document is interactive.

The physical gallery uses a responsive 2-by-2 layout, with a synchronized
global view and wrist carousel in each item. Simulation clips use a single
four-column row on desktop. All videos are demand-loaded and playback pauses
outside the viewport.

See [media.md](media.md) for recording provenance, privacy processing,
evaluation distinctions, and the browser validation procedure. Camera
identifiers, private experiment paths, model weights, and raw wrist videos
are not part of the website.
