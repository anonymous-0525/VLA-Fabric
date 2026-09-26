import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { dedup } from '@gltf-transform/functions';
import { readFile, writeFile } from 'node:fs/promises';
import { gzipSync } from 'node:zlib';

const [input, output] = process.argv.slice(2);
if (!input || !output) throw new Error('Usage: node optimize-scene.mjs input.glb output.glb');
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS);
const document = await io.read(input);
// Share identical geometry and accessors, without changing hierarchy or motion.
await document.transform(dedup());
await io.write(output, document);
const binary = await readFile(output);
const compressed = gzipSync(binary, { level: 9 });
await writeFile(`${output}.gz`, compressed);
console.log(`${binary.length} bytes GLB; ${compressed.length} bytes compressed`);
