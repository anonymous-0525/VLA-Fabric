# Reproducibility Protocol

This document records the frozen protocols represented by the public configs.
It does not include paper result tables or checkpoints.

## Part I: Eagle interaction study

For each of Handover Box and Shoes Table, the matrix contains nine Single-stage
and seven Common-first models:

| ID | Common | Private K/V | Action interaction |
|---|---|---|---|
| I0 | none | none | none |
| I1 | none | enabled | none |
| I2 | one-way | enabled | none |
| I3-AVG | symmetric average | enabled | none |
| I3-RAW | operator-level Raw | enabled | none |
| I4-AVG | symmetric average | enabled | Linear |
| I4-RAW | operator-level Raw | enabled | Linear |
| I5-AVG | symmetric average | enabled | Transformer |
| I5-RAW | operator-level Raw | enabled | Transformer |

Single-stage trains the complete configured interaction directly. Common-first
trains the Common path for 10k updates, then starts a fresh optimizer, schedule,
and RNG for a 40k full-interaction stage. I0 and I1 are shared controls because
they do not define a Common-first transition. The selected I4-RAW and I5-RAW
families have targeted 100k follow-ups; these follow-ups do not reopen the
architecture search.

Machine-readable matrices are under `eagle/configs/matrix/`, and every released
run has a YAML under `eagle/configs/runs/`.

## Part II: pi0.5 transfer

The final pi0.5 realization keeps the same functional coordination roles while
moving them to backbone-native boundaries:

- Common: symmetric operator-level fusion over Common multimodal-prefix tokens;
- latent peer context: Private Prefix K/V exposed to the local action query;
- action coordination: zero-initialized gated residual peer-action attention,
  recomputed per layer and flow step using receiver-local projections.

The protocol uses prediction horizon 50, execution horizon 25, and 10 flow
steps. Four training processes use global batch 12. Common-first Stage 1 runs
10k updates; Stage 2 loads Stage-1 model weights and runs 40k updates with all
three functions enabled. Exact settings are in
`pi05/configs/training/pi_native_v2_h50_4gpu.yaml`.

## Part III: multi-arm extension

The shared N-agent implementation assigns one complete pi0.5 policy process to
each role and preserves local observation and action ownership. StackCube uses
three ranks with H50/E25 and ten flow steps. Its validated operating point is
team microbatch 6 with accumulation 3, giving global team batch 18. Common-only
Stage 1 runs 10k updates and Full Stage 2 runs 50k updates.

Frame Insertion and Arch Assembly use four ranks with the same horizon and flow
protocol. B6A3 has passed a local four-rank training-path check for Frame
Insertion, but formal four-agent training must still follow target-HPC
Stage-1/Stage-2/restore/finite-optimization readiness testing. The checked-in
readiness command records the target allocation and memory evidence without
starting a formal run.
Task and hardware contracts are under `pi05/configs/multiarm/`.

## Paired evaluation

Both implementations preserve condition pairing between candidate models:

| Split | Condition IDs | Role |
|---|---:|---|
| Development | 0--199 | checkpoint and architecture analysis |
| Disjoint paired | 200--399 | frozen-checkpoint comparison |

Environment initialization and model-side stochastic seeds are paired by
condition ID. The evaluator rejects missing IDs, duplicates, non-finite outputs,
and unexpected checkpoint metadata. See `configs/evaluation/paired200.yaml` in
each package for the machine-readable protocol.

## Deliberate exclusions

The release includes focused three- and four-arm workflows but excludes
RoboTwin2 pi0.5 adapters, NSPR/network profiling, private cluster schedulers,
datasets, checkpoints, readiness records, and rollout outputs. Their absence is enforced by the root audit and
scope tests.
