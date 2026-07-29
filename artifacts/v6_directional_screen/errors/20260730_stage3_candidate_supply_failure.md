# Stage 3 Error — Candidate Supply Failure

Classification: `SCIENTIFIC_DATA_SUPPLY_FAILURE`

The frozen generator required a successful clean trajectory containing an
observed tool call for every registered task. For the first task,
`airline:12`, both registered seeds produced tool-using trajectories with
official reward `0.0`. The script exited nonzero before emitting any candidate.

Infrastructure checks excluded SSH, model health, CUDA, OOM, API, and parsing
failures. An eight-attempt post-failure diagnostic also produced eight rewards
of `0.0`; it is quarantined and has no formal scientific status.

Recovery would require changing the generation attempt contract, task set,
teacher, or seed population. Such a change cannot be called the frozen V6
experiment.

