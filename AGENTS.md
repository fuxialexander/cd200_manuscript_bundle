# Working on local MoDL data

For a request to prepare data, train, or interpret a model, read
[AGENT_WORKFLOW.md](AGENT_WORKFLOW.md) first. Use `scripts/modl.py` through the
existing environment, or `bash scripts/modl.sh`. Reuse the shared training and
deletion engines; do not substitute a new model or change scientific definitions
while orchestrating a user's analysis.

Inspect real files before selecting reference assembly, barcode groups, peaks,
holdout chromosomes or scoring heads. Preserve input files and existing outputs.
Keep data local and network logging disabled. Record exact commands, manifests,
checkpoint selection, validation evidence and remaining limitations. A smoke
test is execution evidence, not a predictive benchmark or biological validation.

For implementation changes, run `tests.test_modl_workflow` and any relevant
existing tests. Do not launch long training jobs merely to verify a code edit.
