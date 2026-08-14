## What this changes

<!-- One or two sentences. -->

## Type

- [ ] Bug fix
- [ ] New material / phantom / artifact / metric
- [ ] Refactor (no behaviour change)
- [ ] Documentation
- [ ] Other:

## Numerical impact

Does this change any number a previous run would have produced?

- [ ] No — refactor, docs, tests, or a purely additive material/preset
- [ ] Yes — described below, with which figures or presets are affected

<!-- If yes: say what moved and by how much. Someone's published figure may
     depend on it. -->

## New physical data

<!-- Delete if not applicable. Cite the source of every number: NIST XCOM
     retrieval date, Sears (1992), a datasheet, a measurement. -->

## Checks

- [ ] `pytest` passes
- [ ] `ruff check neutron_xray_sim tests` is clean
- [ ] `MATERIALS.audit()` reports nothing new (if materials changed)
- [ ] Notebook outputs stripped (if notebooks changed)
- [ ] A test covers the change; regression tests name what used to break
