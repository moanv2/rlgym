## What

<!-- Short description of the change. -->

## Why

<!-- Motivation, link to roadmap milestone if applicable. -->

## Training impact

- [ ] No effect on training (refactor, docs, tests, CI)
- [ ] Behavior change — requires a fresh experiment (new exp_NNN_*.yaml)
- [ ] Behavior change — compatible with existing checkpoints

## Checklist

- [ ] `make lint` clean
- [ ] `make test-fast` passes
- [ ] If new component: registered in the right registry, has a unit test
- [ ] If hyperparam change: lives in a config, not hardcoded
- [ ] Roadmap doc updated if a milestone shifted
