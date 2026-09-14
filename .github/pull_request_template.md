## What and why

<!-- What changes, and what problem it solves. This becomes the squash commit
     message on main, so write it for someone reading `git log` in a year. -->

## How it was verified

<!-- Commands run, systems tested, reference programs compared against, numbers
     measured. "Tests pass" is what CI is for; say what you checked that CI
     cannot (msys, mdtraj, gemmi or ChimeraX oracles run locally, benchmarks). -->

## Notes for the reviewer

<!-- Trade-offs, anything deliberately left out, follow-up work. Delete if
     there is nothing to flag. -->

---

- [ ] `pytest tests` passes locally, including the oracle tests that CI skips
- [ ] Documentation updated (README and `docs/`) if behavior changed
- [ ] Generated pages refreshed if the API or examples changed
      (`python docs/gen_reference.py`, `python docs/gen_examples.py`)
