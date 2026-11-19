# Golden test inputs

- `force_inputs/` contains local copies of the input JSON files used by the
  golden tests.
- Expected outputs are read directly from
  `ref_outputs/<swing_id>/<swing_id>_forcescreen.json`,
  `*_basicscreen.json`, and `*_clubFitting.json`.
- `tests/test_force_output_golden.py` calls individual modular basic, force, and
  movement-applicable club methods. It compares recursive keys and every
  returned value. Numeric values use `rel_tol=1e-9` and `abs_tol=1e-12`.
- The basic screen's `error_messages` metadata is excluded because no modular
  method returns it.

Run from the repository root with:

```powershell
$env:PYTHONPATH = "src;src/biomechanics"
python -m pytest tests/test_force_output_golden.py
```
