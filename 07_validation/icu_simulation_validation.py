"""Audit the ACTUAL simulate_priority() in staylength.py without launching Streamlit.
Run in ICU_STREAMLIT_APP: python step74_validate_actual_simulation.py --app staylength.py
Does not load model files or patient records.
"""
import argparse
import ast
import inspect
from pathlib import Path
import numpy as np


def extract_function(path):
    source = path.read_text(encoding='utf-8-sig')
    tree = ast.parse(source, filename=str(path))
    matches = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'simulate_priority']
    if len(matches) != 1 or not isinstance(matches[0], ast.FunctionDef):
        raise RuntimeError('Expected exactly one top-level def simulate_priority(...) in the supplied app.')
    fn = matches[0]
    # Instrument the real function at each iteration of its hour loop, without changing its decisions.
    class Instrument(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == 'simulate_priority':
                node.body.insert(0, ast.parse('_audit_occupancies = []').body[0])
                for inner in ast.walk(node):
                    if isinstance(inner, ast.For) and isinstance(inner.target, ast.Name) and inner.target.id == 'hour':
                        inner.body.append(ast.parse('_audit_occupancies.append(len(busy_until))').body[0])
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Return) and isinstance(inner.value, ast.Dict):
                        inner.value.keys.append(ast.Constant('_audit_occupancies'))
                        inner.value.values.append(ast.Name('_audit_occupancies', ast.Load()))
            return node
    fn = Instrument().visit(fn)
    ast.fix_missing_locations(fn)
    namespace = {'np': np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['simulate_priority']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--app', default='staylength.py')
    args = parser.parse_args()
    path = Path(args.app).resolve()
    if not path.is_file():
        raise SystemExit(f'ERROR: App not found: {path}')
    try:
        simulate = extract_function(path)
    except Exception as e:
        raise SystemExit(f'ERROR loading actual simulation function: {e}')
    params = inspect.signature(simulate).parameters
    expected = {'risks', 'beds', 'occupied', 'mean_stay_days', 'max_wait_hours', 'runs'}
    if not expected.issubset(params):
        raise SystemExit('ERROR: This script supports simulate_priority(risks, beds, occupied, mean_stay_days, max_wait_hours, runs, seed=...). Actual function signature differs. Please share staylength.py.')

    failures = []
    def check(name, condition, detail=''):
        passed = bool(condition)
        print(f"{'PASS' if passed else 'FAIL'} | {name}" + (f' | {detail}' if detail else ''))
        if not passed:
            failures.append(name)

    def run(p, beds, occupied, stay, wait, runs=300, seed=42):
        return simulate(np.asarray(p, dtype=float), beds, occupied, stay, wait, runs, seed=seed)

    print('Auditing ACTUAL function from:', path)
    print('This uses synthetic probabilities only; no MIMIC-IV records.\n')
    try:
        z = run([0, 0, 0], 2, 2, 7, 0)
        check('Zero-risk patients produce zero requests', np.isclose(np.sum(z['need_chance']), 0) and np.sum(z['immediate'] + z['delayed'] + z['unmet']) == 0)

        full = run([1, 1, 1, 1], 1, 1, 1000000, 0, runs=200)
        check('Full bed + no discharges + zero wait -> all unmet', np.all(full['unmet'] == 4) and np.all(full['immediate'] == 0) and np.all(full['delayed'] == 0))
        check('Occupied beds never exceed capacity', max(full['_audit_occupancies'], default=0) <= 1)
        check('Every request accounted for (all-positive scenario)', np.all(full['immediate'] + full['delayed'] + full['unmet'] == 4))

        free = run([1, 1], 3, 0, 1000000, 0, runs=200)
        check('Available capacity admits requests immediately', np.all(free['immediate'] == 2) and np.all(free['unmet'] == 0))
        check('Capacity invariant under available-bed scenario', max(free['_audit_occupancies'], default=0) <= 3)

        a = run([0.7, 0.4, 0.2], 2, 1, 7, 12, runs=250, seed=91)
        b = run([0.7, 0.4, 0.2], 2, 1, 7, 12, runs=250, seed=91)
        keys = ['need_chance', 'admit_chance', 'unmet_chance', 'immediate', 'delayed', 'unmet']
        check('Same random seed -> reproducible results', all(np.array_equal(a[k], b[k]) for k in keys))
        check('Per-patient request accounting', np.allclose(a['need_chance'], a['admit_chance'] + a['unmet_chance'], atol=1e-10))
        check('Mean request accounting', np.isclose(np.sum(a['need_chance']), np.mean(a['immediate'] + a['delayed'] + a['unmet']), atol=1e-10))
        check('All occupancy observations <= capacity', max(a['_audit_occupancies'], default=0) <= 2)
        check('Wait metrics are nonnegative', a['avg_wait_hours'] >= 0 and np.all(a['immediate'] >= 0) and np.all(a['delayed'] >= 0))

        # The function assumes input is already sorted highest-risk first.
        # This tests priority only when requests compete for a single bed.
        # Different randomly sampled arrival times can outweigh priority, so a
        # monotonic admission-rate test would incorrectly fail a valid queue.
        print('NOTE | Priority tie-breaking requires a controlled same-hour-arrival test; not certified here.')
        check('Input probabilities are valid output fractions', np.all((a['admit_chance'] >= 0) & (a['admit_chance'] <= 1)))
    except Exception as e:
        print('ERROR while auditing:', repr(e))
        raise SystemExit(2)

    print('\n' + ('FINAL RESULT: PASS (tested simulation invariants)' if not failures else f'FINAL RESULT: FAIL ({len(failures)} checks)'))
    print('Scope: This audits the actual extracted simulate_priority() logic, including hourly occupancy. It does not verify any separate random-allocation implementation, clinical validity, or calibration.')
    if failures:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
