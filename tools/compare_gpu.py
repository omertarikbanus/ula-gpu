#!/usr/bin/env python3
"""Compare errors against ULA's double run, with a review flag at 4x CPU float."""
import argparse
import math
from pathlib import Path

import controller_replay as cr


def check(ref, candidate, limits, label):
    result = cr.compare_traces(ref, candidate, limits, cr.FieldTolerance(0, 0))
    if result.first_divergence:
        d = result.first_divergence
        print(f'{label}: {d.category}.{d.field} at episode={d.episode} tick={d.tick}: {d.message}')
        raise SystemExit(1)
    return result.field_stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference', type=Path)
    p.add_argument('candidate', type=Path)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--cpu-float', type=Path)
    mode.add_argument('--double-check', action='store_true')
    args = p.parse_args()
    ref = cr.load_and_validate_trace(args.reference)
    cand = cr.load_and_validate_trace(args.candidate)
    fields = tuple(cr.OUTPUT_CONTINUOUS_VECTOR_LENGTHS)
    if args.double_check:
        stats = check(ref, cand, {f: cr.FieldTolerance(1e-9, 0) for f in fields}, 'double logic check')
        for name, s in stats.items():
            print(f'{name}: max abs error {s.max_abs_err:.3e} ({s.max_abs_loc})')
        print('PASS: double logic check, exact gait/contact/WBIC flags')
        return

    # Let the existing checker enforce reset/input/discrete parity while
    # gathering maxima. JSON validation separately rejects NaN/Inf.
    unlimited = {f: cr.FieldTolerance(math.inf, 0) for f in fields}
    cpu = check(ref, cr.load_and_validate_trace(args.cpu_float), unlimited, 'CPU float')
    port = check(ref, cand, unlimited, 'candidate')
    review = False
    for f in fields:
        baseline, actual = cpu[f].max_abs_err, port[f].max_abs_err
        flag = actual > 4 * baseline
        review |= flag
        print(f'{f}: CPU {baseline:.3e}, candidate {actual:.3e} ({port[f].max_abs_loc}), '
              f'4x CPU {4*baseline:.3e}' + (' [REVIEW]' if flag else ''))
    print('REVIEW: float error exceeds 4x; discuss before accepting' if review else
          'PASS: float errors within 4x CPU; exact gait/contact/WBIC flags')


if __name__ == '__main__':
    main()
