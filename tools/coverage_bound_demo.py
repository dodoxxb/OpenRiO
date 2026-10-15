"""Check the manuscript's fixed-pool coverage bound on exhaustive tiny graphs.

Pure Python; no detector, data, or GPU is loaded. This is a mathematical
sanity check, not a complete-dataset oracle evaluation. Values use [0, 1].
"""
import argparse
import itertools
import json
from pathlib import Path


def maximum_matching(edges, n_gt):
    owners = [-1] * n_gt

    def augment(candidate, seen):
        for gt in edges[candidate]:
            if gt in seen:
                continue
            seen.add(gt)
            if owners[gt] == -1 or augment(owners[gt], seen):
                owners[gt] = candidate
                return True
        return False

    return sum(augment(i, set()) for i in range(len(edges)))


def brute_matching(edges, n_gt):
    def visit(i, occupied):
        if i == len(edges):
            return len(occupied)
        result = visit(i + 1, occupied)
        for gt in edges[i]:
            if gt not in occupied:
                result = max(result, visit(i + 1, occupied | {gt}))
        return result

    return visit(0, set())


def ap11(tp_flags, n_gt):
    tp = 0
    curve = []
    for i, flag in enumerate(tp_flags, 1):
        tp += flag
        curve.append((tp, tp / i))
    # Integer recall comparisons avoid a floating-point boundary at j/10.
    return sum(max((p for t, p in curve if 10*t >= j*n_gt), default=0.0)
               for j in range(11)) / 11


def ap11_upper(matched, n_gt):
    if matched == 0:
        return 0.0
    return sum(j*n_gt <= 10*matched for j in range(11)) / 11


def check():
    graphs = cases = 0
    for n_pred in range(1, 4):
        for n_gt in range(1, 4):
            for bits in itertools.product((False, True), repeat=n_pred*n_gt):
                edges = [[g for g in range(n_gt) if bits[p*n_gt+g]]
                         for p in range(n_pred)]
                m = maximum_matching(edges, n_gt)
                assert m == brute_matching(edges, n_gt)
                coverage = len(set(itertools.chain.from_iterable(edges)))
                assert m <= coverage
                graphs += 1
                for order in itertools.permutations(range(n_pred)):
                    for cap in range(n_pred + 1):
                        # Edges have equal eligible overlaps. Use a fixed
                        # target tie order, including duplicate assignments.
                        used = set()
                        tp_flags = []
                        for p in order[:cap]:
                            target = edges[p][0] if edges[p] else None
                            flag = target is not None and target not in used
                            tp_flags.append(int(flag))
                            if flag:
                                used.add(target)
                        bound = min(cap, m)
                        assert sum(tp_flags) <= bound
                        assert ap11(tp_flags, n_gt) <= ap11_upper(bound, n_gt) + 1e-12
                        cases += 1
    example = {'edges': [[0, 1]], 'gt': 2, 'covered_gt': 2,
               'maximum_matching': 1, 'recall_bound': 0.5,
               'ap11_bound': ap11_upper(1, 2), 'achieved_ap11': ap11([1], 2)}
    assert graphs == 682
    assert example['achieved_ap11'] > example['recall_bound']
    return {'status': 'PASS', 'binary_graphs': graphs,
            'order_cap_cases': cases, 'maximum_matching_checked_against': 'exhaustive assignment',
            'example': example, 'scope': 'synthetic graphs only; no dataset result',
            'conditions': 'fixed candidate geometries and labels; one-to-one true positives; VOC2007 AP11'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.dumps(check(), indent=2) + '\n'
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
    print(result, end='')
