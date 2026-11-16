"""Reproduce the synthetic AIS watermark experiments in the course report."""
import argparse
import csv
import math
import random
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

WINDOW_S = 600
WARMUP_START_S = -1200
SCORE_END_S = 28800
HOLD_START_S = 11400
RELEASE_TIMES_S = tuple(minute * 60 for minute in (228, 230, 232, 234))
CUTOFF_S = 32400
SEEDS = {"tune": 15402, "test": 15403}
DEFAULT_Q = Fraction("0.95")
FIXED_CANDIDATES_MIN = (0, 0.5, 1, 2, 3, 5, 10, 20, 30, 40)


@dataclass(frozen=True)
class Report:
    report_id: int
    mmsi: int
    event_s: int
    arrival_s: int
    is_backfill: bool = False


@dataclass(frozen=True)
class FirstRelease:
    count: int
    arrival_s: int
    step: int


def window_start(event_s):
    return event_s // WINDOW_S * WINDOW_S


def ordered_reports(reports):
    return sorted(reports, key=lambda report: (report.arrival_s, report.report_id))


def nearest_rank(values, q):
    return sorted(values)[math.ceil(q * len(values)) - 1]


def replay(reports, starts, policy, initial_s, n, q, r, on_step=None):
    counts = dict.fromkeys(starts, 0)
    first = {}
    next_window = 0
    max_seen = watermark = -math.inf
    previous_arrival = None
    delta = initial_s
    recent = deque(maxlen=n)
    for step, report in enumerate(reports, 1):
        start = window_start(report.event_s)
        if start in counts and watermark < start + WINDOW_S:
            counts[start] += 1
        disorder = max(0, max_seen - report.event_s)
        target = None
        if policy != "A":
            recent.append(disorder)
            if len(recent) == n:
                target = nearest_rank(recent, q)
                if policy == "B":
                    delta = target
                else:
                    elapsed = 0 if previous_arrival is None else report.arrival_s - previous_arrival
                    delta = max(target, delta - r * elapsed)
        previous_arrival = report.arrival_s
        max_seen = max(max_seen, report.event_s)
        watermark = max(watermark, max_seen - delta)
        while next_window < len(starts):
            closing = starts[next_window]
            if closing + WINDOW_S > watermark:
                break
            first[closing] = FirstRelease(counts[closing], report.arrival_s, step)
            next_window += 1
        if on_step is not None:
            on_step(step, report, disorder, max_seen, target, delta, watermark)
    return first


def offline_counts(reports, starts):
    counts = dict.fromkeys(starts, 0)
    for report in reports:
        start = window_start(report.event_s)
        if start in counts:
            counts[start] += 1
    return counts


def synthetic_reports(seed):
    rng = random.Random(seed)
    reports = []
    event_s = WARMUP_START_S - 20 * 60
    while event_s < CUTOFF_S:
        lag = rng.randint(40, 150) if rng.random() < 0.06 else rng.randint(0, 25)
        held = (HOLD_START_S <= event_s < RELEASE_TIMES_S[0] - 150
                and rng.random() < 0.35)
        arrival_s = rng.choice(RELEASE_TIMES_S) if held else event_s + lag
        reports.append(Report(len(reports) + 1, rng.randint(412000001, 412000150),
                              event_s, arrival_s, held))
        event_s += rng.randint(6, 12)
    return ordered_reports(reports)


def initial_delta_s(reports, q):
    maximum = -math.inf
    samples = []
    for report in reports:
        if report.arrival_s >= WARMUP_START_S:
            break
        samples.append(max(0, maximum - report.event_s))
        maximum = max(maximum, report.event_s)
    return nearest_rank(samples, q)


def arrival_phase(arrival_s):
    for limit, name in ((WARMUP_START_S, "calibration"), (0, "warmup"),
                        (HOLD_START_S, "normal"), (RELEASE_TIMES_S[0], "holding"),
                        (RELEASE_TIMES_S[-1] + 1, "backfill"),
                        (SCORE_END_S, "recovery"), (CUTOFF_S + 1, "tail")):
        if arrival_s < limit:
            return name
    return "after_cutoff"


def run_experiment(args):
    seed = SEEDS[args.command]
    all_reports = synthetic_reports(seed)
    reports = [r for r in all_reports if WARMUP_START_S <= r.arrival_s <= CUTOFF_S]
    starts = range(0, SCORE_END_S, WINDOW_S)
    truth = offline_counts(all_reports, starts)
    truth_total = sum(truth.values())
    initial = initial_delta_s(all_reports, args.q) if args.initial_delta_s is None else args.initial_delta_s
    if args.command == "tune" and args.c_rates:
        runs = [(f"C{i}", "C", initial, rate)
                for i, rate in enumerate(dict.fromkeys(args.c_rates), 1)]
    else:
        fixed = args.fixed_deltas if args.command == "tune" else [args.fixed_delta]
        runs = [(f"A{i}", "A", value * 60, args.r)
                for i, value in enumerate(dict.fromkeys(fixed), 1)]
        runs.extend([("B", "B", initial, args.r), ("C", "C", initial, args.r)])
    args.out.mkdir(parents=True, exist_ok=True)
    headers = {
        "reports": "report_id,mmsi,event_time_s,arrival_time_s,is_backfill,arrival_phase,replayed",
        "windows": "run_id,window_start_s,window_end_s,truth_count,status,first_count,first_arrival_s,first_step,first_delay_s,missed_count",
        "trace": "run_id,step,report_id,arrival_phase,arrival_time_s,event_time_s,disorder_s,max_event_time_s,quantile_s,delta_s,watermark_s",
        "summary": ("split,seed,run_id,policy,fixed_delta_min,initial_delta_s,n,q,r_min_per_min,cutoff_s,"
                    "total_windows,published_windows,unpublished_windows,truth_messages,"
                    "unpublished_truth_messages,p95_first_delay_s,missed_messages,first_miss_rate"),
    }
    with ExitStack() as files:
        writers = {}
        for name, header in headers.items():
            stream = files.enter_context((args.out / f"{name}.csv").open("w", encoding="utf-8-sig", newline=""))
            writers[name] = csv.writer(stream)
            writers[name].writerow(header.split(","))
        for report in all_reports:
            writers["reports"].writerow((
                report.report_id, report.mmsi, report.event_s, report.arrival_s, int(report.is_backfill),
                arrival_phase(report.arrival_s), int(WARMUP_START_S <= report.arrival_s <= CUTOFF_S)))
        for run_id, policy, start_delta, rate in runs:
            def write_trace(step, report, disorder, maximum, target, delta, watermark):
                writers["trace"].writerow((
                    run_id, step, report.report_id, arrival_phase(report.arrival_s), report.arrival_s,
                    report.event_s, disorder, maximum, target, delta, watermark))

            first = replay(reports, starts, policy, start_delta, args.n, args.q, rate, write_trace)
            delays = []
            missed = unpublished_truth = 0
            for start in starts:
                end = start + WINDOW_S
                release = first.get(start)
                if release is None:
                    unpublished_truth += truth[start]
                    writers["windows"].writerow((run_id, start, end, truth[start], "unpublished",
                                                 None, None, None, None, None))
                    continue
                delay = release.arrival_s - end
                shortfall = truth[start] - release.count
                delays.append(delay)
                missed += shortfall
                writers["windows"].writerow((run_id, start, end, truth[start], "published",
                                             release.count, release.arrival_s, release.step, delay, shortfall))
            unpublished = len(starts) - len(first)
            p95 = nearest_rank(delays, DEFAULT_Q) if not unpublished else None
            miss_rate = missed / truth_total if not unpublished and truth_total else None
            fixed = policy == "A"
            writers["summary"].writerow((
                args.command, seed, run_id, policy, start_delta / 60 if fixed else None,
                None if fixed else start_delta, None if fixed else args.n, None if fixed else float(args.q),
                rate if policy == "C" else None, CUTOFF_S, len(starts), len(first), unpublished,
                truth_total, unpublished_truth, p95, None if unpublished else missed, miss_rate))
            print(f"{run_id}: P95={p95} s, missed={missed}, unpublished={unpublished}")
    print(f"CSV directory: {args.out.resolve()}")


def self_check():
    events = (6, 9, 4, 5, 12, 8, 11, 7, 14)
    reports = [Report(i, 412000001, t * 60, (i + 8) * 60) for i, t in enumerate(events, 1)]
    assert offline_counts(reports, [0]) == {0: 6}
    for policy, step, count in (("A", 5, 4), ("B", 7, 5), ("C", 9, 6)):
        result = replay(reports, [0], policy, 120, 3, Fraction(1, 2), 1)
        assert result == {0: FirstRelease(count, (step + 8) * 60, step)}
    edge = ordered_reports([Report(3, 412000001, 1200, 1200), Report(2, 412000001, 600, 600),
                            Report(1, 412000001, 540, 600)])
    assert [report.report_id for report in edge] == [1, 2, 3]
    assert offline_counts(edge, [0, 600, 1200]) == {0: 1, 600: 1, 1200: 1}
    assert replay(edge, [0, 600, 1200], "A", 0, 3, DEFAULT_Q, 0.5) == {
        0: FirstRelease(1, 600, 2), 600: FirstRelease(1, 1200, 3)}
    print("Self-check passed: nine reports, window boundaries, arrival ties, and no EOF flush.")


def nonnegative(value):
    number = float(value)
    if number < 0 or not math.isfinite(number * 60):
        raise argparse.ArgumentTypeError("expected a finite nonnegative number")
    return number


def quantile(value):
    try:
        q = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise argparse.ArgumentTypeError("expected a quantile in (0, 1]") from error
    if not 0 < q <= 1:
        raise argparse.ArgumentTypeError("expected a quantile in (0, 1]")
    return q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-check")
    tune, test = sub.add_parser("tune"), sub.add_parser("test")
    for command, locked in ((tune, False), (test, True)):
        command.add_argument("--out", type=Path)
        command.add_argument("--n", type=int, default=100, required=locked)
        command.add_argument("--q", type=quantile, default=DEFAULT_Q, required=locked)
        command.add_argument("--r", type=nonnegative, default=0.5, required=locked)
        command.add_argument("--initial-delta-s", type=nonnegative, required=locked)
    tune.add_argument("--fixed-deltas", type=nonnegative, nargs="+", default=FIXED_CANDIDATES_MIN)
    tune.add_argument("--c-rates", type=nonnegative, nargs="+")
    test.add_argument("--fixed-delta", type=nonnegative, required=True)
    args = parser.parse_args()
    if args.command == "self-check":
        self_check()
        return
    if args.n < 1:
        parser.error("--n must be positive")
    if args.command == "tune":
        if not args.c_rates and len(set(args.fixed_deltas)) < 2:
            tune.error("--fixed-deltas requires at least two distinct candidates")
        default_out = "ais_tuning_c" if args.c_rates else "ais_tuning"
    else:
        default_out = "ais_test"
    if args.out is None:
        args.out = Path(default_out)
    run_experiment(args)


if __name__ == "__main__":
    main()
