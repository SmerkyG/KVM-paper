"""Collect the matched 13-task expansion of the K2 BF16 page-budget study."""

import argparse
from pathlib import Path
from statistics import mean
import time

from .amd_k2_followup_ablation import PAGE_TASKS, page_work
from .amd_k2_threshold_ablation import validate_requests, verified_cohort
from .amd_threshold_prefill_quality_panel import job_state, write
from .amd_threshold_ruler_panel import TASKS, digest, paired_interval, read


def collect(root, manifest):
    if digest(root / "settings.json") != manifest["settings_sha256"]:
        raise ValueError("Changed page-budget settings")
    source = Path(manifest["source"])
    source_manifest = read(root / "source-manifest.json")
    if source_manifest["source_sha256"] != manifest["source_sha256"]:
        raise ValueError("Unexpected page-budget source")
    for file, checksum in source_manifest["source_files"].items():
        if digest(source / file) != checksum:
            raise ValueError(f"Changed source file {file}")
    diagnostic_path = Path(manifest["diagnostic_report"])
    if digest(diagnostic_path) != manifest["diagnostic_sha256"]:
        raise ValueError("Changed page-budget diagnostic")
    diagnostic = read(diagnostic_path)
    prior = Path(manifest["prior"])
    cohorts, tokens = verified_cohort(prior)
    settings = read(root / "settings.json")
    dense = read(prior / "REPORT.json")["models"]["k2"]
    dense_tasks = {entry["task"]: entry["archived_dense"] for entry in dense["tasks"]}
    jobs = [*manifest["archived_jobs"], *manifest["jobs"]]
    expected = {(variant, task) for variant in settings for task in TASKS}
    if len(jobs) != len(expected) or {(j["variant"], j["task"]) for j in jobs} != expected:
        raise ValueError("Missing or duplicate page-budget tasks")
    rows = {}
    for job in jobs:
        fields, data = job_state(job)
        if not fields or fields[7:9] != ["finished", "0"]:
            raise ValueError(f"Incomplete run {job['id']}")
        variant, task = job["variant"], job["task"]
        options = settings[variant]
        scores = validate_requests(data, task, cohorts[task], tokens[task], options["env"],
                                   manifest["source_sha256"], mode=options["mode"])
        rows[variant, task] = scores
        if task in PAGE_TASKS:
            expected_diagnostic = {
                **page_work(scores, variant, task, options),
                "score": 100 * mean(r["score"] for r in scores.values()),
            }
            if diagnostic["retrieval"][variant][task] != expected_diagnostic:
                raise ValueError("Changed results for the original diagnostic prompts")
    scores = {
        variant: {task: 100 * mean(r["score"] for r in rows[variant, task].values()) for task in TASKS}
        for variant in settings
    }
    means = {variant: mean(tasks.values()) for variant, tasks in scores.items()}
    results = {}
    lines = [
        "# K2 BF16 three-tier page budget on 64K RULER", "",
        "All five arms use the same 32 prompts for each of the 13 tasks, native top-eight "
        "prefill with the 16K exact prefix, and region c=16 in decode. The Multikey-2 and "
        "Multikey-3 results come from the original diagnostic on the same frozen source. "
        "The other 11 tasks are fresh runs. Exact-work columns use the Multikey-2 and "
        "Multikey-3 diagnostic, averaged over requests, query heads, and layers at the last "
        "decode step. Dense quality uses the matched archived cohort.", "",
        "| Refinement | RULER mean | Delta vs two-tier (pp) | Paired 95% interval | Exact pages per head | Exact remote entries per head |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in settings:
        changes = [[rows[variant, task][i]["score"] - rows["two-tier", task][i]["score"]
                    for i in sorted(cohorts[task])] for task in TASKS]
        bounds = paired_interval(changes) if variant != "two-tier" else None
        delta = means[variant] - means["two-tier"]
        work = diagnostic["retrieval"][variant]
        entries = mean(work[task]["exact_entries_per_head"] for task in PAGE_TASKS)
        pages = (None if variant == "two-tier" else
                 mean(work[task]["exact_pages_per_head"] for task in PAGE_TASKS))
        results[variant] = {
            "ruler_mean": means[variant], "delta_points": delta,
            "paired_bootstrap_95_interval_points": bounds,
            "exact_entries_per_head": entries, "exact_pages_per_head": pages,
        }
        interval = f"[{bounds[0]:+.2f}, {bounds[1]:+.2f}]" if bounds else "—"
        page_count = f"{pages:.1f}" if pages is not None else "—"
        lines.append(f"| {variant} | {means[variant]:.2f}% | {delta:+.2f} | {interval} | {page_count} | {entries:.1f} |")
    lines += ["", "| Task | Dense | " + " | ".join(settings) + " |",
              "|---|---:|" + "---:|" * len(settings)]
    for task in TASKS:
        lines.append(f"| {task} | {dense_tasks[task]:.2f}% | " +
                     " | ".join(f"{scores[v][task]:.2f}%" for v in settings) + " |")
    lines += [f'| **Macro-average** | **{dense["mean"]["archived_dense"]:.2f}%** | ' +
              " | ".join(f"**{means[v]:.2f}%**" for v in settings) + " |", "",
              "[New and reused commands](jobs.json) · "
              "[Matched decode timings](../k2-page-budget-fixed1-v1/REPORT.md)", ""]
    write(root / "REPORT.json", {
        "stage": "complete", "dense_score": dense["mean"]["archived_dense"],
        "results": results, "tasks": scores, "speed": diagnostic["speed"],
    })
    (root / "REPORT.md").write_text("\n".join(lines))
    print(root / "REPORT.md", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    root = parser.parse_args().output.resolve()
    manifest = read(root / "jobs.json")
    jobs = manifest["jobs"]
    expected = 5 * (len(TASKS) - len(PAGE_TASKS))
    if len(jobs) != expected:
        raise ValueError("Incomplete new job manifest")
    while True:
        complete, failures, examples = 0, [], 0
        for job in jobs:
            fields, data = job_state(job)
            examples += len(data.get("records", []))
            if fields and fields[7:9] == ["finished", "0"] and data.get("stage") == "complete":
                complete += 1
            elif fields:
                failures.append(job["id"])
        status = {"jobs": len(jobs), "complete": complete, "examples": examples, "failures": failures}
        write(root / "status.json", status)
        print(status, flush=True)
        if failures:
            raise RuntimeError(f"Failed runs {failures}")
        if complete == expected:
            collect(root, manifest)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
