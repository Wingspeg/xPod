#!/usr/bin/env python3
# collect_metrics.py: 从集群查询 pod/vcjob 时间戳并落盘 metrics.csv；支持按 run_id 关联 replay_log 字段以补全可观测性指标。Modified: 2026-04-28
import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


def parse_rfc3339(s: Any) -> Optional[datetime]:
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None
    try:
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        return datetime.fromisoformat(t)
    except Exception:
        return None


def dt_to_ms(dt: Optional[datetime]) -> Optional[int]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def parse_number(x: Any, default: Optional[float] = None) -> Optional[float]:
    if x is None:
        return default
    s = str(x).strip()
    if s == "":
        return default
    try:
        return float(s)
    except Exception:
        return default


def kubectl_get_json(resource: str, kube_context: str, namespace: str, selector: str) -> Optional[Dict[str, Any]]:
    cmd = ["kubectl"]
    if kube_context:
        cmd += ["--context", kube_context]
    if namespace:
        cmd += ["-n", namespace]
    cmd += ["get", resource]
    if selector:
        cmd += ["-l", selector]
    cmd += ["-o", "json"]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        return None
    try:
        return json.loads(res.stdout.decode("utf-8"))
    except Exception:
        return None


def get_vcjobs_json(kube_context: str, namespace: str) -> Optional[Dict[str, Any]]:
    j = kubectl_get_json("vcjob", kube_context, namespace, "")
    if j is not None:
        return j
    return kubectl_get_json("jobs.batch.volcano.sh", kube_context, namespace, "")


def get_pods_json(kube_context: str, namespace: str) -> Optional[Dict[str, Any]]:
    j = kubectl_get_json("pods", kube_context, namespace, "volcano.sh/job-name")
    if j is not None:
        return j
    return kubectl_get_json("pods", kube_context, namespace, "xpod")


def first_container_status(pod: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sts = (((pod.get("status") or {}).get("containerStatuses")) or [])
    if not sts:
        return None
    return sts[0]


def pod_scheduled_node(pod: Dict[str, Any]) -> str:
    return str((((pod.get("spec") or {}).get("nodeName")) or "")).strip()


def pod_scheduler_name(pod: Dict[str, Any]) -> str:
    return str((((pod.get("spec") or {}).get("schedulerName")) or "")).strip()


def pod_start_time(pod: Dict[str, Any]) -> Optional[datetime]:
    st = parse_rfc3339(((pod.get("status") or {}).get("startTime")))
    c0 = first_container_status(pod)
    if c0:
        running = ((c0.get("state") or {}).get("running")) or {}
        rt = parse_rfc3339(running.get("startedAt"))
        if rt is not None:
            return rt
    return st


def pod_succeeded_time(pod: Dict[str, Any]) -> Optional[datetime]:
    c0 = first_container_status(pod)
    if c0:
        term = ((c0.get("state") or {}).get("terminated")) or {}
        ft = parse_rfc3339(term.get("finishedAt"))
        if ft is not None:
            return ft
    conds = ((pod.get("status") or {}).get("conditions")) or []
    for c in conds:
        if str(c.get("type") or "") == "Ready" and str(c.get("status") or "") == "False":
            dt = parse_rfc3339(c.get("lastTransitionTime"))
            if dt is not None:
                return dt
    return None


def volcano_job_phase(vcjob: Dict[str, Any]) -> str:
    status = vcjob.get("status") or {}
    state = status.get("state") or {}
    p = state.get("phase")
    if p:
        return str(p)
    p = status.get("phase")
    if p:
        return str(p)
    return ""


def volcano_job_completed_time(vcjob: Dict[str, Any]) -> Optional[datetime]:
    status = vcjob.get("status") or {}
    for k in ["completionTime", "completedTime", "finishTime", "finishedAt", "endTime"]:
        dt = parse_rfc3339(status.get(k))
        if dt is not None:
            return dt
    state = status.get("state") or {}
    dt = parse_rfc3339(state.get("lastTransitionTime"))
    if dt is not None:
        return dt
    dt = parse_rfc3339(status.get("lastTransitionTime"))
    if dt is not None:
        return dt
    return None


def vcjob_name(vcjob: Dict[str, Any]) -> str:
    return str((((vcjob.get("metadata") or {}).get("name")) or "")).strip()


def pod_job_name(pod: Dict[str, Any]) -> str:
    labels = ((pod.get("metadata") or {}).get("labels")) or {}
    for k in ["volcano.sh/job-name", "xpod"]:
        v = labels.get(k)
        if v:
            return str(v)
    return ""


@dataclass
class TaskRecord:
    xpod_id: str
    submit_time_s: float
    source: str
    scene: str
    scheduler_name: str
    first_seen_wall_s: float
    estimated_transfer_ms: str
    cold_start: str
    sla_ms: str
    mode: str
    run_id: str


def read_replay_log(path: str, watch_existing: bool) -> Dict[str, TaskRecord]:
    tasks: Dict[str, TaskRecord] = {}
    if not os.path.exists(path):
        return tasks
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            xpod_id = (row.get("xpod_id") or "").strip()
            if not xpod_id:
                continue
            ts = parse_number(row.get("wall_time"), None)
            if ts is None:
                ts = parse_number(row.get("submit_time"), None)
            if ts is None:
                continue
            source = (row.get("source") or "").strip()
            scene = (row.get("scene") or "").strip()
            scheduler_name = (row.get("scheduler_name") or "").strip()
            estimated_transfer_ms = (row.get("estimated_transfer_ms") or "").strip()
            cold_start = (row.get("cold_start") or "").strip()
            sla_ms = (row.get("sla_ms") or "").strip()
            mode = (row.get("mode") or "").strip()
            run_id = (row.get("run_id") or "").strip()
            tasks[xpod_id] = TaskRecord(
                xpod_id=xpod_id,
                submit_time_s=float(ts),
                source=source,
                scene=scene,
                scheduler_name=scheduler_name,
                first_seen_wall_s=time.time(),
                estimated_transfer_ms=estimated_transfer_ms,
                cold_start=cold_start,
                sla_ms=sla_ms,
                mode=mode,
                run_id=run_id,
            )
    return tasks


def write_header_if_needed(path: str, fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()


def append_rows(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    write_header_if_needed(path, fieldnames)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        for r in rows:
            w.writerow(r)


def main(argv: List[str]) -> int:
    default_log_file = "results/raw/replay_log.csv"
    default_output = "results/raw/metrics.csv"

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="")
    ap.add_argument("--replay-log", default=default_log_file)
    ap.add_argument("--output", default=default_output)
    ap.add_argument("--poll-interval", type=float, default=5.0)
    ap.add_argument("--job-timeout", type=float, default=30.0)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--namespace", default="default")
    ap.add_argument("--kube-context", default="")
    ap.add_argument("--watch-existing", action="store_true")
    ap.add_argument("--scene", default="")
    ap.add_argument("--scheduler", dest="scheduler_name", default="")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args(argv)

    from scheduler.logging_config import setup_logging

    setup_logging(args.log_level, args.log_file)
    logger.info("main start args=%s", vars(args))

    if args.run_id:
        run_log = os.path.join("results", "raw", str(args.run_id), "replay_log.csv")
        run_out = os.path.join("results", "raw", str(args.run_id), "metrics.csv")
        if args.replay_log == default_log_file:
            args.replay_log = run_log
        if args.output == default_output:
            args.output = run_out

    fieldnames = [
        "run_id",
        "mode",
        "xpod_id",
        "submit_time",
        "pod_start_time",
        "pod_succeeded_time",
        "volcano_job_completed_time",
        "queue_wait_ms",
        "execution_ms",
        "jct_ms",
        "scheduled_node",
        "scheduler_name",
        "source",
        "scene",
        "status",
        "estimated_transfer_ms",
        "cold_start",
        "sla_ms",
        "sla_violated",
    ]

    write_header_if_needed(args.output, fieldnames)

    replay_by_id = read_replay_log(args.replay_log, False)
    tasks = {} if args.watch_existing else dict(replay_by_id)
    start_wall = time.time()
    completed: Set[str] = set()

    while True:
        now = time.time()
        if args.timeout > 0 and (now - start_wall) > args.timeout * 60.0:
            break

        if args.watch_existing:
            j = get_vcjobs_json(args.kube_context, args.namespace)
            if j and isinstance(j.get("items"), list):
                for it in j["items"]:
                    name = vcjob_name(it)
                    if not name or name in tasks:
                        continue
                    ct = parse_rfc3339(((it.get("metadata") or {}).get("creationTimestamp")))
                    submit_time_s = float(ct.timestamp()) if ct else now
                    tasks[name] = TaskRecord(
                        xpod_id=name,
                        submit_time_s=submit_time_s,
                        source="",
                        scene=args.scene or "",
                        scheduler_name=args.scheduler_name or "",
                        first_seen_wall_s=now,
                        estimated_transfer_ms="",
                        cold_start="",
                        sla_ms="",
                        mode="",
                        run_id="",
                    )

        if not tasks:
            time.sleep(args.poll_interval)
            tasks = read_replay_log(args.replay_log, args.watch_existing)
            continue

        vcjobs = get_vcjobs_json(args.kube_context, args.namespace) or {"items": []}
        pods = get_pods_json(args.kube_context, args.namespace) or {"items": []}

        vcjob_by_name: Dict[str, Dict[str, Any]] = {}
        for it in vcjobs.get("items") or []:
            name = vcjob_name(it)
            if name:
                vcjob_by_name[name] = it

        pod_by_job: Dict[str, Dict[str, Any]] = {}
        for p in pods.get("items") or []:
            name = pod_job_name(p)
            if not name:
                continue
            phase = str(((p.get("status") or {}).get("phase")) or "")
            if name not in pod_by_job:
                pod_by_job[name] = p
                continue
            if phase == "Running":
                pod_by_job[name] = p
            elif phase == "Succeeded" and str(((pod_by_job[name].get("status") or {}).get("phase")) or "") != "Running":
                pod_by_job[name] = p

        out_rows: List[Dict[str, Any]] = []
        to_remove: List[str] = []

        for xpod_id, rec in list(tasks.items()):
            if xpod_id in completed:
                to_remove.append(xpod_id)
                continue

            submit_dt = datetime.fromtimestamp(rec.submit_time_s, tz=timezone.utc)
            submit_ms = int(rec.submit_time_s * 1000)

            vcjob = vcjob_by_name.get(xpod_id)
            vc_phase = volcano_job_phase(vcjob) if vcjob else ""
            vc_done_dt = volcano_job_completed_time(vcjob) if vcjob else None

            pod = pod_by_job.get(xpod_id)
            pod_start_dt = pod_start_time(pod) if pod else None
            pod_succ_dt = pod_succeeded_time(pod) if pod else None

            status = ""
            if vc_phase:
                status = vc_phase
            elif pod:
                status = str(((pod.get("status") or {}).get("phase")) or "")

            timed_out = False
            if args.job_timeout > 0:
                if (now - rec.first_seen_wall_s) > args.job_timeout * 60.0:
                    timed_out = True

            is_done = False
            if vc_phase in {"Completed", "Failed"}:
                is_done = True
            if timed_out:
                is_done = True
                status = "Timeout"

            if not is_done:
                continue

            vc_done_ms = dt_to_ms(vc_done_dt)
            pod_start_ms = dt_to_ms(pod_start_dt)
            pod_succ_ms = dt_to_ms(pod_succ_dt)

            queue_wait_ms = (pod_start_ms - submit_ms) if (pod_start_ms is not None) else ""
            execution_ms = (pod_succ_ms - pod_start_ms) if (pod_succ_ms is not None and pod_start_ms is not None) else ""
            jct_ms = (vc_done_ms - submit_ms) if (vc_done_ms is not None) else ""

            extra = replay_by_id.get(xpod_id)
            estimated_transfer_ms = extra.estimated_transfer_ms if extra else ""
            cold_start = extra.cold_start if extra else ""
            sla_ms = extra.sla_ms if extra else ""
            mode = extra.mode if extra else ""
            run_id = extra.run_id if extra else ""

            sla_violated: Any = ""
            sla_ms_num = parse_number(sla_ms, None)
            jct_ms_num = parse_number(jct_ms, None)
            if sla_ms_num is not None and jct_ms_num is not None:
                sla_violated = bool(jct_ms_num > sla_ms_num)

            out_rows.append(
                {
                    "run_id": run_id,
                    "mode": mode,
                    "xpod_id": xpod_id,
                    "submit_time": submit_dt.isoformat(),
                    "pod_start_time": pod_start_dt.isoformat() if pod_start_dt else "",
                    "pod_succeeded_time": pod_succ_dt.isoformat() if pod_succ_dt else "",
                    "volcano_job_completed_time": vc_done_dt.isoformat() if vc_done_dt else "",
                    "queue_wait_ms": queue_wait_ms,
                    "execution_ms": execution_ms,
                    "jct_ms": jct_ms,
                    "scheduled_node": pod_scheduled_node(pod) if pod else "",
                    "scheduler_name": rec.scheduler_name or args.scheduler_name or "",
                    "source": rec.source or "",
                    "scene": rec.scene or args.scene or "",
                    "status": status,
                    "estimated_transfer_ms": estimated_transfer_ms,
                    "cold_start": cold_start,
                    "sla_ms": sla_ms,
                    "sla_violated": ("" if sla_violated == "" else str(bool(sla_violated))),
                }
            )
            completed.add(xpod_id)
            to_remove.append(xpod_id)

        if out_rows:
            append_rows(args.output, fieldnames, out_rows)

        for k in to_remove:
            tasks.pop(k, None)

        if not tasks:
            if not args.watch_existing:
                break

        time.sleep(max(0.1, float(args.poll_interval)))
        latest = read_replay_log(args.replay_log, args.watch_existing)
        if latest:
            replay_by_id.update(latest)
        for xpod_id, rec in latest.items():
            if xpod_id not in tasks and xpod_id not in completed:
                tasks[xpod_id] = rec

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
