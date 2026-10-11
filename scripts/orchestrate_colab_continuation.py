"""Run one Colab Adam update, archive its checkpoint locally, then run the next.

This trusted host-side script talks to an existing authenticated CDP daemon and
Colab tab. Every successful update is copied to a new local directory and
SHA-256 verified before another process is started. A disconnect or mismatch
stops the loop. Run with Python -I; never execute code from downloaded files.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_colab_checkpoint_download import _read as read_json


def cdp_eval(session_id, expression, *, await_promise=True, timeout=50):
    payload = json.dumps({"sessionId": session_id, "expression": expression,
                          "awaitPromise": await_promise, "timeout_seconds": timeout}).encode()
    request = urllib.request.Request("http://127.0.0.1:7799/eval", data=payload,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout + 15) as response:
        record = json.load(response)
    result = record.get("result")
    if not isinstance(result, dict) or "exceptionDetails" in result:
        raise RuntimeError(f"Colab browser execution failed: {str(record)[:600]}")
    return result.get("value")


def download(session_id, path):
    expression = ("colab.global.notebook.kernelFiles.download(" + json.dumps(str(path))
                  + ",{canceled:false,update:()=>{}}).then(b=>b.text())")
    content = cdp_eval(session_id, expression, timeout=60)
    if not isinstance(content, str):
        raise RuntimeError(f"Colab download did not return file text: {path}")
    return content.encode("utf-8")


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_new(path, contents):
    with Path(path).open("xb") as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(Path(path).parent)


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def read_remote_json(session_id, path):
    raw = download(session_id, path)
    return json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def verify_downloaded_checkpoint(folder, identity, expected_iteration):
    from verify_colab_checkpoint_download import main as verify
    verify(["--download-dir", str(folder), "--expected-identity", identity,
            "--expected-iteration", str(expected_iteration)])
    checkpoint = read_json(folder / "optimizer_state.json")
    last = checkpoint["state"]["history"][-1]
    if last["iteration"] != expected_iteration or last["diagnostics"].get("replay_consistent") is not True:
        raise ValueError(f"Update {expected_iteration} has no consistent strict backward replay")
    if checkpoint["state"].get("continuation_mismatches") != 0:
        raise ValueError("An earlier continued update had a replay mismatch")
    return checkpoint


def archive_checkpoint(session_id, run_dir, mirror_dir, archive_dir, identity, iteration):
    folder = archive_dir / f"update_{iteration:04d}"
    if folder.exists():
        return verify_downloaded_checkpoint(folder, identity, iteration)
    folder.mkdir(exist_ok=False)
    try:
        save_new(folder / "optimizer_state.json", download(session_id, run_dir / "optimizer_state.json"))
        save_new(folder / "checkpoint_sha256.json", download(session_id, mirror_dir / "checkpoint_sha256.json"))
        checkpoint = verify_downloaded_checkpoint(folder, identity, iteration)
        save_new(folder / "result.json", download(session_id, run_dir / "result.json"))
        result = read_json(folder / "result.json")
        if result.get("iterations") != iteration or result.get("status") not in {"paused", "budget_exhausted"}:
            raise ValueError(f"Update {iteration} has no completed fit result")
        _fsync_directory(folder)
        return checkpoint
    except BaseException:
        # Keep the incomplete directory for diagnosis; a retry must use a new archive root.
        raise


def _wrapper_code(command, status_path, logfile, iteration):
    return f'''import datetime,json,os,pathlib,subprocess,sys,uuid
cmd={command!r}
cmd[0]=sys.executable
status=pathlib.Path({str(status_path)!r})
logfile=pathlib.Path({str(logfile)!r})
iteration={iteration!r}

def write_status(record):
    temp=status.with_name("."+status.name+"."+uuid.uuid4().hex+".tmp")
    with temp.open("x") as stream:
        json.dump(record,stream,sort_keys=True)
        stream.write("\\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp,status)

with logfile.open("x") as output:
    child=subprocess.Popen(cmd,stdout=output,stderr=subprocess.STDOUT,
                           cwd={str(Path(command[2]).resolve().parents[1])!r},start_new_session=True)
    write_status({{"iteration":iteration,"phase":"running","pid":child.pid,
                  "started_at":datetime.datetime.now(datetime.timezone.utc).isoformat()}})
    code=child.wait()
    write_status({{"iteration":iteration,"phase":"finished","pid":child.pid,
                  "exit_code":code,"finished_at":datetime.datetime.now(datetime.timezone.utc).isoformat()}})
'''


def launch_one(session_id, driver, dataset, original_run, run_dir, mirror_dir, iteration, *, partial):
    status_dir = run_dir / "host_orchestration"
    status_path = status_dir / f"update_{iteration:04d}.json"
    logfile = status_dir / f"update_{iteration:04d}.log"
    command = [sys.executable, "-I", str(driver), "--dataset", str(dataset),
               "--original-run", str(original_run), "--output-dir", str(run_dir),
               "--mirror-dir", str(mirror_dir), "--target-updates", "171",
               "--max-updates-this-run", "1", "--resume"]
    if partial:
        command.append("--allow-partial-evaluation-after-user-request")
    wrapper = _wrapper_code(command, status_path, logfile, iteration)
    notebook_code = ("import pathlib,subprocess,sys\n"
                     f"d=pathlib.Path({str(status_dir)!r});d.mkdir(exist_ok=True)\n"
                     f"p=d / {status_path.name!r}\n"
                     "assert not p.exists(), 'Update already submitted'\n"
                     f"w=subprocess.Popen([sys.executable,'-I','-c',{wrapper!r}],"
                     "cwd=str(d.parent),stdout=subprocess.DEVNULL,"
                     "stderr=subprocess.DEVNULL,start_new_session=True)\n"
                     "print('CONTINUATION_WRAPPER_PID',w.pid)\n")
    expression = ("(()=>{colab.global.notebook.kernel.execute(" + json.dumps(notebook_code)
                  + ");return 'submitted'})()")
    if cdp_eval(session_id, expression, await_promise=False, timeout=30) != "submitted":
        raise RuntimeError("Colab did not accept one-update launch")
    return status_path, logfile


def wait_one(session_id, status_path, iteration, *, max_seconds):
    deadline = time.monotonic() + max_seconds
    first_status_deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            status = read_remote_json(session_id, status_path)
        except Exception:
            if time.monotonic() > first_status_deadline:
                raise  # A browser or Colab disconnect stops all further updates.
            time.sleep(5)
            continue
        if status.get("iteration") != iteration:
            raise ValueError("Colab process status has a different iteration")
        if status.get("phase") == "finished":
            if status.get("exit_code") != 0:
                raise RuntimeError(f"Colab update {iteration} failed with exit {status.get('exit_code')}")
            print(f"Update {iteration} process exited 0 (PID {status.get('pid')})", flush=True)
            return status
        if status.get("phase") != "running" or type(status.get("pid")) is not int:
            raise ValueError("Invalid Colab process status")
        time.sleep(30)
    raise TimeoutError(f"Colab update {iteration} did not finish within {max_seconds}s")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--driver", type=Path, required=True,
                        help="Trusted driver path inside the existing Colab workspace")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mirror-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True,
                        help="New LOCAL host directory; never a path on the Colab runtime")
    parser.add_argument("--start-at", type=int, default=22, help="First already accepted update to archive")
    parser.add_argument("--stop-at", type=int, default=171, help="Last update to run and archive")
    parser.add_argument("--max-update-seconds", type=int, default=14400)
    parser.add_argument("--allow-partial-evaluation-after-user-request", action="store_true")
    args = parser.parse_args(argv)
    if args.start_at < 22 or not args.start_at <= args.stop_at <= 171:
        parser.error("Require completed start update >=22 and target <=171")
    if args.max_update_seconds < 60:
        parser.error("max-update-seconds must be at least 60")
    run_dir = args.run_dir
    mirror_dir = args.mirror_dir
    local_root = args.archive_dir.resolve()
    if local_root.exists() and any(local_root.iterdir()):
        parser.error("Archive directory must be new and empty")
    local_root.mkdir(parents=True, exist_ok=True)
    manifest_bytes = download(args.session_id, run_dir / "run_manifest.json")
    manifest = json.loads(manifest_bytes)
    identity = manifest.get("identity", {})
    if (identity.get("action") != "dataset-fit-minibatch-continuation"
            or identity.get("origin_iteration") != 21 or identity.get("target_iteration") != 171
            or identity.get("partial_evaluation_after_user_request")
            is not args.allow_partial_evaluation_after_user_request):
        raise ValueError("Remote run is not the requested exact 21-to-171 continuation")
    run_hash = manifest["identity_sha256"]
    save_new(local_root / "run_manifest.json", manifest_bytes)
    save_new(local_root / "validated_inputs.json", download(args.session_id, run_dir / "validated_inputs.json"))
    print(f"Archiving continuation identity {run_hash} to {local_root}", flush=True)
    for iteration in range(args.start_at, args.stop_at + 1):
        if iteration > args.start_at:
            status_path, log_path = launch_one(args.session_id, args.driver, args.dataset,
                                               args.original_run, run_dir, mirror_dir,
                                               iteration, partial=args.allow_partial_evaluation_after_user_request)
            print(f"Launched update {iteration}; status {status_path}; log {log_path}", flush=True)
            wait_one(args.session_id, status_path, iteration, max_seconds=args.max_update_seconds)
        checkpoint = archive_checkpoint(args.session_id, run_dir, mirror_dir, local_root,
                                        run_hash, iteration)
        e = checkpoint["state"]["physical_parameters"].get("youngs_modulus")
        if e is not None and (e <= 7800 or e >= 54200):
            print(f"WARNING: Young's modulus {e:.6g} Pa near [2000, 60000] Pa bound at update {iteration}",
                  flush=True)
        print(f"LOCAL_CHECKPOINT_VERIFIED iteration={iteration} E={e} "
              f"at={datetime.now(timezone.utc).isoformat()}", flush=True)
    print(f"All updates through {args.stop_at} copied to local host and verified", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
