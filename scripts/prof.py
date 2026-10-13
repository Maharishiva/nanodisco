"""Where the GPU time of a script goes: runs it under jax.profiler and splits the kernel time into parts.

usage: python scripts/prof.py OUT [--graphs] -- scripts/rule_paper.py ARGS...
Wraps mini's parts in named scopes (env, act, agent, q_stats, rule, grad, adam, replay, critic, meta), runs the script
with these scopes, dumps the optimized HLO (to map kernels to scopes) and traces it, then prints the GPU time per part:
fwd = forward ops, bwd = their transposes (the gradient), "meta" marks the outer meta-gradient pass. Idle is the time in
which no kernel runs (launch latency, host syncs, loop control). By default CUDA graphs are off, so every kernel maps
to its HLO op; --graphs keeps them (as in normal runs: the idle share is then the real one, but library kernels such as
convolutions and matmuls inside a graph show as "graph fprop|dgrad|wgrad|gemm|other", by kernel name).
The trace keeps about 1M kernels: keep runs short, or trace a window of the run with --start S --duration D (seconds
after launch, e.g. past the compile). --again: analyze OUT's trace again.
"""

import argparse
import collections
import functools
import glob
import gzip
import json
import os
import re
import runpy
import sys
import threading
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("out", type=Path)
p.add_argument("--graphs", action="store_true", help="keep CUDA graphs (command buffers)")
p.add_argument("--top", type=int, default=0, help="also print the op names with the most GPU time")
p.add_argument("--again", action="store_true", help="only analyze the trace already in OUT")
p.add_argument("--start", type=float, default=0, help="seconds after launch at which the trace starts")
p.add_argument("--duration", type=float, default=0, help="seconds of trace (0: until the script ends)")
p.add_argument("script")
p.add_argument("args", nargs=argparse.REMAINDER)
a = p.parse_args()
hlo = a.out / "hlo"
flags = "" if a.again else f" --xla_dump_to={hlo} --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=^$"
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + flags + ("" if a.graphs else
                                                                     " --xla_gpu_enable_command_buffer=")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import jax

import mini as m

def scoped(name, f):
    @functools.wraps(f)
    def g(*args, **kwargs):
        with jax.named_scope(name):
            return f(*args, **kwargs)
    return g

PARTS = dict(agent_step="act", unroll="agent", q_stats="q_stats", features="rule", rule="rule",
             inner_grad="grad", inner_loss="grad", adam="adam", store="replay", sample="replay", fit_critic="critic",
             critic_targets="critic", draw_slots="other", init_agents="other")
for f, name in PARTS.items():
    setattr(m, f, scoped(name, getattr(m, f)))
m.collect = scoped("rollout", m.collect)
if "import bench" in Path(a.script).read_text():  # bench sets mini's B, ROLL, RETRACE and obs: only for its users
    import bench
    make_env = bench.make_env
    def scoped_env(name):  # rule_paper sets bench.ENV_STEP from make_env
        obs, reset, step = make_env(name)
        return obs, reset, scoped("env", step)
    bench.make_env, bench.rollout = scoped_env, scoped("rollout", bench.rollout)
for module in ("craftax_pixels", "arc", "minatar"):  # env modules, if their imports work here
    try:
        mod = __import__(module)
        mod.step = scoped("env", mod.step)
    except Exception:  # noqa: BLE001 (craftax needs its deps)
        pass

sys.argv = [a.script] + [x for x in a.args if x != "--"]
trace = a.out / "trace"
options = jax.profiler.ProfileOptions()
options.host_tracer_level, options.python_tracer_level = 0, 0  # GPU events only: the trace holds at most 1M events
tracing = threading.Lock()  # held while a trace runs

def start():
    with tracing:
        jax.profiler.start_trace(str(trace), create_perfetto_trace=True, profiler_options=options)
        start.on = True

def stop():
    with tracing:
        if getattr(start, "on", False):
            jax.profiler.stop_trace()
            start.on = False

if not a.again:
    threading.Timer(a.start, start).start()
    if a.duration:
        threading.Timer(a.start + a.duration, stop).start()
    runpy.run_path(a.script, run_name="__main__")
    stop()

ops = {}  # (module, op) -> op_name
for path in glob.glob(str(hlo / "*after_optimizations.txt")):
    module = re.search(r"module_\d+\.(.+?)\.sm_", Path(path).name)
    if module is None:  # not a GPU program
        continue
    module = module.group(1)
    ops.setdefault(module, {})
    for line in open(path):
        x = re.match(r"\s*(?:ROOT )?%?([\w.\-]+) = .*?op_name=\"([^\"]*)\"", line)
        if x:
            ops[module][x.group(1)] = ops[module][x.group(1).replace(".", "_")] = x.group(2)
events = json.load(gzip.open(glob.glob(str(trace / "**" / "perfetto_trace.json.gz"), recursive=True)[-1]))
events = events["traceEvents"] if isinstance(events, dict) else events
gpu = {e["pid"] for e in events if e.get("ph") == "M" and "GPU" in str(e.get("args", {}).get("name", ""))}
by_size = sorted(ops, key=lambda k: -len(ops[k]))

def module_of(e):  # a trace started mid-run has no module names: the largest program that has the kernel's name
    args = e.get("args", {})
    if "hlo_module" in args:
        return args["hlo_module"] if args["hlo_module"] in ops else None  # None: autotuning during compiles
    return next((k for k in by_size if e["name"] in ops[k] or args.get("hlo_op") in ops[k]), None)

kernels = [e for e in events if e.get("ph") == "X" and e["pid"] in gpu and module_of(e)]
for e in kernels:
    e["module"] = module_of(e)
SCOPES = ("env", "act", "rollout", "q_stats", "rule", "adam", "replay", "critic", "agent", "grad", "other")

def op_name(e):  # None: a library kernel inside a CUDA graph
    table = ops[e["module"]]
    return table.get(e["args"].get("hlo_op")) or table.get(e["name"])

def part(e):  # innermost scope; bwd: under a transpose at or inside the outermost scope; meta: transposed outside it
    op = op_name(e)
    if op is None:  # library kernels: by name
        kind = next((k for k in ("fprop", "dgrad", "wgrad", "gemm") if k in e["name"].lower()), "other")
        return ("graph " if "command_buffer" in e["args"].get("hlo_op", "") else "unmapped ") + kind
    if "Memcpy" in e["name"]:
        return "memcpy"
    path = [(x, re.findall(r"\w+", x)[-1] if re.findall(r"\w+", x) else "") for x in op.split("/")]
    where = [i for i, (_, name) in enumerate(path) if name in SCOPES]
    if not where:
        return "other"
    names = [path[i][1] for i in where]
    scope = names[-1] if names[-1] not in ("act", "env") else "rollout:" + names[-1]
    if scope == "agent" and "grad" in names:
        scope = "grad:agent"  # the agent's forward pass inside the gradient of the inner loss
    meta = "meta " if any("transpose(" in x for x, _ in path[: where[0]]) else ""
    direction = "bwd" if any("transpose(" in x for x, _ in path[where[0] :]) else "fwd"
    return f"{meta}{scope} {direction}"

busy, names = collections.Counter(), collections.Counter()
for e in kernels:
    busy[part(e)] += e["dur"]
    names[op_name(e) or e["name"]] += e["dur"]
per_module = collections.Counter()
for e in kernels:
    per_module[e["module"]] += e["dur"]
main = {k for k, t in per_module.items() if t > 0.05 * sum(per_module.values())}  # the span: from the main programs
spans = sorted((e["ts"], e["ts"] + e["dur"]) for e in kernels if e["module"] in main)
start, end, covered, last = spans[0][0], max(s[1] for s in spans), 0.0, spans[0][0]
for s, t in spans:  # union of kernel intervals
    covered += max(0.0, t - max(s, last))
    last = max(last, t)
wall = end - start
print(f"GPU span of {', '.join(sorted(main))}: {wall / 1e6:.2f}s, busy {covered / wall:.1%}, "
      f"idle {1 - covered / wall:.1%}; {len(kernels)} kernels")
for name, t in busy.most_common():
    print(f"  {name:24s} {t / 1e6:8.3f}s  {t / covered:6.1%} of busy")
for name, t in names.most_common(a.top):
    print(f"  {t / 1e6:8.3f}s  {name}")
