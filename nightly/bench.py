#!/usr/bin/env python3
"""
bench.py — feasibility harness for the Sunday-night local-model report.

The question this answers: given a 6-hour window (01:00-07:00 Sunday), how large a context
and how long a report can this machine actually produce with a given model — including the
disk-offloaded-MoE case where routed experts stream from SSD instead of living in RAM?

Two things dominate and BOTH are measured:
  pp (prompt processing / prefill)  — cost of reading the assembled context. Usually the
                                      binding constraint; every 1k of context costs real minutes.
  tg (token generation)             — cost of writing the report.

  total_seconds ≈ context_tokens / pp_rate  +  output_tokens / tg_rate

MoE offload is simulated with llama-bench's -ot: forcing `\\.ffn_.*_exps\\.weight=CPU` pushes
routed-expert FFN tensors out of Metal and into CPU/mmap, which is the same mechanism a
too-large model uses. Measuring the resident-vs-offloaded ratio on a model we already have
gives a scaling estimate WITHOUT first downloading 241GB.

IMPORTANT: uses the arm64 Homebrew binary explicitly. The x86_64 build in /usr/local/bin
shadows it in PATH and runs under Rosetta with no Metal device.

Usage:
  python nightly/bench.py --list                    # identify local GGUFs (arch, params, experts)
  python nightly/bench.py -m <gguf> --quick         # fast sanity run
  python nightly/bench.py -m <gguf> --offload       # add the expert-offload case
  python nightly/bench.py -m <gguf> --envelope      # print the feasibility envelope
"""
import os, sys, glob, json, struct, argparse, subprocess, time

LLAMA_BENCH = '/opt/homebrew/bin/llama-bench'      # arm64 + Metal. NOT the PATH one.
LLAMA_CLI = '/opt/homebrew/bin/llama-cli'
BLOBS = os.path.expanduser('~/.ollama/models/blobs')
WINDOW_SECONDS = 6 * 3600                          # 01:00 -> 07:00
EXPERT_PATTERN = r'\.ffn_.*_exps\.weight=CPU'

# ---------------------------------------------------------------- gguf metadata
_GT = {0: ('B', 1), 1: ('b', 1), 2: ('H', 2), 3: ('h', 2), 4: ('I', 4), 5: ('i', 4),
       6: ('f', 4), 7: ('?', 1), 10: ('Q', 8), 11: ('q', 8), 12: ('d', 8)}


def _skip_val(f, t):
    """Advance past a value without materialising it. Tokenizer vocabs are ~150k strings —
    reading them is slow and forgetting to skip them corrupts every subsequent offset."""
    if t == 8:
        f.seek(struct.unpack('<Q', f.read(8))[0], 1)
    elif t == 9:
        et, n = struct.unpack('<IQ', f.read(12))
        if et in _GT:
            f.seek(_GT[et][1] * n, 1)               # fixed width -> one seek
        else:
            for _ in range(n):
                _skip_val(f, et)                    # strings/nested -> walk it
    else:
        f.seek(_GT[t][1], 1)


def _read_val(f, t):
    if t == 8:                                      # string
        n = struct.unpack('<Q', f.read(8))[0]
        return f.read(n).decode('utf-8', 'replace')
    if t == 9:                                      # array — sample the head, skip the tail
        et, n = struct.unpack('<IQ', f.read(12))
        head = [_read_val(f, et) for _ in range(min(n, 8))]
        for _ in range(max(0, n - 8)):
            _skip_val(f, et)
        return head + (['...'] if n > 8 else [])
    fmt, sz = _GT[t]
    return struct.unpack('<' + fmt, f.read(sz))[0]


def gguf_meta(path, want=('general.architecture', 'general.name', 'general.size_label',
                          'general.file_type')):
    """Read GGUF header KVs. Returns dict of the interesting keys + any *expert* keys."""
    out = {}
    with open(path, 'rb') as f:
        if f.read(4) != b'GGUF':
            return None
        ver, ntensor, nkv = struct.unpack('<IQQ', f.read(20))
        out['_gguf_version'], out['_tensors'] = ver, ntensor
        for _ in range(nkv):
            klen = struct.unpack('<Q', f.read(8))[0]
            key = f.read(klen).decode('utf-8', 'replace')
            vt = struct.unpack('<I', f.read(4))[0]
            if key in want or 'expert' in key or key.endswith('.block_count'):
                out[key] = _read_val(f, vt)
            else:
                _skip_val(f, vt)                    # don't pay for what we don't want
    return out


def list_models():
    rows = []
    for p in sorted(glob.glob(os.path.join(BLOBS, 'sha256-*')), key=os.path.getsize, reverse=True):
        if os.path.getsize(p) < 1 << 30:
            continue
        try:
            m = gguf_meta(p)
        except Exception:
            m = None
        if not m:
            continue
        exp = next((v for k, v in m.items() if k.endswith('expert_count')), 0)
        rows.append(dict(path=p, gb=os.path.getsize(p) / 1e9,
                         arch=m.get('general.architecture', '?'),
                         name=m.get('general.name', '?'),
                         size=m.get('general.size_label', '?'),
                         experts=exp))
    return rows


# ---------------------------------------------------------------- benchmark
def run_bench(model, prompts, gens, offload=False, reps=2, timeout=1800):
    """Invoke llama-bench, return list of dicts with pp/tg throughput."""
    cmd = [LLAMA_BENCH, '-m', model, '-r', str(reps), '-o', 'json',
           '-p', ','.join(str(p) for p in prompts),
           '-n', ','.join(str(g) for g in gens)]
    if offload:
        cmd += ['-ot', EXPERT_PATTERN]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  !! timed out after {timeout}s"); return []
    if r.returncode != 0:
        print('  !! llama-bench failed:'); print('   ', (r.stderr or '').strip()[-600:]); return []
    try:
        data = json.loads(r.stdout)
    except Exception:
        print('  !! could not parse json:'); print(r.stdout[-600:]); return []
    print(f"  ({time.time()-t0:.0f}s wall)")
    return data


def summarise(data):
    """-> (pp_rate, tg_rate) tok/s, averaged over reported rows."""
    pp = [d['avg_ts'] for d in data if d.get('n_prompt', 0) > 0]
    tg = [d['avg_ts'] for d in data if d.get('n_gen', 0) > 0]
    return (sum(pp) / len(pp) if pp else None, sum(tg) / len(tg) if tg else None)


# ---------------------------------------------------------------- envelope
def envelope(pp_rate, tg_rate, window=WINDOW_SECONDS):
    print(f"\n  FEASIBILITY ENVELOPE — {window/3600:.0f}h window "
          f"(prefill {pp_rate:.1f} tok/s, gen {tg_rate:.2f} tok/s)")
    print(f"  {'context':>10}{'prefill':>12}{'left for gen':>15}{'max report':>13}")
    for ctx in (2000, 4000, 8000, 16000, 32000, 64000):
        pre = ctx / pp_rate
        left = window - pre
        if left <= 0:
            print(f"  {ctx:>10,}{pre/60:>10.0f}m{'— EXCEEDS WINDOW':>15}{'':>13}")
            continue
        print(f"  {ctx:>10,}{pre/60:>10.0f}m{left/60:>13.0f}m{left*tg_rate:>12,.0f} tok")
    print(f"\n  (a useful report is ~1,500-3,000 tokens; budget 30% headroom for load time)")


# ---------------------------------------------------------------- disk + projection
F_NOCACHE = 48   # macOS fcntl: bypass the unified buffer cache


def disk_throughput(path, chunk_mb=8, n=256, seed=1):
    """Random-access read throughput — the physical floor for any disk-streamed model.

    F_NOCACHE is essential: without it a file touched by an earlier read is served from
    page cache and reports 2-3x the true device throughput, which would make a
    disk-streamed model look far more feasible than it is."""
    import random, fcntl
    sz = os.path.getsize(path); chunk = chunk_mb << 20
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.fcntl(fd, F_NOCACHE, 1)
    except OSError:
        print('  (warning: F_NOCACHE unavailable — number may be cache-inflated)')
    random.seed(seed); t0 = time.time(); tot = 0
    for _ in range(n):
        os.lseek(fd, random.randrange(0, sz - chunk), 0)
        tot += len(os.read(fd, chunk))
    dt = time.time() - t0; os.close(fd)
    return tot / 1e9 / dt


def project_streamed(gb_total, b_total, b_active, gbps, metal_gb=53.0, window_h=6.0,
                     shared_frac=0.20):
    """Estimate prefill/gen throughput for a model too large to hold resident.

    Generation streams the ROUTED-EXPERT slice of the active params for every token.
    Prefill streams each needed expert ONCE per batch and amortises it over the batch,
    so it is dramatically cheaper per token — which is why context is survivable here."""
    bytes_per_param = gb_total * 1e9 / b_total
    active_gb = b_active * bytes_per_param / 1e9
    routed_gb = active_gb * (1 - shared_frac)          # shared weights stay resident
    resident_frac = min(1.0, metal_gb / gb_total)
    stream_gb = routed_gb * (1 - resident_frac)

    tg = gbps / stream_gb                              # tokens/sec, I/O-bound
    batch = 512
    per_batch_gb = (gb_total - metal_gb) * 0.75        # a batch touches most experts once
    pp = batch / (per_batch_gb / gbps)

    print(f"\n  PROJECTION — {gb_total:.0f}GB model, {b_total/1e9:.0f}B total / {b_active/1e9:.0f}B active")
    print(f"    effective {bytes_per_param*8:.2f} bits/param | active slice {active_gb:.1f} GB/token")
    print(f"    resident in Metal: {resident_frac*100:.0f}% ({metal_gb:.0f}GB) | streamed/token: {stream_gb:.1f} GB")
    print(f"    measured disk: {gbps:.2f} GB/s")
    print(f"    -> prefill ~{pp:.0f} tok/s (amortised) | generate ~{tg:.2f} tok/s")
    print(f"    NOTE: I/O-bound ceiling. Real throughput will be LOWER — page-fault")
    print(f"          overhead, compute, and imperfect expert locality are not modelled.")
    for factor, lbl in ((1.0, 'ceiling'), (0.5, 'realistic'), (0.25, 'pessimistic')):
        envelope(pp * factor, tg * factor, window_h * 3600)
        print(f"    ^^^ {lbl} ({factor:.0%} of ceiling)")
    return pp, tg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='store_true', help='identify local GGUF models')
    ap.add_argument('-m', '--model', help='path to GGUF')
    ap.add_argument('--quick', action='store_true', help='tiny run to prove the harness works')
    ap.add_argument('--offload', action='store_true', help='also measure with experts forced to CPU')
    ap.add_argument('--envelope', action='store_true', help='print feasibility envelope')
    ap.add_argument('--window', type=float, default=6.0, help='hours available (default 6)')
    ap.add_argument('--disk', action='store_true', help='measure random-read throughput')
    ap.add_argument('--project', action='store_true',
                    help='project feasibility for a disk-streamed model (default: GLM-5.2 2-bit)')
    ap.add_argument('--gb', type=float, default=241.0, help='projected model size on disk, GB')
    ap.add_argument('--btotal', type=float, default=744e9, help='total params')
    ap.add_argument('--bactive', type=float, default=40e9, help='active params per token')
    ap.add_argument('--gbps', type=float,
                    help='override measured disk throughput (use when page cache inflates it; '
                         'M1 Max SSD spec is ~7.4 GB/s, so 5.0 is a defensible conservative figure)')
    a = ap.parse_args()

    if a.disk or a.project:
        if a.gbps:
            gbps = a.gbps
            print(f"\n  using supplied disk throughput: {gbps:.2f} GB/s")
        else:
            probe = a.model or max(glob.glob(os.path.join(BLOBS, 'sha256-*')), key=os.path.getsize)
            gbps = disk_throughput(probe)
            print(f"\n  measured random-read throughput: {gbps:.2f} GB/s  ({os.path.basename(probe)[:20]})")
            print("  (if this exceeds ~7 GB/s on an M1 Max it is page-cache inflated — "
                  "re-run with --gbps 5.0)")
        if a.project:
            project_streamed(a.gb, a.btotal, a.bactive, gbps, window_h=a.window)
        if not a.model:
            return

    if a.list or not a.model:
        rows = list_models()
        print(f"{'GB':>7}  {'arch':<12}{'experts':>8}  {'size':<8} name")
        for r in rows:
            print(f"{r['gb']:>7.1f}  {r['arch']:<12}{str(r['experts']):>8}  {r['size']:<8} {r['name'][:38]}")
            print(f"         {r['path']}")
        if not a.model:
            print("\nPick one with -m <path>.  MoE models (experts>0) are the relevant test.")
            return

    prompts = [512] if a.quick else [512, 2048, 8192]
    gens = [16] if a.quick else [32, 128]

    print(f"\n=== RESIDENT (Metal) — {os.path.basename(a.model)[:24]} ===")
    base = run_bench(a.model, prompts, gens)
    if not base:
        sys.exit('resident benchmark failed; fix that before interpreting anything else')
    pp0, tg0 = summarise(base)
    print(f"  prefill {pp0:>8.1f} tok/s | generate {tg0:>8.2f} tok/s")

    pp1 = tg1 = None
    if a.offload:
        print(f"\n=== EXPERTS OFFLOADED TO CPU ({EXPERT_PATTERN}) ===")
        off = run_bench(a.model, prompts, gens, offload=True, timeout=3600)
        if off:
            pp1, tg1 = summarise(off)
            print(f"  prefill {pp1:>8.1f} tok/s | generate {tg1:>8.2f} tok/s")
            print(f"\n  OFFLOAD PENALTY: prefill {pp0/pp1:.1f}x slower | generate {tg0/tg1:.1f}x slower")
            print("  ^ use this ratio to estimate a model that CANNOT fit resident at all")

    if a.envelope:
        w = a.window * 3600
        envelope(pp0, tg0, w)
        if pp1:
            print("\n  --- with experts offloaded ---")
            envelope(pp1, tg1, w)


if __name__ == '__main__':
    main()
