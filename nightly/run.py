#!/usr/bin/env python3
"""
run.py — orchestrate the weekly report: assemble context -> generate -> score -> write.

Deliberately model-agnostic. The point is an A/B: the same context goes to every model,
each report is scored by the same objective checks, and the outputs sit side by side so
they can be read blind. Models are tried FASTEST FIRST so that a report always exists
even if a slow one blows its deadline.

  ollama:<name>   -> local ollama HTTP API (fast, Metal-resident)
  gguf:<path>     -> llama.cpp CPU-only mmap streaming (for models too large for Metal;
                     ~0.2 tok/s on a 64GB M1 Max — see nightly/bench.py)

Each model gets a hard deadline. Blowing it is a logged result, not a crash.

Usage:
  python nightly/run.py                                  # default model set
  python nightly/run.py --models ollama:qwen3.6:35b-a3b-coding-mxfp8
  python nightly/run.py --deadline 3600                  # per-model seconds
"""
import os, sys, json, time, argparse, subprocess, datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import assemble as asm
import score as sc

LLAMA_CLI = '/opt/homebrew/bin/llama-cli'      # arm64+Metal build; NOT the x86 one in PATH
OLLAMA = 'http://localhost:11434/api/generate'
REPORTS = os.path.join(ROOT, 'reports')

DEFAULT_MODELS = [
    'ollama:qwen3.6:35b-a3b-coding-mxfp8',
    'ollama:llama3.3:70b',
]


def gen_ollama(model, prompt, deadline):
    """NOTE: qwen3.x are thinking models. Without think=false the entire num_predict budget
    is spent in the `thinking` field and `response` comes back EMPTY — same trap documented
    in oracle_debate.py. We disable thinking, and fall back to the thinking text only if a
    model ignores the flag."""
    import requests
    t0 = time.time()
    body = {'model': model, 'prompt': prompt, 'stream': False, 'think': False,
            'options': {'temperature': 0.3, 'num_ctx': 16384, 'num_predict': 4096}}
    r = requests.post(OLLAMA, json=body, timeout=deadline)
    d = r.json()
    txt = (d.get('response') or '').strip()
    if not txt:
        txt = (d.get('thinking') or '').strip()
        if txt:
            print("    (warning: model ignored think=false; using thinking field)")
    return txt, time.time() - t0


def gen_gguf(path, prompt, deadline):
    """CPU-only mmap streaming. -ngl 0 is REQUIRED for models >> RAM: on Apple Silicon,
    Metal memory is system RAM, so reserving it starves the page cache that mmap needs.
    Every GPU-offload variant OOMs at 254GB (see the config matrix in the repo notes)."""
    pf = os.path.join(HERE, '.prompt.tmp')
    open(pf, 'w').write(prompt)
    cmd = [LLAMA_CLI, '-m', path, '-ngl', '0', '-c', '16384', '-n', '3000',
           '--no-warmup', '-st', '--simple-io', '--temp', '0.3', '-f', pf]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=deadline)
        out = r.stdout
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b'').decode('utf-8', 'replace') if isinstance(e.stdout, bytes) else (e.stdout or '')
        out += "\n\n[TRUNCATED — deadline exceeded]"
    # strip llama.cpp banner/log lines
    lines = [l for l in out.splitlines() if not l.startswith(('build ', 'model ', 'ftype '))
             and not l.strip().startswith(('>', '[ Prompt:', 'available commands', '/'))]
    return '\n'.join(lines).strip(), time.time() - t0


def run_model(spec, prompt, deadline):
    kind, _, ref = spec.partition(':')
    print(f"\n>>> {spec}  (deadline {deadline}s)")
    try:
        if kind == 'ollama':
            txt, el = gen_ollama(ref, prompt, deadline)
        elif kind == 'gguf':
            txt, el = gen_gguf(ref, prompt, deadline)
        else:
            return None, f'unknown model kind: {kind}'
        print(f"    {len(txt):,} chars in {el/60:.1f} min "
              f"(~{len(txt)/3.8/max(el,1):.2f} tok/s)")
        return txt, None
    except Exception as e:
        print(f"    FAILED: {e}")
        return None, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', help='comma-separated; default: fast local set')
    ap.add_argument('--deadline', type=int, default=5400, help='seconds per model')
    ap.add_argument('--budget', type=int, default=asm.DEFAULT_BUDGET)
    a = ap.parse_args()

    models = a.models.split(',') if a.models else DEFAULT_MODELS
    today = dt.date.today().isoformat()
    outdir = os.path.join(REPORTS, today)
    os.makedirs(outdir, exist_ok=True)

    print(f"=== weekly report run {today} ===")
    ctx, toks = asm.assemble(a.budget)
    ctxf = os.path.join(outdir, 'context.txt')
    open(ctxf, 'w').write(ctx)
    print(f"context: ~{toks:,} tokens -> {ctxf}")

    prompt = open(os.path.join(HERE, 'prompt.md')).read() + "\n\n" + ctx

    results = {}
    for spec in models:
        txt, err = run_model(spec, prompt, a.deadline)
        safe = spec.replace(':', '_').replace('/', '_')
        if err or not txt:
            results[spec] = {'error': err or 'empty output'}
            continue
        rf = os.path.join(outdir, f'report__{safe}.md')
        open(rf, 'w').write(txt)
        s = sc.score(txt, ctx)
        results[spec] = s
        print(sc.render(s, spec))
        open(os.path.join(outdir, f'score__{safe}.json'), 'w').write(json.dumps(s, indent=2))

    # leaderboard
    print(f"\n{'='*58}\nSUMMARY — {today}\n{'='*58}")
    print(f"{'model':<44}{'trust':>7}{'tok':>7}")
    for spec, s in results.items():
        if 'error' in s:
            print(f"{spec:<44}{'ERR':>7}  {s['error'][:40]}")
        else:
            print(f"{spec:<44}{s['trust_score']:>6}%{s['tokens']:>7,}")
    open(os.path.join(outdir, 'summary.json'), 'w').write(json.dumps(results, indent=2))
    print(f"\nwritten -> {outdir}")


if __name__ == '__main__':
    main()
