#!/usr/bin/env python3
"""
oracle_debate.py — multi-agent DEBATE Oracle (LangGraph + local ollama). Agents live in oracle_agents/*.md.

Agentic-layer LEARNING build:
  - Each agent is a MARKDOWN file: frontmatter (name/role/model/type/temperature) + body = its SYSTEM PROMPT.
    Tune an agent by editing its .md; ADD an agent by dropping a new .md in oracle_agents/. No code changes.
  - The debate graph is built DYNAMICALLY from whatever analyst agents exist.
  - Proper system/user separation via ollama /api/chat (system = persona, user = today's data).
  - Full observability: every turn (system model, exact user prompt, response, parsed lean/conviction, latency)
    is logged to oracle_debate_log.jsonl so you can inspect and A/B-tune prompts.
  - Still writes an Oracle-compatible record to oracle_forecasts.jsonl so the 16:05 verify job scores it.

Debate flow (LangGraph): START -> dispatch -> [analysts in parallel] -> tally -> (converged/max? judge : loop) -> judge -> END
Migrate to Claude Agent SDK later by swapping chat() for a Claude call — agents, graph, and logging are unchanged.

Usage:  python oracle_debate.py            # run, log, email (the scheduled path)
        python oracle_debate.py --show     # print every agent turn (for tuning), no email
        python oracle_debate.py --no_email
"""
import os, re, json, time, argparse, warnings, operator, requests
from datetime import datetime
from typing import TypedDict, Annotated
warnings.filterwarnings('ignore')
from langgraph.graph import StateGraph, START, END
import oracle_forecast as of

HERE = os.path.dirname(os.path.abspath(__file__))
AGENTS_DIR = os.path.join(HERE, 'oracle_agents')
OLLAMA_CHAT = 'http://localhost:11434/api/chat'
DEBATE_LOG = os.path.join(HERE, 'oracle_debate_log.jsonl')
MAX_ROUNDS = 2


# ---------------- agent registry: load .md files ----------------
def parse_agent(path):
    txt = open(path).read()
    m = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)$', txt, re.DOTALL)
    if not m:
        raise ValueError(f"{os.path.basename(path)}: missing --- frontmatter ---")
    fm = {}
    for line in m.group(1).splitlines():
        if ':' in line:
            k, v = line.split(':', 1); fm[k.strip()] = v.strip()
    think = fm.get('think')
    think = {'false': False, 'true': True}.get(think.lower()) if think else None
    return dict(name=fm['name'], role=fm.get('role', fm['name']), model=fm['model'],
                type=fm.get('type', 'analyst'), temperature=float(fm.get('temperature', 0.4)),
                think=think, system=m.group(2).strip())


def load_agents(d=AGENTS_DIR):
    agents = [parse_agent(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.endswith('.md')]
    analysts = [a for a in agents if a['type'] == 'analyst']
    judge = next(a for a in agents if a['type'] == 'judge')
    return analysts, judge


# ---------------- llm (system/user separated) ----------------
NUM_CTX = 8192  # cap context so KV cache stays small; our prompts are ~2-4k tokens. Default 131k = a memory bomb.


def chat(model, system, user, temperature=0.4, num_predict=340, think=None):
    body = {'model': model, 'stream': False,
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
            'options': {'temperature': temperature, 'num_predict': num_predict, 'num_ctx': NUM_CTX}}
    if think is not None:  # thinking models (qwen3) must be told think:false or they burn the budget with no answer
        body['think'] = think
    try:
        r = requests.post(OLLAMA_CHAT, json=body, timeout=300)
        t = r.json().get('message', {}).get('content', '')
    except Exception as e:
        return f"(model error: {e})"
    return re.sub(r'<think>.*?</think>', '', t, flags=re.DOTALL).strip()


def p_lean(t): m = re.search(r'LEAN[:\s]+(BULL|BEAR)', t.upper()); return m.group(1) if m else None
def p_conv(t): m = re.search(r'CONVICTION[:\s]+(\d+)', t.upper()); return int(m.group(1)) if m else None
def p_field(t, k): m = re.search(k + r'[:\s]+(.+)', t, re.IGNORECASE); return m.group(1).strip() if m else ''


def fmt_signals(s):
    try:
        return json.dumps(s, indent=2, default=str)[:1300]
    except Exception:
        return str(s)[:1300]


def fmt_news(n):
    return '\n'.join('- ' + str(x.get('headline') if isinstance(x, dict) else x) for x in (n or [])[:12]) or '(no news)'


# ---------------- graph ----------------
class DState(TypedDict):
    signals: dict
    news: list
    transcript: Annotated[list, operator.add]
    round: int
    converged: bool
    decision: dict


def analyst_node(agent):
    def fn(state):
        rnd = state['round']; t0 = time.time()
        data = f"TODAY'S SIGNALS:\n{fmt_signals(state['signals'])}\n\nNEWS HEADLINES:\n{fmt_news(state['news'])}"
        if rnd == 1:
            user = f"{data}\n\nRound 1. Give your opening read from your lens, in your required output format."
        else:
            prior = '\n'.join(f"[{t['role']}] {t['lean']} ({t['conviction']}): {t['points']}"
                              for t in state['transcript'] if t['round'] == rnd - 1 and t['name'] != agent['name'])
            user = (f"{data}\n\nThe panel's round {rnd-1} positions:\n{prior}\n\n"
                    f"Round {rnd}. Hold or revise your view, rebutting the weakest point you see. Use your required output format.")
        resp = chat(agent['model'], agent['system'], user, agent['temperature'], think=agent.get('think'))
        return {"transcript": [dict(round=rnd, name=agent['name'], role=agent['role'], model=agent['model'],
                                    lean=p_lean(resp), conviction=p_conv(resp), points=p_field(resp, 'POINTS'),
                                    user=user, response=resp, latency=round(time.time() - t0, 1))]}
    return fn


def tally(state):
    rnd = state['round']
    leans = [t['lean'] for t in state['transcript'] if t['round'] == rnd and t['lean'] in ('BULL', 'BEAR')]
    unanimous = len(leans) >= 3 and len(set(leans)) == 1
    return {"round": rnd + 1, "converged": bool(unanimous or rnd >= MAX_ROUNDS)}


def route(state):
    return 'judge' if state['converged'] else 'continue'


def judge_node(agent):
    def fn(state):
        t0 = time.time()
        debate = '\n\n'.join(f"[R{t['round']} {t['role']} -> {t['lean']} ({t['conviction']})]\n{t['response']}"
                             for t in sorted(state['transcript'], key=lambda x: (x['round'], x['name'])))
        user = f"THE DEBATE:\n{debate}\n\nRender the panel's final verdict in your required output format."
        resp = chat(agent['model'], agent['system'], user, agent['temperature'], think=agent.get('think'))
        dfield = p_field(resp, 'DECISION').upper()
        call = 'BULL' if 'BULL' in dfield else ('BEAR' if 'BEAR' in dfield else None)
        dec = dict(call=call, lean=call, conviction=p_conv(resp), key_driver=p_field(resp, 'KEY_DRIVER'),
                   rationale=p_field(resp, 'RATIONALE'), matrix={}, latency=round(time.time() - t0, 1), raw=resp)
        return {"decision": dec,
                "transcript": [dict(round=99, name='judge', role=agent['role'], model=agent['model'],
                                    lean=call, conviction=dec['conviction'], points=dec['key_driver'],
                                    user=user, response=resp, latency=dec['latency'])]}
    return fn


def build_graph(analysts, judge):
    g = StateGraph(DState)
    g.add_node('dispatch', lambda s: {})
    for a in analysts:
        g.add_node(a['name'], analyst_node(a))
        g.add_edge('dispatch', a['name']); g.add_edge(a['name'], 'tally')
    g.add_node('tally', tally); g.add_node('judge', judge_node(judge))
    g.add_edge(START, 'dispatch')
    g.add_conditional_edges('tally', route, {'continue': 'dispatch', 'judge': 'judge'})
    g.add_edge('judge', END)
    return g.compile()


# ---------------- run / log / email ----------------
def run_debate():
    analysts, judge = load_agents()
    sig = of.build_signals(); news = of.get_news()
    out = build_graph(analysts, judge).invoke(
        {'signals': sig, 'news': news, 'transcript': [], 'round': 1, 'converged': False, 'decision': {}},
        {'recursion_limit': 40})
    return analysts, judge, sig, news, out


def log_run(analysts, judge, out):
    rec = dict(ts=datetime.now().isoformat(timespec='seconds'),
               agents=[{'name': a['name'], 'role': a['role'], 'model': a['model']} for a in analysts + [judge]],
               rounds=max((t['round'] for t in out['transcript'] if t['name'] != 'judge'), default=1),
               turns=out['transcript'],
               decision={k: v for k, v in out['decision'].items() if k != 'raw'})
    open(DEBATE_LOG, 'a').write(json.dumps(rec) + '\n')


def email_body(out):
    d = out['decision']; finals = {}
    for t in out['transcript']:
        if t['name'] != 'judge':
            finals[t['role']] = (t['lean'], t['conviction'])
    L = [f"DEBATE VERDICT — {d['call']} · conviction {d['conviction']} · {d.get('key_driver')}",
         d.get('rationale', ''), "", "Final panel positions:"]
    for role, (lean, conv) in finals.items():
        L.append(f"  {role:<22} {lean} ({conv})")
    L += ["", "───── debate ─────"]
    for t in sorted(out['transcript'], key=lambda x: (x['round'], x['name'])):
        if t['name'] == 'judge':
            continue
        L.append(f"[R{t['round']} {t['role']} -> {t['lean']} ({t['conviction']})] {t['points']}")
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--show', action='store_true', help='print every agent turn (tuning) and skip email')
    ap.add_argument('--no_email', action='store_true')
    a = ap.parse_args()

    analysts, judge, sig, news, out = run_debate()
    log_run(analysts, judge, out)
    dec = out['decision']
    rec = {'date': of.now_et().strftime('%Y-%m-%d'), 'ts_forecast': of.now_et().isoformat(timespec='seconds'),
           'provider': 'debate', 'model': '+'.join(a['model'] for a in analysts) + f"|judge={judge['model']}",
           'method': 'debate', 'signals': sig, 'news': [n['headline'] for n in news[:25]] if news else [],
           'feedback_used': False, 'decision': {k: v for k, v in dec.items() if k not in ('latency', 'raw')},
           'raw_response': dec.get('raw', ''), 'parse_ok': dec['call'] is not None,
           'prompt': '(multi-agent debate)', 'actual': None, 'hit': None}
    of.append(rec)

    body = email_body(out)
    print(body if a.show else f"verdict {dec['call']} ({dec['conviction']}) · {dec.get('key_driver')}")
    if a.show:
        print("\n=== FULL TURNS (for tuning) ===")
        for t in sorted(out['transcript'], key=lambda x: (x['round'], x['name'])):
            print(f"\n── R{t['round']} {t['role']} ({t['model']}) {t['latency']}s → {t['lean']} ({t['conviction']}) ──\n{t['response']}")
    if not (a.show or a.no_email):
        of.send_email(f"Oracle debate {rec['date']} — {dec['call']} (conv {dec['conviction']})", body)
    print(f"\nlogged -> {of.LOG} + {DEBATE_LOG}")


if __name__ == '__main__':
    main()
