"""Markdown export of the report (results/report/report.json + analysis/journal.json + VLM benchmark) for the repo:
docs/RESULTS.md (human-readable) and docs/results_summary.json (numbers without videos / curves). Called at the end of
fawam/report.py, so it is refreshed with every report build.

  python -m fawam.report_md [results/report/report.json]
"""
import json
import sys
from pathlib import Path

from fawam.paths import ROOT

SETTINGS = ("table_h_m05", "table_h_m10")
SN = {"table_h_z00": "0 cm", "table_h_m05": "−5 cm", "table_h_m10": "−10 cm", "hid_trans": "hid trans"}
SWEEP_TITLE = {
    "p0_baseline_table": "P0 · Basismodell, Tischhöhe",
    "p5a_signals": "A · Adapter-Update nach dem Fehlversuch, verschiedene Signale",
    "p5b_vlm": "A · Adapter-Update mit VLM",
    "p5f_paper": "B · Dual-Site-Adapter (Paper-Protokoll: Erfolg nach Adaption)",
    "p6a_icl_content": "C · Kontext ohne Training (Orakel-Text)",
    "p6v_icl_free_vlm": "C · Kontext ohne Training (VLM)",
    "p8_icl_vlm_meta": "D · Meta-trainierte Kontext-Adapter, Testtyp (Tischhöhe) im Training",
    "p8f_finetune": "D · Nur Fine-Tuning",
    "p8_icl_oracle_meta": "D · Orakel (nur Referenz, keine Methode)",
    "p8d_heldout_type": "D · Ungesehener Fehlertyp, Training auf 2 Typen (Startpose, Kamera)",
    "p8t_heldout_types": "D · Ungesehener Fehlertyp, Training auf „7 Typen“ (P7b + P7c)",
    "p8s_strong": "D · Mehr Fehlschläge (P7d), VLM detail, mit/ohne Basis-Training",
    "p9_related": "E · Verwandte Arbeiten nachgebaut",
    "p12_hid_eval": "hid · Versteckte Offsets, Held-out-Aufgaben (T1)",
}


def cell(c, proto):
    if not c:
        return "–"
    if proto == "single":
        k, n = c.get("first_ok"), c.get("first_n")
    elif proto == "paper":
        k, n = c.get("eval_ok"), c.get("eval_n")
    else:
        k, n = c.get("fixed"), c.get("retried")
    if not n:
        return "–"
    s = f"{k}/{n} ({100 * k / n:.0f} %)"
    if c.get("flung"):
        s += f", {c['flung']} weggeflogen"
    return s


def variants_md(r):
    by = {}
    for k, v in r["variants"].items():
        by.setdefault(v["sweep"], []).append(v)
    out = []
    for sweep in [s for s in SWEEP_TITLE if s in by] + sorted(s for s in by if s not in SWEEP_TITLE):
        vs = by[sweep]
        proto = vs[0]["protocol"]
        sets = sorted({s for v in vs for s in v["settings"]}, key=lambda s: list(SN).index(s) if s in SN else 9)
        what = {"single": "Erfolg im ersten Versuch", "paper": "Erfolg nach 3 Adaptions-Episoden",
                "retry": "gelöste Fehlversuche im zweiten Versuch"}[proto]
        out.append(f"### {SWEEP_TITLE.get(sweep, sweep)}\n\n`{sweep}` · {what}\n")
        out.append("| Methode | " + " | ".join(SN.get(s, s) for s in sets) + " |")
        out.append("|---|" + "---|" * len(sets))
        for v in sorted(vs, key=lambda v: v["method"]):
            out.append(f"| `{v['method']}` | " + " | ".join(cell(v["settings"].get(s), proto) for s in sets) + " |")
        out.append("")
    return "\n".join(out)


def journal_md(j):
    if not j:
        return "_kein Journal_\n"
    out = [f"Stand: {j.get('updated', '–')}\n", "### Was gerade läuft\n", "| Schritt | Status |", "|---|---|"]
    out += [f"| {x['s']} | {x['st']} |" for x in j.get("running", [])]
    out += ["", "### Analysen\n"]
    for a in j.get("analyses", []):
        out.append(f"**{a['t']}**\n")
        out += [f"- {b}" for b in a["b"]]
        out.append("")
    out += ["### Probleme\n", "| Problem | Ursache | Maßnahme | Status |", "|---|---|---|---|"]
    out += [f"| {p['p']} | {p['why']} | {p['fix']} | {p['status']} |" for p in j.get("problems", [])]
    return "\n".join(out) + "\n"


def vlm_md(rows, vo):
    out = []
    if vo:
        out += ["### VLM gegen Orakel (Qwen3-VL-8B, offline)\n", "| Tisch / Eingabe / Aufgaben | n | Typ richtig | "
                "Richtung richtig | Δz-Fehler |", "|---|---|---|---|---|"]
        for k, x in vo.items():
            out.append(f"| {k} | {x['n']} | {100 * (x.get('type_agree') or 0):.0f} % | "
                       f"{100 * (x.get('sign_agree') or 0):.0f} % | {x.get('abs_err_cm')} cm |")
        out.append("")
    if rows:
        out += ["### VLM-Benchmark: 105 Fehlschläge (fawam/vlm_bench.py)\n",
                "tiefer = sagt „tiefer greifen“, wo der Greifer zu hoch war · tiefer falsch = sagt es ohne Höhenfehler\n",
                "| Modell | Prompt | n | Ausfälle | Typ | tiefer | tiefer falsch | Lokalisierung |", "|---|---|---|---|---|---|---|---|"]
        p = lambda x: "–" if x is None else f"{100 * x:.0f} %"
        for x in rows:
            out.append(f"| {x['model']} | {x['variant']} | {x['n']} | {x['errors']} | {p(x['type'])} | {p(x['lower'])} | "
                       f"{p(x['lower_false'])} | {p(x.get('loc'))} |")
    return "\n".join(out) + "\n"


def training_md(t):
    out = ["| Adapter | Schritte | Val-Loss mit Kontext | ohne Kontext | Kontext hilft? |", "|---|---|---|---|---|"]
    for k, x in sorted(t.items()):
        if not x.get("val"):
            continue
        a, b = x["val"][-1], (x.get("val_noctx") or [None])[-1]
        out.append(f"| `{k}` | {x.get('last_step')} | {a} | {b} | "
                   f"{'ja' if (b is not None and a < 0.9 * b) else 'nein'} |")
    return "\n".join(out) + "\n"


def summary(r):
    keep = ("n_seeds", "first_ok", "first_n", "fixed", "retried", "eval_ok", "eval_n", "adapt_ok", "adapt_n", "flung",
            "above1_mm", "above2_mm", "above_eval_mm", "gap1", "gap2", "fix_ci", "first_ci", "eval_ci", "vs_control",
            "same_failure")
    return dict(generated=r.get("generated"), test_task=r.get("test_task"),
                variants={k: dict(sweep=v["sweep"], method=v["method"], protocol=v["protocol"],
                                  settings={s: {f: c.get(f) for f in keep if f in c} for s, c in v["settings"].items()})
                          for k, v in r["variants"].items()},
                training={k: {f: x.get(f) for f in ("last_step", "steps", "val", "val_noctx")} for k, x in r["training"].items()},
                vlm_vs_oracle={k: {f: x.get(f) for f in x if f != "examples"} for k, x in (r.get("vlm_vs_oracle") or {}).items()},
                vlm_bench=[{f: x.get(f) for f in x if f != "examples"} for x in (r.get("vlm_bench") or [])])


def main(path=None):
    path = Path(path or ROOT / "results/report/report.json")
    r = json.loads(path.read_text())
    md = [f"# Ergebnisse Failure-Aware WAM\n",
          f"Automatisch erzeugt aus dem Report (`fawam/report_md.py`) · Stand der Daten: {r.get('generated')} · "
          f"Testaufgabe `{r.get('test_task')}`.\n",
          "Erfolg zählt nur, wenn das Objekt auf dem Tisch bleibt („weggeflogen“ = vom Simulator als Erfolg gewertet, "
          "hier Fehlschlag). Brüche: gelöst / versucht. Ziele und Plan: [GOAL.md](GOAL.md), [PLAN.md](PLAN.md), "
          "[PLAN_HID.md](PLAN_HID.md); ältere Befunde: [FINDINGS.md](FINDINGS.md).\n",
          "## Probleme, Analysen, laufende Arbeit\n", journal_md(r.get("journal")),
          "## Ergebnisse je Sweep\n", variants_md(r),
          "## Meta-Training: Validierungs-Loss mit gegen ohne Kontext\n", training_md(r.get("training") or {}),
          "## VLM\n", vlm_md(r.get("vlm_bench") or [], r.get("vlm_vs_oracle") or {})]
    (ROOT / "docs/RESULTS.md").write_text("\n".join(md))
    (ROOT / "docs/results_summary.json").write_text(json.dumps(summary(r), indent=1, ensure_ascii=False))
    print("wrote docs/RESULTS.md, docs/results_summary.json")


if __name__ == "__main__":
    main(*sys.argv[1:])
