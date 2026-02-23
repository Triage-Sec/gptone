"""Demonstrate human-guided evasion using diagnose output.

Takes 10 AI-generated texts, shows what diagnose recommends, provides
hand-crafted human edits following those recommendations, and verifies
each against GPTZero's live API.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Each entry: (original_ai_text, human_edited_version, edit_notes)
# The human edits follow diagnose guidance: shorten words, add contractions,
# remove hedging, vary sentence lengths, add personal voice.
EVASION_PAIRS: List[Tuple[str, str, str]] = [
    (
        "The implementation of sustainable development strategies requires comprehensive analysis of environmental, economic, and social factors. A systematic approach to resource allocation ensures optimal outcomes for all stakeholders involved in the process.",
        "Sustainable development isn't something you can just wing. You've got to look at the environmental side, the money side, and how it affects actual people. Get the resource allocation right and everyone benefits -- mess it up and you're just wasting time.",
        "Shortened words, added contractions, removed 'systematic approach' hedging, added casual voice"
    ),
    (
        "Recent advances in machine learning have demonstrated significant potential for transforming healthcare delivery systems. Diagnostic algorithms now achieve accuracy levels comparable to trained specialists in specific imaging tasks.",
        "ML in healthcare has come a long way. Some of these diagnostic algorithms? They're matching trained specialists on certain imaging tasks. Not all of them, but enough that hospitals are paying attention. It's pretty wild honestly.",
        "Shortened 'demonstrated significant potential', added questions, contractions, personal reaction"
    ),
    (
        "The regulatory framework governing emerging technologies must balance innovation incentives with consumer protection requirements. Policymakers face the challenge of crafting legislation that remains relevant amid rapid technological change.",
        "Here's the thing about regulating new tech: you can't just lock everything down or nobody innovates. But you can't let companies run wild either. Policymakers are basically trying to write rules for stuff that didn't exist last year. Good luck with that.",
        "Complete restructure with casual voice, rhetorical framing, humor"
    ),
    (
        "Urbanization trends indicate that approximately 68% of the global population will reside in metropolitan areas by 2050. Infrastructure planning must account for this demographic shift through investments in transportation, housing, and public services.",
        "By 2050, roughly 68% of people will live in cities. That's a lot of folks needing buses, apartments, and parks. Cities that plan ahead will be fine. The ones that don't? They'll feel it. We're already seeing what happens when infrastructure can't keep up.",
        "Kept the statistic, casualized everything else, added prediction/opinion"
    ),
    (
        "Effective communication in professional settings requires attention to clarity, tone, and audience awareness. The ability to convey complex information in an accessible manner distinguishes exceptional leaders from their peers.",
        "Good communicators at work do three things: they're clear, they match their tone to the room, and they know who they're talking to. That's it. The people who can take complicated stuff and make it simple? Those are the ones who get promoted.",
        "Turned abstract into concrete list, added direct address, kept meaning"
    ),
    (
        "The intersection of ethics and artificial intelligence raises fundamental questions about accountability, transparency, and human autonomy. As automated decision-making systems become more prevalent, establishing clear governance frameworks becomes increasingly critical.",
        "Who's responsible when an AI makes a bad call? That's the question nobody has a great answer to yet. These automated systems are everywhere now -- hiring, lending, even criminal justice. We need rules. Real ones. Not just guidelines that nobody follows.",
        "Led with a question, removed all hedging, added urgency and specifics"
    ),
    (
        "Climate change mitigation strategies encompass a broad spectrum of approaches, from renewable energy adoption to carbon capture technologies. The effectiveness of these strategies depends on coordinated international cooperation and sustained financial commitment.",
        "There's no single fix for climate change. Solar and wind help. Carbon capture might help too. But none of it works unless countries actually cooperate and keep funding it year after year. That's the hard part -- not the tech, the politics.",
        "Broke into short declarative sentences, added contrast, personal assessment"
    ),
    (
        "The digital transformation of educational institutions has accelerated significantly in recent years, driven by advances in learning management systems and adaptive assessment tools. Student engagement metrics suggest improved outcomes when technology is thoughtfully integrated.",
        "Schools went digital fast, especially after COVID. The LMS platforms and adaptive testing tools are actually pretty decent now. And yeah, when you use the tech right, students do better. The key word is 'right' -- plenty of schools just threw iPads at the problem and called it innovation.",
        "Added temporal reference, contractions throughout, critical aside"
    ),
    (
        "Biodiversity conservation efforts must address the interconnected challenges of habitat fragmentation, invasive species management, and climate-induced ecosystem shifts. Protected area networks alone are insufficient without landscape-level connectivity planning.",
        "Saving biodiversity isn't just about drawing lines on a map and calling it protected. Habitats are fragmented. Invasive species keep showing up. The climate is literally shifting what can survive where. You need connected corridors between reserves or the animals can't move. It's that simple.",
        "Removed nominalization, added 'you', concrete imagery, emphatic ending"
    ),
    (
        "The evolution of cybersecurity threats necessitates a paradigm shift from reactive defense mechanisms to proactive threat hunting methodologies. Organizations must invest in continuous monitoring capabilities and develop incident response playbooks that account for emerging attack vectors.",
        "Cybersecurity can't just be about building walls anymore. The threats evolve too fast. Companies need people actively hunting for problems, not waiting for alarms to go off. And when something does hit? You better have a playbook ready. The organizations that figure this out survive. The ones that don't learn the hard way.",
        "Removed 'paradigm shift' and jargon, added direct warnings, varied rhythm"
    ),
]


def run_demo(
    api_key: str,
    output_dir: str = "/artifacts/gptzero-audit",
    delay: float = 0.5,
) -> Dict[str, Any]:
    from src.client import GPTZeroClient

    client = GPTZeroClient(api_key=api_key)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    print("=== HUMAN-GUIDED EVASION DEMONSTRATION ===\n")

    for idx, (original, edited, notes) in enumerate(EVASION_PAIRS):
        print(f"--- Text {idx + 1}/{len(EVASION_PAIRS)} ---")
        print(f"  Edits: {notes}")

        orig_resp = client.predict(original, delay=delay)
        edit_resp = client.predict(edited, delay=delay)

        orig_prob = float(orig_resp.get("average_generated_prob", -1))
        orig_cls = orig_resp.get("predicted_class", "error")
        edit_prob = float(edit_resp.get("average_generated_prob", -1))
        edit_cls = edit_resp.get("predicted_class", "error")

        evaded = edit_cls in ("human", "mixed") or edit_prob < 0.5
        delta = orig_prob - edit_prob

        print(f"  Original:  {orig_prob:.4f} ({orig_cls})")
        print(f"  Edited:    {edit_prob:.4f} ({edit_cls})  delta={delta:+.4f}  {'EVADED' if evaded else 'CAUGHT'}")
        print()

        results.append({
            "index": idx + 1,
            "original_prob": round(orig_prob, 4),
            "original_class": orig_cls,
            "edited_prob": round(edit_prob, 4),
            "edited_class": edit_cls,
            "delta": round(delta, 4),
            "evaded": evaded,
            "edit_notes": notes,
        })

    evaded_count = sum(1 for r in results if r["evaded"])
    avg_delta = sum(r["delta"] for r in results) / len(results) if results else 0.0

    summary = {
        "total_texts": len(results),
        "evaded": evaded_count,
        "caught": len(results) - evaded_count,
        "evasion_rate": round(evaded_count / len(results), 2) if results else 0.0,
        "avg_prob_reduction": round(avg_delta, 4),
        "results": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    print("=== SUMMARY ===")
    print(f"  Evaded:    {evaded_count}/{len(results)} ({summary['evasion_rate']*100:.0f}%)")
    print(f"  Caught:    {len(results) - evaded_count}/{len(results)}")
    print(f"  Avg delta: {avg_delta:+.4f}")

    results_path = out / "human_evasion_demo.json"
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nResults saved to {results_path}")

    try:
        from modal import Volume
        Volume.from_name("gptzero-audit-artifacts").commit()
    except Exception:
        pass

    return summary
