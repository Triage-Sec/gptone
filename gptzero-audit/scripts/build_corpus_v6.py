"""Build v6 training corpus: diverse text sourcing + GPTZero labeling + perturbation variants.

Runs on Modal with GPU (for GPT-2 text generation) and API key (for GPTZero labeling).
Produces a labeled JSONL dataset with base texts and perturbation pairs.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MIN_WORDS = 80
MAX_WORDS = 600
VARIANTS_PER_BASE = 5


def _word_count(text: str) -> int:
    return len(text.split())


def _clean_paragraph(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\s*=+\s*[^=]+=+\s*", " ", text)
    return text.strip()


def _is_good_paragraph(text: str) -> bool:
    wc = _word_count(text)
    if wc < MIN_WORDS or wc > MAX_WORDS:
        return False
    if text.count("=") > 3:
        return False
    alpha_ratio = sum(c.isalpha() for c in text) / max(1, len(text))
    if alpha_ratio < 0.6:
        return False
    return True


def _source_human_pre_llm(target_count: int, seed: int) -> List[Dict[str, Any]]:
    """Source pre-LLM human text from WikiText-103."""
    from datasets import load_dataset

    print(f"[human_pre_llm] Loading WikiText-103 (target={target_count})...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    rng = random.Random(seed)

    paragraphs: List[str] = []
    for row in ds:
        text = row.get("text", "")
        if not text or not text.strip():
            continue
        cleaned = _clean_paragraph(text)
        if _is_good_paragraph(cleaned):
            paragraphs.append(cleaned)

    rng.shuffle(paragraphs)
    selected = paragraphs[:target_count]
    print(f"[human_pre_llm] Selected {len(selected)} paragraphs from WikiText-103")

    return [
        {
            "sample_id": f"wiki103_{i:04d}",
            "text": t,
            "stratum": "human_pre_llm",
            "ground_truth_label": "human",
            "source": "wikitext-103-raw-v1",
        }
        for i, t in enumerate(selected)
    ]


def _source_human_conversational(target_count: int, seed: int) -> List[Dict[str, Any]]:
    """Source conversational human text from IMDB reviews and AG News."""
    from datasets import load_dataset

    rng = random.Random(seed)
    paragraphs: List[str] = []

    print(f"[human_conversational] Loading IMDB reviews...")
    try:
        imdb = load_dataset("imdb", split="train")
        for row in imdb:
            text = _clean_paragraph(row.get("text", ""))
            if _is_good_paragraph(text):
                paragraphs.append(text)
    except Exception as e:
        print(f"  IMDB load failed: {e}, continuing with other sources...")

    half = target_count // 2
    if len(paragraphs) < half:
        print(f"[human_conversational] Loading AG News as supplement...")
        try:
            ag = load_dataset("ag_news", split="train")
            for row in ag:
                text = _clean_paragraph(row.get("text", ""))
                if _is_good_paragraph(text):
                    paragraphs.append(text)
        except Exception as e:
            print(f"  AG News load failed: {e}")

    rng.shuffle(paragraphs)
    selected = paragraphs[:target_count]
    print(f"[human_conversational] Selected {len(selected)} paragraphs")

    return [
        {
            "sample_id": f"conv_{i:04d}",
            "text": t,
            "stratum": "human_conversational",
            "ground_truth_label": "human",
            "source": "imdb_or_agnews",
        }
        for i, t in enumerate(selected)
    ]


def _source_ai_from_hc3(target_count: int, seed: int, style: str) -> List[str]:
    """Source AI-generated text from HC3 (Human ChatGPT Comparison) dataset."""
    from datasets import load_dataset

    print(f"[ai_{style}] Loading HC3 dataset...")
    rng = random.Random(seed)
    paragraphs: List[str] = []

    try:
        ds = load_dataset("Hello-SimpleAI/HC3", "all", split="train", trust_remote_code=True)
        for row in ds:
            answers = row.get("chatgpt_answers", [])
            for ans in answers:
                if isinstance(ans, str):
                    cleaned = _clean_paragraph(ans)
                    if _is_good_paragraph(cleaned):
                        paragraphs.append(cleaned)
    except Exception as e:
        print(f"  HC3 load failed: {e}")

    if len(paragraphs) < target_count:
        print(f"[ai_{style}] HC3 yielded {len(paragraphs)}, supplementing with synthetic...")
        paragraphs.extend(_generate_synthetic_ai(target_count - len(paragraphs), seed, style))

    rng.shuffle(paragraphs)
    print(f"[ai_{style}] Selected {min(len(paragraphs), target_count)} texts")
    return paragraphs[:target_count]


def _generate_synthetic_ai(count: int, seed: int, style: str) -> List[str]:
    """Fallback: generate synthetic AI-like text using templates."""
    rng = random.Random(seed)
    templates = _FORMAL_TEMPLATES if style == "formal" else _INSTRUCTIONAL_TEMPLATES
    results: List[str] = []
    for i in range(count):
        template = templates[i % len(templates)]
        text = _clean_paragraph(template)
        if _is_good_paragraph(text):
            results.append(text)
    return results


_INSTRUCTIONAL_TEMPLATES = [
    "To effectively manage your time, it is essential to prioritize tasks based on their urgency and importance. Begin by creating a comprehensive list of all pending tasks. Next, categorize each task using a priority matrix. Allocate specific time blocks for high-priority items and ensure that you minimize distractions during these focused work periods. Regular review of your progress will help you adjust your approach as needed. Consider using digital tools or applications that support task management and scheduling. The key to successful time management is consistency and discipline in following your established routine.",
    "The process of learning a new programming language requires a structured approach. Start with understanding the fundamental syntax and basic data types. Practice writing simple programs that demonstrate core concepts such as variables, loops, and conditional statements. Gradually progress to more complex topics including functions, object-oriented programming, and error handling. Working through practical projects will reinforce your understanding and build confidence. Online resources, documentation, and community forums provide valuable support throughout the learning journey. Regular practice and code review are essential for developing proficiency.",
    "Effective communication in professional settings requires attention to clarity, tone, and audience awareness. Begin by identifying the purpose of your communication and the key message you want to convey. Structure your content logically with clear transitions between points. Use appropriate language that matches the formality level expected by your audience. Active listening is equally important in two-way communication, as it demonstrates respect and facilitates understanding. Regular feedback and reflection on communication outcomes will help you refine your approach over time.",
    "Maintaining physical health requires a balanced approach that incorporates regular exercise, proper nutrition, and adequate rest. Establish a consistent exercise routine that includes both cardiovascular and strength training activities. Pay attention to your dietary intake, focusing on whole foods, lean proteins, and adequate hydration. Sleep is a critical component of overall health, and adults should aim for seven to nine hours per night. Regular health screenings and preventive care measures contribute to long-term wellbeing. Mental health should also be prioritized through stress management techniques and social connections.",
    "The fundamentals of project management involve planning, execution, monitoring, and closure. A successful project begins with clear objectives and well-defined scope. Create a detailed project plan that outlines tasks, timelines, and resource allocation. During execution, maintain open communication with all stakeholders and address issues promptly. Regular progress monitoring ensures that the project stays on track and within budget. Document lessons learned throughout the process to improve future project outcomes. Effective risk management and contingency planning are essential components of project success.",
]

_FORMAL_TEMPLATES = [
    "The intersection of artificial intelligence and healthcare presents both significant opportunities and complex challenges that require careful consideration. Recent advances in machine learning algorithms have demonstrated remarkable capability in diagnostic imaging analysis, drug discovery, and patient outcome prediction. However, the deployment of these technologies in clinical settings raises important questions regarding data privacy, algorithmic bias, and the appropriate role of automation in medical decision-making. A comprehensive regulatory framework must balance innovation with patient safety, ensuring that AI-assisted healthcare solutions meet rigorous standards of accuracy, reliability, and ethical compliance. The ongoing collaboration between technologists, healthcare professionals, and policymakers is essential for navigating these challenges effectively.",
    "Contemporary urban planning methodologies must account for the multifaceted challenges posed by climate change, population growth, and evolving social dynamics. The integration of sustainable infrastructure design with community-centered development approaches represents a paradigm shift in how cities are conceptualized and constructed. Evidence-based planning frameworks that incorporate environmental impact assessments, transportation modeling, and social equity analyses provide a foundation for informed decision-making. The adoption of smart city technologies offers promising pathways for optimizing resource utilization and improving quality of life. However, these technological solutions must be implemented with careful attention to accessibility, privacy, and the preservation of community identity.",
    "The evolution of global financial systems in the twenty-first century has been characterized by increasing interconnectedness, technological disruption, and regulatory complexity. The emergence of digital currencies, decentralized finance platforms, and algorithmic trading systems has fundamentally altered the landscape of financial intermediation. These developments have created new opportunities for financial inclusion while simultaneously introducing novel systemic risks that traditional regulatory frameworks were not designed to address. The challenge for policymakers lies in developing adaptive governance structures that foster innovation while maintaining the stability and integrity of financial markets. International coordination and information sharing among regulatory bodies have become essential components of effective financial oversight.",
    "Recent advances in neuroscience have substantially enhanced our understanding of the neural mechanisms underlying cognitive function, emotional regulation, and behavioral adaptation. Neuroimaging technologies, including functional magnetic resonance imaging and electroencephalography, have provided unprecedented insights into brain activity patterns associated with various cognitive tasks and psychological states. The integration of computational modeling approaches with empirical neuroscience research has facilitated the development of more comprehensive theories of brain function. These advances have important implications for clinical applications, including the treatment of neurological disorders and the development of brain-computer interface technologies.",
    "The relationship between economic development and environmental sustainability represents one of the most pressing challenges facing contemporary societies. Traditional models of economic growth, predicated on the extensive exploitation of natural resources, have resulted in significant environmental degradation and biodiversity loss. The transition to sustainable development pathways requires fundamental shifts in production and consumption patterns, supported by appropriate policy instruments and institutional frameworks. Carbon pricing mechanisms, green technology investments, and circular economy principles offer promising approaches for reconciling economic objectives with environmental imperatives. The success of these transitions depends on coordinated action across multiple levels of governance and active engagement from both public and private sector stakeholders.",
]


def _source_ai_instructional(target_count: int, seed: int) -> List[Dict[str, Any]]:
    texts = _source_ai_from_hc3(target_count, seed, "instructional")
    return [
        {
            "sample_id": f"ai_instr_{i:04d}",
            "text": t,
            "stratum": "ai_instructional",
            "ground_truth_label": "ai",
            "source": "hc3_chatgpt_instructional",
        }
        for i, t in enumerate(texts)
    ]


def _source_ai_formal(target_count: int, seed: int) -> List[Dict[str, Any]]:
    texts = _source_ai_from_hc3(target_count, seed + 100, "formal")
    return [
        {
            "sample_id": f"ai_formal_{i:04d}",
            "text": t,
            "stratum": "ai_formal",
            "ground_truth_label": "ai",
            "source": "hc3_chatgpt_formal",
        }
        for i, t in enumerate(texts)
    ]


def _create_mixed_blended(
    human_texts: List[Dict[str, Any]],
    ai_texts: List[Dict[str, Any]],
    target_count: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Create mixed text by interleaving human and AI sentences."""
    rng = random.Random(seed)
    from src.pattern_benchmark import _split_sentences

    results: List[Dict[str, Any]] = []
    h_pool = [r["text"] for r in human_texts if r.get("ground_truth_label") == "human"]
    a_pool = [r["text"] for r in ai_texts if r.get("ground_truth_label") == "ai"]
    rng.shuffle(h_pool)
    rng.shuffle(a_pool)

    for i in range(min(target_count, len(h_pool), len(a_pool))):
        h_sents = _split_sentences(h_pool[i])
        a_sents = _split_sentences(a_pool[i])
        if len(h_sents) < 2 or len(a_sents) < 2:
            continue

        blend_ratio = rng.uniform(0.3, 0.7)
        n_ai = max(1, int(len(h_sents) * blend_ratio))
        replacement_indices = sorted(rng.sample(range(len(h_sents)), min(n_ai, len(h_sents))))
        mixed_sents = list(h_sents)
        for j, idx in enumerate(replacement_indices):
            if j < len(a_sents):
                mixed_sents[idx] = a_sents[j]
        mixed_text = " ".join(mixed_sents)

        if _is_good_paragraph(mixed_text):
            results.append(
                {
                    "sample_id": f"mixed_{i:04d}",
                    "text": mixed_text,
                    "stratum": "mixed_blended",
                    "ground_truth_label": "mixed",
                    "source": "human_ai_interleave",
                }
            )
        if len(results) >= target_count:
            break

    print(f"[mixed_blended] Created {len(results)} blended texts")
    return results


def _create_boundary_near(
    ai_texts: List[Dict[str, Any]],
    target_count: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Create boundary-case text by lightly humanizing AI text."""
    from src.boundary_mining import humanize_variants

    rng = random.Random(seed)
    pool = [r["text"] for r in ai_texts if r.get("ground_truth_label") == "ai"]
    rng.shuffle(pool)

    results: List[Dict[str, Any]] = []
    for i, text in enumerate(pool):
        if len(results) >= target_count:
            break
        variants = humanize_variants(text)
        if variants:
            picked = rng.choice(variants)
            picked = _clean_paragraph(picked)
            if _is_good_paragraph(picked):
                results.append(
                    {
                        "sample_id": f"bound_{i:04d}",
                        "text": picked,
                        "stratum": "boundary_near",
                        "ground_truth_label": None,
                        "source": "humanized_ai",
                    }
                )

    print(f"[boundary_near] Created {len(results)} boundary texts")
    return results


def _generate_perturbation_variants(
    base_records: List[Dict[str, Any]], variants_per_base: int, seed: int
) -> List[Dict[str, Any]]:
    """Generate perturbation variants for each base record."""
    from src.pattern_benchmark import generate_variants

    rng = random.Random(seed)
    all_rows: List[Dict[str, Any]] = []
    for rec in base_records:
        variants = generate_variants(
            rec["text"], rng=rng, max_variants=variants_per_base + 1
        )
        for v_idx, var in enumerate(variants):
            all_rows.append(
                {
                    "base_sample_id": rec["sample_id"],
                    "stratum": rec["stratum"],
                    "ground_truth_label": rec.get("ground_truth_label"),
                    "source": rec.get("source", ""),
                    "variant_order": v_idx,
                    "variant_id": var["variant_id"],
                    "perturbation_kind": var["perturbation_kind"],
                    "perturbation_strength": var["perturbation_strength"],
                    "text": var["text"],
                }
            )
    return all_rows


def _query_gptzero_batched(
    client: Any,
    rows: List[Dict[str, Any]],
    *,
    batch_size: int = 64,
    delay: float = 0.25,
    cache: Optional[Dict[str, Dict]] = None,
    flush_callback: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Query GPTZero for all rows, with caching and batched flushing."""
    if cache is None:
        cache = {}

    labeled: List[Dict[str, Any]] = []
    queries_made = 0
    cache_hits = 0

    for i, row in enumerate(rows):
        text = row["text"]
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        if text_hash in cache:
            resp = cache[text_hash]
            cache_hits += 1
        else:
            resp = client.predict(text, delay=delay)
            if "error" not in resp:
                cache[text_hash] = resp
                queries_made += 1
            else:
                print(f"  ERROR on {row.get('base_sample_id', '?')}/{row.get('variant_id', '?')}: {resp['error']}")
                queries_made += 1

        gptzero_prob = None
        gptzero_class = None
        if "error" not in resp:
            gptzero_prob = resp.get("average_generated_prob")
            gptzero_class = resp.get("predicted_class")

        labeled.append(
            {
                **row,
                "text_hash": text_hash,
                "gptzero_prob": gptzero_prob,
                "gptzero_class": gptzero_class,
                "gptzero_class_probs": resp.get("class_probabilities"),
                "gptzero_burstiness": resp.get("overall_burstiness"),
                "gptzero_sentences": resp.get("sentences"),
                "has_error": "error" in resp,
            }
        )

        if (i + 1) % batch_size == 0:
            pct = (i + 1) / len(rows) * 100
            print(
                f"  [{i+1}/{len(rows)}] ({pct:.0f}%) queries={queries_made} "
                f"cache_hits={cache_hits}"
            )
            if flush_callback:
                flush_callback(cache)

    if flush_callback:
        flush_callback(cache)

    print(
        f"  Labeling complete: {len(labeled)} rows, {queries_made} API queries, "
        f"{cache_hits} cache hits"
    )
    return labeled


def build_and_label(
    api_key: str,
    config: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    """Full Phase 1: source text, generate variants, label with GPTZero."""
    from src.client import GPTZeroClient

    seed = int(config.get("seed", 42))
    target_per_stratum = int(config.get("target_per_stratum", 350))
    boundary_count = int(config.get("boundary_count", 250))
    mixed_count = int(config.get("mixed_count", 250))
    variants_per_base = int(config.get("variants_per_base", VARIANTS_PER_BASE))
    delay = float(config.get("delay", 0.25))
    batch_size = int(config.get("batch_size", 64))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "gptzero_corpus_cache.json"
    cache: Dict[str, Dict] = {}
    if cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            cache = raw.get("responses", {})
            print(f"Loaded {len(cache)} cached responses from {cache_path}")
        except Exception:
            pass

    def _flush_cache(c: Dict) -> None:
        cache_path.write_text(
            json.dumps({"saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "responses": c}),
            encoding="utf-8",
        )

    print("=" * 60)
    print("PHASE 1a: Sourcing diverse base texts")
    print("=" * 60)

    human_pre = _source_human_pre_llm(target_per_stratum, seed)
    human_conv = _source_human_conversational(target_per_stratum, seed)
    ai_instr = _source_ai_instructional(target_per_stratum, seed)
    ai_formal = _source_ai_formal(target_per_stratum, seed)

    all_human = human_pre + human_conv
    all_ai = ai_instr + ai_formal
    mixed = _create_mixed_blended(all_human, all_ai, mixed_count, seed)
    boundary = _create_boundary_near(all_ai, boundary_count, seed)

    base_records = human_pre + human_conv + ai_instr + ai_formal + mixed + boundary
    print(f"\nTotal base records: {len(base_records)}")
    for stratum in sorted(set(r["stratum"] for r in base_records)):
        count = sum(1 for r in base_records if r["stratum"] == stratum)
        print(f"  {stratum}: {count}")

    base_path = out / "corpus_v6_base.jsonl"
    with base_path.open("w", encoding="utf-8") as fh:
        for rec in base_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nBase records saved to {base_path}")

    print("\n" + "=" * 60)
    print("PHASE 1a.2: Generating perturbation variants")
    print("=" * 60)

    all_rows = _generate_perturbation_variants(base_records, variants_per_base, seed)
    print(f"Total rows (base + variants): {len(all_rows)}")

    print("\n" + "=" * 60)
    print("PHASE 1b: Labeling with GPTZero API")
    print("=" * 60)

    client = GPTZeroClient(api_key=api_key)
    labeled = _query_gptzero_batched(
        client,
        all_rows,
        batch_size=batch_size,
        delay=delay,
        cache=cache,
        flush_callback=_flush_cache,
    )

    labeled_path = out / "corpus_v6_labeled.jsonl"
    with labeled_path.open("w", encoding="utf-8") as fh:
        for rec in labeled:
            row_out = {k: v for k, v in rec.items() if k != "gptzero_sentences"}
            fh.write(json.dumps(row_out, ensure_ascii=False) + "\n")

    full_labeled_path = out / "corpus_v6_labeled_full.jsonl"
    with full_labeled_path.open("w", encoding="utf-8") as fh:
        for rec in labeled:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    error_count = sum(1 for r in labeled if r.get("has_error"))
    strata_counts = {}
    for r in labeled:
        strata_counts.setdefault(r["stratum"], 0)
        strata_counts[r["stratum"]] += 1

    _flush_cache(cache)

    try:
        from modal import Volume
        Volume.from_name("gptzero-audit-artifacts").commit()
    except Exception:
        pass

    summary = {
        "total_base": len(base_records),
        "total_rows": len(labeled),
        "total_errors": error_count,
        "api_queries": client.call_count,
        "cache_size": len(cache),
        "strata_counts": strata_counts,
        "labeled_path": str(labeled_path),
        "full_labeled_path": str(full_labeled_path),
    }

    summary_path = out / "corpus_v6_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("CORPUS BUILD COMPLETE")
    print("=" * 60)
    print(json.dumps(summary, indent=2))

    return summary
