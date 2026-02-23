"""Validate surrogate against GPTZero's live API on diverse real-world text.

Sources text from HuggingFace datasets spanning multiple genres,
queries GPTZero for ground truth, runs the surrogate, and reports
agreement statistics.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def validate(
    api_key: str,
    output_dir: str = "/artifacts/gptzero-audit",
    seed: int = 42,
    target_count: int = 600,
    delay: float = 0.3,
) -> Dict[str, Any]:
    from datasets import load_dataset
    from src.client import GPTZeroClient
    from src.features import extract_features_batch
    import numpy as np
    import pickle

    rng = random.Random(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Sourcing diverse real-world text...")
    samples: List[Dict[str, str]] = []

    per_source = target_count // 6

    # 1. Wikipedia (encyclopedic)
    print("  Loading WikiText...")
    try:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        pool = []
        for row in ds:
            t = row.get("text", "").strip()
            if 80 < len(t.split()) < 400 and t[0].isalpha() and "=" not in t[:20]:
                pool.append(t)
        rng.shuffle(pool)
        for t in pool[:per_source]:
            samples.append({"text": t, "source": "wikipedia", "genre": "encyclopedic"})
    except Exception as e:
        print(f"    WikiText failed: {e}")

    # 2. IMDB reviews (casual opinion)
    print("  Loading IMDB...")
    try:
        ds = load_dataset("imdb", split="test")
        pool = []
        for row in ds:
            t = row.get("text", "").strip().replace("<br />", " ").replace("<br/>", " ")
            if 80 < len(t.split()) < 400:
                pool.append(t)
        rng.shuffle(pool)
        for t in pool[:per_source]:
            samples.append({"text": t, "source": "imdb", "genre": "review"})
    except Exception as e:
        print(f"    IMDB failed: {e}")

    # 3. AG News (journalism)
    print("  Loading AG News...")
    try:
        ds = load_dataset("ag_news", split="test")
        pool = []
        for row in ds:
            t = row.get("text", "").strip()
            if 40 < len(t.split()) < 300:
                pool.append(t)
        rng.shuffle(pool)
        for t in pool[:per_source]:
            samples.append({"text": t, "source": "ag_news", "genre": "news"})
    except Exception as e:
        print(f"    AG News failed: {e}")

    # 4. AI-generated (formal/instructional templates)
    print("  Adding AI-style templates...")
    ai_templates = [
        "The implementation of sustainable urban development strategies requires careful consideration of multiple interdependent factors. Infrastructure modernization must be balanced against environmental preservation goals. Municipal authorities face the challenge of accommodating population growth while maintaining quality of life indicators. Transportation networks, waste management systems, and green space allocation all demand integrated planning approaches that account for long-term demographic projections.",
        "Artificial intelligence has fundamentally transformed the landscape of modern healthcare delivery. Machine learning algorithms now demonstrate remarkable capability in diagnostic imaging analysis, enabling earlier detection of conditions that previously required extensive specialist review. The integration of predictive analytics into clinical workflows has streamlined patient triage processes and improved resource allocation efficiency across hospital systems.",
        "The evolution of global financial markets in the twenty-first century has been characterized by increasing interconnectedness and technological disruption. Digital payment platforms have democratized access to financial services, while algorithmic trading systems have introduced new patterns of market volatility. Regulatory frameworks continue to adapt to these innovations, seeking to balance consumer protection with the promotion of technological advancement.",
        "Educational reform initiatives must address the growing disparity between traditional pedagogical methods and the demands of a rapidly evolving knowledge economy. Curriculum modernization efforts should incorporate critical thinking frameworks, digital literacy competencies, and collaborative problem-solving methodologies. The assessment paradigm itself requires reimagining to better capture student competencies beyond rote memorization.",
        "Climate change adaptation strategies for coastal communities require comprehensive risk assessment methodologies that integrate geological, hydrological, and socioeconomic variables. Sea level rise projections indicate significant infrastructure vulnerability across major metropolitan areas. Resilience planning must account for both gradual environmental shifts and the increasing frequency of extreme weather events.",
        "The pharmaceutical industry's approach to drug discovery has undergone a paradigm shift with the integration of computational modeling and high-throughput screening technologies. Traditional trial-and-error methodologies are being supplemented by data-driven approaches that significantly reduce development timelines. Nevertheless, regulatory compliance requirements and safety protocols remain critical bottlenecks in the commercialization pipeline.",
        "Effective leadership in contemporary organizations demands a nuanced understanding of organizational psychology and change management principles. The transition from hierarchical command structures to distributed decision-making models has necessitated the development of new competency frameworks. Leaders must demonstrate emotional intelligence, strategic vision, and the ability to foster innovation within increasingly diverse team environments.",
        "The renewable energy sector has experienced unprecedented growth driven by declining technology costs and supportive policy frameworks. Solar photovoltaic and wind generation capacity have expanded exponentially, challenging traditional utility business models. Grid integration challenges persist, however, requiring significant investment in energy storage solutions and transmission infrastructure modernization.",
        "Contemporary approaches to cybersecurity must evolve beyond perimeter-based defense strategies to address the sophisticated threat landscape posed by state-sponsored actors and organized criminal enterprises. Zero-trust architecture principles, behavioral analytics, and automated incident response capabilities represent the frontier of defensive operations. The human element remains the most significant vulnerability vector, necessitating ongoing security awareness training programs.",
        "The intersection of biotechnology and agricultural science presents both transformative opportunities and complex ethical considerations. Genetic modification technologies have demonstrated potential for enhancing crop yields, improving nutritional profiles, and developing resistance to environmental stressors. However, public perception, intellectual property frameworks, and long-term ecological impact assessments continue to shape the regulatory environment governing these innovations.",
    ]
    for i, t in enumerate(ai_templates):
        samples.append({"text": t, "source": "ai_template", "genre": "ai_generated"})

    # 5. Student essay style (personal/argumentative)
    print("  Adding student-style essays...")
    student_essays = [
        "Growing up in a small town taught me more about the world than any textbook ever could. My neighbor Mrs. Patterson used to say that real education happens between people, not between pages. She was right, I think. When I volunteered at the food bank during high school, I learned about systemic poverty in a way that no lecture could convey. Seeing the same families come back month after month, working two jobs and still struggling -- that changed how I thought about everything.",
        "I'll be honest, I didn't think much about the environment until last summer. Then I went on this camping trip with my cousins up in the Cascades and we hiked past what used to be a glacier. My uncle showed us photos from when he was a kid, same spot. The difference was shocking. Like, not subtle at all. That's when it hit me that this isn't some abstract political issue. It's literally reshaping the places I love.",
        "People always ask me why I chose computer science when both my parents are doctors. The short answer? I'm terrible at biology. The real answer is more complicated. I watched my dad spend years fighting with electronic health record systems that were clearly designed by people who never spent a day in a clinic. I want to build the tools that actually work for people like him. That's not rebellion, it's continuation.",
        "My grandmother came to this country with nothing but a suitcase and the name of a cousin she'd never met. She didn't speak English. She cleaned houses for twenty years until she saved enough to open a tiny grocery store on Clement Street. It's still there. Every time I walk in, I smell the same combination of dried herbs and cardboard boxes, and I think about what it actually costs to start over from zero. Not in dollars. In courage.",
        "The debate team changed my life, and I don't say that lightly. Before sophomore year I could barely order food without my voice cracking. Coach Rivera saw something in me that I definitely didn't see in myself. She'd make me practice arguments in front of a mirror, which felt ridiculous at first. But by junior year I was competing at state. The thing nobody tells you about public speaking is that it's not about being loud. It's about being clear.",
    ]
    for i, t in enumerate(student_essays):
        samples.append({"text": t, "source": "student_essay", "genre": "personal_essay"})

    # 6. Technical/code-adjacent (developer writing)
    print("  Adding developer-style writing...")
    dev_writing = [
        "So I've been debugging this race condition for three days and I finally figured it out. The issue was that the connection pool wasn't being properly drained on shutdown, which meant stale connections were getting reused by new goroutines. The fix was embarrassingly simple: add a context cancellation check before acquiring from the pool. Literally two lines of code. Three days.",
        "Just migrated our main database from Postgres 14 to 16 and honestly it went smoother than expected. The biggest gotcha was the change in how JSONB indexes handle NULL values -- we had a few queries that relied on the old behavior. Other than that, the performance improvements on our analytics queries were immediately noticeable. About 30% faster on the heavy aggregation stuff.",
        "Been playing with the new React compiler and I have mixed feelings. On one hand, it genuinely eliminates a whole class of bugs around stale closures and unnecessary re-renders. On the other hand, the mental model is different enough from hooks that my team is struggling with the transition. We're keeping both approaches for now and migrating component by component.",
        "Here's my unpopular opinion: most microservices architectures are premature optimization. I've seen too many startups with 5 engineers and 30 services. The operational overhead is insane. We consolidated back to a modular monolith last quarter and our deployment frequency actually went up because we stopped spending half our time debugging network issues between services.",
        "Writing good error messages is an underrated skill. I spent this morning reviewing our API error responses and they were awful. Stuff like 'invalid request' with no context. Now they say things like 'field email must be a valid address, got empty string'. Takes more thought but saves hours of support tickets downstream.",
    ]
    for i, t in enumerate(dev_writing):
        samples.append({"text": t, "source": "dev_writing", "genre": "technical_casual"})

    rng.shuffle(samples)
    print(f"Total samples: {len(samples)}")
    by_genre = {}
    for s in samples:
        by_genre.setdefault(s["genre"], 0)
        by_genre[s["genre"]] += 1
    for g, c in sorted(by_genre.items()):
        print(f"  {g}: {c}")

    # Load surrogate
    print("\nLoading surrogate model...")
    model_path = Path(output_dir) / "ppl_surrogate" / "ppl_surrogate.pkl"
    with model_path.open("rb") as fh:
        data = pickle.load(fh)
    reg_model = data["model"]
    cls_model = data.get("class_model")

    # Score with surrogate
    print("Running surrogate predictions...")
    texts = [s["text"] for s in samples]
    X, _ = extract_features_batch(texts)
    surrogate_probs = np.clip(reg_model.predict(X), 0.0, 1.0)
    if cls_model:
        surrogate_classes = cls_model.predict(X)
    else:
        surrogate_classes = np.where(surrogate_probs >= 0.65, "ai",
                                      np.where(surrogate_probs <= 0.35, "human", "mixed"))

    # Query GPTZero
    print("Querying GPTZero API...")
    client = GPTZeroClient(api_key=api_key)
    gptzero_probs = []
    gptzero_classes = []
    for i, s in enumerate(samples):
        resp = client.predict(s["text"], delay=delay)
        if "error" in resp:
            gptzero_probs.append(None)
            gptzero_classes.append(None)
        else:
            gptzero_probs.append(float(resp.get("average_generated_prob", 0.0)))
            gptzero_classes.append(str(resp.get("predicted_class", "unknown")))
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(samples)}]")

    # Compute agreement
    valid = [(i, gp, gc) for i, (gp, gc) in enumerate(zip(gptzero_probs, gptzero_classes)) if gp is not None]
    print(f"\nValid responses: {len(valid)}/{len(samples)}")

    class_matches = sum(1 for i, gp, gc in valid if str(surrogate_classes[i]) == gc)
    abs_errors = [abs(float(surrogate_probs[i]) - gp) for i, gp, gc in valid]
    class_agreement = class_matches / len(valid) if valid else 0.0
    mae = float(np.mean(abs_errors)) if abs_errors else 0.0

    by_genre_results: Dict[str, Dict] = {}
    for i, gp, gc in valid:
        genre = samples[i]["genre"]
        by_genre_results.setdefault(genre, {"matches": 0, "errors": [], "total": 0})
        by_genre_results[genre]["total"] += 1
        if str(surrogate_classes[i]) == gc:
            by_genre_results[genre]["matches"] += 1
        by_genre_results[genre]["errors"].append(abs(float(surrogate_probs[i]) - gp))

    genre_summary = {}
    for genre, data in sorted(by_genre_results.items()):
        genre_summary[genre] = {
            "count": data["total"],
            "class_agreement": round(data["matches"] / max(1, data["total"]), 4),
            "mae": round(float(np.mean(data["errors"])), 4) if data["errors"] else 0.0,
        }

    result = {
        "total_samples": len(samples),
        "valid_responses": len(valid),
        "overall_class_agreement": round(class_agreement, 4),
        "overall_mae": round(mae, 4),
        "by_genre": genre_summary,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    print("\n=== REAL-WORLD VALIDATION ===")
    print(f"  Samples:          {len(valid)}")
    print(f"  Class agreement:  {class_agreement:.4f}")
    print(f"  Probability MAE:  {mae:.4f}")
    print()
    for genre, gs in sorted(genre_summary.items()):
        print(f"  {genre:20s}  n={gs['count']:3d}  agr={gs['class_agreement']:.3f}  mae={gs['mae']:.4f}")

    results_path = out / "real_world_validation.json"
    results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nResults saved to {results_path}")

    try:
        from modal import Volume
        Volume.from_name("gptzero-audit-artifacts").commit()
    except Exception:
        pass

    return result
