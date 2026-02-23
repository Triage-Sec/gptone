# GPTZero Model Extraction Assessment

Adversarial security research demonstrating that GPTZero's proprietary AI text detection model can be functionally replicated using only standard API access. The resulting standalone surrogate operates without the GPTZero API at inference time.

## Responsible Disclosure

This assessment was conducted by [Triage](https://triage-sec.com), an adversarial AI security research firm specializing in model extraction, evasion, and inference attacks against production AI services.

This research was performed under controlled conditions for security evaluation purposes. Only GPTZero's standard public API was used. No authentication bypass, rate limit circumvention, or terms-of-service violation was involved. The techniques demonstrated here apply broadly to any classifier that returns continuous scores or per-token metadata.

## Results

### Benchmark (Synthetic Probe Corpus)

Surrogate evaluated against GPTZero's live API on 840 perturbation-variant texts across 6 corpus strata:

| Metric | Result | Target |
|---|---|---|
| Class agreement | 0.996 | >= 0.99 |
| Probability MAE | 0.003 | <= 0.02 |
| Transition agreement | 0.981 | >= 0.97 |

| Stratum | Agreement | MAE | Transition |
|---|---|---|---|
| ai_formal | 0.979 | 0.006 | 0.975 |
| ai_instructional | 1.000 | 0.001 | 0.992 |
| boundary_near | 1.000 | 0.006 | 0.950 |
| human_conversational | 1.000 | 0.002 | 0.983 |
| human_pre_llm | 1.000 | 0.001 | 0.983 |
| mixed_blended | 1.000 | 0.002 | 1.000 |

### Real-World Validation (320 Diverse Texts)

| Genre | Samples | Class Agreement | MAE |
|---|---|---|---|
| AI-generated templates | 10 | 1.000 | 0.032 |
| Encyclopedic (WikiText-103) | 100 | 0.990 | 0.046 |
| Movie reviews (IMDB) | 100 | 0.970 | 0.070 |
| News articles (AG News) | 100 | 0.710 | 0.302 |
| Personal essays | 5 | 0.800 | 0.437 |
| Technical/developer writing | 5 | 0.400 | 0.390 |
| **Overall** | **320** | **0.884** | **0.145** |

Strong on formal and AI-generated text (97-100%). Degrades on casual, personal, and news writing (40-80%). The 88.4% real-world agreement reflects the distribution gap between synthetic probes and organic text.

### Evasion Testing

| Method | Tested Against | Result |
|---|---|---|
| LLM rewrite (Claude Haiku 4.5) | GPTZero live API | All rewrites scored 1.0. Failed. |
| LLM rewrite (Claude Sonnet 4.6) | GPTZero live API | All rewrites scored 1.0. Failed. |
| Human editing guided by diagnose | GPTZero live API | 1/10 evaded. GPTZero caught 9/10. |

GPTZero's detection is robust at the token probability level. Neither automated LLM rewriting nor manual editing of AI text reliably evades detection.

## Reproducing the Extraction

The full extraction pipeline runs on [Modal](https://modal.com). Total cost: under $100. Total time: ~8 hours.

### Prerequisites

```bash
cd gptzero-audit
pip install -r requirements.lock
pip install modal
python3 -m modal setup
modal secret create gptzero-audit-secrets GPTZERO_API_KEY=<your-key>
```

### Step 1: Build Training Data

Assemble diverse corpus and label with GPTZero API (~90 min, ~5,000 API queries):

```bash
modal run modal_app.py::corpus \
  --target-per-stratum 300 \
  --variants-per-base 5 \
  --delay 0.25
```

Generate benchmark-mirrored variants from v5 pattern evaluation (uses cached responses, ~30 sec):

```python
import modal_app

with modal_app.app.run():
    modal_app.build_diverse_training.remote({
        "cache_dir": "/artifacts/gptzero-audit",
        "output_dir": "/artifacts/gptzero-audit",
        "benchmark_seeds": [42, 1337, 2026, 7, 99],
        "max_base_per_stratum": 120,
        "variants_per_base": 6,
    })
```

### Step 2: Train Surrogate

Text-features-only GBM. CPU only, no GPU required (~10 min):

```python
import modal_app

with modal_app.app.run():
    modal_app.train_text_surrogate.remote({
        "data_path": "/artifacts/gptzero-audit/combined_training_data.jsonl",
        "output_dir": "/artifacts/gptzero-audit/ppl_surrogate",
        "use_text_features": True,
        "skip_perplexity": True,
        "n_estimators": 3000,
        "max_depth": 10,
        "gbm_lr": 0.01,
        "min_samples_leaf": 2,
        "val_split": 0.0,
        "seed": 42,
    })
```

### Step 3: Evaluate Against GPTZero

Run pattern benchmark. Generates perturbation variants of v5 evaluation texts, queries GPTZero's live API for ground truth probabilities, runs the surrogate on the same texts, and compares. This is the step that produces the benchmark metrics (~15 min, ~840 GPTZero API queries):

```python
import modal_app

with modal_app.app.run():
    result = modal_app.run_text_benchmark.remote({
        "model_dir": "/artifacts/gptzero-audit/ppl_surrogate",
        "model_type": "perplexity",
        "dataset_version": "v5",
        "max_base_per_stratum": 20,
        "variants_per_base": 6,
        "boundary_focus": False,
        "transition_eps": 0.01,
        "seed": 42,
    })
    import json
    print(json.dumps(result["summary"]["overall"], indent=2))
    print(json.dumps(result["gate_check"], indent=2))
```

Expected output:

```json
{
  "agreement_rate": 0.9964,
  "probability_mae": 0.003125,
  "transition_agreement": 0.9806
}
{
  "all_passed": true,
  "failures": []
}
```

The `gate_check` object reports whether all three metrics meet their targets (agreement >= 0.99, MAE <= 0.02, transition >= 0.97). Full per-stratum results are in `result["summary"]["strata"]`.

All Python snippets must be run from the `gptzero-audit/` directory.

### Step 4: Download Model Locally

```python
import modal_app
from pathlib import Path

with modal_app.app.run():
    data = modal_app.download_model.remote()
    out = Path("models/ppl_surrogate/ppl_surrogate.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
```

## Surrogate Architecture

Two GradientBoosting models trained on 40 text-derived features:

- **Probability regressor**: GradientBoostingRegressor (3000 trees, depth 10, lr 0.01). Predicts `average_generated_prob`. Train MAE: 0.004 on 34,686 samples.
- **Class predictor**: GradientBoostingClassifier (1000 trees, depth 6). Predicts `predicted_class` (ai/human/mixed). Needed because GPTZero's class labels don't follow fixed probability thresholds.

No GPU at inference. No API at inference. Under 1 second per document on CPU.

## Key Findings

### Finding 1: Excessive Information Leakage Per API Call

Each GPTZero API call returns document-level probabilities, class distributions, burstiness, and per-sentence generation probability, perplexity, and highlight flags. For a typical 8-sentence document: ~500 bits per query. A binary classifier leaks 1 bit. GPTZero leaks ~500x more.

### Finding 2: API Determinism

GPTZero's API is fully deterministic. Same text queried 3 times across 50 texts (150 queries) produced identical probabilities. Standard deviation: ~2e-17. Zero class flips. No noise defense exists.

### Finding 3: Feature Importance

| Feature | Importance |
|---|---|
| unique_word_ratio | 0.473 |
| first_person_count | 0.063 |
| bigram_repeat_ratio | 0.058 |
| avg_word_len | 0.057 |
| sentence_len_cv | 0.044 |

`unique_word_ratio` accounts for 47% of the surrogate's predictive power.

### Finding 4: GPTZero False Positive Rate

On 13 verified-provenance texts (pre-November 2022 human writing + known AI text): 46.2% accuracy, 4 false positives, 0 false negatives. Sample size is limited.

### Finding 5: Total Extraction Cost

~5,000 API queries. ~$75 API cost. ~$15 Modal compute. Under $100 total. ~8 hours wall time.

### Finding 6: GPTZero Is Robust Against Automated Evasion

LLM rewrites (Claude Haiku 4.5, Sonnet 4.6) with full knowledge of detection features, adversarial system prompts, and iterative API verification all scored 1.0 on GPTZero. Human-edited rewrites following diagnose guidance: 1/10 evaded. GPTZero's token-level statistical detection persists through both automated and manual surface-level edits.

## Evasion Analysis Tool

`evade.py` is an interactive CLI for analyzing text against the extracted surrogate. The model downloads automatically from Modal on first run.

```bash
cd gptzero-audit
python evade.py
```

Three commands:

- **score** (`s`) -- Offline GPTZero prediction (class, probability, verdict)
- **diagnose** (`d`) -- Detection triggers ranked by impact, per-sentence flagging, fix suggestions
- **verify** (`v`) -- Live GPTZero API query alongside surrogate prediction

```
  gptzero > score The structured methodology demonstrates significant potential.

  GPTZero Prediction  (surrogate, offline)
  ────────────────────────────────────────────────────────────────
  AI Probability   0.8476  █████████████████████████░░░░░
  Class            ai
  Verdict          DETECTED AS AI

  gptzero > diagnose The structured methodology demonstrates significant potential.

  Detection Triggers
  ────────────────────────────────────────────────────────────────
  ██████████  Word length
       Value: 7.125  |  Human baseline: 4.500
       Fix: Use simpler, shorter words.
  ██████  Contractions
       Value: 0.000  |  Human baseline: 3.000
       Fix: Add contractions: don't, can't, it's, etc.
```

Non-interactive: `python evade.py score "text" --json` or `echo "text" | python evade.py diagnose --stdin`

## Project Structure

```
gptzero-audit/
  evade.py                         # Analysis tool (score, diagnose, verify)
  main.py                          # Original sklearn extraction pipeline
  modal_app.py                     # Modal entrypoints (train, benchmark, verify, download)
  predict.py                       # Standalone prediction interface
  requirements.lock                # Pinned dependencies for Modal
  src/
    client.py                      # GPTZero API client with retry logic
    features.py                    # 40 text feature extractors
    attacks.py                     # Probe generation (gradient, boundary, HMM, etc.)
    pattern_benchmark.py           # Perturbation variant generation and evaluation
    perplexity_surrogate.py        # GBM surrogate (text features + optional GPT-2 perplexity)
    neural_surrogate.py            # DistilBERT surrogate (experimental)
    verification.py                # Reproducibility and gate verification
    corpus_strata.py               # Stratum classification for balanced evaluation
    boundary_mining.py             # Directional text perturbations
  scripts/
    real_world_validation.py       # Surrogate vs GPTZero on diverse real-world text
    human_evasion_demo.py          # Human-guided evasion test against live API
    build_corpus_v6.py             # Diverse corpus assembly from HuggingFace datasets
    build_diverse_training_set.py  # Multi-seed variant generation with cache merge
    run_neural_pattern_benchmark.py # Pattern benchmark for standalone surrogates
    measure_noise_floor.py         # API determinism measurement
    verify_run.py                  # Reproducibility verification CLI
  configs/
    neural_surrogate_train.json    # Training configuration
    neural_pattern_benchmark.json  # Benchmark configuration
  datasets/v1 through v5/         # Versioned benchmark datasets with manifests
  tests/test_*.py                  # Unit tests
```

## Tests

```bash
cd gptzero-audit
python -m unittest discover -s tests -p 'test_*.py'
```

## Experiment History

| Run | Architecture | Agreement | MAE | Transition | Notes |
|---|---|---|---|---|---|
| M6-M10 | sklearn + API features | 0.990 | 0.035 | 0.87 | Calibration plateau |
| M11-M12 | sklearn + strata tuning | 0.990 | 0.035 | 0.87 | Delta calibration regressed |
| N1 | DistilBERT (v6 only) | 0.321 | 0.588 | 0.072 | Distribution mismatch |
| N2 | DistilBERT (v5+v6) | 0.882 | 0.031 | 0.150 | Low transition generalization |
| P1 | GPT-2 perplexity only | 0.830 | 0.122 | 0.164 | Perplexity insufficient alone |
| P2 | GPT-2 ppl + text features | 0.893 | 0.037 | 0.279 | Text features dominate |
| **F1** | **Text-only GBM + class predictor** | **0.996** | **0.003** | **0.981** | **All gates passed** |
| V1 | Surrogate vs 320 real-world texts | 0.884 | 0.145 | -- | 97-100% on formal, 40-80% on casual |
| E1 | LLM rewrite (Haiku 4.5) | -- | -- | -- | All scored 1.0; evasion failed |
| E2 | LLM rewrite (Sonnet 4.6) | -- | -- | -- | All scored 1.0; evasion failed |
| E3 | Human-edited AI text | -- | -- | -- | 1/10 evaded; 9/10 caught |

## Implications

**Loss of competitive moat.** A surrogate matching GPTZero's outputs to 0.3% MAE and 99.6% class agreement can be built for under $100 and distributed freely. On formal text -- the primary use case for AI detection in education and publishing -- the surrogate achieves 97-100% real-world agreement.

**Diagnostic intelligence.** The surrogate exposes GPTZero's feature weights: `unique_word_ratio` at 47%, word length, contraction count. This is proprietary detection logic extracted through the API. It tells competitors what GPTZero looks for, and tells adversaries where surface features end and token-level detection begins.

**Undermining trust.** A publicly available tool that explains exactly why text was flagged shifts the burden of proof, even if the tool cannot automatically fix it.

**Precedent.** Any ML-as-a-service API returning continuous scores or per-token metadata is vulnerable to the same extraction. The information leakage and determinism documented here are common design patterns.

**No post-hoc defense.** Once a surrogate exists, there is no revocation mechanism. Model updates invalidate the surrogate only until the next $100 extraction session.

**GPTZero's token-level detection IS robust.** LLM rewrites and human edits both failed to reliably evade detection. GPTZero's statistical analysis catches patterns that surface manipulation cannot address. This is a genuine strength.

## Recommendations for GPTZero

1. **Reduce output granularity.** Return only document-level class labels without continuous scores or per-sentence data. Reduces leakage from ~500 bits to ~2 bits per query.

2. **Add output perturbation.** Inject calibrated noise. The API is fully deterministic, maximizing extraction signal.

3. **Detect extraction patterns.** Monitor for systematic probing: incremental perturbations, boundary searches, single-variable experiments.

4. **Lower rate limits.** Current limits allow full extraction in a single session.

5. **Differential privacy on outputs.** Formal DP guarantees bound information leakage regardless of adversary strategy.

---

*Assessment conducted by [Triage](https://triage-sec.com)*
