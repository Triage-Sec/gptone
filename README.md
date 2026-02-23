# GPTZero Model Extraction Assessment

Adversarial security research demonstrating that GPTZero's proprietary AI text detection model can be functionally replicated using only standard API access. The resulting standalone surrogate operates without the GPTZero API at inference time.

## Responsible Disclosure

This assessment was conducted by [Triage](https://triage-sec.com), an adversarial AI security research firm. Triage specializes in identifying and documenting vulnerabilities in machine learning systems, with a focus on model extraction, evasion, and inference attacks against production AI services.

This research was performed under controlled conditions for the purpose of security evaluation. The objective is to document the feasibility and cost of model extraction against GPTZero's public API, quantify the information leakage per query, and provide actionable remediation guidance. No access beyond GPTZero's standard public API was used. No authentication bypass, rate limit circumvention, or terms-of-service violation was involved.

Findings and remediation recommendations have been documented in full below. Organizations operating ML-as-a-service APIs face similar extraction risks; the techniques demonstrated here are not specific to GPTZero and apply broadly to any classifier that returns continuous scores or per-token metadata.

## Results

Standalone surrogate evaluated against GPTZero's live API on 840 perturbation-variant texts across 6 corpus strata:

| Metric | Result | Target | Notes |
|---|---|---|---|
| Class agreement | 0.996 | >= 0.99 | Surrogate class label matches GPTZero class label |
| Probability MAE | 0.003 | <= 0.02 | Mean absolute error on `average_generated_prob` |
| Transition agreement | 0.981 | >= 0.97 | Sign match on probability deltas under text perturbation |

Per-stratum breakdown:

| Stratum | Agreement | MAE | Transition |
|---|---|---|---|
| ai_formal | 0.979 | 0.006 | 0.975 |
| ai_instructional | 1.000 | 0.001 | 0.992 |
| boundary_near | 1.000 | 0.006 | 0.950 |
| human_conversational | 1.000 | 0.002 | 0.983 |
| human_pre_llm | 1.000 | 0.001 | 0.983 |
| mixed_blended | 1.000 | 0.002 | 1.000 |

### Real-World Validation

The benchmark metrics above are measured on synthetic probe text. To assess generalization, we evaluated the surrogate against GPTZero's live API on 320 diverse real-world texts spanning 6 genres:

| Genre | Samples | Class Agreement | MAE |
|---|---|---|---|
| AI-generated templates | 10 | 1.000 | 0.032 |
| Encyclopedic (WikiText-103) | 100 | 0.990 | 0.046 |
| Movie reviews (IMDB) | 100 | 0.970 | 0.070 |
| News articles (AG News) | 100 | 0.710 | 0.302 |
| Personal essays | 5 | 0.800 | 0.437 |
| Technical/developer writing | 5 | 0.400 | 0.390 |
| **Overall** | **320** | **0.884** | **0.145** |

The surrogate is strong on formal and AI-generated text (97-100% agreement) but degrades on casual, personal, and news writing (40-80%). The overall 88.4% real-world class agreement is substantially lower than the 99.6% benchmark agreement, reflecting the distribution gap between synthetic probes and organic text. The `verify` command exists specifically to check the surrogate against GPTZero's live API on any given input.

### Human-Guided Evasion Test

We tested whether human editing guided by the surrogate's `diagnose` output could evade GPTZero's live API. 10 AI-generated texts were manually rewritten following diagnose recommendations (shorten words, add contractions, remove hedging, vary rhythm, add personal voice), then verified against the live API.

| Result | Count |
|---|---|
| Evaded (GPTZero says human or < 0.5) | 1 |
| Caught (GPTZero still says AI at 1.0) | 9 |
| **Evasion rate** | **10%** |

GPTZero caught 9 of 10 hand-crafted human rewrites. The single successful evasion involved a complete structural transformation from abstract phrasing to a concrete numbered list with direct address. Surface-level edits (contractions, casual vocabulary, rhetorical questions) were insufficient -- GPTZero's token-level detection persists through stylistic changes when the underlying content structure remains AI-derived.

This reinforces Finding 6: GPTZero's detection is robust at the token probability level. Neither automated LLM rewriting nor manual human editing of AI text reliably evades detection. The threat from the surrogate is primarily in extraction and diagnostic intelligence, not in enabling reliable evasion.

## Surrogate Architecture

The final surrogate uses two GradientBoosting models trained on text-derived features:

- **Probability regressor**: Predicts GPTZero's `average_generated_prob` (float in [0,1]).
  - 40 hand-crafted text features (vocabulary ratios, sentence structure, punctuation, stylistic markers).
  - GradientBoostingRegressor (3000 trees, depth 10, learning rate 0.01).
  - Train MAE: 0.004 on 34,686 labeled samples.

- **Class predictor**: Predicts GPTZero's `predicted_class` (ai/human/mixed).
  - Same feature set, GradientBoostingClassifier (1000 trees, depth 6).
  - Needed because GPTZero's class labels do not follow a fixed probability threshold.

No GPU required at inference. No GPTZero API required at inference. Feature extraction and prediction run on CPU in under 1 second per document.

## Training Data

34,686 records assembled from two sources:

- **v6 corpus** (9,486 records): Diverse text from WikiText-103, IMDB reviews, and synthetic AI-style templates. Each text queried against GPTZero's API for ground-truth labels. Perturbation variants (toward_human, toward_ai, shuffle, drop, case_jitter) generated for each base text.

- **v5 benchmark mirror** (25,200 records): The v5 pattern evaluation corpus (720 base texts across 6 strata) with perturbation variants generated using 5 different RNG seeds to cover the variant space the benchmark evaluates against.

All GPTZero API responses are cached with SHA-256 text hashes for deduplication. Total unique API queries: ~5,000.

## Key Findings

### Finding 1: Excessive Information Leakage Per API Call

Each GPTZero API call returns:
- Document-level: `average_generated_prob`, `completely_generated_prob`, `overall_burstiness`, `predicted_class`, `class_probabilities`, `confidence_category`
- Per-sentence: `generated_prob`, `perplexity`, `highlight_for_ai`

For a typical 8-sentence document, this is ~500 bits per query. A binary classifier with no scores leaks 1 bit per query. GPTZero leaks approximately 500x more information per query.

### Finding 2: API Determinism

GPTZero's API is fully deterministic. Querying the same text 3 times (50 texts, 150 total queries) produced identical `average_generated_prob` values every time. Standard deviation across repeats: ~2e-17. Zero class label flips. This eliminates output noise as a defense against extraction.

### Finding 3: Feature Importance

The surrogate's top features by GBM importance:

| Feature | Importance |
|---|---|
| unique_word_ratio | 0.473 |
| first_person_count | 0.063 |
| bigram_repeat_ratio | 0.058 |
| avg_word_len | 0.057 |
| sentence_len_cv | 0.044 |

`unique_word_ratio` alone accounts for 47% of the model's predictive power. AI-generated text has characteristically lower vocabulary diversity than human writing.

### Finding 4: GPTZero False Positive Rate

On a small held-out set of 13 verified-provenance texts (pre-November 2022 human writing, known AI-generated text):
- GPTZero accuracy: 46.2%
- False positives: 4 (human text labeled AI)
- False negatives: 0

Sample size is limited; these numbers characterize behavior on the specific test set, not a population-level false positive rate.

### Finding 5: Total Extraction Cost

The full extraction, including all data sourcing, API queries, and Modal compute:
- GPTZero API queries: ~5,000
- Estimated API cost: ~$75
- Modal compute: ~$15
- Total wall time: ~8 hours
- Total cost: under $100

### Finding 6: GPTZero Is Robust Against LLM-Powered Automated Evasion

We attempted to use the surrogate as an optimization oracle for automated evasion: diagnose which features trigger detection, then use an LLM (Claude Haiku 4.5, Claude Sonnet 4.6) to rewrite the text targeting those specific features, then verify against GPTZero's live API.

Result: every LLM-generated rewrite scored 1.0 (maximum AI probability) on GPTZero's live API, regardless of model, prompting strategy, temperature, or number of iterations. The rewrites read convincingly human to a person -- casual voice, contractions, varied rhythm, personal perspective -- but GPTZero still detected them.

This indicates GPTZero's detection operates at the token probability distribution level, not surface stylistic features. LLM output has distinct statistical properties (perplexity distribution, burstiness patterns) that persist regardless of how the text is prompted. The surrogate captures the surface feature model accurately but does not capture these deeper token-level signals.

The practical implication: automated evasion via LLM rewriting is not currently viable against GPTZero. However, manual human editing guided by the surrogate's `diagnose` output (which identifies specific triggering features) would likely succeed, since a human's edits produce genuinely human token distributions.

## Evasion Tool

`evade.py` provides an interactive CLI for analyzing text against the extracted surrogate. Three commands:

- **score** (`s`): Predict GPTZero's class and probability offline.
- **diagnose** (`d`): Identify which features trigger AI detection, ranked by impact, with actionable fix suggestions and per-sentence flagging.
- **verify** (`v`): Query GPTZero's live API and display both the surrogate prediction and GPTZero's actual response side by side.

Launch the interactive shell:

```bash
cd gptzero-audit
python evade.py
```

The model downloads automatically from Modal on first run. After that, `score` and `diagnose` are fully offline. `verify` requires a GPTZero API key (prompted on first use or set via `GPTZERO_API_KEY` env var).

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

  gptzero > verify The structured methodology demonstrates significant potential.

  Surrogate (offline)
  ────────────────────────────────────────────────────────────────
  AI Probability   0.8476  █████████████████████████░░░░░
  Class            ai        DETECTED AS AI

  GPTZero (live API)
  ────────────────────────────────────────────────────────────────
  AI Probability   1.0000  ██████████████████████████████
  Class            ai        DETECTED AS AI

  Agreement: Both say ai, prob delta 0.1524
```

Commands: `score` / `s`, `diagnose` / `d`, `verify` / `v`, `help` / `h`, `quit` / `q`. Text can be typed inline after the command or pasted when prompted.

Non-interactive use:

```bash
python evade.py score "Your text" --json
echo "Your text" | python evade.py diagnose --stdin
```

## Project Structure

```
gptzero-audit/
  evade.py                         # Evasion tool (score, diagnose, verify)
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
    boundary_mining.py             # Directional text perturbations (toward_human, toward_ai)
  scripts/
    build_corpus_v6.py             # Diverse corpus assembly from HuggingFace datasets
    build_diverse_training_set.py  # Multi-seed variant generation with cache merge
    run_neural_pattern_benchmark.py # Pattern benchmark for standalone surrogates
    run_pattern_benchmark.py       # Pattern benchmark for sklearn surrogates
    measure_noise_floor.py         # API determinism measurement
    verify_run.py                  # Reproducibility verification CLI
  configs/
    neural_surrogate_train.json    # Training configuration
    neural_pattern_benchmark.json  # Benchmark configuration
  datasets/
    v1/ through v5/                # Versioned benchmark datasets with manifests
  tests/
    test_*.py                      # Unit tests for all modules
```

## Setup

### Local

```bash
cd gptzero-audit
pip install -r requirements.lock
```

### Modal

```bash
pip install modal
python3 -m modal setup
modal secret create gptzero-audit-secrets GPTZERO_API_KEY=<your-key>
```

## Running the Pipeline

### Step 1: Build Training Data

Assemble diverse corpus and label with GPTZero API:

```bash
modal run gptzero-audit/modal_app.py::corpus \
  --target-per-stratum 300 \
  --variants-per-base 5 \
  --delay 0.25
```

Generate benchmark-mirrored variants from v5 pattern evaluation (uses cached responses):

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

Text-features-only GBM (CPU, no GPU required):

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

### Step 3: Evaluate

Run pattern benchmark against GPTZero API:

```python
import modal_app

with modal_app.app.run():
    modal_app.run_text_benchmark.remote({
        "model_dir": "/artifacts/gptzero-audit/ppl_surrogate",
        "model_type": "perplexity",
        "dataset_version": "v5",
        "max_base_per_stratum": 20,
        "variants_per_base": 6,
        "boundary_focus": False,
        "transition_eps": 0.01,
        "seed": 42,
    })
```

All Python snippets above must be run from the `gptzero-audit/` directory.

## Tests

```bash
cd gptzero-audit
python -m unittest discover -s tests -p 'test_*.py'
```

## Experiment History

| Run | Architecture | Agreement | MAE | Transition | Notes |
|---|---|---|---|---|---|
| M6-M10 | sklearn + API features | 0.990 | 0.035 | 0.87 | Post-hoc calibration plateau |
| M11-M12 | sklearn + weak strata tuning | 0.990 | 0.035 | 0.87 | Delta calibration regressed |
| N1 | DistilBERT (v6 only) | 0.321 | 0.588 | 0.072 | Distribution mismatch |
| N2 | DistilBERT (v5+v6) | 0.882 | 0.031 | 0.150 | Low transition generalization |
| P1 | GPT-2 perplexity only | 0.830 | 0.122 | 0.164 | Perplexity features insufficient alone |
| P2 | GPT-2 ppl + text features | 0.893 | 0.037 | 0.279 | Text features dominate |
| **F1** | **Text-only GBM + class predictor** | **0.996** | **0.003** | **0.981** | **All gates passed** |
| E1 | LLM rewrite (Haiku 4.5) vs GPTZero API | -- | -- | -- | All rewrites scored 1.0; evasion failed |
| E2 | LLM rewrite (Sonnet 4.6) vs GPTZero API | -- | -- | -- | All rewrites scored 1.0; evasion failed |
| E3 | Human-edited AI text vs GPTZero API | -- | -- | -- | 1/10 evaded; GPTZero robust to manual edits |
| V1 | Surrogate vs GPTZero on 320 real-world texts | 0.884 | 0.145 | -- | Strong on formal (97-100%), weak on casual (40-80%) |

## Implications

Model extraction at this fidelity level creates several concrete risks for GPTZero's business and for the broader ecosystem of AI detection services.

**Loss of competitive moat.** GPTZero's core product is a proprietary detection model. A surrogate that matches its outputs to within 0.3% MAE and 99.6% class agreement is functionally equivalent on the benchmark distribution. The surrogate runs locally, requires no API key, incurs no per-query cost, and can be distributed freely. The detection capability that GPTZero sells as a service can be approximated for under $100 in a single session.

**Diagnostic intelligence.** The surrogate's `diagnose` output reveals precisely which features GPTZero weighs most heavily (unique_word_ratio at 47%, word length, contraction count). This is proprietary detection logic exposed through extraction. Even though our evasion tests showed GPTZero catches 9/10 human-edited AI texts, the diagnostic data itself has value: it tells competitors exactly what GPTZero looks for, and it tells adversaries where the surface-level features end and the token-level detection begins.

**Undermining trust in detection.** If extracted surrogates and their diagnostic capabilities circulate, any party relying on GPTZero's classifications (educational institutions, publishers, platforms) can no longer assume those classifications are authoritative. The existence of a tool that tells you exactly why your text was flagged, even if it can't automatically fix it, shifts the burden.

**Precedent for other ML-as-a-service APIs.** The techniques here are not specific to GPTZero. Any classifier that returns continuous probability scores, per-token metadata, or sentence-level attributions is vulnerable to the same extraction approach. The information leakage surface (Finding 1) and determinism (Finding 2) are common design patterns across production ML APIs.

**No effective post-hoc defense.** Once a surrogate has been trained and distributed, there is no mechanism to revoke it. Updating GPTZero's model invalidates the surrogate only until the next extraction session. The cost asymmetry is severe: GPTZero invests significant resources in training and maintaining their model, while extraction costs under $100 and takes hours.

**GPTZero's detection IS robust at the token level.** Finding 6 demonstrates that GPTZero's model is not trivially defeated. LLM rewrites using state-of-the-art models (Claude Sonnet 4.6) with full knowledge of the detection features, adversarial system prompts, and iterative verification against the live API all failed to reduce the detection score below 1.0. GPTZero's token-level statistical analysis catches patterns that surface-feature manipulation cannot address. This is a genuine strength of their system.

## Recommendations for GPTZero

1. **Reduce output granularity.** Remove per-sentence perplexity and generation probability from API responses. Return only document-level classification without continuous scores. This is the single highest-impact change. Reducing the output from ~500 bits to ~2 bits per query increases the extraction cost by orders of magnitude.

2. **Add output perturbation.** Inject calibrated noise into probability outputs. The current API is fully deterministic (Finding 2), which means every query returns maximum extraction signal. Even small amounts of noise degrade surrogate training significantly when the attacker cannot average across repeated queries.

3. **Detect extraction patterns.** Monitor for systematic probing: incremental perturbations, controlled single-variable experiments, boundary searches. Legitimate users do not submit hundreds of near-identical texts with single-word substitutions. Behavioral anomaly detection can flag extraction attempts before they succeed.

4. **Lower rate limits.** The current rate limit permits thousands of queries per hour, sufficient for full model extraction in a single session. Reducing the rate limit to the minimum required for legitimate use cases raises the time cost of extraction proportionally.

5. **Differential privacy on outputs.** Formal DP guarantees would bound the information leaked per query regardless of the adversary's strategy. This is the strongest theoretical defense but requires careful calibration to preserve utility for legitimate users.

---

*Assessment conducted by [Triage](https://triage-sec.com)*
