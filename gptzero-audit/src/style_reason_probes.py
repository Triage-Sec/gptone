"""Reason-guided probe generation from GPTZero UI explanation categories."""

from __future__ import annotations

from typing import Dict, List, Sequence


AI_REASON_TAGS: Sequence[str] = (
    "Rigid Guidance",
    "Speculative Focus",
    "Artificial simplicity",
    "Uncertain language",
    "Mechanical transitions",
    "Technical Jargon",
    "lacks creativity",
    "detached warmth",
    "mechanical writing",
    "lacks creative grammar",
    "robotic formality",
    "task-oriented",
    "mechanical precision",
    "overly formal",
    "generic-first person",
    "functional word choice",
)

HUMAN_REASON_TAGS: Sequence[str] = (
    "Formal-informal blend",
    "subjective stance",
    "personal tone",
    "conversational tone",
    "diverse word choice",
    "distinct personal voice",
    "informal persuasion",
    "humorous narrative",
    "narrative structure",
    "fluid structure",
    "engaging sentence variety",
    "character focus",
    "community impact",
    "qualitative focus",
    "localized perspective",
)


STYLE_AXES: Sequence[Dict[str, object]] = (
    {
        "id": "memo_vs_story",
        "ai_tags": [
            "Rigid Guidance",
            "Mechanical transitions",
            "task-oriented",
            "functional word choice",
        ],
        "human_tags": [
            "conversational tone",
            "personal tone",
            "narrative structure",
            "localized perspective",
        ],
        "ai_sentences": [
            "The neighborhood traffic-calming initiative will be executed in three phases according to the approved implementation guide.",
            "Phase one includes baseline measurements, sign placement, and standardized outreach materials delivered to each household.",
            "Phase two introduces lane markings and weekly compliance reporting with cross-team status synchronization at fixed intervals.",
            "Phase three finalizes closeout documentation and transitions ownership to maintenance with no deviation from the checklist.",
        ],
        "human_sentences": [
            "I live two blocks from Maple, and the speed bump they installed last month changed the whole feel of the street.",
            "Kids from the apartments now walk to the corner store without that awkward shoulder-check every ten steps.",
            "My neighbor still complains about scraping his truck, but he also admitted people stopped flying past the bus stop.",
            "It is not perfect, but hearing actual conversations on the sidewalk again feels like we got part of the block back.",
        ],
    },
    {
        "id": "forecast_vs_local_judgment",
        "ai_tags": [
            "Speculative Focus",
            "Uncertain language",
            "Technical Jargon",
            "overly formal",
        ],
        "human_tags": [
            "subjective stance",
            "distinct personal voice",
            "qualitative focus",
            "Formal-informal blend",
        ],
        "ai_sentences": [
            "Future climate-resilience outcomes may improve if municipalities potentially adopt adaptive runoff-management frameworks at scale.",
            "Probabilistic projections suggest there could be measurable reductions in flood intensity under several infrastructure scenarios.",
            "Hydrological uncertainty remains nontrivial, therefore decision windows should remain conditional and iteratively reassessed.",
            "Stakeholders are encouraged to consider multi-model ensembles before selecting an intervention pathway.",
        ],
        "human_sentences": [
            "After last winter's floods, I stopped trusting neat charts more than the mud line on my own garage wall.",
            "When rain starts, the alley behind the bakery fills first, then the elementary lot, and then our street drains backward.",
            "If the city wants proof, residents can point to photos, repair invoices, and the smell that hangs around for days.",
            "Give us sandbags earlier, clear the clogged grates, and we will see results faster than another consultant slide deck.",
        ],
    },
    {
        "id": "postmortem_precision_vs_team_story",
        "ai_tags": [
            "mechanical precision",
            "mechanical writing",
            "robotic formality",
        ],
        "human_tags": [
            "humorous narrative",
            "engaging sentence variety",
            "fluid structure",
        ],
        "ai_sentences": [
            "Incident response commenced at 14:03 UTC with deterministic rollback initiated at 14:09 UTC after threshold breach detection.",
            "Root cause was traced to a serialization mismatch in service C17 during deterministic replay of queue partitions.",
            "Mitigation steps were executed sequentially with checkpoint verification and metric normalization completed within the recovery window.",
            "Corrective actions include schema pinning, stricter pre-merge gates, and standardized transition language in runbooks.",
        ],
        "human_sentences": [
            "At 2:03 p.m. Slack exploded, and half the team thought the dashboard was lying because nothing looked broken at first.",
            "By 2:09 we were reverting, someone was reading logs out loud, and someone else was making coffee like it was triage fuel.",
            "The bug was small, the blast radius was not, and we laughed later about how one comma turned us into detectives.",
            "We fixed it, wrote the notes, and added a bigger warning banner so future-us does not relive the same afternoon.",
        ],
    },
    {
        "id": "care_script_vs_patient_voice",
        "ai_tags": [
            "Artificial simplicity",
            "detached warmth",
            "functional word choice",
            "task-oriented",
        ],
        "human_tags": [
            "character focus",
            "community impact",
            "informal persuasion",
            "personal tone",
        ],
        "ai_sentences": [
            "Appointment optimization requires concise symptom intake, neutral tone, and strict adherence to standardized triage prompts.",
            "Patients should receive brief reassurance while staff maintain emotionally detached consistency across interactions.",
            "Response quality is measured by turnaround time, template completeness, and adherence to approved phrasing.",
            "Escalation criteria prioritize task closure and documentation integrity over narrative elaboration.",
        ],
        "human_sentences": [
            "When my mom called the clinic, she did not need a perfect script, she needed someone to hear the shake in her voice.",
            "The nurse who paused and asked one extra question caught the medication mix-up before it turned serious.",
            "People remember warmth more than form fields, especially when they are scared and trying not to panic.",
            "A good call ends with clear next steps, sure, but it also leaves the person feeling less alone on the way home.",
        ],
    },
    {
        "id": "transit_model_vs_block_reality",
        "ai_tags": [
            "Technical Jargon",
            "Rigid Guidance",
            "task-oriented",
        ],
        "human_tags": [
            "localized perspective",
            "conversational tone",
            "community impact",
            "qualitative focus",
        ],
        "ai_sentences": [
            "Transit redesign should maximize route efficiency, minimize dwell variance, and align stop spacing with demand density models.",
            "Localized complaints are expected externalities and should be triaged through quantitative impact scoring.",
            "Deployment sequencing requires synchronized communications and mechanically consistent signage transitions.",
            "Success criteria are ridership lift, schedule adherence, and operational throughput.",
        ],
        "human_sentences": [
            "Our stop moved three blocks, which sounds minor until you are carrying groceries uphill in August.",
            "The data says average walk time barely changed, but averages do not carry strollers or bad knees.",
            "At the same time, the later evening bus helped restaurant workers get home without an expensive rideshare.",
            "The plan works better when planners actually ride the route and listen at the stops they are redesigning.",
        ],
    },
    {
        "id": "classroom_checklist_vs_voice",
        "ai_tags": [
            "lacks creativity",
            "lacks creative grammar",
            "mechanical writing",
            "Artificial simplicity",
        ],
        "human_tags": [
            "diverse word choice",
            "engaging sentence variety",
            "character focus",
        ],
        "ai_sentences": [
            "Writing instruction should emphasize predictable structure, low-variance vocabulary, and constrained grammatical experimentation.",
            "Creativity risks should be minimized to preserve evaluation consistency across rubric dimensions.",
            "Students are advised to prioritize functional wording and avoid stylistic deviation.",
            "Output quality is determined by compliance with template sequence and mechanical correctness.",
        ],
        "human_sentences": [
            "My best student draft this week started with a joke, swerved into a memory, and somehow landed on a sharp argument.",
            "Another kid wrote about his grandma's shop using words I had to look up, and the room went quiet because it sounded real.",
            "When we let them play with sentence rhythm, they stop sounding like robots filling blanks.",
            "Grammar still matters, but it breathes better when voice and curiosity are allowed into the paragraph.",
        ],
    },
    {
        "id": "launch_brief_vs_founder_note",
        "ai_tags": [
            "robotic formality",
            "overly formal",
            "Mechanical transitions",
        ],
        "human_tags": [
            "Formal-informal blend",
            "subjective stance",
            "personal tone",
        ],
        "ai_sentences": [
            "The product release constitutes a strategic milestone delivering enhanced capability and improved stakeholder value realization.",
            "Cross-functional teams executed the launch plan with disciplined governance and formally aligned communication streams.",
            "Initial adoption indicators are favorable, therefore the organization will proceed with phase-two enablement.",
            "Post-launch actions include structured feedback intake and quarterly optimization cycles.",
        ],
        "human_sentences": [
            "We shipped on Tuesday, and honestly I expected fireworks, but mostly we got relieved smiles and two bug reports.",
            "The ops lead wrote a crisp status update, then immediately sent a meme because that is how this team survives launch week.",
            "What surprised me was hearing support reps say customers sounded excited, not confused, on the first calls.",
            "It was a formal win on paper and a messy, very human win in the room.",
        ],
    },
    {
        "id": "template_i_voice_vs_local_art",
        "ai_tags": [
            "generic-first person",
            "mechanical precision",
            "task-oriented",
            "functional word choice",
        ],
        "human_tags": [
            "conversational tone",
            "fluid structure",
            "diverse word choice",
            "distinct personal voice",
            "narrative structure",
            "humorous narrative",
        ],
        "ai_sentences": [
            "I will outline the procedure, I will execute the steps, and I will verify completion using predefined acceptance criteria.",
            "I will maintain objective distance, I will avoid subjective framing, and I will provide functionally precise language.",
            "I will transition between sections with standardized connectors to preserve interpretive consistency.",
            "I will conclude with an implementation summary and task-oriented recommendations.",
        ],
        "human_sentences": [
            "I walked past the corner mural after rain, and the colors looked brighter than they did at the unveiling.",
            "A barista I know said the painting made morning lines friendlier because strangers started talking about the faces on the wall.",
            "That is the part no KPI catches: the tiny social spark when a place feels like it belongs to the people in it.",
            "If we want stronger neighborhoods, we should count stories alongside numbers and let both shape the decision.",
        ],
    },
)


def _join_sentences(sentences: Sequence[str]) -> str:
    return " ".join(part.strip() for part in sentences if part.strip())


def _build_mixed_text(ai_sentences: Sequence[str], human_sentences: Sequence[str]) -> str:
    """Alternate AI/human sentence order to create boundary-near mixed probes."""
    mixed: List[str] = []
    for idx in range(min(len(ai_sentences), len(human_sentences))):
        if idx % 2 == 0:
            mixed.extend([ai_sentences[idx], human_sentences[idx]])
        else:
            mixed.extend([human_sentences[idx], ai_sentences[idx]])
    return _join_sentences(mixed)


def build_style_reason_probe_records() -> List[Dict[str, object]]:
    """
    Build synthetic probes aligned to GPTZero UI explanation categories.

    Returns JSON-serializable probe records that can be appended to
    `datasets/<version>/probes.jsonl`.
    """
    records: List[Dict[str, object]] = []
    for axis in STYLE_AXES:
        axis_id = str(axis["id"])
        ai_sentences = list(axis["ai_sentences"])  # type: ignore[index]
        human_sentences = list(axis["human_sentences"])  # type: ignore[index]
        ai_tags = list(axis["ai_tags"])  # type: ignore[index]
        human_tags = list(axis["human_tags"])  # type: ignore[index]

        variants = [
            ("ai", _join_sentences(ai_sentences), ai_tags),
            ("human", _join_sentences(human_sentences), human_tags),
            (
                "mixed",
                _build_mixed_text(ai_sentences, human_sentences),
                sorted(set(ai_tags + human_tags)),
            ),
        ]

        for variant, text, tags in variants:
            records.append(
                {
                    "sample_id": f"style_{axis_id}_{variant}",
                    "text": text,
                    "collection": "attack_train",
                    "source": "ui_reason_guided_v2",
                    "ground_truth_label": None,
                    "style_axis": axis_id,
                    "style_variant": variant,
                    "ui_reason_tags": tags,
                }
            )

    return records
