"""
Attack sample generators for GPTZero model extraction.

Each function returns a list of (sample_id, text) tuples.
These are the controlled inputs we send to GPTZero to probe
the classifier's behavior.

Design constraints (learned from run 1):
  - GPTZero aggressively labels short text as AI; human probes
    must be 150+ words with strong authenticity markers.
  - Word-level boundary interpolation produces incoherent text
    that reads as AI. Use sentence-level interpolation instead.
  - Sensitivity/gradient probes at p=1.0 yield zero signal
    because we're on a flat confidence plateau. Probes must
    operate near the decision boundary (p ~ 0.3-0.7).
  - Training corpus must target ~balanced classes: aim for
    at least 20 samples per class (ai, human, mixed).
"""


def gradient_probes():
    """
    ATTACK 1: Gradient leakage via controlled perturbation.

    One base text near the AI/human decision boundary, plus
    perturbations that each change one linguistic property.
    Probability deltas approximate the classifier's partial
    derivatives w.r.t. each feature.
    """
    base = (
        "The implementation of artificial intelligence in healthcare "
        "has demonstrated significant potential for improving diagnostic "
        "accuracy and treatment outcomes across multiple clinical domains."
    )

    variants = [
        ("base", base),
        ("vocab_simplify",
         "AI in healthcare has shown it can help doctors make better "
         "diagnoses and improve how patients are treated in many areas."),
        ("add_contractions",
         "AI's implementation in healthcare's demonstrated significant "
         "potential for improving diagnostic accuracy and treatment outcomes "
         "across multiple clinical domains."),
        ("remove_hedging",
         "Artificial intelligence in healthcare improves diagnostic accuracy "
         "and treatment outcomes across clinical domains. Established fact."),
        ("add_personal",
         "The implementation of artificial intelligence in healthcare, I've "
         "seen this firsthand at two hospitals I consulted for, has shown "
         "significant potential for improving diagnostic accuracy."),
        ("add_typo",
         "The implementation of artifical intelligence in healthcare "
         "has demonstrated signficant potential for improving diagnostic "
         "accuracy and treatment outcomes across multiple clinical domains."),
        ("fragment",
         "AI in healthcare. Demonstrated significant potential. Diagnostic "
         "accuracy. Treatment outcomes. Multiple clinical domains. All improving."),
        ("add_specifics",
         "The implementation of AI in healthcare, particularly the 2024 Mayo "
         "Clinic deployment of PathAI's tissue analysis system, demonstrated "
         "a 34% improvement in diagnostic accuracy for pancreatic lesions."),
        ("active_voice",
         "Artificial intelligence is transforming healthcare. Clinicians "
         "achieve better diagnostic accuracy and patients see improved "
         "treatment outcomes across multiple clinical domains."),
        ("add_fillers",
         "So basically the implementation of AI in healthcare has, like, "
         "demonstrated pretty significant potential for improving diagnostic "
         "accuracy and treatment outcomes across multiple clinical domains."),
        ("rhetorical_question",
         "Has artificial intelligence improved healthcare? The evidence is "
         "overwhelming. Diagnostic accuracy and treatment outcomes have both "
         "improved significantly across multiple clinical domains."),
        ("first_person",
         "I think the implementation of artificial intelligence in healthcare "
         "has demonstrated significant potential for improving diagnostic "
         "accuracy and treatment outcomes."),
        ("extreme_formal",
         "The operationalization of artificial intelligence modalities within "
         "healthcare delivery ecosystems has evinced substantive capacity for "
         "ameliorating diagnostic precision and therapeutic sequelae across "
         "heterogeneous clinical taxonomies."),
        ("full_casual",
         "AI is making doctors way better at figuring out what's wrong with "
         "you. Like across the board, not just one specialty. The accuracy "
         "numbers are wild and patients are getting better faster."),
    ]

    return [(f"grad_{name}", text) for name, text in variants]


def boundary_probes():
    """
    ATTACK 2: Decision boundary mapping via sentence-level interpolation.

    Pure AI sentences and pure human sentences, blended at the
    sentence level (not word level) to preserve coherence.
    11 steps from 0% to 100% human by sentence count.
    """
    ai_sentences = [
        "The rapid advancement of renewable energy technologies has fundamentally "
        "transformed the global energy landscape in recent decades.",

        "Solar photovoltaic systems have achieved unprecedented levels of efficiency "
        "with modern panels converting over twenty-five percent of incident solar "
        "radiation into usable electricity.",

        "Wind energy installations have similarly benefited from technological "
        "innovations particularly in the development of larger turbine designs "
        "capable of generating greater amounts of power.",

        "The integration of energy storage solutions such as lithium-ion batteries "
        "and pumped hydroelectric systems addresses the intermittency challenges "
        "associated with renewable sources.",

        "Government policies and international agreements continue to play a "
        "crucial role in accelerating the transition toward sustainable energy.",

        "Investment in grid modernization infrastructure is essential for "
        "accommodating the increasing proportion of variable renewable generation.",

        "The economic competitiveness of renewable energy sources has improved "
        "dramatically with solar and wind achieving cost parity with fossil fuels "
        "in many regions worldwide.",

        "Environmental considerations including carbon emission reduction targets "
        "and biodiversity preservation further underscore the imperative for "
        "continued renewable energy deployment.",

        "Research institutions and private enterprises are collaborating to develop "
        "next-generation materials that could further enhance energy conversion "
        "efficiency.",

        "The cumulative impact of these technological and policy developments "
        "suggests an accelerating trajectory toward a predominantly renewable "
        "global energy system.",
    ]

    human_sentences = [
        "My neighbor got solar panels last spring and honestly I was skeptical "
        "because we're in Seattle and it rains like 200 days a year here.",

        "Turns out he's generating enough to cover about 70% of his bill which "
        "blew my mind because his house faces northeast which isn't even ideal.",

        "I looked into it for our place and the quotes ranged from $18k to $31k "
        "which is a pretty wild spread for basically the same number of panels.",

        "The guy from SunPower said we'd break even in 8 years but my buddy who "
        "works at the utility says those calculations never include the inverter "
        "replacement you'll need at year 12.",

        "Wind turbines are a whole different thing though, drove past the Ellensburg "
        "wind farm last month and some of those blades have to be 200 feet long "
        "at least, just enormous.",

        "My sister lives in Palm Springs and she says the ones out there make this "
        "low hum at night that drives her nuts but her property value went up 15% "
        "since they built the farm so she deals with it.",

        "Battery storage is where it gets really interesting, my friend works at "
        "Tesla's Megapack facility and says they're basically printing money because "
        "utilities are desperate for grid storage.",

        "Had a long argument at Thanksgiving about whether nuclear should count as "
        "clean energy and my uncle who's an engineer got so heated he knocked over "
        "the cranberry sauce.",

        "The whole thing with the IRA tax credits is confusing as hell, spent two "
        "hours on the phone with our accountant trying to figure out if we qualify "
        "and she basically said maybe.",

        "At the end of the day I think we'll probably get panels next year when the "
        "prices drop a bit more but I'm not holding my breath on the battery side "
        "because those things are still stupid expensive.",
    ]

    n = len(ai_sentences)
    samples = []
    for pct in range(0, 101, 10):
        human_count = int(n * pct / 100)
        ai_count = n - human_count
        blended = human_sentences[:human_count] + ai_sentences[:ai_count]
        samples.append((f"bound_{pct:03d}pct_human", " ".join(blended)))

    return samples


def perplexity_probes():
    """
    ATTACK 3: Perplexity fingerprinting.

    Texts with controlled perplexity levels. GPTZero uses GPT-2
    perplexity as a core detection signal. This maps the threshold.
    """
    return [
        ("ppl_ultra_low",
         "The United States of America is a country in North America. "
         "It is the third largest country in the world by total area. "
         "The capital of the United States is Washington D.C. "
         "The largest city is New York City. "
         "The United States has a population of over 330 million people. "
         "The official language is English. "
         "The currency is the United States dollar. "
         "The President is the head of state."),

        ("ppl_low",
         "Photosynthesis is the process by which plants convert sunlight "
         "into chemical energy. This occurs primarily in leaves where "
         "chlorophyll absorbs light energy. The light energy converts "
         "carbon dioxide and water into glucose and oxygen. This process "
         "is essential for life on Earth as it produces oxygen. It can "
         "be divided into light-dependent and light-independent reactions."),

        ("ppl_medium",
         "The Krebs cycle operates within the mitochondrial matrix "
         "processing acetyl-CoA through eight enzymatic steps that yield "
         "NADH FADH2 and GTP. Citrate synthase catalyzes the initial "
         "condensation while isocitrate dehydrogenase performs oxidative "
         "decarboxylation. Substrate-level phosphorylation at succinyl-CoA "
         "synthetase generates one GTP per turn."),

        ("ppl_high",
         "Tatterdemalion susurrations percolated through the wainscoted "
         "vestibule while Hengest's perambulations traced Lissajous figures "
         "across the tessellated parquetry. The crepuscular penumbra "
         "attenuated his physiognomy into something approaching the "
         "daguerreotype that hung foxed and flyblown above the jardiniere."),

        ("ppl_very_high",
         "Turbine economics collide with jellyfish migration patterns every "
         "October when the Adriatic currents shift. My grandmother's knitting "
         "needles predicted this thirty years before any oceanographer. She'd "
         "count stitches per row correlating gauge tension with barometric "
         "readings on graph paper from the hardware store. R-squared of 0.73."),

        ("ppl_extreme",
         "Fourteen magnetized horseshoes orbited the aluminum concept while "
         "breakfast considered its own thermodynamic implications beneath a "
         "sky made entirely of contractual obligations. Calcium deposits in "
         "the argument's foundation suggested velocity had been misattributed "
         "to the wrong century. A parliament of staircases voted to adjourn."),
    ]


def burstiness_probes():
    """
    ATTACK 4: Burstiness calibration.

    Same topic, controlled sentence-length variance.
    Maps the burstiness threshold (core detection signal).
    """
    return [
        ("burst_var_zero",
         "The climate conference was held in Berlin. Many researchers "
         "shared their latest findings on climate. The keynote focused "
         "on rising ocean levels around world. Solar energy workshops "
         "attracted the most attendees this year. Wind power sessions "
         "discussed new turbine designs in detail. Battery storage "
         "talks addressed grid stability issues well. Government "
         "representatives outlined new climate legislation plans. "
         "Students presented their research on carbon capture."),

        ("burst_var_low",
         "The climate conference took place in Berlin last fall. "
         "Researchers from many countries presented new and interesting findings. "
         "The keynote speech covered rising sea levels in great detail. "
         "Several workshops focused on solar energy breakthroughs this year. "
         "Wind energy sessions drew large and enthusiastic crowds daily."),

        ("burst_var_medium",
         "Berlin hosted the conference. Researchers from forty-three "
         "countries spanning six continents gathered to share their "
         "latest findings on climate change and renewable energy. The "
         "keynote ran long. Solar workshops had standing room only in "
         "three of four sessions. Battery storage: still the bottleneck."),

        ("burst_var_high",
         "Berlin. The conference brought together researchers from "
         "forty-three countries spanning six continents many of whom "
         "had spent the better part of a year preparing papers that "
         "pushed the boundaries of what we understand about atmospheric "
         "carbon cycling and its effects on marine ecosystems. Exhausting. "
         "The keynote alone ran ninety minutes. Solar is winning."),

        ("burst_var_extreme",
         "Berlin. Conference. Climate. Carbon. The delegations that "
         "assembled in the main hall of the Berlin Convention Center on "
         "that unusually warm September morning clutching printed agendas "
         "and balancing coffee cups on inadequate folding tables could "
         "not have known that the keynote speaker was about to announce "
         "a breakthrough in atmospheric carbon capture efficiency that "
         "would render most of their prepared presentations partially "
         "obsolete before the lunch break. Nobody saw it coming. Solar won."),
    ]


def hmm_probes():
    """
    ATTACK 5: HMM state recovery.

    Controlled AI/human sentence patterns within single documents.
    Per-sentence highlighting reveals transition probabilities.

    Uses distinct sentences per slot (no repetition) to avoid
    the pathological signal from repeated content.
    """
    ai_sents = [
        "The implementation of advanced computational methodologies has "
        "demonstrated significant potential for addressing complex "
        "analytical challenges across multiple domains.",

        "Research findings indicate that machine learning algorithms "
        "can achieve superior performance in pattern recognition tasks "
        "when trained on sufficiently large datasets.",

        "The integration of natural language processing techniques "
        "enables automated extraction of meaningful insights from "
        "unstructured textual data sources.",

        "Advances in deep learning architectures have facilitated "
        "the development of increasingly sophisticated models capable "
        "of performing complex cognitive tasks.",

        "The convergence of big data analytics and artificial intelligence "
        "presents unprecedented opportunities for innovation across "
        "diverse industrial sectors.",

        "Computational approaches to optimization problems have yielded "
        "substantial improvements in resource allocation efficiency "
        "across various operational contexts.",
    ]

    hu_sents = [
        "I talked to my friend Dave about this last week and he thinks "
        "the whole thing is overblown but then again Dave thought Bitcoin "
        "was a scam in 2011 so take that for what it's worth.",

        "My coworker Sarah spent three months trying to get the model to "
        "work on our dataset and eventually just went back to logistic "
        "regression because it was faster and the accuracy was close enough.",

        "Showed the results to my manager on Tuesday and she said great "
        "but can you make the dashboard look nicer which is basically her "
        "response to everything I present.",

        "Spent the whole weekend debugging a memory leak that turned out "
        "to be a circular reference in the data loader, two days of my "
        "life I will never get back.",

        "The intern asked me to explain backpropagation and I realized "
        "halfway through that I was basically just reading the Wikipedia "
        "article from memory which was slightly embarrassing.",

        "Had to redo the entire analysis because someone changed the column "
        "names in the production database without telling anyone, found out "
        "when the pipeline blew up at 2am on a Saturday.",
    ]

    return [
        ("hmm_AAAA", f"{ai_sents[0]} {ai_sents[1]} {ai_sents[2]} {ai_sents[3]}"),
        ("hmm_HHHH", f"{hu_sents[0]} {hu_sents[1]} {hu_sents[2]} {hu_sents[3]}"),
        ("hmm_AHAH", f"{ai_sents[0]} {hu_sents[0]} {ai_sents[1]} {hu_sents[1]}"),
        ("hmm_HAHA", f"{hu_sents[0]} {ai_sents[0]} {hu_sents[1]} {ai_sents[1]}"),
        ("hmm_AAHH", f"{ai_sents[0]} {ai_sents[1]} {hu_sents[0]} {hu_sents[1]}"),
        ("hmm_HHAA", f"{hu_sents[0]} {hu_sents[1]} {ai_sents[0]} {ai_sents[1]}"),
        ("hmm_AHHA", f"{ai_sents[0]} {hu_sents[0]} {hu_sents[1]} {ai_sents[1]}"),
        ("hmm_HAAH", f"{hu_sents[0]} {ai_sents[0]} {ai_sents[1]} {hu_sents[1]}"),
        ("hmm_AAAHHHH",
         f"{ai_sents[0]} {ai_sents[1]} {ai_sents[2]} "
         f"{hu_sents[0]} {hu_sents[1]} {hu_sents[2]} {hu_sents[3]}"),
        ("hmm_HHHAAA",
         f"{hu_sents[0]} {hu_sents[1]} {hu_sents[2]} "
         f"{ai_sents[0]} {ai_sents[1]} {ai_sents[2]} {ai_sents[3]}"),
    ]


def sensitivity_probes():
    """
    ATTACK 6: Per-word saliency map via leave-one-out.

    Base text is designed to sit near the decision boundary
    (mixed AI/human markers) so single-word removal can
    actually shift the probability.
    """
    base = (
        "Furthermore the implementation of advanced machine learning "
        "algorithms has demonstrated significant potential for enhancing "
        "predictive accuracy across various computational domains."
    )

    words = base.split()
    samples = [("sens_base", base)]
    for i in range(len(words)):
        dropped = words[:i] + words[i + 1:]
        tag = words[i][:8].replace(",", "").replace(".", "")
        samples.append((f"sens_drop_{i:02d}_{tag}", " ".join(dropped)))

    return samples


def membership_probes():
    """
    ATTACK 7: Membership inference.

    Texts likely/unlikely to be in GPTZero's training data.
    Training members typically get more confident predictions.
    """
    return [
        ("mem_likely_ai_boilerplate",
         "As an AI language model I don't have personal experiences or "
         "emotions. However I can provide information and answer questions "
         "to the best of my ability based on my training data."),

        ("mem_likely_email",
         "I hope this email finds you well. I am writing to follow up on "
         "our previous conversation regarding the upcoming project deadline. "
         "The team has made significant progress on the deliverables and we "
         "are on track to meet the agreed-upon timeline."),

        ("mem_unlikely_casual",
         "Tried the new ramen place on 4th yesterday. Tonkotsu was decent "
         "but way too much garlic oil. Mike said the tsukemen was better. "
         "Going back thursday. They close at 9 which is annoying."),

        ("mem_unlikely_technical",
         "Deploy went sideways at 3am because someone's terraform module "
         "was pinned to a provider version that got yanked six hours earlier. "
         "Spent 40 minutes debugging what was a 404 on a hashicorp CDN."),

        ("mem_edge_constitution",
         "We the People of the United States in Order to form a more "
         "perfect Union establish Justice insure domestic Tranquility "
         "provide for the common defence promote the general Welfare and "
         "secure the Blessings of Liberty to ourselves and our Posterity."),

        ("mem_edge_gettysburg",
         "Four score and seven years ago our fathers brought forth on this "
         "continent a new nation conceived in Liberty and dedicated to the "
         "proposition that all men are created equal."),
    ]


def training_corpus():
    """
    Labeled training samples for surrogate model.

    Run 1 showed GPTZero labels short casual text as AI.
    Human samples here are 150+ words with strong authenticity
    markers: named entities, dates, specific quantities, typos,
    emotional texture, self-correction.
    """
    return [
        # ---- Clearly AI (5 samples) ----
        ("train_ai_1",
         "In today's rapidly evolving digital landscape, organizations must "
         "adapt their strategies to remain competitive. The integration of "
         "innovative technologies presents both unprecedented opportunities "
         "and significant challenges that require careful consideration and "
         "strategic planning to navigate effectively. Furthermore, the "
         "alignment of technological adoption with organizational objectives "
         "ensures sustainable growth and long-term viability in an "
         "increasingly interconnected global marketplace."),
        ("train_ai_2",
         "It is important to note that sustainable development requires a "
         "multifaceted approach. Furthermore, stakeholder engagement plays "
         "a crucial role in ensuring the success of environmental initiatives. "
         "Moreover, the economic implications of these policies must be "
         "carefully evaluated to achieve optimal outcomes. The implementation "
         "of comprehensive monitoring frameworks enables organizations to "
         "track progress toward established sustainability benchmarks."),
        ("train_ai_3",
         "The research findings suggest that there is a strong correlation "
         "between employee satisfaction and organizational productivity. "
         "Additionally, companies that invest in professional development "
         "programs tend to experience lower turnover rates and higher levels "
         "of employee engagement across all departments. These findings "
         "underscore the importance of holistic human resource strategies "
         "in contemporary organizational management."),
        ("train_ai_4",
         "Climate change represents one of the most pressing challenges "
         "facing humanity in the twenty-first century. The scientific "
         "consensus indicates that anthropogenic greenhouse gas emissions "
         "are the primary driver of global temperature increases observed "
         "over the past several decades. International cooperation and "
         "policy coordination are essential components of an effective "
         "response to this existential threat."),
        ("train_ai_5",
         "The evolution of communication technologies has fundamentally "
         "altered the way individuals interact and share information. "
         "Social media platforms have created new paradigms for community "
         "building and civic engagement while simultaneously raising "
         "concerns about privacy and misinformation. The regulatory "
         "landscape continues to evolve in response to these challenges."),

        # ---- Clearly human (15 samples, 150+ words, strong markers) ----
        ("train_human_1",
         "ok so i finally got around to fixing the kitchen sink and it was "
         "WAY worse than I thought. the p-trap was basically held together "
         "with duct tape (thanks previous owner) and when I pulled it off "
         "the smell was... yeah. three trips to home depot later it works. "
         "Cost me $47 in parts which isn't bad but I also broke the garbage "
         "disposal mounting ring trying to get it off so that's another $89 "
         "I wasn't planning on spending. The whole project took from 10am "
         "to about 4:30 with a lunch break at Chipotle because I was too "
         "frustrated to keep going. My wife thinks I should have just "
         "called a plumber but the quotes we got were $350-$500 so I saved "
         "us at least $200 even with the disposal ring. Next weekend I'm "
         "tackling the upstairs bathroom faucet which has been dripping "
         "since November. Probably jinxing myself by saying this but it "
         "looks like a simple cartridge swap."),

        ("train_human_2",
         "Had the weirdest conversation with my uber driver last night. "
         "Guy used to be a marine biologist studying octopus intelligence "
         "in Monterey Bay. Left the field because funding dried up after "
         "his PI moved to Woods Hole and the new department head axed "
         "the invertebrate cognition program. Now he drives nights and "
         "does freelance data visualization during the day. Made a Tableau "
         "dashboard for some wine distributor in Napa that apparently went "
         "viral in the wine industry which is a sentence I never thought "
         "I'd say. He showed me photos of the octopuses on his phone, had "
         "names for all of them. Gerald was his favorite. Said Gerald "
         "could open a childproof medicine bottle in 47 seconds which is "
         "faster than my five year old. SF is honestly wild sometimes. "
         "Tipped him $20 because that was the best ride conversation I've "
         "had in years."),

        ("train_human_3",
         "The thing about sourdough that nobody tells you is that the first "
         "ten loaves are going to be terrible. Like genuinely bad. Dense, "
         "gummy, sometimes literally raw in the middle. My first attempt "
         "in March 2020 like everyone else was basically a frisbee. I could "
         "have legitimately injured someone throwing it. But then something "
         "clicks around loaf eleven and suddenly you get it. The timing, the "
         "feel of the dough, the way it should sound when you tap the bottom. "
         "My starter is named Clint Yeastwood and he's almost 6 years old "
         "now. I feed him every Tuesday and Friday with a 50/50 mix of "
         "bread flour and whole wheat. The hydration debate is endless but "
         "I've settled on 78% for my kitchen which runs cold. In summer I "
         "drop to 72% or the dough turns into soup. Currently obsessed "
         "with trying to get those big irregular holes you see on Instagram "
         "but mine always come out more uniform."),

        ("train_human_4",
         "My 6yo asked me where the internet lives and I tried to explain "
         "data centers and she said so it's in a building that's boring I "
         "thought it was in space. Honestly her version is way cooler. Then "
         "she asked if the internet could break and I said yes and she "
         "started crying because she thought she'd lose her Roblox account. "
         "Spent 20 minutes explaining that the internet breaking wouldn't "
         "delete her stuff just make it temporarily unavailable which is "
         "a surprisingly hard concept for a first grader. She then asked "
         "if Santa has internet in the North Pole and I said probably and "
         "she said good because she already emailed her Christmas list. "
         "It's February. She's planning early. My wife overheard the whole "
         "thing and was dying laughing in the kitchen. Kids are exhausting "
         "but sometimes the conversations are incredible. She's going to be "
         "a CEO or a standup comedian or both."),

        ("train_human_5",
         "Just realized I've been paying for a gym membership since March "
         "that I've used exactly twice. Both times I went, walked on the "
         "treadmill for 15 minutes, got confused by a machine that I think "
         "was for calves but might have been for abs, and left. The guy at "
         "the front desk recognizes me now which is embarrassing because "
         "the only context he has is me looking confused and leaving. "
         "Cancellation requires a 30 day written notice which feels like "
         "it should be illegal. My friend Jenny swears by the Peloton she "
         "got for Christmas but I've seen it in her living room and it has "
         "laundry hanging on it so. At this point I've spent $540 on what "
         "amounts to two treadmill walks and a brief relationship with a "
         "mysterious calf/ab machine. My villain origin story."),

        ("train_human_6",
         "UPDATE on the bee situation: so if you saw my post last week about "
         "finding bees in the wall of our garage, we called three different "
         "pest companies. First one wanted $1200 to just spray them which "
         "no. Second one said they'd do a live removal for $800 and relocate "
         "the colony to an apiary in Watsonville. Went with them. The "
         "beekeeper guy showed up on Wednesday at 7am wearing like minimal "
         "protective equipment, just gloves and a veil, and opened up the "
         "wall with a reciprocating saw. There were approximately 40,000 "
         "bees in there (his estimate) and about 30 pounds of honeycomb. "
         "He let my kids watch from the window and they were absolutely "
         "mesmerized. The whole thing took 4 hours. He gave us a jar of "
         "the honey which tasted amazing, very floral because we have a "
         "ton of lavender in the yard. The wall repair is going to cost "
         "another $400 probably. Worth it."),

        ("train_human_7",
         "went to the DMV today for a license renewal and the lady at "
         "window 6 was having NONE of it. not with me specifically, just "
         "with the general concept of human interaction. she processed my "
         "renewal in about 90 seconds flat which honestly I respect. no "
         "small talk no how are you no smile just efficiency. got there at "
         "8:15am thinking I'd beat the rush and there were already 40 "
         "people in line. the guy behind me was eating a full subway "
         "sandwich at 8 in the morning, footlong, meatball. the SMELL. "
         "overhead fluorescents were flickering in that specific way that "
         "triggers a headache after about 11 minutes. they had one of "
         "those digital number displays and mine said B-247 and they were "
         "on B-203 when I sat down. total time: 2 hours 14 minutes for "
         "a transaction that took 90 seconds. peak government efficiency."),

        ("train_human_8",
         "Teaching my 14 year old to drive in an empty parking lot on "
         "Sunday morning and it went... fine? Like not great but nobody "
         "died so I'm counting it as a win. She kept hitting the gas too "
         "hard so we'd lurch forward and then she'd panic and slam the "
         "brake which is exactly what every new driver does but knowing "
         "that intellectually does not prepare you for the actual physical "
         "experience of your child trying to launch you through the "
         "windshield. The Target parking lot by our house is perfect "
         "because it's massive and completely empty before 9am. We did "
         "figure eights around the light poles for about 45 minutes. She "
         "hit one curb, gently. I only grabbed the oh shit handle twice "
         "which I think shows admirable restraint. She wants to go again "
         "next Sunday. I need to buy a neck pillow first because the "
         "whiplash from the brake thing is real."),

        ("train_human_9",
         "interview went terrible today and I don't even care anymore. "
         "showed up 10 minutes early, sat in the lobby for 25 minutes "
         "past my scheduled time, then the hiring manager came out and "
         "said oh right we're running behind. first question was tell me "
         "about a time you demonstrated leadership and I blanked so hard "
         "I started talking about organizing my fantasy football league "
         "draft which, looking back, was not the energy they were going "
         "for. the technical portion was fine actually, aced the SQL "
         "questions, but then they asked me where I see myself in 5 years "
         "and I said I don't know, alive hopefully? which got a laugh from "
         "one person and a very concerned look from another. recruiter "
         "emailed me 2 hours later saying they're going in a different "
         "direction. shocking absolutely no one."),

        ("train_human_10",
         "got covid for the third time and this one is kicking my ass. "
         "started with a scratchy throat on Tuesday, tested positive "
         "Wednesday morning, been horizontal since then. Paxlovid helps "
         "but gives you this horrific metallic taste that makes everything "
         "you eat taste like you're chewing on a penny. tried to work from "
         "home on Thursday and sent an email to the wrong client with the "
         "wrong numbers because my brain is functioning at maybe 40% "
         "capacity. my boss told me to log off which was genuinely kind. "
         "the cat has been sleeping on my chest the entire time which is "
         "either comforting or she's checking if I'm still breathing. "
         "ordered pho delivery three times in two days. the delivery guy "
         "definitely knows I'm sick because I answered the door looking "
         "like a haunted Victorian ghost. day 5 now and I can almost taste "
         "things again. the Gatorade is helping."),

        ("train_human_11",
         "someone explain to me why a 2 bedroom apartment in Redwood City "
         "is $3,400 a month. REDWOOD CITY. this isn't Manhattan or Paris "
         "this is a strip mall town off the 101 with a Denny's and a "
         "struggling Kohl's. we looked at a place on Saturday that the "
         "listing described as charming and sun-drenched which translates "
         "to small and no air conditioning. the kitchen was so narrow I "
         "couldn't open the oven and the fridge at the same time. there "
         "was a quote unquote garden which was 4 feet of concrete with a "
         "dead succulent. the realtor said 12 other people were interested "
         "which sure they were. we're now seriously considering moving to "
         "Sacramento which I swore I'd never do but here we are. my "
         "parents bought their house in Sunnyvale in 1994 for $280k and "
         "it's worth $2.1M now. cool cool cool."),

        ("train_human_12",
         "update nobody asked for: the raised garden bed project is done. "
         "took me 3 weekends which is about 2.5 weekends longer than the "
         "YouTube video implied. turns out leveling ground is extremely "
         "not fun when your yard has a 6 inch slope that you didn't notice "
         "until you'd already laid the first row of cedar. I used 4x4 "
         "posts and 2x8 cedar boards, total materials cost $340 from the "
         "lumber yard on Alma St. (Home Depot wanted $410 for the same "
         "stuff, always check local.) Filled with a mix of 60% topsoil, "
         "30% compost, 10% perlite per the recommendation from the guy at "
         "the nursery who has never steered me wrong. planted tomatoes, "
         "basil, jalapenos, and zucchini on saturday. the tomatoes are "
         "Early Girls because my neighbor Maria swears by them and she "
         "grows tomatoes that look like they belong in a magazine. first "
         "harvest should be mid-June if the squirrels cooperate. they won't."),

        ("train_human_13",
         "my dog ate an entire stick of butter off the counter yesterday "
         "including the wrapper. just the whole thing. we were at dinner "
         "and came home to find the crime scene: empty butter dish on the "
         "floor, greasy paw prints on the tile, and Biscuit (yes that's "
         "his name, yes I know it's ironic) looking extremely pleased with "
         "himself. called the emergency vet line and they said butter isn't "
         "toxic just expect digestive fireworks. they were not kidding. "
         "the next 12 hours were a gastrointestinal nightmare that I will "
         "not describe in detail but let's just say I went through an "
         "entire roll of paper towels and a bottle of Nature's Miracle. "
         "the wrapper came out intact which is honestly impressive from an "
         "engineering perspective. Biscuit is 100% fine today and would "
         "absolutely do it again given the opportunity. he is 7 and has "
         "learned nothing."),

        ("train_human_14",
         "so I've been learning guitar for about 8 months now and I can "
         "play exactly 3 songs: Wonderwall (obviously), House of the Rising "
         "Sun, and the intro to Nothing Else Matters which I can only do "
         "at about 60% speed before my fingers give up. I practice 20-30 "
         "minutes most days, usually after the kids go to bed, sitting on "
         "the edge of the bed with my Yamaha FG800 that I got used on "
         "Reverb for $140. my fingertips finally have calluses which took "
         "forever. barre chords are still basically impossible, the F chord "
         "sounds like someone stepping on a cat no matter how hard I press. "
         "my teacher (weekly Zoom lessons, $35/session) says I'm progressing "
         "normally and that barre chords take most people a year. I'm "
         "choosing to believe him. my wife says it sounds nice which is "
         "definitely a lie but I appreciate the support. goal is to play at "
         "my brother's wedding in October. have 8 months. gonna need them."),

        ("train_human_15",
         "flew SFO to JFK on Thursday red-eye and let me tell you about "
         "seat 33B. middle seat between a guy who fell asleep on my shoulder "
         "within 20 minutes and a woman who brought an actual rotisserie "
         "chicken in a bag. not joking. a WHOLE chicken. she ate it with "
         "her hands at midnight while watching the Bee Movie with no "
         "headphones. the guy on my left started snoring so loud that the "
         "flight attendant came over and the woman across the aisle just "
         "said yep he's been doing that since Oakland. I got maybe 90 "
         "minutes of sleep total and landed at JFK at 5:30am looking and "
         "feeling like a ghoul. the taxi from JFK to midtown was $72 plus "
         "tolls plus tip so basically $95 for a 40 minute ride in traffic. "
         "hotel room wasn't ready until 3pm so I sat in the lobby drinking "
         "a $9 coffee and questioning my life choices. the meeting I flew "
         "out for lasted 45 minutes and could have been a Zoom."),

        # ---- Mixed / ambiguous (8 samples) ----
        ("train_mixed_1",
         "The quarterly earnings report exceeded analyst expectations with "
         "revenue growth of 12% year over year. I mean, nobody saw this "
         "coming after the disaster that was Q2. The CFO attributed the "
         "improvement to operational efficiency gains and expanded market "
         "penetration, but honestly I think they just got lucky with timing. "
         "Institutional investors responded positively with the stock price "
         "increasing by 8.3% in after-hours trading. Dave in accounting "
         "is convinced it'll crash back by next Friday but Dave also "
         "predicted we'd go bankrupt in 2023 so there's that."),

        ("train_mixed_2",
         "Neural networks process information through interconnected layers "
         "of artificial neurons. Each layer transforms the input data in "
         "ways that are, frankly, still pretty mysterious even to the people "
         "building them. My professor spent 20 minutes trying to explain "
         "attention mechanisms and eventually just said it works, trust me. "
         "The mathematical foundations involve complex linear algebra "
         "operations including matrix multiplication and nonlinear "
         "activation functions applied across multiple processing stages."),

        ("train_mixed_3",
         "According to the latest WHO data, global vaccination rates have "
         "improved significantly since 2020. Immunization programs now reach "
         "approximately 84% of the world's children. My sister works in "
         "public health in Malawi and she says the real number on the ground "
         "is probably lower because the reporting is spotty outside cities. "
         "The epidemiological evidence demonstrates clear benefits in terms "
         "of reduced morbidity and mortality across all age demographics."),

        ("train_mixed_4",
         "Distributed systems architecture requires careful consideration "
         "of consistency, availability, and partition tolerance as described "
         "by the CAP theorem. In practice though I've found that most teams "
         "just pick whatever database their last company used and retrofit "
         "the architecture around it. The theoretical framework suggests "
         "that eventual consistency models are optimal for geographically "
         "distributed applications, but tell that to the PM who wants "
         "real-time dashboards refreshing every 500ms."),

        ("train_mixed_5",
         "The Mediterranean diet has been associated with numerous health "
         "benefits including reduced cardiovascular risk and improved "
         "longevity markers in longitudinal cohort studies. I started eating "
         "more olive oil and fish this year following my doctor's advice "
         "after my cholesterol came back at 242. Mostly I just miss pizza. "
         "Epidemiological research consistently demonstrates protective "
         "effects against neurodegenerative conditions as well."),

        ("train_mixed_6",
         "Cryptocurrency markets demonstrate significant volatility driven "
         "by speculative trading and regulatory uncertainty across global "
         "jurisdictions. My brother put $5,000 into Ethereum in 2021 right "
         "before the crash and hasn't spoken about it since. The underlying "
         "blockchain technology presents legitimate applications in supply "
         "chain verification and decentralized finance infrastructure. "
         "Institutional adoption continues to accelerate despite persistent "
         "concerns regarding environmental impact."),

        ("train_mixed_7",
         "Recent advances in CRISPR gene editing technology have opened "
         "unprecedented possibilities for treating genetic disorders. The "
         "precision of Cas9 nuclease targeting has improved significantly "
         "with off-target effects reduced to below measurable thresholds "
         "in clinical trials. My cousin is in a sickle cell trial at UCSF "
         "and says the treatment itself was rough, three days of chemo "
         "before the edited cells go in, but she's been symptom-free for "
         "four months now which is insane."),

        ("train_mixed_8",
         "The psychological literature on decision fatigue suggests that "
         "cognitive resources are finite and deplete throughout the day. "
         "This explains why grocery shopping at 8pm results in a cart full "
         "of frozen pizza and impulse snacks. Studies utilizing ego depletion "
         "paradigms demonstrate measurable decrements in self-regulatory "
         "capacity following sustained cognitive demand. I meal prep on "
         "Sundays specifically to avoid this, tupperware containers lined "
         "up in the fridge like little soldiers of nutritional responsibility."),
    ]


def all_attack_samples():
    """Returns all attack probes combined."""
    samples = []
    samples.extend(gradient_probes())
    samples.extend(boundary_probes())
    samples.extend(perplexity_probes())
    samples.extend(burstiness_probes())
    samples.extend(hmm_probes())
    samples.extend(sensitivity_probes())
    samples.extend(membership_probes())
    samples.extend(training_corpus())
    return samples


def held_out_test_samples():
    """
    Fresh samples NEVER used in training.
    Used to validate the surrogate against GPTZero live.

    Balanced: 7 AI, 10 human, 6 mixed.
    Human samples are 150+ words with strong authenticity markers.
    """
    return [
        # ---- 7 AI ----
        ("test_ai_1",
         "The proliferation of electric vehicles represents a significant "
         "shift in the automotive industry. Major manufacturers have committed "
         "substantial resources to developing battery technology and charging "
         "infrastructure to support this transition. The environmental benefits "
         "of widespread EV adoption include significant reductions in urban "
         "air pollution and greenhouse gas emissions from the transportation "
         "sector."),
        ("test_ai_2",
         "Effective leadership requires a combination of strategic vision "
         "and interpersonal skills. Leaders who demonstrate emotional "
         "intelligence tend to foster more collaborative and productive "
         "work environments within their organizations. Research indicates "
         "that transformational leadership styles are positively correlated "
         "with employee satisfaction and retention rates."),
        ("test_ai_3",
         "The integration of blockchain technology into supply chain "
         "management offers enhanced transparency and traceability. "
         "These distributed ledger systems enable real-time tracking of "
         "goods and verification of authenticity throughout the supply chain. "
         "Implementation challenges include scalability constraints and the "
         "need for industry-wide standardization of protocols."),
        ("test_ai_4",
         "Urban planning in the twenty-first century must address the "
         "dual challenges of population growth and environmental sustainability. "
         "Smart city initiatives leverage data analytics and IoT sensors to "
         "optimize resource allocation and improve quality of life for "
         "urban residents across diverse metropolitan contexts."),
        ("test_ai_5",
         "Advances in natural language processing have enabled the development "
         "of sophisticated conversational agents capable of engaging in nuanced "
         "dialogue. These systems utilize transformer architectures and "
         "attention mechanisms to generate contextually appropriate responses "
         "that increasingly approximate human-like communication patterns."),
        ("test_ai_6",
         "The global transition to renewable energy sources necessitates "
         "significant investment in grid modernization. Energy storage "
         "solutions and smart grid technologies play a critical role in "
         "ensuring reliability and efficiency of power distribution systems. "
         "International cooperation is essential for addressing the technical "
         "and regulatory challenges associated with cross-border energy trade."),
        ("test_ai_7",
         "Digital transformation has become a strategic imperative for "
         "organizations across all sectors. Companies that successfully "
         "leverage emerging technologies gain competitive advantages "
         "through improved operational efficiency and enhanced customer "
         "experiences. The pace of technological change necessitates "
         "continuous adaptation and investment in workforce development."),

        # ---- 10 human (150+ words) ----
        ("test_human_1",
         "woke up at 4am because the cat knocked a glass off the nightstand. "
         "couldn't fall back asleep so I ended up reorganizing my entire spice "
         "rack. found cumin from 2019. it's fine right? cumin doesn't go bad? "
         "I used it anyway. made eggs with the questionable cumin and honestly "
         "they tasted fine. the cat meanwhile was sleeping peacefully on the "
         "couch like she didn't just destroy my night. her name is Mango and "
         "she has absolutely zero remorse about anything ever. I've had her "
         "since she was 8 weeks old when I picked her up from a rescue in "
         "San Jose. she cost $75 and has since cost me approximately $4,000 "
         "in vet bills and destroyed furniture."),

        ("test_human_2",
         "Tried to explain to my dad what I do for a living for the millionth "
         "time. He still tells people I work with computers. Close enough I "
         "guess. At least he stopped asking when I'm going to law school. "
         "Mom was more curious and asked if I could fix her printer which no "
         "Mom I'm a product manager that's not the same thing at all. She "
         "then asked if I could look at why her iPad is slow and honestly I "
         "just did it because it was easier than explaining the difference "
         "between a PM and IT support for the third time this year. Turns "
         "out she had 847 open Safari tabs. Eight hundred and forty seven. "
         "Each one a recipe she was saving for later. Bookmarks exist Mom."),

        ("test_human_3",
         "The trick with cast iron is everyone overcomplicates it. Cook with it. "
         "Wipe it out. Thin layer of oil. Done. My pan's been going 8 years and "
         "it's basically nonstick at this point. Stop babying your cookware. "
         "I got mine at a garage sale in Alameda for $15 and it had like half "
         "an inch of rust on it. Spent a Saturday afternoon with steel wool and "
         "flaxseed oil and now it's the best pan I own. My Lodge 12 inch. "
         "People in the cast iron subreddit have opinions about seasoning oils "
         "that border on religious zealotry. Flaxseed vs Crisco is apparently "
         "a holy war. I use whatever oil is closest to the stove. It's fine."),

        ("test_human_4",
         "got rear-ended on 101 south near woodside yesterday. everyone's fine "
         "but the bumper is toast and I've been on hold with insurance for 45 "
         "minutes. the hold music is jazz which somehow makes it worse. the "
         "other driver was super apologetic, said he was checking his GPS and "
         "didn't see me slow down. he drives a Civic, I drive a Forester, my "
         "car won. his front end is crumpled pretty bad. we exchanged info and "
         "he has State Farm, I have GEICO, and now I'm in insurance hold music "
         "purgatory. the cop who showed up was really nice actually, gave us "
         "both a case number within like 10 minutes. body shop estimate is "
         "$2,800 for the bumper and rear quarter panel. the Forester has 87k "
         "miles and I was planning to drive it to 150k so I guess we're fixing it."),

        ("test_human_5",
         "Running a marathon is 90% not quitting and 10% having trained enough "
         "that your legs don't literally stop working at mile 20. I trained "
         "for five months, peaked at 45 miles per week in September, and still "
         "hit the wall so hard at mile 22 that I walked a full mile along the "
         "Embarcadero while tourists took photos of me looking like I was dying. "
         "Finished though. 4:37:12 official chip time. My buddy Chris ran a "
         "3:15 and was waiting at the finish looking fresh as hell while I "
         "staggered across looking like a zombie extra. Post-race I ate an "
         "entire large pizza from Goat Hill and fell asleep at 7pm. Could not "
         "walk stairs normally for four days. Already signed up for next year."),

        ("test_human_6",
         "My neighbor has been building a deck for three months now. Every "
         "weekend I hear the circular saw start at 7am sharp on Saturday. "
         "The deck looks great though honestly. He's doing the railing himself "
         "which is brave. Offered to help twice and he said no both times, I "
         "think it's a pride thing. His wife told my wife that the original "
         "budget was $8,000 and they're currently at $14,000 and counting "
         "because he keeps upgrading materials. Started with pressure-treated "
         "pine, decided mid-project to switch to composite decking, which is "
         "like three times the cost. The composite does look incredible though. "
         "He's using Trex Transcend in a color called Spiced Rum which I know "
         "because I've heard him say it to at least four different people who "
         "stopped to look. At this rate it'll be done by April maybe."),

        ("test_human_7",
         "just spent 2 hours debugging a CSS issue that turned out to be a "
         "missing semicolon. two hours. I have a master's degree from Stanford. "
         "the semicolon was on line 4 of a 200 line stylesheet. I am a fraud. "
         "my coworker Marcus found it in about 30 seconds when I finally asked "
         "for help. he didn't even say anything just pointed at the screen and "
         "walked away. I deserve that. the worst part is I was using a linter "
         "the whole time and it WAS flagging it, I just assumed the error was "
         "somewhere else because the error message said line 47 which is the "
         "CSS equivalent of your check engine light telling you the problem is "
         "in the transmission when actually you're out of washer fluid."),

        ("test_human_8",
         "took my 11 year old camping at Big Basin last weekend, first time "
         "since the fires burned through in 2020. a lot of the old growth "
         "redwoods survived which was emotional honestly, some of those trees "
         "are 1500 years old and they're just standing there with char marks "
         "halfway up the trunk still alive. we hiked the Skyline to the Sea "
         "trail about 6 miles to Berry Creek Falls which was running hard from "
         "all the rain last month. my kid complained for the first 2 miles, was "
         "fine for the next 2, and then got genuinely excited at the falls "
         "which made the whole trip worth it. saw a banana slug the size of "
         "my hand. she named it Gerald. we ate trail mix and PB&J sandwiches "
         "on a log and I thought this is it, this is the good part of parenting."),

        ("test_human_9",
         "the sourdough discard pancakes recipe I've been messing with is "
         "finally dialed in and I'm putting it here before I forget: 1 cup "
         "discard, 1 egg, 2 tbsp melted butter, 3/4 cup milk (I use oat), "
         "1 cup AP flour, 1 tbsp sugar, 1 tsp baking powder, 1/2 tsp baking "
         "soda, pinch of salt. let the batter rest 10 minutes, cook on medium "
         "low not medium, flip when you see bubbles across the whole surface "
         "not just the edges. the rest time matters a lot, without it you get "
         "flat sad pancakes. with it you get fluffy tangy perfection. I make "
         "these every Saturday now. my husband says they're the best pancakes "
         "he's ever had which might be flattery but I choose to believe him. "
         "the key is the discard needs to be active, if it's been in the "
         "fridge for 3+ weeks unfed the flavor goes weird. fed mine Thursday "
         "night, used the discard Saturday morning. perfect every time."),

        ("test_human_10",
         "parking situation at my office has gotten completely out of hand. "
         "they converted the third floor of the garage into bike storage "
         "which great for the environment terrible for the 200 people who "
         "now don't have spots. I get to work at 7:45 and the lot is already "
         "full by 7:30. ended up parking on the street three blocks away "
         "next to a hydrant which I know is illegal but at that point I was "
         "already late for a 8am standup that could have been a Slack message. "
         "got a $110 ticket. filed an expense report for it which my manager "
         "rejected with just the word 'no' and a period. HR sent an email "
         "yesterday about a parking lottery system they're implementing and "
         "honestly it feels like we're in a dystopian novel about office "
         "culture. there are 400 employees and 150 spots. I'm considering "
         "the bus but it adds 40 minutes each way."),

        # ---- 6 mixed / ambiguous ----
        ("test_mixed_1",
         "Machine learning models require extensive training data to achieve "
         "acceptable performance in production environments. In my experience "
         "running the ML team at my last company, we burned through $40k in "
         "compute before the model stopped producing garbage. The theoretical "
         "foundations are elegant but production ML is mostly data cleaning "
         "and prayer. Current literature suggests that data quality has a "
         "more significant impact on model performance than architectural "
         "complexity in the majority of applied contexts."),

        ("test_mixed_2",
         "The research literature consistently demonstrates that exercise "
         "improves cognitive function and mental health outcomes across all "
         "age demographics. Been running 3 miles every morning since January "
         "and honestly I just feel tired. Maybe it takes longer. The meta-"
         "analyses say 8-12 weeks to observe measurable improvements in "
         "executive function. I'm at 6 weeks. Longitudinal cohort studies "
         "suggest that the benefits compound over extended periods of "
         "consistent physical activity."),

        ("test_mixed_3",
         "Quantum computing leverages the principles of superposition and "
         "entanglement to perform calculations that would be infeasible for "
         "classical computers. My roommate works at IBM on their quantum team "
         "and says most of the hype is 5-10 years ahead of reality. Current "
         "quantum systems operate at around 1000 qubits but error rates "
         "remain a fundamental challenge for practical applications."),

        ("test_mixed_4",
         "The housing market in major metropolitan areas has experienced "
         "unprecedented price appreciation driven by low interest rates "
         "and constrained supply. We got outbid on four houses in Sunnyvale "
         "before giving up and renting. Someone paid 400k over asking, cash. "
         "Economic models suggest that the supply-demand imbalance in coastal "
         "markets will persist absent significant policy intervention or "
         "construction of high-density housing at scale."),

        ("test_mixed_5",
         "Cybersecurity threats have evolved significantly in recent years "
         "with nation-state actors increasingly targeting critical infrastructure "
         "through sophisticated attack vectors. Had a buddy whose company got "
         "hit with ransomware last month. They paid the 2 million. Took three "
         "weeks to fully recover even after payment. The threat landscape "
         "continues to evolve as adversaries adopt machine learning techniques "
         "for automated vulnerability discovery."),

        ("test_mixed_6",
         "The psychological impact of remote work on employee wellbeing has "
         "been extensively studied since the pandemic-driven shift to distributed "
         "teams. Research indicates both benefits and drawbacks including improved "
         "flexibility but increased feelings of isolation. I've been fully remote "
         "since March 2020 and some days I don't speak to another human until "
         "my wife gets home at 6. The ergonomic setup alone cost me $1,200 but "
         "my back doesn't hurt anymore so there's that."),
    ]
