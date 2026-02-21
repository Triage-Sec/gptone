"""
Attack sample generators.

Each function returns a list of (sample_id, text) tuples.
These are the controlled inputs we send to GPTZero to probe
the classifier's behavior.
"""


def gradient_probes():
    """
    ATTACK 1: Gradient leakage via single-sentence perturbation.
    
    One base AI sentence + 14 perturbations that each change exactly
    one linguistic property. The probability delta per perturbation
    approximates the classifier's gradient w.r.t. that feature.
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
    ATTACK 2: Decision boundary mapping via word-level interpolation.
    
    Pure AI paragraph -> pure human paragraph at 10% increments.
    Finds the exact ratio where classification flips.
    """
    ai_text = (
        "The rapid advancement of renewable energy technologies has fundamentally "
        "transformed the global energy landscape. Solar photovoltaic systems have "
        "achieved unprecedented levels of efficiency with modern panels converting "
        "over twenty-five percent of incident solar radiation into usable electricity. "
        "Wind energy installations have similarly benefited from technological "
        "innovations particularly in the development of larger turbine designs "
        "capable of generating greater amounts of power. The integration of energy "
        "storage solutions such as lithium-ion batteries and pumped hydroelectric "
        "systems addresses the intermittency challenges associated with renewable "
        "sources. Government policies and international agreements continue to play "
        "a crucial role in accelerating the transition toward sustainable energy."
    )
    
    human_text = (
        "Renewables have gotten weird fast. Five years ago nobody I knew had solar "
        "and now three of my neighbors do and one guy has a battery the size of a "
        "fridge in his garage. Panel efficiency numbers are honestly kind of nuts. "
        "My uncle's setup in Tucson pulls 8kW on a clear day from like 20 panels. "
        "Wasn't possible a decade ago. Wind turbines keep getting bigger too and "
        "it's slightly terrifying. Drove past that farm outside Palm Springs last "
        "month blades looked taller than the mountains behind them. Battery tech "
        "is where it gets interesting though. The duck curve problem basically "
        "disappears once you can store 4 hours of grid load. Policy matters more "
        "than people think. Germany went all in on feed-in tariffs and got results."
    )
    
    ai_words = ai_text.split()
    human_words = human_text.split()
    min_len = min(len(ai_words), len(human_words))
    ai_words = ai_words[:min_len]
    human_words = human_words[:min_len]
    
    samples = []
    for pct in range(0, 101, 10):
        cutoff = int(min_len * pct / 100)
        mixed = human_words[:cutoff] + ai_words[cutoff:]
        samples.append((f"bound_{pct:03d}pct_human", " ".join(mixed)))
    
    return samples


def perplexity_probes():
    """
    ATTACK 3: Perplexity fingerprinting.
    
    Texts with controlled perplexity levels. GPTZero uses GPT-2
    perplexity as a core signal. This maps the threshold.
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
    Maps the burstiness threshold (core detection signal #2).
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
    
    Controlled AI/human sentence patterns. The per-sentence
    highlighting reveals HMM transition probabilities.
    """
    AI = ("The implementation of advanced computational methodologies has "
          "demonstrated significant potential for addressing complex "
          "analytical challenges across multiple domains.")
    
    HU = ("I talked to my friend Dave about this last week and he thinks "
          "the whole thing is overblown but then again Dave thought Bitcoin "
          "was a scam in 2011 so take that for what it's worth.")
    
    return [
        ("hmm_AAAA", f"{AI} {AI} {AI} {AI}"),
        ("hmm_HHHH", f"{HU} {HU} {HU} {HU}"),
        ("hmm_AHAH", f"{AI} {HU} {AI} {HU}"),
        ("hmm_HAHA", f"{HU} {AI} {HU} {AI}"),
        ("hmm_AAHH", f"{AI} {AI} {HU} {HU}"),
        ("hmm_HHAA", f"{HU} {HU} {AI} {AI}"),
        ("hmm_AHHA", f"{AI} {HU} {HU} {AI}"),
        ("hmm_HAAH", f"{HU} {AI} {AI} {HU}"),
        ("hmm_long_block", f"{AI} {AI} {AI} {HU} {HU} {HU} {AI} {AI} {AI} {HU} {HU}"),
        ("hmm_long_alt", f"{AI} {HU} {AI} {HU} {AI} {HU} {AI} {HU} {AI} {HU} {AI}"),
    ]


def sensitivity_probes():
    """
    ATTACK 6: Per-word sensitivity (saliency map).
    
    Drop one word at a time from a sentence. The probability
    delta per removal reveals which words matter most.
    """
    base = (
        "Furthermore the implementation of advanced machine learning "
        "algorithms has demonstrated significant potential for enhancing "
        "predictive accuracy across various computational domains."
    )
    
    words = base.split()
    samples = [("sens_base", base)]
    for i in range(len(words)):
        dropped = words[:i] + words[i+1:]
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
    Extra labeled texts for beefing up surrogate training data.
    Mix of clearly-AI, clearly-human, and mixed content.
    """
    return [
        # Clearly AI
        ("train_ai_1",
         "In today's rapidly evolving digital landscape, organizations must "
         "adapt their strategies to remain competitive. The integration of "
         "innovative technologies presents both unprecedented opportunities "
         "and significant challenges that require careful consideration and "
         "strategic planning to navigate effectively."),
        ("train_ai_2",
         "It is important to note that sustainable development requires a "
         "multifaceted approach. Furthermore, stakeholder engagement plays "
         "a crucial role in ensuring the success of environmental initiatives. "
         "Moreover, the economic implications of these policies must be "
         "carefully evaluated to achieve optimal outcomes."),
        ("train_ai_3",
         "The research findings suggest that there is a strong correlation "
         "between employee satisfaction and organizational productivity. "
         "Additionally, companies that invest in professional development "
         "programs tend to experience lower turnover rates and higher levels "
         "of employee engagement across all departments."),
        ("train_ai_4",
         "Climate change represents one of the most pressing challenges "
         "facing humanity in the twenty-first century. The scientific "
         "consensus indicates that anthropogenic greenhouse gas emissions "
         "are the primary driver of global temperature increases observed "
         "over the past several decades."),
        ("train_ai_5",
         "The evolution of communication technologies has fundamentally "
         "altered the way individuals interact and share information. "
         "Social media platforms have created new paradigms for community "
         "building and civic engagement while simultaneously raising "
         "concerns about privacy and misinformation."),
        
        # Clearly human
        ("train_human_1",
         "ok so i finally got around to fixing the kitchen sink and it was "
         "WAY worse than I thought. the p-trap was basically held together "
         "with duct tape (thanks previous owner) and when I pulled it off "
         "the smell was... yeah. three trips to home depot later it works."),
        ("train_human_2",
         "Had the weirdest conversation with my uber driver last night. Guy "
         "used to be a marine biologist studying octopus intelligence in "
         "Monterey Bay. Left because the funding dried up. Now he drives "
         "nights and does freelance data viz during the day. SF is wild."),
        ("train_human_3",
         "The thing about sourdough that nobody tells you is that the first "
         "ten loaves are going to be terrible. Like genuinely bad. Dense, "
         "gummy, sometimes literally raw in the middle. But then something "
         "clicks around loaf eleven and suddenly you get it. The timing, the "
         "feel of the dough, the way it should sound when you tap the bottom."),
        ("train_human_4",
         "My 6yo asked me where the internet lives and I tried to explain "
         "data centers and she said 'so it's in a building? that's boring. "
         "I thought it was in space.' Honestly her version is way cooler."),
        ("train_human_5",
         "Just realized I've been paying for a gym membership since March "
         "that I've used exactly twice. Both times I went, walked on the "
         "treadmill for 15 minutes, got confused by a machine, and left. "
         "This is my villain origin story."),
        
        # Mixed/ambiguous
        ("train_mixed_1",
         "The quarterly earnings report exceeded analyst expectations with "
         "revenue growth of 12% year over year. I mean, nobody saw this "
         "coming after the disaster that was Q2. The CFO attributed the "
         "improvement to operational efficiency gains and expanded market "
         "penetration, but honestly I think they just got lucky with timing."),
        ("train_mixed_2",
         "Neural networks process information through interconnected layers "
         "of artificial neurons. Each layer transforms the input data in "
         "ways that are, frankly, still pretty mysterious even to the people "
         "building them. My professor spent 20 minutes trying to explain "
         "attention mechanisms and eventually just said 'it works, trust me.'"),
        ("train_mixed_3",
         "According to the latest WHO data, global vaccination rates have "
         "improved significantly since 2020. Immunization programs now reach "
         "approximately 84% of the world's children. My sister works in "
         "public health in Malawi and she says the real number on the ground "
         "is probably lower because the reporting is spotty outside cities."),
    ]


def all_attack_samples():
    """Returns all attack samples combined."""
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
    20 fresh samples NEVER used in training.
    Used to validate the surrogate against GPTZero live.
    """
    return [
        # 7 AI
        ("test_ai_1",
         "The proliferation of electric vehicles represents a significant "
         "shift in the automotive industry. Major manufacturers have committed "
         "substantial resources to developing battery technology and charging "
         "infrastructure to support this transition."),
        ("test_ai_2",
         "Effective leadership requires a combination of strategic vision "
         "and interpersonal skills. Leaders who demonstrate emotional "
         "intelligence tend to foster more collaborative and productive "
         "work environments within their organizations."),
        ("test_ai_3",
         "The integration of blockchain technology into supply chain "
         "management offers enhanced transparency and traceability. "
         "These distributed ledger systems enable real-time tracking of "
         "goods and verification of authenticity throughout the supply chain."),
        ("test_ai_4",
         "Urban planning in the twenty-first century must address the "
         "dual challenges of population growth and environmental sustainability. "
         "Smart city initiatives leverage data analytics and IoT sensors to "
         "optimize resource allocation and improve quality of life."),
        ("test_ai_5",
         "Advances in natural language processing have enabled the development "
         "of sophisticated conversational agents capable of engaging in nuanced "
         "dialogue. These systems utilize transformer architectures and "
         "attention mechanisms to generate contextually appropriate responses."),
        ("test_ai_6",
         "The global transition to renewable energy sources necessitates "
         "significant investment in grid modernization. Energy storage "
         "solutions and smart grid technologies play a critical role in "
         "ensuring reliability and efficiency of power distribution systems."),
        ("test_ai_7",
         "Digital transformation has become a strategic imperative for "
         "organizations across all sectors. Companies that successfully "
         "leverage emerging technologies gain competitive advantages "
         "through improved operational efficiency and enhanced customer "
         "experiences."),
        
        # 7 human
        ("test_human_1",
         "woke up at 4am because the cat knocked a glass off the nightstand. "
         "couldn't fall back asleep so I ended up reorganizing my entire spice "
         "rack. found cumin from 2019. it's fine right? cumin doesn't go bad? "
         "I used it anyway."),
        ("test_human_2",
         "Tried to explain to my dad what I do for a living for the millionth "
         "time. He still tells people I 'work with computers.' Close enough I "
         "guess. At least he stopped asking when I'm going to law school."),
        ("test_human_3",
         "The trick with cast iron is everyone overcomplicates it. Cook with it. "
         "Wipe it out. Thin layer of oil. Done. My pan's been going 8 years and "
         "it's basically nonstick at this point. Stop babying your cookware."),
        ("test_human_4",
         "got rear-ended on 101 south near woodside yesterday. everyone's fine "
         "but the bumper is toast and I've been on hold with insurance for 45 "
         "minutes. the hold music is jazz which somehow makes it worse."),
        ("test_human_5",
         "Running a marathon is 90% not quitting and 10% having trained enough "
         "that your legs don't literally stop working at mile 20. I trained "
         "for five months and still hit the wall so hard at 22 that I walked "
         "a full mile. Finished though. 4:37."),
        ("test_human_6",
         "My neighbor has been building a deck for three months now. Every "
         "weekend I hear the circular saw start at 7am sharp. The deck looks "
         "great though honestly. He's doing the railing himself which is brave. "
         "Offered to help twice and he said no both times."),
        ("test_human_7",
         "just spent 2 hours debugging a CSS issue that turned out to be a "
         "missing semicolon. two hours. I have a master's degree. the semicolon "
         "was on line 4. I am a fraud."),

        # 6 mixed / ambiguous
        ("test_mixed_1",
         "Machine learning models require extensive training data to achieve "
         "acceptable performance. In my experience running the ML team at "
         "my last company, we burned through $40k in compute before the model "
         "stopped producing garbage. The theoretical foundations are elegant "
         "but production ML is mostly data cleaning and prayer."),
        ("test_mixed_2",
         "The research literature consistently demonstrates that exercise "
         "improves cognitive function and mental health outcomes. Been running "
         "3 miles every morning since January and honestly I just feel tired. "
         "Maybe it takes longer. The meta-analyses say 8-12 weeks. I'm at 6."),
        ("test_mixed_3",
         "Quantum computing leverages the principles of superposition and "
         "entanglement to perform calculations that would be infeasible for "
         "classical computers. My roommate works at IBM on their quantum team "
         "and says most of the hype is 5-10 years ahead of reality."),
        ("test_mixed_4",
         "The housing market in major metropolitan areas has experienced "
         "unprecedented price appreciation driven by low interest rates "
         "and constrained supply. We got outbid on four houses in Sunnyvale "
         "before giving up and renting. Someone paid 400k over asking. Cash."),
        ("test_mixed_5",
         "Cybersecurity threats have evolved significantly in recent years "
         "with nation-state actors increasingly targeting critical infrastructure. "
         "Had a buddy whose company got hit with ransomware last month. They "
         "paid the 2 million. Took three weeks to fully recover anyway."),
        ("test_mixed_6",
         "The Mediterranean diet has been associated with numerous health "
         "benefits including reduced cardiovascular risk and improved "
         "longevity markers. Started eating more olive oil and fish this "
         "year. Mostly I just miss pizza. The studies never mention how "
         "boring grilled vegetables get after week four."),
    ]
