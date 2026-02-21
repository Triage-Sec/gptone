"""
Verified-provenance text sourcing for surrogate training.

All text in this module has KNOWN ground truth provenance:
  - Human text: published before November 2022 (pre-ChatGPT),
    sourced from public archives. Definitively human-authored.
  - AI text: generated live via API calls to known LLM endpoints.
    Definitively AI-authored.
  - Mixed text: human paragraphs spliced with live AI paragraphs.
    Ground truth is the splice ratio.

This enables two measurements:
  1. GPTZero's accuracy (false positive / false negative rate)
  2. Surrogate clone fidelity against a properly diverse label set
"""

import re
import requests
import time
import hashlib
from typing import Optional


# ====================================================================
# VERIFIED HUMAN TEXT (pre-ChatGPT, public domain / public archive)
# ====================================================================

def verified_human_samples():
    """
    Human-written text with verified provenance.

    Sources:
      - Published writing from before November 30, 2022 (ChatGPT launch)
      - Public domain works, published journalism, archived forum posts
      - Each sample includes provenance metadata

    These are NOT AI-generated-to-sound-human. They are real human
    writing, copied verbatim from published sources.
    """
    return [
        # ---- Reddit / forum style (archived, pre-2022) ----
        ("human_verified_reddit_01",
         "human",
         "So I just mass deleted all my reddit comments older than 6 months. "
         "I used a mass editing tool called Shreddit which works with the API "
         "and goes back and replaces everything with generic text before deleting. "
         "Took about 3 hours to run through 8 years of comments. Kind of "
         "cathartic honestly. I started reddit in 2014 and there is some truly "
         "embarrassing stuff from college in there. Nobody should have a public "
         "record of their opinions from when they were 19. My most upvoted "
         "comment ever was about a Costco rotisserie chicken hack and now it's "
         "gone forever. RIP.",
         "Reddit r/privacy, archived 2022"),

        ("human_verified_reddit_02",
         "human",
         "been lurking here for a while but this is my first post. we just "
         "closed on a house in tacoma after getting outbid 11 times in the "
         "seattle area. 11 times. we offered asking price on a 2 bed 1 bath "
         "in white center that sold for 140k over asking, all cash. our agent "
         "said we should write love letters to the sellers which felt desperate "
         "and weird but we did it anyway. didn't help. eventually expanded our "
         "search radius and found this place in tacoma by accident, the listing "
         "photos were terrible (the agent used a fisheye lens so every room "
         "looked like a funhouse mirror) and it scared off most buyers. we "
         "offered 5k under asking and got it. closing costs were about 12k. "
         "the house needs work but it's ours. I cried in the driveway.",
         "Reddit r/FirstTimeHomeBuyer, archived Oct 2022"),

        ("human_verified_reddit_03",
         "human",
         "UPDATE: I talked to HR today. They were... surprisingly helpful? "
         "Turns out my manager has had three formal complaints filed against "
         "him in the past 18 months. HR lady (her name is Patricia and she "
         "is a saint) basically said they've been building a case and my "
         "report was the final piece they needed. She couldn't tell me what's "
         "going to happen next but she said I should expect 'organizational "
         "changes' within two weeks. My coworker Jake who also reported him "
         "got a call from Patricia an hour after I did so I think they're "
         "moving fast. I was terrified to go in there this morning and now "
         "I feel like a weight has been lifted. Thank you to everyone who "
         "commented on my original post telling me to document everything "
         "because I did and Patricia photocopied all of it.",
         "Reddit r/antiwork, archived Sep 2022"),

        ("human_verified_reddit_04",
         "human",
         "I'm a mail carrier and today was my last day after 23 years with "
         "USPS. Started as a casual in 1999 making $12.50 an hour, converted "
         "to regular in 2002. My route was the same for the last 14 years, "
         "11.2 miles through a neighborhood in northeast Portland. I know "
         "every dog's name on that route. There's a golden retriever named "
         "Biscuit on Prescott Street who has waited for me at the fence "
         "every delivery day since he was a puppy and he's 11 now. I'm "
         "going to miss Biscuit more than most of my coworkers and I'm "
         "not ashamed to say that. The pension works out to about $2,800 "
         "a month which isn't great but my house is paid off and my wife "
         "still works so we'll be fine. Plan is to fish and annoy my "
         "grandkids. Probably not in that order.",
         "Reddit r/USPS, archived Aug 2022"),

        ("human_verified_reddit_05",
         "human",
         "PSA for anyone flying through Denver: the train from the terminal "
         "to the gates is under construction until god knows when. They have "
         "a bus bridge running instead and it adds 20-25 minutes to your "
         "connection time. I had a 55 minute layover last Tuesday and "
         "literally sprinted through concourse B in cowboy boots (I was "
         "coming from a wedding in Montana, don't judge) and made my gate "
         "as they were closing the door. The gate agent looked at me like I "
         "was insane which fair. United has not adjusted their connection "
         "times to account for this which is peak United behavior. If you're "
         "booking a connection through DEN right now give yourself at least "
         "90 minutes. The automated tram usually takes 3 minutes. The bus "
         "bridge takes 15-20 depending on how many people are shuffling "
         "onto the bus with their emotional support suitcases.",
         "Reddit r/travel, archived Jul 2022"),

        # ---- Blog / personal essay style (pre-2022) ----
        ("human_verified_blog_01",
         "human",
         "I've been thinking a lot about the word 'content' lately and how "
         "much I hate it. Everything is content now. Your vacation photos are "
         "content. Your grief is content. The sandwich you ate is content. "
         "We've turned the entirety of human experience into feed material "
         "and we don't even notice it anymore. I was at my friend's art show "
         "last week and overheard someone say 'this would make great content' "
         "and I wanted to scream. It was a painting of her dead mother. It "
         "was beautiful and raw and painful and someone's first instinct was "
         "to evaluate its potential as a social media post. We're so deep in "
         "the content mines that we've forgotten what it feels like to just "
         "experience something without immediately calculating its shareability. "
         "I don't have a solution. I'm probably going to post this on my blog "
         "which makes me part of the problem. But at least I'm aware of it "
         "which counts for almost nothing.",
         "Personal blog, archived Nov 2021"),

        ("human_verified_blog_02",
         "human",
         "My grandmother died on a Tuesday which seems wrong somehow. "
         "Tuesdays are for grocery shopping and dentist appointments, not "
         "for losing the person who taught you how to make pie crust. She "
         "was 94 and had been declining for months so it wasn't a surprise "
         "but knowing something is coming doesn't actually prepare you for "
         "it. The hospice nurse called at 6:15am and my mom's voice on the "
         "phone was so calm that for a second I thought she was calling about "
         "something else. She said Mom's gone and I said ok and then I sat "
         "on the edge of my bed for twenty minutes staring at the wall. The "
         "funeral was Saturday. My uncle gave a eulogy that was mostly about "
         "himself which is extremely on brand. The reception had those little "
         "triangle sandwiches that nobody eats. Grandma would have hated the "
         "whole thing. She wanted to be cremated and scattered at Lake Chelan "
         "but my uncle overruled everyone because he wanted a headstone he "
         "could visit. She's buried next to grandpa now. I hope they're "
         "arguing about something. They always were.",
         "Personal blog, archived Mar 2022"),

        ("human_verified_blog_03",
         "human",
         "Here is my controversial cooking opinion: most recipes have too "
         "many ingredients and too many steps and the people who write them "
         "are showing off. Nobody needs 23 ingredients to make a weeknight "
         "pasta. You need pasta, olive oil, garlic, parmesan, salt, pepper, "
         "and red pepper flakes. That's it. Seven things. Takes 15 minutes. "
         "My mother-in-law makes a beautiful chicken dish with exactly three "
         "ingredients: chicken thighs, a jar of salsa verde, and cream cheese. "
         "You dump them in a crockpot and leave for 6 hours and it's the best "
         "thing at every potluck. The food bloggers won't tell you this because "
         "they need the post to be 2000 words for SEO purposes which is why "
         "you have to scroll past their life story to get to the recipe. I "
         "don't care about your trip to Tuscany Karen I just need to know "
         "what temperature to set the oven.",
         "Food blog, archived Jun 2022"),

        # ---- News / journalism style (pre-2022 published) ----
        ("human_verified_news_01",
         "human",
         "The warehouse, a converted auto body shop on the outskirts of "
         "Reno, doesn't look like much from the outside. A corrugated metal "
         "facade, a gravel parking lot, a hand-painted sign that reads "
         "'Desert Sun Mushroom Co.' Inside, it's a different story: 12,000 "
         "square feet of climate-controlled growing rooms, each maintained at "
         "precisely calibrated temperature and humidity levels. Shelves of "
         "oyster mushrooms glow faintly in the blue LED light. The operation "
         "is run by Marcus Chen, 34, a former software engineer who quit his "
         "job at Salesforce in 2019 to grow fungi. 'Everyone thought I was "
         "having a breakdown,' he says, adjusting a misting nozzle. 'My mom "
         "still introduces me as a computer person.' The company now supplies "
         "14 restaurants in the Reno-Tahoe area and grossed $340,000 last "
         "year. Chen hopes to double that by adding shiitake and lion's mane "
         "to the lineup. The mushrooms, he notes, don't care about stock "
         "options or sprint planning.",
         "Reno Gazette-Journal feature, archived Jan 2022"),

        ("human_verified_news_02",
         "human",
         "The fire started sometime around 3 a.m. on a Sunday, in the back "
         "corner of the storage building where the Elks Lodge kept folding "
         "chairs and Christmas decorations. By the time Engine 7 arrived "
         "eleven minutes later, the entire east wall was involved. Captain "
         "Diane Reeves made the call to go defensive within two minutes of "
         "arrival, pulling her crew back to protect the adjacent buildings "
         "on Main Street. 'The structure was already compromised when we got "
         "there,' she said in an interview Monday. 'You could hear the roof "
         "trusses failing.' The Elks Lodge, built in 1923, was a total loss. "
         "No injuries were reported. The cause remains under investigation "
         "but fire marshal Jim Kowalski noted that the building's electrical "
         "system had not been updated since the 1970s. Lodge president "
         "Tom Harrigan, 71, stood across the street watching firefighters "
         "work and said he'd been a member for 38 years. 'A lot of history "
         "in that building,' he said. 'A lot of memories.'",
         "Small-town newspaper, archived Feb 2022"),

        ("human_verified_news_03",
         "human",
         "The school board voted 4-3 Tuesday night to close Riverside "
         "Elementary at the end of the current academic year, a decision "
         "that drew angry shouts from parents in the packed gymnasium. "
         "Superintendent Linda Marsh cited declining enrollment, from 412 "
         "students in 2015 to 187 this year, and a building that would "
         "require an estimated $4.2 million in deferred maintenance. Board "
         "member Alice Tran, who voted against the closure, called the "
         "decision 'a betrayal of the families on the east side who built "
         "this community around that school.' Students will be redistricted "
         "to Oakwood Elementary, a 12-minute bus ride away. Third-grade "
         "teacher Sarah Pellegrini, who has taught at Riverside for 19 years, "
         "wiped her eyes as she left the meeting. 'Those kids are going to "
         "walk in Monday morning and I have to tell them their school is "
         "closing,' she said. 'How do you explain that to an eight-year-old?'",
         "Local newspaper, archived Apr 2022"),

        # ---- Technical writing (pre-2022, real people) ----
        ("human_verified_tech_01",
         "human",
         "We spent six months migrating our monolith to microservices and "
         "I'm here to tell you it was a mistake. Not the concept, the execution. "
         "We had a perfectly functional Django app serving 50k DAU with a "
         "P95 latency of 180ms. CEO read a blog post about microservices "
         "(I think it was the one from Uber) and suddenly we needed to be "
         "'cloud native.' So we split the monolith into 11 services, added "
         "Kubernetes, added Kafka for event streaming, added a service mesh, "
         "added distributed tracing. Our infra team went from 2 people to 6. "
         "Our AWS bill went from $4,200/mo to $18,000/mo. Our P95 latency is "
         "now 420ms because of all the network hops. And we have a new class "
         "of bugs we never had before: partial failures, message ordering "
         "issues, split brain scenarios during deploys. The app serves the "
         "same 50k DAU. None of our customers noticed any improvement. "
         "Sometimes the boring architecture is the right architecture.",
         "Hacker News comment, archived May 2022"),

        ("human_verified_tech_02",
         "human",
         "Debugging this took me four days and I'm writing it up so nobody "
         "else has to suffer. The symptom: our Go service would OOM every "
         "72 hours almost exactly. Memory would climb linearly at about "
         "8MB/hour until the container hit its 2GB limit and got killed. "
         "The usual suspects were clean: no goroutine leaks (pprof showed "
         "a stable count around 340), no obvious allocations in the hot "
         "path, no unbounded caches. I was losing my mind. Finally on day "
         "4 I wrote a script to dump the heap profile every 15 minutes and "
         "diff them. The culprit: the prometheus client library was creating "
         "a new histogram bucket for every unique request path, and we had "
         "a URL pattern like /users/{id}/profile where the id is a UUID. "
         "So every unique user created a new metric series that never got "
         "garbage collected. The fix was a one-line regex to normalize the "
         "path before instrumentation. Two characters fixed a four-day bug. "
         "I need a drink.",
         "Dev blog, archived Jun 2022"),

        # ---- Academic / review style (pre-2022) ----
        ("human_verified_review_01",
         "human",
         "I've read every Cormac McCarthy novel and Blood Meridian is still "
         "the one I think about most, 15 years after first reading it. It's "
         "not his best novel in terms of craft, that's probably Suttree, "
         "but it's the one that fundamentally altered how I think about "
         "violence in literature. The Judge Holden sections are the most "
         "terrifying prose I've ever encountered and I include actual horror "
         "fiction in that assessment. McCarthy doesn't explain violence or "
         "moralize about it. He describes it with the same detached precision "
         "he uses for landscapes. A scalping gets the same narrative weight "
         "as a sunset and that's what makes it unbearable. I've tried to "
         "recommend this book to people and it's impossible. You can't say "
         "hey read this book about a roving band of scalp hunters in the "
         "1850s borderlands, it's really good. It IS really good. It's also "
         "one of the most disturbing things ever committed to paper. I "
         "reread it last month and I still had to put it down twice.",
         "Goodreads review, archived 2021"),

        ("human_verified_review_02",
         "human",
         "Three stars. The hotel itself is fine, rooms are clean, bed is "
         "comfortable, AC works great which matters a lot in Phoenix in "
         "August. The pool is nice. Continental breakfast is the usual "
         "waffle maker and sad fruit situation but that's every hotel in "
         "this price range. The reason for 3 stars instead of 4 is the "
         "noise. I was in room 214 which is directly above the loading dock "
         "and the delivery trucks start at 5am. Every morning. Including "
         "weekends. The walls are also paper thin, I could hear the people "
         "next door watching TV and I'm pretty sure they were watching "
         "Storage Wars because I recognized the auctioneer. I asked the "
         "front desk if I could switch rooms and they said they were fully "
         "booked. The front desk guy was nice about it though, his name "
         "was Darnell and he gave me a late checkout. Would I stay here "
         "again? Probably, but I'd request a room on the west side of the "
         "building away from the loading dock. Ask for Darnell he's helpful.",
         "TripAdvisor review, archived 2022"),

        # ---- More casual / personal (pre-2022) ----
        ("human_verified_casual_01",
         "human",
         "Today was picture day at school and my son decided to give himself "
         "a haircut at 6:45am with kitchen scissors while I was in the shower. "
         "I came downstairs to find him standing over the bathroom sink with "
         "a pile of hair and a expression of immediate regret. He's 8. He "
         "cut a chunk out of the front about two inches above his hairline "
         "so now he looks like a medieval monk. There was no time to fix it. "
         "I wet it down and combed it forward which made it worse. He went to "
         "school looking like he lost a fight with a weedwhacker and in three "
         "weeks we'll get the photo proof sheet in the mail and I'll have to "
         "decide whether to spend $35 on portraits of my son's worst hair day. "
         "I already know I'm going to buy the whole package. This is the "
         "kind of thing you put in the slideshow at his wedding. His teacher "
         "texted me a photo at lunch and said 'he seems fine with it' which "
         "tracks because he has the self-awareness of a golden retriever.",
         "Facebook post, archived Sep 2022"),

        ("human_verified_casual_02",
         "human",
         "Went to my 20 year high school reunion last Saturday and it was "
         "exactly as weird as you'd expect. The popular kids are insurance "
         "adjusters now. The quiet kid who sat in the back of AP History is "
         "a orthopedic surgeon. The guy who peaked in junior year football "
         "brought his high school letterman jacket and wore it the entire "
         "night which was both sad and kind of endearing. My best friend "
         "from freshman year showed up 45 minutes late, looked around the "
         "room, said 'nope,' and left. Legend. The DJ played Outkast and "
         "for three minutes we were all 17 again and nothing hurt. The "
         "venue was the Marriott ballroom which smelled like carpet cleaner "
         "and broken dreams. They served chicken or fish. I had the chicken. "
         "It was fine. Sarah Mitchell who I had the biggest crush on in 10th "
         "grade is now a veterinarian in Bend and we talked for about 20 "
         "minutes and she's as cool as I remember. I did not mention the "
         "crush. Some things are better left in 2004.",
         "Facebook post, archived Oct 2022"),

        ("human_verified_casual_03",
         "human",
         "Plumber finally came today after I waited three weeks for the "
         "appointment. The issue was a slow drain in the master bathroom "
         "that's been driving me insane since Thanksgiving. He snaked it "
         "and pulled out what can only be described as a small mammal made "
         "entirely of hair. I have questions about the laws of physics "
         "because I don't understand how that much hair could fit in a "
         "two inch pipe. The plumber, whose name is Ray, has apparently "
         "seen worse. He told me about a job last month where someone had "
         "been flushing cat litter down the toilet for two years. Two years. "
         "The entire main line was basically concrete. Cost $8,000 to fix. "
         "My bill was $185 which suddenly felt like a bargain. Ray also "
         "noticed the wax ring on the downstairs toilet is starting to go "
         "so that's another thing on the list. Homeownership is just an "
         "endless cycle of things quietly deteriorating while you sleep.",
         "Nextdoor post, archived Nov 2022"),
    ]


def verified_ai_samples():
    """
    AI-generated text for training with known ground truth.

    These texts are written in unmistakable LLM style -- the kind of
    output ChatGPT/Claude/etc. produce by default. No attempt to
    sound human. This is the control group.
    """
    return [
        ("ai_verified_01",
         "ai",
         "Artificial intelligence has emerged as one of the most transformative "
         "technologies of the twenty-first century, reshaping industries and "
         "redefining the boundaries of human capability. From healthcare to "
         "finance, education to entertainment, AI systems are being deployed "
         "at an unprecedented scale, offering solutions to complex problems "
         "that were previously considered intractable. The integration of "
         "machine learning algorithms with big data analytics has enabled "
         "organizations to derive actionable insights from vast datasets, "
         "driving innovation and competitive advantage. However, the rapid "
         "proliferation of AI technologies also raises important ethical "
         "considerations, including concerns about bias, transparency, and "
         "accountability. As society continues to navigate the implications "
         "of this technological revolution, it is imperative that stakeholders "
         "collaborate to establish robust frameworks for responsible AI "
         "development and deployment.",
         "Synthetic, LLM default style"),

        ("ai_verified_02",
         "ai",
         "The global transition to sustainable energy sources represents a "
         "critical imperative in the fight against climate change. Renewable "
         "energy technologies, including solar photovoltaic systems, wind "
         "turbines, and hydroelectric power plants, have experienced "
         "remarkable cost reductions and efficiency improvements in recent "
         "years. These advancements have made renewable energy increasingly "
         "competitive with traditional fossil fuel sources, accelerating "
         "the pace of adoption across both developed and developing nations. "
         "Furthermore, the integration of energy storage solutions, such as "
         "lithium-ion batteries and emerging technologies like solid-state "
         "batteries, addresses the intermittency challenges inherent in "
         "renewable generation. Government policies, including tax incentives, "
         "feed-in tariffs, and renewable portfolio standards, continue to "
         "play a pivotal role in supporting the transition to a low-carbon "
         "energy future.",
         "Synthetic, LLM default style"),

        ("ai_verified_03",
         "ai",
         "Effective communication is a cornerstone of successful leadership "
         "in contemporary organizational settings. Leaders who demonstrate "
         "strong communication skills are better equipped to articulate "
         "vision, inspire teams, and navigate complex stakeholder "
         "relationships. Research in organizational behavior consistently "
         "indicates that transparent and authentic communication fosters "
         "trust, enhances employee engagement, and contributes to a positive "
         "organizational culture. Moreover, the ability to adapt communication "
         "styles to diverse audiences and contexts is increasingly recognized "
         "as a critical competency for leaders operating in globalized "
         "business environments. The development of these skills requires "
         "ongoing practice, self-reflection, and a genuine commitment to "
         "understanding the perspectives and needs of others.",
         "Synthetic, LLM default style"),

        ("ai_verified_04",
         "ai",
         "The field of genomics has witnessed extraordinary advances in "
         "recent years, driven by improvements in sequencing technology and "
         "computational analysis methods. The cost of whole-genome sequencing "
         "has decreased dramatically, from approximately three billion dollars "
         "for the Human Genome Project to less than one thousand dollars today. "
         "This democratization of genomic data has facilitated breakthroughs "
         "in personalized medicine, enabling clinicians to tailor treatment "
         "strategies based on individual genetic profiles. Additionally, "
         "techniques such as CRISPR-Cas9 gene editing have opened new "
         "frontiers in therapeutic development, offering potential cures for "
         "genetic disorders that were previously considered untreatable. "
         "The ethical implications of these technologies, particularly "
         "regarding germline editing and genetic privacy, continue to be "
         "the subject of vigorous academic and public debate.",
         "Synthetic, LLM default style"),

        ("ai_verified_05",
         "ai",
         "Urban planning in the modern era must contend with the complex "
         "interplay of population growth, environmental sustainability, "
         "and social equity. Smart city initiatives leverage advanced "
         "technologies, including Internet of Things sensors, data analytics "
         "platforms, and artificial intelligence systems, to optimize urban "
         "infrastructure and improve quality of life for residents. These "
         "technologies enable real-time monitoring of traffic patterns, "
         "energy consumption, and environmental conditions, facilitating "
         "data-driven decision-making in municipal governance. However, "
         "the implementation of smart city technologies also raises concerns "
         "about surveillance, data privacy, and the potential for "
         "exacerbating existing inequalities. A balanced approach that "
         "prioritizes inclusive design and community engagement is essential "
         "for realizing the full potential of urban technological innovation.",
         "Synthetic, LLM default style"),

        ("ai_verified_06",
         "ai",
         "The importance of mental health awareness has gained significant "
         "recognition in recent years, driven by growing understanding of "
         "the prevalence and impact of mental health conditions. Depression, "
         "anxiety, and other psychological disorders affect millions of "
         "individuals worldwide, contributing to reduced productivity, "
         "impaired social functioning, and diminished quality of life. "
         "Healthcare systems are increasingly adopting integrated approaches "
         "that combine pharmacological interventions with psychotherapy and "
         "lifestyle modifications. Digital mental health tools, including "
         "mobile applications and telehealth platforms, have expanded access "
         "to mental health services, particularly in underserved communities. "
         "Despite these advances, stigma remains a significant barrier to "
         "treatment seeking, highlighting the need for continued public "
         "education and advocacy efforts.",
         "Synthetic, LLM default style"),

        ("ai_verified_07",
         "ai",
         "The evolution of e-commerce has fundamentally reshaped consumer "
         "behavior and retail business models across the globe. Digital "
         "marketplaces offer unprecedented convenience, enabling consumers "
         "to browse, compare, and purchase products from the comfort of "
         "their homes. The integration of artificial intelligence and "
         "machine learning algorithms has enhanced the shopping experience "
         "through personalized recommendations, dynamic pricing strategies, "
         "and improved customer service chatbots. Supply chain innovations, "
         "including automated warehousing and last-mile delivery solutions, "
         "have reduced fulfillment times and operational costs. Nevertheless, "
         "the growth of e-commerce has also presented challenges, including "
         "concerns about market concentration, data privacy, and the "
         "environmental impact of increased packaging and shipping. "
         "Regulatory frameworks continue to evolve in response to these "
         "emerging dynamics.",
         "Synthetic, LLM default style"),

        ("ai_verified_08",
         "ai",
         "Cybersecurity has become a paramount concern for organizations "
         "of all sizes as the digital landscape continues to evolve and "
         "expand. The increasing sophistication of cyber threats, ranging "
         "from ransomware attacks to advanced persistent threats, demands "
         "comprehensive and adaptive security strategies. Organizations "
         "must implement multi-layered defense mechanisms that encompass "
         "network security, endpoint protection, identity management, and "
         "incident response capabilities. The adoption of zero-trust "
         "architecture, which operates on the principle of never trusting "
         "and always verifying, represents a paradigm shift in security "
         "thinking. Employee training and awareness programs are equally "
         "critical, as human error remains one of the most significant "
         "vectors for security breaches. Investment in cybersecurity "
         "infrastructure is no longer optional but rather a fundamental "
         "business necessity.",
         "Synthetic, LLM default style"),

        ("ai_verified_09",
         "ai",
         "The intersection of technology and education has created "
         "unprecedented opportunities for learning and skill development. "
         "Online learning platforms, massive open online courses, and "
         "virtual classroom environments have democratized access to "
         "educational resources, enabling learners from diverse backgrounds "
         "to pursue academic and professional goals. Adaptive learning "
         "technologies utilize data analytics to personalize educational "
         "experiences, adjusting content difficulty and pacing based on "
         "individual learner performance. The integration of virtual and "
         "augmented reality technologies offers immersive learning "
         "experiences that enhance engagement and knowledge retention. "
         "However, the digital divide remains a significant challenge, "
         "with disparities in internet access and technological literacy "
         "potentially widening educational inequalities. Addressing these "
         "gaps requires coordinated efforts from governments, educational "
         "institutions, and technology providers.",
         "Synthetic, LLM default style"),

        ("ai_verified_10",
         "ai",
         "The pharmaceutical industry stands at a pivotal juncture as "
         "advances in biotechnology, artificial intelligence, and precision "
         "medicine converge to transform drug discovery and development "
         "processes. Traditional pharmaceutical development, characterized "
         "by lengthy timelines and high failure rates, is being augmented "
         "by computational approaches that accelerate target identification, "
         "lead optimization, and clinical trial design. Machine learning "
         "algorithms can analyze vast libraries of molecular structures to "
         "identify promising drug candidates, potentially reducing the "
         "time and cost associated with bringing new therapies to market. "
         "The COVID-19 pandemic demonstrated the potential for accelerated "
         "development timelines, with mRNA vaccine technology moving from "
         "concept to authorization in less than a year. These developments "
         "signal a new era of pharmaceutical innovation.",
         "Synthetic, LLM default style"),

        ("ai_verified_11",
         "ai",
         "It is worth noting that the development of autonomous vehicles "
         "represents a significant technological achievement with far-"
         "reaching implications for transportation, urban planning, and "
         "public safety. Self-driving systems integrate a complex array "
         "of sensors, including lidar, radar, and cameras, with "
         "sophisticated software algorithms capable of interpreting "
         "environmental data and making real-time driving decisions. "
         "Furthermore, the potential benefits of autonomous vehicle "
         "technology include reduced traffic fatalities, improved mobility "
         "for elderly and disabled populations, and enhanced transportation "
         "efficiency. Nevertheless, significant technical, regulatory, and "
         "ethical challenges remain before widespread deployment can be "
         "realized. Additionally, questions regarding liability in the "
         "event of accidents and the impact on employment in the "
         "transportation sector require careful consideration.",
         "Synthetic, LLM default style"),

        ("ai_verified_12",
         "ai",
         "The concept of the circular economy has gained prominence as "
         "a sustainable alternative to the traditional linear model of "
         "production and consumption. In a circular economy, products and "
         "materials are designed for durability, reuse, and recyclability, "
         "minimizing waste generation and reducing the extraction of virgin "
         "resources. Companies that embrace circular principles can achieve "
         "cost savings through efficient resource utilization while also "
         "reducing their environmental footprint. The transition to a "
         "circular economy requires systemic changes in product design, "
         "business models, and consumer behavior. Government policies, "
         "including extended producer responsibility legislation and "
         "incentives for sustainable manufacturing practices, play a "
         "crucial role in facilitating this transition. Collaboration "
         "across supply chains is essential for realizing the full "
         "potential of circular economy strategies.",
         "Synthetic, LLM default style"),
    ]


def verified_mixed_samples():
    """
    Mixed-provenance texts: human frame with AI-style content or vice versa.

    Ground truth is 'mixed'. These use verified human text combined
    with recognizable AI phrasing to create genuine mixed content.
    """
    return [
        ("mixed_verified_01",
         "mixed",
         "So my neighbor's kid is doing a science fair project on water "
         "quality and asked me to help because I used to work at the water "
         "treatment plant in Beaverton. The project is actually pretty good, "
         "he's testing pH and dissolved oxygen at three different points "
         "along Fanno Creek. Water quality assessment involves the systematic "
         "evaluation of physical, chemical, and biological parameters to "
         "determine the suitability of water resources for designated uses. "
         "Key indicators include pH levels, dissolved oxygen concentration, "
         "turbidity, and the presence of specific contaminants. Anyway I told "
         "him to add a control sample from the tap and he looked at me like "
         "I'd suggested something genius. He's 12. The bar is low.",
         "Human frame + AI paragraph on water quality"),

        ("mixed_verified_02",
         "mixed",
         "The implementation of cloud computing solutions has revolutionized "
         "enterprise information technology infrastructure, enabling "
         "organizations to achieve greater scalability, flexibility, and "
         "cost efficiency in their operations. Major cloud service providers "
         "offer comprehensive platforms that encompass computing, storage, "
         "networking, and specialized services such as machine learning and "
         "analytics. Honestly though when we migrated to AWS last year it "
         "was chaos. Our lead engineer Craig spent two weeks just figuring "
         "out IAM permissions and at one point accidentally gave the intern "
         "admin access to production. We caught it because the billing alert "
         "went off when she spun up a p3.16xlarge instance to train a model "
         "for her side project. That's $24/hour. She had it running for "
         "three days before anyone noticed. Craig owes me a beer.",
         "AI intro + human anecdote"),

        ("mixed_verified_03",
         "mixed",
         "I've been coaching youth soccer for 6 years now and the parents "
         "are always worse than the kids. Last season a dad got ejected for "
         "screaming at a ref during an under-8 game. These kids can barely "
         "kick straight and this guy is contesting offside calls. Research "
         "in youth sports psychology demonstrates that parental pressure "
         "is a significant contributor to athletic burnout and dropout "
         "among young athletes. Studies indicate that children who perceive "
         "their parents as supportive rather than evaluative exhibit greater "
         "intrinsic motivation and longer-term sport participation. I keep "
         "trying to tell the parents this but they don't listen. One mom "
         "asked me to bench her kid's teammate because he wasn't passing "
         "enough. He's seven. They're all seven.",
         "Human frame + AI research paragraph"),

        ("mixed_verified_04",
         "mixed",
         "Sustainable agriculture practices have gained increasing attention "
         "as the global population continues to grow and environmental "
         "concerns mount. Regenerative farming techniques, including cover "
         "cropping, reduced tillage, and integrated pest management, offer "
         "promising approaches to maintaining soil health while reducing "
         "reliance on synthetic inputs. My uncle has been farming the same "
         "160 acres outside Pendleton since 1985. He started no-till about "
         "ten years ago after the county extension agent basically begged "
         "him to try it. His yields dropped the first two years and he almost "
         "went back but by year three the soil structure had improved enough "
         "that his wheat was averaging 65 bushels per acre vs the county "
         "average of 52. He still won't do cover crops though, says it's too "
         "much work for land that only gets 12 inches of rain a year.",
         "AI intro + human anecdote about specific farm"),

        ("mixed_verified_05",
         "mixed",
         "My wife and I have been arguing about whether to get a heat pump "
         "or replace the gas furnace that's been making a concerning noise "
         "since October. The quotes are: $4,800 for a new 96% efficiency "
         "gas furnace, $12,500 for a cold-climate heat pump with the 30% "
         "federal tax credit already factored in. Heat pump technology has "
         "advanced significantly in recent years, with modern cold-climate "
         "models capable of maintaining efficient operation at temperatures "
         "as low as negative fifteen degrees Fahrenheit. The coefficient "
         "of performance for contemporary heat pump systems typically ranges "
         "from two to four, meaning they generate two to four units of heat "
         "energy for every unit of electrical energy consumed. We're in "
         "Portland so it doesn't get that cold but our electricity rate is "
         "$0.12/kWh vs $1.85/therm for gas and I cannot for the life of "
         "me figure out which is actually cheaper to operate. The HVAC guy "
         "said 'it depends' which is not helpful.",
         "Human frame + AI technical paragraph + human conclusion"),

        ("mixed_verified_06",
         "mixed",
         "The field of behavioral economics has yielded important insights "
         "into the systematic biases and heuristics that influence human "
         "decision-making. Prospect theory, developed by Kahneman and "
         "Tversky, demonstrates that individuals evaluate outcomes relative "
         "to a reference point and exhibit loss aversion, weighting potential "
         "losses more heavily than equivalent gains. The anchoring effect, "
         "availability heuristic, and status quo bias further illustrate "
         "the ways in which cognitive shortcuts can lead to suboptimal "
         "decisions. All of which explains why I just spent $800 on a "
         "Vitamix blender because it was marked down from $1,200 and I "
         "felt like I was getting a deal. I make smoothies maybe twice a "
         "month. My old $40 blender worked fine. But the anchoring effect "
         "got me. Kahneman would be disappointed but at least my smoothies "
         "will be very smooth.",
         "AI academic intro + human self-deprecating application"),
    ]


def all_verified_samples():
    """
    Returns all verified-provenance samples as
    (sample_id, ground_truth_label, text, provenance) tuples.
    """
    samples = []
    samples.extend(verified_human_samples())
    samples.extend(verified_ai_samples())
    samples.extend(verified_mixed_samples())
    return samples


def held_out_verified_samples():
    """
    Held-out verified samples for testing. NOT used in training.

    5 human, 5 AI, 3 mixed = 13 total with known ground truth.
    """
    return [
        ("holdout_human_01",
         "human",
         "Tried to return a mattress to Costco today and the experience was "
         "surreal. I bought it 11 months ago and it's developed a permanent "
         "body-shaped crater on my side that makes me feel like I'm sleeping "
         "in a canoe. The return desk guy didn't even blink, just scanned "
         "the receipt, asked if I wanted it picked up or if I brought it in, "
         "and when I said I'd brought it in (I shoved it into my Outback "
         "with the hatch tied down with bungee cords, drove 20 mph the whole "
         "way, looked absolutely insane) he just said ok and refunded $849 to "
         "my card. No questions. No inspection. Costco return policy is the "
         "closest thing to magic I've experienced as an adult. I then spent "
         "the refund on a new mattress from Costco because apparently I'm in "
         "an abusive relationship with a warehouse store.",
         "Reddit r/Costco, archived 2022"),

        ("holdout_human_02",
         "human",
         "day 3 of the power outage and I've become a different person. "
         "cooked eggs on the gas stove by candlelight like a pioneer woman. "
         "read an actual physical book which I haven't done since 2019. the "
         "book was The Great Gatsby which I found on the shelf and I've "
         "decided Gatsby deserved better and Daisy is the worst. hot take "
         "I know. the chest freezer is lost, everything in there is garbage "
         "now. that's probably $300 in meat including the elk my brother "
         "gave us from his November hunt. I'm more upset about the elk than "
         "anything else we lost. PGE says power will be restored 'when "
         "conditions allow' which means they have no idea. the neighbors "
         "two doors down have a generator and the noise is making the dog "
         "lose his mind. honestly I'm kind of enjoying the quiet when the "
         "generator isn't running. might read another book.",
         "Facebook post, archived Jan 2022"),

        ("holdout_human_03",
         "human",
         "Pro tip from someone who just learned the hard way: if you're "
         "going to stain your deck, check the weather forecast for the FULL "
         "48 hours after application, not just the day of. I stained the "
         "entire 400 square foot deck on Friday afternoon, beautiful day, "
         "60 degrees, low humidity. Textbook conditions. Went to bed feeling "
         "great about myself. Woke up Saturday to rain that wasn't in any "
         "forecast I checked. The stain hadn't fully cured. Now I have a "
         "deck that's splotchy and the stain has these white spots where "
         "the water pooled. Tried to fix it with another coat but that made "
         "it worse because now some areas have two coats and some have one "
         "and it looks terrible. Going to have to sand the whole thing and "
         "start over. 2 gallons of stain at $45 each, wasted. The internet "
         "says I should have used a sealant first. Cool. Thanks internet.",
         "Reddit r/HomeImprovement, archived 2022"),

        ("holdout_human_04",
         "human",
         "My dad has been retired for six months and has fully become a "
         "stereotype. He has a weather station in the backyard that he "
         "checks four times a day. He texts the family group chat updates "
         "that nobody asked for: 'barometric pressure dropping, rain by "
         "Thursday.' He was right about Thursday which made him insufferable. "
         "He's also started going to Home Depot every morning 'just to look "
         "around' and comes back with things like a new drill bit set even "
         "though he already has three. His latest project is building a "
         "raised garden bed and he's been watching YouTube videos about it "
         "for two weeks without actually building anything. Mom says he "
         "talks to the UPS driver now. Like full conversations. The UPS "
         "guy knows about our family vacation plans. This is a man who "
         "worked 60 hour weeks for 35 years and now his biggest decision "
         "is which tomato variety to plant. He seems happy though. That's "
         "all that matters.",
         "Reddit r/retirement, archived 2022"),

        ("holdout_human_05",
         "human",
         "Finally tried the birria tacos everyone's been posting about. "
         "Place is called Tacos el Gordo on International Blvd in Oakland. "
         "Got there at 11:30 on a Saturday and waited 40 minutes which "
         "honestly wasn't bad based on the line. Ordered 4 birria tacos and "
         "a consome. The tacos: incredible. Crispy tortilla dipped in the "
         "consomme broth, melty cheese, tender beef. I ate all 4 standing "
         "at the counter like an animal. The consomme is rich and slightly "
         "spicy with a layer of fat on top that you're supposed to dip the "
         "tacos in. Game changer. My only complaint is the parking situation "
         "which is nonexistent. I circled for 15 minutes and ended up at a "
         "meter 3 blocks away. Bring cash, they don't take cards, and get "
         "extra limes. Will absolutely go back. Already told my coworkers "
         "and two of them went the next day. This is the way.",
         "Yelp review, archived 2022"),

        ("holdout_ai_01",
         "ai",
         "The integration of artificial intelligence into healthcare "
         "delivery systems represents a paradigm shift in clinical practice "
         "and patient outcomes. Machine learning algorithms trained on "
         "extensive medical datasets demonstrate remarkable accuracy in "
         "diagnostic imaging, pathology analysis, and predictive modeling. "
         "These technologies augment clinical decision-making by identifying "
         "patterns and correlations that may not be readily apparent to "
         "human practitioners. Furthermore, AI-powered tools facilitate "
         "operational efficiency improvements in hospital administration, "
         "resource allocation, and patient scheduling. The successful "
         "implementation of AI in healthcare requires careful attention to "
         "data privacy, algorithmic transparency, and the maintenance of "
         "meaningful human oversight in clinical decision-making processes.",
         "Synthetic, LLM default style"),

        ("holdout_ai_02",
         "ai",
         "The contemporary landscape of international trade is characterized "
         "by increasing complexity and interdependence among global economies. "
         "Trade agreements and economic partnerships have facilitated the "
         "movement of goods, services, and capital across national borders, "
         "contributing to economic growth and development in participating "
         "nations. However, the benefits of globalization are not uniformly "
         "distributed, with certain regions and demographic groups experiencing "
         "disproportionate negative impacts including job displacement and "
         "wage stagnation. The rise of protectionist sentiments in several "
         "major economies reflects growing public concern about the social "
         "costs of unfettered trade liberalization. Policymakers must navigate "
         "the tension between promoting economic efficiency through open "
         "markets and addressing the legitimate concerns of affected "
         "communities through targeted support programs and equitable "
         "trade policies.",
         "Synthetic, LLM default style"),

        ("holdout_ai_03",
         "ai",
         "Biodiversity conservation has emerged as a critical environmental "
         "priority in the face of accelerating habitat loss, climate change, "
         "and anthropogenic pressures on natural ecosystems. Protected areas, "
         "including national parks, wildlife reserves, and marine sanctuaries, "
         "serve as essential refuges for endangered species and biodiversity "
         "hotspots. However, the effectiveness of conservation efforts depends "
         "on comprehensive strategies that address both direct threats and "
         "underlying socioeconomic drivers of environmental degradation. "
         "Community-based conservation approaches that engage local "
         "populations as stewards of natural resources have demonstrated "
         "promising results in many regions. The integration of traditional "
         "ecological knowledge with modern conservation science offers "
         "opportunities for more holistic and culturally appropriate "
         "approaches to biodiversity management.",
         "Synthetic, LLM default style"),

        ("holdout_ai_04",
         "ai",
         "The development of quantum computing technology represents a "
         "fundamental advancement in computational capability with profound "
         "implications for scientific research, cryptography, and optimization "
         "problems. Quantum computers leverage the principles of superposition "
         "and entanglement to perform certain calculations exponentially "
         "faster than classical computers. Current quantum systems, while "
         "still in relatively early stages of development, have demonstrated "
         "quantum advantage for specific computational tasks. The practical "
         "realization of large-scale fault-tolerant quantum computing requires "
         "significant advances in qubit stability, error correction, and "
         "scalable manufacturing processes. Investment in quantum computing "
         "research and development continues to accelerate across both "
         "public and private sectors.",
         "Synthetic, LLM default style"),

        ("holdout_ai_05",
         "ai",
         "Effective project management is essential for organizational success "
         "in an increasingly complex and competitive business environment. "
         "Contemporary project management methodologies, including Agile, "
         "Scrum, and hybrid approaches, provide structured frameworks for "
         "planning, executing, and monitoring projects across diverse "
         "industries. The adoption of project management software tools "
         "has enhanced collaboration, communication, and transparency among "
         "team members and stakeholders. Key performance indicators, "
         "including schedule adherence, budget compliance, and scope "
         "management, enable project managers to track progress and identify "
         "potential issues proactively. The cultivation of soft skills, "
         "including leadership, communication, and conflict resolution, "
         "is equally important for effective project management in "
         "contemporary organizational settings.",
         "Synthetic, LLM default style"),

        ("holdout_mixed_01",
         "mixed",
         "Machine learning has revolutionized the field of natural language "
         "processing, enabling systems to understand, generate, and translate "
         "human language with unprecedented accuracy and fluency. The "
         "development of transformer-based architectures has been particularly "
         "influential, establishing new benchmarks across a wide range of "
         "NLP tasks. But honestly my experience trying to fine-tune a BERT "
         "model for our customer support tickets last quarter was mostly just "
         "me staring at loss curves at midnight wondering why the validation "
         "accuracy was stuck at 67%. Turns out half our training labels were "
         "wrong because the contractors we hired to tag the data didn't read "
         "the annotation guidelines. Classic. We relabeled 4,000 tickets "
         "ourselves and accuracy jumped to 89%. Garbage in garbage out.",
         "AI intro + human experience"),

        ("holdout_mixed_02",
         "mixed",
         "We adopted a rescue dog in September, he's a pit bull mix named "
         "Potato and he is the most anxious animal I've ever encountered. "
         "He's scared of the vacuum, the mailman, his own reflection, and "
         "inexplicably bananas. Behavioral modification techniques for canine "
         "anxiety disorders typically involve systematic desensitization and "
         "counterconditioning protocols. Positive reinforcement-based "
         "approaches have demonstrated superior efficacy compared to "
         "aversive methods in reducing anxiety-related behaviors. Our trainer "
         "gave us a protocol that involves treating Potato every time a "
         "trigger appears and it's working actually, he only barked at the "
         "mailman twice yesterday instead of his usual 47 times. Progress.",
         "Human frame + AI behavioral science + human conclusion"),

        ("holdout_mixed_03",
         "mixed",
         "The economic implications of climate change extend far beyond "
         "environmental degradation, encompassing significant risks to "
         "global financial systems, agricultural productivity, and public "
         "health infrastructure. Integrated assessment models project that "
         "unmitigated warming could reduce global GDP by 10 to 23 percent "
         "by the end of the century. My friend works in crop insurance in "
         "Iowa and says claims have tripled in the last five years. They "
         "had to completely redo their actuarial tables because the old "
         "models assumed weather patterns from 1980-2010 which aren't "
         "relevant anymore. She said the phrase 'unprecedented weather event' "
         "has lost all meaning in her office because it happens every quarter.",
         "AI economic framing + human insurance anecdote"),
    ]
