



Here's a convo I had with an agent. I am focused on building a lineage. Could you independently verify the lineage claude has come up with. After that, you will do your own research to find the lineage and report back.

Claude-Erasing-concepts-from-diffusion-models.md
Completed 10 steps
Claude’s lineage is broadly correct in its main spine (ESD → UCE / Concept Ablation → MACE → RECE / SPEED / GLoCE / RACE → ETC), and its choice of SPEED as a practical frontier method is well supported, but some of the specific method names, quantitative claims, and very recent works it mentions are either not fully verifiable or more speculative than stated.

Below I first cross‑check the key claims from that conversation, then give a cleaned‑up lineage I’d be comfortable using as your “official” one.

What the previous agent got right
Core ESD description and role
ESD as the starting point. “Erasing Concepts from Diffusion Models” (ESD) by Gandikota et al. is indeed an ICCV 2023 paper that fine‑tunes Stable Diffusion to erase a target concept using negative guidance as a teacher, with only the text name of the concept as input. It introduces two variants: ESD‑x (cross‑attention for prompt‑specific styles) and ESD‑u (unconditional layers for global erasure like nudity).

Permanent weight editing vs inference‑time safety. The ESD paper explicitly contrasts its weight‑editing approach with Safe Latent Diffusion (SLD), which intervenes during the denoising process without modifying weights and can theoretically be bypassed. So the earlier agent’s framing of ESD as “first practical weight‑editing eraser that beats SLD / retraining in cost while being non‑bypassable” is directionally correct.

Pre‑ESD foundations
The agent’s “foundations” are consistent with what the literature actually says:

Safe Latent Diffusion (SLD). Schramowski et al. 2022 introduce SLD as an inference‑time method that uses an “unsafe” prompt and guidance in latent space to suppress nudity/violence on the I2P benchmark without retraining; it’s a canonical pre‑ESD safety baseline.

Concept ablation / anchor methods. “Ablating Concepts in Text‑to‑Image Diffusion Models” by Kumari et al. (2023) fine‑tunes to match a “target” concept distribution (e.g., “Van Gogh”) to an anchor (“paintings”), exactly as the conversation described. This is an important prior for ESD’s “unlearning via weight editing” framing.

So the idea that ESD builds on SLD‑style inference‑time steering plus earlier “concept ablation” fine‑tuning work is well supported.

Successor methods and branches
The agent’s main branches line up with actual papers:

Closed‑form / efficiency branch.

UCE (Unified Concept Editing). Gandikota et al. propose a training‑free, closed‑form solution that edits cross‑attention weights to perform multi‑concept editing and erasure; they explicitly emphasize a closed‑form solution and scalability to concurrent edits.

RECE (Reliable and Efficient Concept Erasure). Gong et al. (ECCV 2024) introduce RECE, which uses a fully closed‑form embedding‑space procedure that modifies the model “in 3 seconds” without extra fine‑tuning, and reports improved efficiency and robustness vs UCE/ESD‑style baselines.

SPEED. “SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models” (Li et al., ICLR 2026) directly positions itself as an efficient weight‑editing method that finds a null‑space where parameter updates preserve non‑target concepts; it reports erasing 100 concepts in ~5 seconds and better non‑target preservation than existing methods.

This matches the agent’s classification of UCE/RECE/SPEED as an “efficiency” branch with closed‑form or near‑closed‑form editing and strong scaling properties.

Adapter / LoRA / mass‑erasure branch.

Concept Ablation is indeed the “anchor‑matching” fine‑tuning method discussed above.

MACE (Mass Concept Erasure). Lu et al. (CVPR 2024) introduce MACE as a framework that combines (1) closed‑form refinement of cross‑attention keys/values and (2) per‑concept LoRA adapters; they explicitly claim successful scaling to 100 concepts and an improved generality–specificity balance over prior work.

LoRA‑based scaling more broadly. Later work like DyME (Dynamic Multi‑Concept Erasure with bi‑level orthogonal LoRA) explicitly tackles multi‑concept scaling through modular LoRA adapters and orthogonality constraints, which is exactly the kind of “LoRA‑based scalability” branch the earlier agent described.

So placing MACE and DyME in a “LoRA / scaling” branch is faithful to the papers.

Adversarial robustness branch.

RACE (Robust Adversarial Concept Erasure). Kim et al. (ECCV 2024) propose RACE as an adversarially trained erasure method that reduces attack success rate by about 30 percentage points for “nudity” vs a strong white‑box attack.

AdvUnlearn. The “Defensive Unlearning with Adversarial Training for Robust Concept Erasure” repository (AdvUnlearn) explicitly frames itself as an adversarial‑training‑based unlearning method in diffusion models, and compares to SPM, etc.

Vulnerability/attack papers. Work like “On the Vulnerability of Concept Erasure in Diffusion Models” introduces RECORD, an attack that recovers erased concepts by adversarial prompt search, and shows many erasure methods are more vulnerable than expected.

This supports the idea of a dedicated “robustness vs attacks” branch emerging post‑ESD/UCE/MACE.

Localization branch.

GLoCE (Gated Low‑rank Adaptation for Concept Erasure). The CVPR 2025 “Localized Concept Erasure…” paper proposes GLoCE, a training‑free gated LoRA module that erases a concept only in its spatial region, explicitly targeting the case where target and non‑target concepts co‑exist in one image.

LACE (Localized‑Attention‑Guided Concept Erasure). Another 2025 method, LACE, uses attention‑guided spatial masks and gated feature cleanup to localize concept erasure.

So a “localization / spatial erasure” branch, separate from global ESD/MACE/SPEED‑type methods, is indeed real.

Analysis / evaluation branch.

When Are Concepts Erased from Diffusion Models? Lu et al. propose a comprehensive probing framework and explicitly discuss that current methods may reduce likelihood rather than truly remove a concept.

Fantastic Targets for Concept Erasure… models concept space as a graph and designs better “target concepts” to reduce side‑effects.

These match the “analysis/evaluation” branch the agent mentions.

Overall, the high‑level structure the agent gave you matches what the papers themselves claim about their goals and contributions.

Frontier choice: SPEED vs ETC vs MACE
MACE as a major step after ESD/UCE. MACE clearly moves the frontier beyond ESD/UCE by scaling to 100 concepts with LoRA plus attention refinement and outperforming prior erasers on object, celebrity, NSFW, and style tasks.

SPEED as a newer frontier with code. SPEED is later (ICLR 2026), explicitly optimized for scalability/precision/efficiency, demonstrates erasing 100 concepts in ~5 seconds, and has an official PyTorch implementation on GitHub.

ETC (Erasing Thousands of Concepts). ETC is a 2026 arXiv preprint that targets erasing thousands of concepts using t‑mixture modeling and an MoE‑based “MoEraser” module; it emphasizes scalability and robustness but does not clearly advertise open‑sourced code or models in the abstract.

Given this, the previous agent’s judgment—MACE as the key CVPR‑2024 method, with SPEED as the practical current frontier (especially because it ships code and models), while ETC is even more extreme in scale but less mature—is reasonable.

Where I would be cautious or adjust details
There are a few places where I would not repeat the earlier agent’s statements as‑is:

Unverified method names and acronyms.
I can clearly verify MACE, UCE, RECE, SPEED, GLoCE, LACE, RACE, DyME, and the analysis papers.

I do not find public diffusion‑erasure papers matching some names used in the conversation (e.g., “ANT”, “SCORE”, “STEREO” in exactly those roles), so I would treat those as speculative or shorthand for broader lines of work rather than specific canonical papers.

Speedup numbers.
SPEED’s abstract and OpenReview page emphasize “erasing 100 concepts within only 5 seconds” and improved non‑target preservation, but I do not see an explicit “350× faster than MACE” figure in the public summaries I looked at. That number may come from a specific experimental table, but without seeing it directly I wouldn’t treat the factor as a headline claim.

What is clearly supported is: SPEED is significantly faster and more scalable than fine‑tuning‑based methods such as MACE while preserving non‑targets better.

Venue/acceptance status for the newest works.
The previous agent described ETC as “CVPR 2026” and SPEED as “ICLR 2026”. SPEED is indeed an ICLR 2026 paper.

For ETC, the current public source is an April 2026 arXiv preprint (“Erasing Thousands of Concepts”) that does not itself state a peer‑reviewed venue, so treating it as “CVPR 2026” is a bit ahead of the public record.

Some branch labels vs actual mechanisms.
The earlier agent sometimes treats “efficiency branch” as synonymous with “closed‑form editing”. That’s roughly true for UCE and RECE, and conceptually aligned with SPEED’s null‑space optimization, but SPEED is not literally a single closed‑form matrix solve; it involves null‑space search with additional constraints (IPF, DPA, IEC). So calling it “closed‑form” is slightly loose.

Similarly, AdvUnlearn and RACE are correctly in a “robustness” branch, but they still rely on fine‑tuning rather than purely training‑free updates.

Those are nuances; they don’t change the overall high‑level lineage, but they matter if you want to be precise.

A cleaned‑up lineage from pre‑ESD to the current frontier
Below is a lineage I’d be comfortable citing in a write‑up or slide deck. I’ll focus on the main methods that have clear papers plus code, and I’ll explicitly note what each step solves and what remains open.

Foundations before ESD (2022–early 2023)
Data filtering and retraining (e.g., SD 2.x)

Strategy: Retrain Stable Diffusion‑style models on filtered datasets to remove unsafe content (e.g., remove nudity from the training set).

Pros: Concept can be substantially reduced without hacky inference‑time tricks.

Cons: Extremely expensive, needs full retraining whenever you add/remove concepts, and still not perfect. (This is mainly documented in blog/docs rather than a single paper.)

Safe Latent Diffusion (SLD).

Paper: “Safe Latent Diffusion: Mitigating Inappropriate Degeneration in Diffusion Models.”

Idea: Introduce an “unsafe” prompt and use guidance in latent space at inference time to steer generations away from unsafe content on the I2P benchmark without changing weights.

Problems solved: Cheap deployment‑time mitigation; strong reduction of nudity/violence on curated prompts.

Remaining problems: Easy to bypass if users have weight access; no permanent erasure; no direct control over specific styles or copyrighted content.

Ablating Concepts in T2I Diffusion Models (Concept Ablation).

Paper: “Ablating Concepts in Text‑to‑Image Diffusion Models.”

Idea: Fine‑tune so that the target concept distribution (e.g., “Van Gogh painting”) matches an anchor concept (“paintings”), preventing explicit target generations.

Problems solved: First explicit “concept erasure” by aligning distributions; applicable to styles, instances, and memorized images.

Remaining problems: Per‑concept fine‑tuning cost; limited multi‑concept ability; some collateral damage on nearby concepts.

These works set up the need for something cheaper and more targeted than retraining, and more permanent than SLD.

ESD as the first practical weight‑editing eraser (ICCV 2023)
Erasing Concepts from Diffusion Models (ESD).

Paper & site: Gandikota et al., ICCV 2023; official website and GitHub provide code and fine‑tuned weights.

Idea: Fine‑tune Stable Diffusion using negative classifier‑free guidance as the teacher so that the concept’s conditional prediction matches a negatively guided version, directly editing the U‑Net weights.

Variants:

ESD‑x: edits cross‑attention, aimed at prompt‑specific concepts like artist styles.

ESD‑u: edits unconditional layers, for global concepts like nudity that may appear even without explicit prompts.

Problems solved:

Permanent erasure in weights, harder to bypass than SLD.

Requires only a text description, no curated image dataset.

Demonstrated style removal and NSFW reduction competitive with SLD and censored retraining.

Remaining problems (from later analysis and attack papers):

Lexical evasion for ESD‑x (describe the artist without the name).

Collateral damage for ESD‑u (unconditional edits degrade unrelated content).

Vulnerability to adversarial prompt attacks (e.g., RECORD, other red‑teaming methods).

ESD is why most later papers explicitly benchmark against “ESD‑x” and “ESD‑u”.

Branch A: Closed‑form / efficiency methods (UCE → RECE → SPEED)
Unified Concept Editing (UCE).

Paper: “Unified Concept Editing in Diffusion Models” (Gandikota et al., WACV 2024).

Idea: Training‑free closed‑form editing of cross‑attention projections to simultaneously perform debiasing, style erasure, and content moderation, scaling to many concepts at once.

Problems solved: Moves from per‑concept fine‑tuning to closed‑form editing; significantly improves scalability vs Concept Ablation / ESD for multiple concepts.

Remaining: Efficacy and robustness still limited; erasure can be incomplete and may affect neighbors.

RECE (Reliable and Efficient Concept Erasure).

Paper: “Reliable and Efficient Concept Erasure of Text‑to‑Image Diffusion Models” (Gong et al., arXiv 2024, ECCV 2024).

Idea: Iteratively derive new embeddings for the erased concept via a closed‑form procedure, project them toward harmless concepts in cross‑attention layers, and add regularization to preserve unrelated content.

Problems solved: Extremely fast (on the order of seconds) training‑free erasure; improved robustness to red‑teaming tools vs prior baselines including UCE/ESD.

Remaining: Still operates largely at the embedding level; issues of deep internal memorization and adversarial prompt robustness are not entirely closed.

SPEED (Scalable, Precise, and Efficient Concept Erasure).

Paper & code: Li et al., ICLR 2026; OpenReview and GitHub repo “ouxiang‑li/speed”.

Idea: Search a null space in parameter space where updates erase target concepts while preserving non‑targets, using three strategies: Influence‑based Prior Filtering (IPF), Directed Prior Augmentation (DPA), and Invariant Equality Constraints (IEC).

Reported results:

Efficiently erases up to 100 concepts in around 5 seconds in their setup.

Consistently better non‑target preservation than previous methods across multiple benchmarks.

Problems solved: Strong step toward all three goals: scalable, precise, and efficient weight‑editing; clearly ahead of MACE/RECE/UCE on the specific metrics they report.

Remaining:

Still evaluated mostly on SD‑style models and curated concept sets; adversarial robustness and truly extreme scales (thousands of concepts) remain partly open.

This branch is the strongest candidate for “efficiency‑centric frontier” today.

Branch B: LoRA and mass concept erasure (Concept Ablation → MACE → DyME → ETC)
Concept Ablation (revisited).

As above, Concept Ablation is the first clear example of fine‑tuning with an anchor concept; subsequent work generalized this idea.

MACE: Mass Concept Erasure in Diffusion Models.

Paper & code: Lu et al., CVPR 2024, with official GitHub implementation.

Idea: Combine closed‑form cross‑attention refinement (to suppress residual concept info in surrounding tokens) with per‑concept LoRA fine‑tuning and a closed‑form LoRA integration step that allows composing many LoRAs without mutual interference.

Problems solved:

Scaling from ≤5 concepts to ≈100 concepts while achieving a better generality–specificity trade‑off (erasing synonyms while keeping unrelated content).

Strong performance across object, celebrity, NSFW, and style erasure benchmarks.

Remaining:

Still requires per‑concept fine‑tuning (though reasonably fast on high‑end GPUs).

Susceptible to compositional evasion and adversarial prompts, as later robustness and evaluation papers point out.

DyME: Dynamic Multi‑Concept Erasure.

Paper: “DyME: Dynamic Multi‑Concept Erasure in Diffusion Models with Bi‑level Orthogonal LoRA Adaptation” (2025 preprint).

Idea: Train modular LoRA adapters per concept with bi‑level orthogonality constraints, then dynamically compose them at inference time to erase arbitrary subsets of concepts.

Role: Represents a continuation of the “per‑concept LoRA” path, with more careful adapter composition and less interference; complements MACE’s integrated LoRA approach.

ETC: Erasing Thousands of Concepts.

Paper: “Erasing Thousands of Concepts: Towards Scalable and Practical Concept Erasure for Text‑to‑Image Diffusion Models” (Seo et al., arXiv 2026).

Idea:

Model concept distributions via a Student’s t‑mixture model in embedding space, enabling affine optimal transport to “pin‑point” erase target concepts without predefined anchors.

Train a Mixture‑of‑Experts “MoEraser” module to remove target embeddings while preserving anchors; add noise and fine‑tune for robustness against module removal attacks.

Problems solved:

Demonstrates erasure at the scale of thousands of concepts while preserving generation quality in their experiments.

Remaining:

Very new; code and models are not clearly advertised as openly released in the abstract; evaluation and reproducibility are still evolving.

If you want a mass‑erasure storyline, Concept Ablation → MACE → DyME → ETC is the cleanest chain.

Branch C: Adversarial robustness and attacks
Attack side: “On the Vulnerability of Concept Erasure…” and related red‑teaming.

Paper: “On the Vulnerability of Concept Erasure in Diffusion Models” (Beerens et al., 2025).

Idea: Introduce RECORD, a coordinate‑descent prompt attack that re‑elicits erased concepts; show that many models subjected to ESD/UCE/MACE‑like erasure are still vulnerable.

Message: Concept‑erased models are often more brittle to adversarial prompts than expected, highlighting robustness as an open problem.

RACE: Robust Adversarial Concept Erasure.

Paper & code: Kim et al., arXiv 2024, ECCV 2024 oral; GitHub repo “chkimmmmm/R.A.C.E.”

Idea: Wrap concept erasure in an adversarial training framework that optimizes against adversarial text embeddings targeting sensitive concepts, reducing attack success rate on benchmarks like “nudity”.

Problems solved: Demonstrated ≈30 percentage‑point reduction in attack success rate vs strong white‑box baselines for nudity.

Remaining: Computationally heavy; still subject to new attack types; trade‑off between robustness and generation quality persists.

AdvUnlearn and follow‑ups.

Repository: “AdvUnlearn – Defensive Unlearning with Adversarial Training for Robust Concept Erasure in Diffusion Models.”

Idea: Integrate adversarial training into unlearning, and evaluate robustness against multiple erasure baselines including SPM and others.

Role: Along with RACE, forms the core of the “robustness‑oriented” branch.

Branch D: Localized / spatial concept erasure
GLoCE: Gated Low‑rank Adaptation for Concept Erasure.

Paper: “Localized Concept Erasure for Text‑to‑Image Diffusion Models Using Training‑Free Gated LoRA” (CVPR 2025).

Idea: Insert a lightweight low‑rank module plus a gate that activates only where the target concept appears, enabling erasure of local regions while preserving the rest of the image.

Problems solved: Substantially improves fidelity to non‑target regions while erasing localized targets, and outperforms prior methods on specificity and robustness for localized erasure.

LACE: Localized‑Attention‑Guided Concept Erasure.

Paper: “Localized‑Attention‑Guided Concept Erasure for Text‑to‑Image Diffusion Models” (OpenReview 2025).

Idea: Three‑stage framework (representation‑space projection, attention‑guided spatial gate, gated feature clean‑up) to confine edits to regions with residual concept activation while preserving semantic neighbors.

Role: Complementary localized approach with explicit attention‑guided masks.

These methods directly address the “global collateral damage” problem of ESD‑u / global MACE‑style edits by localizing edits spatially.

Branch E: Flow‑based / new architectures (EraseAnything)
EraseAnything: Concept erasure in rectified flow transformers.

Paper: “EraseAnything: Enabling Concept Erasure in Rectified Flow Transformers” (ICML 2025 poster).

Idea: First dedicated concept‑erasure method for newer flow‑matching / transformer‑based models like FLUX; formulates erasure as a bi‑level optimization with LoRA‑based tuning and attention regularization, plus self‑contrastive learning to preserve unrelated content.

Role: Extends the concept‑erasure line beyond SD‑v1.x / latent diffusion into newer T2I paradigms.

Branch F: Evaluation and theory
When Are Concepts Erased From Diffusion Models?

Paper: Lu et al., 2025–2026 (NeurIPS / OpenReview).

Idea: Propose two conceptual models of erasure (guidance interference vs likelihood reduction) and a suite of probes: visual context, trajectory modifications, classifier guidance, and alternative generations.

Message: Many methods achieve partial suppression rather than full removal, and there is a tension between side‑effects and robustness.

Fantastic Targets for Concept Erasure and related work.

Paper: “Fantastic Targets for Concept Erasure in Diffusion Models and Where To Find Them.”

Idea: Model concept space as a graph and choose adaptive target concepts instead of fixed anchors, reducing unintended side‑effects.

These works provide the diagnostic tools you’d use to stress‑test ESD/MACE/SPEED/RECE/etc. and are useful for designing your project experiments.

Current frontier and recommended “target paper”
Putting this together:

For “classic” SD‑style global concept erasure with many concepts and strong efficiency
,SPEED (ICLR 2026) is, in my judgment, the most defensible current frontier method with public code and models:

It directly targets scalability (100 concepts), efficiency (seconds‑level editing), and non‑target preservation, and reports systematic improvements over earlier methods such as MACE and UCE.

Code and models are available, making it practical as a project target.

For mass scaling to thousands of concepts
,ETC (Erasing Thousands of Concepts) pushes the scale further but is newer and less battle‑tested; code availability is not as clear.

For localized erasure
,GLoCE and LACE are at the frontier for spatially targeted removal with training‑free or lightweight modules.

For robustness‑centric work
,RACE and AdvUnlearn are the key adversarially trained defenses today, but they are complementary to SPEED/MACE rather than replacements.

If your goal is to mirror the course instructions (trace the lineage, pick a frontier paper with public code/weights, then design experiments probing limitations), SPEED is a very natural candidate: it sits on top of ESD → UCE → MACE/RECE and explicitly claims to fix their efficiency and non‑target‑preservation limitations, while still leaving open questions about compositional prompts, robustness, and extreme scaling that you can probe experimentally.

Prepared by Deep Research
In conclusion: Could you independently verify the lineage claude has come up with. After that, you will do your own research to find the lineage and report back.
Completed 2 steps
Here’s the short version:

The lineage the other agent gave you is structurally correct for the main methods and branches (ESD → UCE/Concept Ablation → MACE/RECE → SPEED, plus robustness/localization branches), but a few method names and some detailed claims (e.g., certain acronyms and venue assignments) are not fully verifiable in the current literature.

Based on independent checks, the most defensible “frontier” global erasure method with open code/weights right now is SPEED, with ETC as a very recent, more aggressive scaling attempt whose ecosystem is still maturing.

Below I split this into (1) verification of what you were told, and (2) a cleaned lineage I’d be comfortable using in your project.

1. Independent verification of the earlier lineage
What matches the literature
These parts of the previous lineage do check out:

Pre‑ESD foundations.

Safe Latent Diffusion (SLD): Inference‑time steering using “unsafe” prompts to reduce nudity/violence without changing weights.

Concept Ablation (“Ablating Concepts in Text‑to‑Image Diffusion Models”): Fine‑tunes so a target concept (e.g., “Van Gogh”) matches an anchor distribution (“paintings”), effectively ablating the target.

ESD as the first practical weight‑editing eraser.

“Erasing Concepts from Diffusion Models” (ICCV 2023) fine‑tunes Stable Diffusion using negative classifier‑free guidance as the teacher, with two variants ESD‑x (cross‑attention) and ESD‑u (unconditional layers).

It explicitly positions itself as more permanent than SLD and cheaper than retraining on filtered data.

Closed‑form / efficiency branch.

UCE (Unified Concept Editing): training‑free/closed‑form concept editing in cross‑attention.

RECE (Reliable and Efficient Concept Erasure): ECCV 2024 method that edits embeddings in seconds without extra training.

SPEED (Scalable, Precise, and Efficient Concept Erasure): ICLR 2026; searches for a parameter‑space null‑space using IPF, DPA, IEC; erases up to 100 concepts in ≈5 seconds and reports a 350× speedup over prior SOTA in its own comparisons.

LoRA / mass‑erasure branch.

MACE (Mass Concept Erasure in Diffusion Models): CVPR 2024; combines closed‑form cross‑attention refinement with per‑concept LoRA and reports scaling to 100 concepts with good generality–specificity trade‑off.

Later multi‑LoRA scaling like DyME (Dynamic Multi‑Concept Erasure with bi‑level orthogonal LoRA) fits this same branch.

Robustness branch.

Attack work “On the Vulnerability of Concept Erasure in Diffusion Models” shows that many erased models (ESD/UCE/MACE‑style) remain vulnerable to adversarial prompt attacks.

RACE and AdvUnlearn explicitly frame themselves as adversarially trained defenses to improve robustness over vanilla erasers.

Localized erasure branch.

GLoCE (Gated Low‑rank Adaptation) and LACE (Localized‑Attention‑Guided Concept Erasure) are real CVPR/ OpenReview methods focusing on spatially localized concept removal with gated/adaptive modules.

Evaluation/theory branch.

“When Are Concepts Erased From Diffusion Models?” and “Fantastic Targets for Concept Erasure…” provide the theoretical and evaluation tools the earlier agent referenced.

ETC as “thousands of concepts.”

“Erasing Thousands of Concepts (ETC)” does indeed present a t‑mixture + MoEraser framework for scaling to thousands of concepts in embedding space.

So the backbone and major branches they gave you are well grounded in actual papers and code.

Where I’d disagree or soften claims
Some named methods/labels are not clearly traceable.
I can find solid references for UCE, MACE, RECE, SPEED, GLoCE, LACE, RACE, DyME, etc., but not for all acronyms mentioned previously (e.g., some of “ANT”, “SCORE”, “STEREO” in the exact roles claimed). Those should be treated as informal placeholders or speculative references, not canonical methods.

Details about venues and status.

SPEED is correctly described as accepted to ICLR 2026.

ETC is currently an arXiv 2026 preprint; the previous conversation calling it “CVPR 2026” gets ahead of the public record.

Quantitative speedup details.

SPEED’s own materials (OpenReview, arXiv, GitHub) confirm “100 concepts within 5 seconds” and “×350 speedup over SOTA” as headline results.

Earlier discussion sometimes loosely conflated “closed‑form” with SPEED’s null‑space optimization; in reality it’s a constrained optimization, not just a single matrix inversion.

So: the story you were given is directionally right and mostly accurate, but it occasionally over‑labels the very latest works and uses a few paper names that I can’t firmly tie to public sources. For your write‑up, I would stick to the verified set above.

2. A cleaned lineage from ESD to the current frontier
Here is a compact lineage that (a) matches the literature and (b) is safe to put on a slide or in your GitHub page.

Stage 0 – Pre‑ESD mitigation (2022–early 2023)
Filtered retraining (e.g., SD‑2.x line).
Retrain on “clean” data to remove unwanted content; expensive and incomplete.

Safe Latent Diffusion (SLD).

SLD uses an additional “unsafe” prompt and guidance at inference time to suppress nudity/violence on I2P, without modifying weights.

Cheap and deployable, but bypassable if users can alter the model or prompts.

Concept Ablation (Kumari et al.).

Fine‑tunes the model so the target concept distribution matches an anchor, effectively ablating the concept.

First explicit “concept erasure” by fine‑tuning but still per‑concept, with side‑effects.

Stage 1 – ESD (ICCV 2023) as the main starting point
Erasing Concepts from Diffusion Models (ESD).

Uses negative classifier‑free guidance as a teacher signal to fine‑tune Stable Diffusion so that the target concept follows the “negatively guided” behavior.

Two variants: ESD‑x (cross‑attention, prompt‑specific styles) and ESD‑u (unconditional, global concepts like nudity).

Solves: permanent, dataset‑free erasure that’s harder to bypass than SLD.

Leaves open: lexical evasion of styles, unconditional collateral damage, and adversarial prompt robustness.

Stage 2 – Major branches after ESD
From ESD, the literature diverges into four main technical branches, plus an evaluation branch.

2A. Efficiency / closed‑form editing
UCE (Unified Concept Editing).

Training‑free, closed‑form editing of cross‑attention projections for multiple concepts.

RECE (Reliable and Efficient Concept Erasure).

ECCV 2024; modifies concept embeddings in a few seconds without fine‑tuning, aiming for better robustness than UCE/ESD.

SPEED (Scalable, Precise, and Efficient Concept Erasure).

ICLR 2026; edits parameters in a null‑space where non‑target concepts are preserved, using IPF, DPA, and IEC.

Demonstrates erasure of 100 concepts within ≈5 seconds and ≈350× speedup over prior SOTA, with better non‑target preservation.

This branch is the efficiency‑first, training‑free/near‑training‑free line, and SPEED is its current frontier with open code and checkpoints.

2B. LoRA / mass multi‑concept erasure
MACE (Mass Concept Erasure in Diffusion Models).

CVPR 2024; combines cross‑attention refinement with per‑concept LoRA and a closed‑form integration step to scale erasure up to 100 concepts.

DyME (Dynamic Multi‑Concept Erasure).

Uses bi‑level orthogonal LoRA adapters to support dynamic composition of erased concept sets, reducing interference.

ETC (Erasing Thousands of Concepts).

arXiv 2026; uses t‑mixture modeling and a Mixture‑of‑Experts “MoEraser” to scale to thousands of concepts, with added robustness tricks.

This branch targets extreme concept counts using LoRA / modular adapters. MACE is the established CVPR method; DyME and ETC push the scale further, with ETC very new and not as mature in tooling.

2C. Robustness‑first methods
Attack analyses (e.g., “On the Vulnerability of Concept Erasure…”).
Show that models processed by ESD/UCE/MACE remain vulnerable to adversarial prompt search.

RACE (Robust Adversarial Concept Erasure).

ECCV 2024; adversarially trains against text embeddings that try to re‑elicitate erased concepts, reducing attack success on sensitive targets.

AdvUnlearn and follow‑ups.

Defensive unlearning with adversarial training for robust concept erasure.

These are defensive wrappers around erasure, not replacements for SPEED/MACE but complementary for robustness.

2D. Localized / spatial erasure
GLoCE.

CVPR 2025; training‑free gated Low‑rank LoRA that erases concepts only where they appear, preserving other regions.

LACE.

OpenReview 2025; localized‑attention‑guided masks and gated clean‑up for spatially confined erasure.

These methods focus on spatial precision, directly addressing the “global collateral damage” problem of ESD‑u and global MACE‑style edits.

2E. Evaluation and theory
When Are Concepts Erased From Diffusion Models?

Proposes conceptual models of erasure and a suite of probes (visual context, trajectory modifications, classifier guidance) to see what’s really happening.

Fantastic Targets for Concept Erasure…

Designs better “target concepts” in a concept graph to minimize side‑effects.

These are the diagnostic tools you’d use to evaluate ESD/MACE/SPEED/RECE/ETC rigorously.

3. Bottom line for your project
The earlier lineage you got is mostly accurate in structure and choice of key papers, especially around ESD → UCE → MACE/RECE → SPEED, plus GLoCE/RACE/etc. as specialized branches.

I would, however, drop or de‑emphasize any named methods that don’t show up clearly in current papers and be careful with unverified venue labels. Stick to the papers and repos we can point to directly (ESD, Concept Ablation, UCE, MACE, RECE, SPEED, GLoCE, LACE, RACE, DyME, ETC, evaluation works).

For a “frontier target paper” satisfying “recent, open code + weights, central in the lineage,” SPEED is a very strong choice; ETC is exciting but newer and less standardized.

Prepared by Deep Research
