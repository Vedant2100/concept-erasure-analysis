# YouTube Code Walkthrough: Testing the Limits of Concept Erasure (SPEED)

## 0:00 – Introduction
Hey everyone, welcome back. Today, we're doing a deep dive into the code behind our latest research project. We've been analyzing a state-of-the-art concept erasure algorithm called **SPEED**—which claims to safely remove concepts from Stable Diffusion without damaging anything else. 

The core of our project was asking: *Does its mathematical guarantee actually hold up under extreme pressure?* Spoiler alert: it doesn't. We discovered two major vulnerabilities: **Rank Saturation** and **Lexical Overfitting**. I’m going to walk you through exactly how we wrote the code to prove this.

## 1:15 – Repository Overview
Let's take a look at the repo structure.
- **`SPEED_repo/`**: This is the engine. It contains the original unlearning implementation from the SPEED authors. 
- **`exp1/`**: This directory contains all the code for our "Rank Saturation" limits test, where we aggressively targeted an entire artistic movement.
- **`exp2/`**: This houses the "Lexical Overfitting" code, where we tested if the model actually destroyed the visual style, or if it just hid it behind a text filter.
- **`index.html` & `report.md`**: Our full write-ups and visual grids.

## 2:30 – Exp 1: The Rank Saturation Test (The Waterbed Effect)
Let's open up `exp1/scripts/`. This directory actually contains the full chronological history of our probing. Before running the final massive test, we built automatic downloaders in the `setup/` folder and wrote early sanity checks in the `early_tests/` folder. These proved that SPEED *does* work perfectly when erasing just 1 to 3 concepts. 

But to break it, we scaled up. Let's look at `slurm_rank_saturation.sh`. 

Instead of erasing a few random concepts, we wrote bash arrays that dynamically build nested lists of Impressionist painters—from N=5 up to N=40. The critical part is that we intentionally held out "canary" concepts like Pissarro and Gauguin. We wanted to see what happens to them when we erase everyone around them.

All our image generations are tied to `neighbor_prompts.json`, guaranteeing every grid is 100% deterministic. Once the models are trained using `SPEED_repo/train_erase_null.py`, we run `probe_rank_saturation.py`. This script utilizes the HuggingFace `diffusers` library to load the base model alongside the edited LoRA weights and generate a grid of images for our canaries.

Finally, we run `analyze_rank_saturation.py`. This script takes the baseline images and the erased images, passes them through a CLIP image encoder, and calculates the cosine distance between them. This is how we mathematically proved that as you erase 40 concepts, Pissarro's CLIP drift monotonically doubles. 

We also have a script called `slurm_dpa_ablation.sh` which turns off SPEED's built-in Prior Augmentation feature, proving that their own protection mechanism actually starves the mathematical capacity and makes the "waterbed effect" worse.

## 4:30 – Exp 2: Lexical Overfitting (The Paraphrase Bypass)
Now let's switch to `exp2/scripts/`. Here we wanted to see if SPEED actually erased the *visual features* of an artist like Van Gogh, or if it just overfit to the text token "Van Gogh".

If you open `slurm_paraphrase.sh`, you'll see we train SPEED alongside an older unlearning method called **ESD-x** for comparison.

Then we hit `probe_paraphrase.py`. We feed the models two types of prompts:
1. **Explicit Prompts**: "a painting in the style of Van Gogh"
2. **Paraphrased Prompts**: "a painting of golden wheat fields under a turbulent blue sky with heavy expressive brushwork"

When we run `analyze_paraphrase.py` over the outputs, the math is staggering. On the explicit prompt, SPEED works flawlessly. But on the paraphrased prompt, it leaks 100% of the style—generating images mathematically identical to the baseline. It proved that SPEED is acting as a brittle text filter, whereas the older ESD-x successfully destroyed the visual features.

## 5:45 – Wrap Up
And that's the code! By writing these targeted probing scripts and custom CLIP analyzers, we were able to expose 5 interlocking mathematical failures in the algorithm. 

All of this code, along with the HTML blog post rendering the image grids, is available on our GitHub repository on the `master` branch. You can clone it, run the Slurm scripts yourself, and reproduce the exact mathematical collapse we documented. 

Thanks for watching, check the link in the description for the repo, and I'll see you in the next one.
