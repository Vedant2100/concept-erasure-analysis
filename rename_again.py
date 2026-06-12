import re

with open("index.html", "r") as f:
    html = f.read()

# Replace H2s
html = html.replace('1. Trace the Lineage & Find the Frontier', '1. Research Lineage & The Frontier')
html = html.replace('3. Articulate Strengths', '3. Algorithmic Strengths')
html = html.replace('4. Expose Limitations Through Experiments', '4. Empirical Limitations')
html = html.replace('Probe 2 — Concentrated Mass Erasure (The Limit Appears)', 'Limitation 1: Rank Saturation Collapse (Probe 2)')
html = html.replace('id="probe-2-concentrated-mass-erasure-the-limit-appears"', 'id="probe-2"')
html = html.replace('Limitation 2: The Evasion Vulnerability (Lexical Overfitting)', 'Limitation 2: Evasion Vulnerability (Lexical Overfitting)')
html = html.replace('id="limitation-2-the-evasion-vulnerability-lexical-overfitting"', 'id="limitation-2"')

# Regenerate the TOC
toc = """        <h3>Contents</h3>
        <ul>
          <li><a href="#tldr">TL;DR</a></li>
          <li><a href="#lineage">1. Research Lineage & The Frontier</a></li>
          <li><a href="#how">2. Target Paper: SPEED (Scalable, Precise, and Efficient Concept Erasure)</a></li>
          <li><a href="#strengths">3. Algorithmic Strengths</a></li>
          <li><a href="#deep-dive-empirical-limitations">4. Empirical Limitations</a>
            <ul style="margin-top: 0.5rem; padding-left: 1rem; border-left: 2px solid var(--border-color); margin-left: 0.5rem;">
              <li><a href="#the-question">The Question</a></li>
              <li><a href="#probe-1-sparse-multi-concept-erasure-speed-holds">Probe 1: Sparse Multi-Concept</a></li>
              <li><a href="#probe-2">Limitation 1: Rank Saturation Collapse</a></li>
              <li><a href="#why-only-one-neighbor-failed">Why Only One Neighbor Failed</a></li>
              <li><a href="#ablation-the-refinement-contradiction">Ablation: Refinement Contradiction</a></li>
              <li><a href="#robustness-fidelity-vs-identity">Robustness: Fidelity vs. Identity</a></li>
              <li><a href="#limitation-2">Limitation 2: Evasion Vulnerability</a></li>
              <li><a href="#methodology-amp-pitfalls-we-corrected">Methodology & Pitfalls</a></li>
            </ul>
          </li>
        </ul>"""

html = re.sub(r'<h3>Contents</h3>.*?</ul>', toc, html, flags=re.DOTALL)

with open("index.html", "w") as f:
    f.write(html)
