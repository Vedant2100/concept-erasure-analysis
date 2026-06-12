import re

with open("index.html", "r") as f:
    html = f.read()

new_toc = """        <h3>Contents</h3>
        <ul>
          <li><a href="#tldr">TL;DR</a></li>
          <li><a href="#lineage">1. Trace the Lineage & Find the Frontier</a></li>
          <li><a href="#how">2. Target Paper: SPEED</a></li>
          <li><a href="#strengths">3. Articulate Strengths</a></li>
          <li><a href="#experiments">4. Expose Limitations Through Experiments</a>
            <ul style="margin-top: 0.5rem; padding-left: 1rem; border-left: 2px solid var(--border-color); margin-left: 0.5rem;">
              <li><a href="#findings">Key Findings Summary</a></li>
              <li><a href="#the-question">The Question</a></li>
              <li><a href="#probe-1-sparse-multi-concept-erasure-speed-holds">Probe 1: Sparse Multi-Concept</a></li>
              <li><a href="#probe-2-concentrated-mass-erasure-the-limit-appears">Probe 2: Concentrated Mass Erasure</a></li>
              <li><a href="#ablation-the-refinement-contradiction">Ablation: Refinement Contradiction</a></li>
              <li><a href="#limitation-2-the-evasion-vulnerability-lexical-overfitting">Limitation 2: Evasion Vulnerability</a></li>
              <li><a href="#methodology-amp-pitfalls-we-corrected">Methodology & Pitfalls</a></li>
            </ul>
          </li>
        </ul>"""

html = re.sub(r'<h3>Contents</h3>.*?</ul>', new_toc, html, flags=re.DOTALL)

with open("index.html", "w") as f:
    f.write(html)
