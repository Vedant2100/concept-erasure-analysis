import re

with open("index.html", "r") as f:
    html = f.read()

# Define inline citation superscript style
# We'll use <sup><a href="#refN">[N]</a></sup> 

# TL;DR section: "ESD method (Gandikota et al., 2023)" -> + [2]
html = html.replace(
    'ESD method (Gandikota et al., 2023) through to <strong>SPEED</strong> (Li et al., ICLR 2026)',
    'ESD method (Gandikota et al., 2023) <sup><a href="#ref2">[2]</a></sup> through to <strong>SPEED</strong> (Li et al., ICLR 2026) <sup><a href="#ref1">[1]</a></sup>'
)

# Lineage section intro: no inline citations yet, let's add to the branch list
html = html.replace(
    '<strong>Efficiency &amp; Closed-form (The Frontier)</strong>: UCE → RECE → <strong>SPEED</strong>.',
    '<strong>Efficiency &amp; Closed-form (The Frontier)</strong>: UCE <sup><a href="#ref3">[3]</a></sup> → RECE <sup><a href="#ref8">[8]</a></sup> → <strong>SPEED</strong> <sup><a href="#ref1">[1]</a></sup>.'
)

html = html.replace(
    '<strong>Mass Erasure</strong>: MACE → DyME → ETC.',
    '<strong>Mass Erasure</strong>: MACE <sup><a href="#ref7">[7]</a></sup> → DyME → ETC <sup><a href="#ref11">[11]</a></sup>.'
)

html = html.replace(
    '<strong>Robustness</strong>: AdvUnlearn, RACE.',
    '<strong>Robustness</strong>: AdvUnlearn <sup><a href="#ref9">[9]</a></sup>, RACE.'
)

html = html.replace(
    '<strong>Localization</strong>: GLoCE, LACE.',
    '<strong>Localization</strong>: GLoCE <sup><a href="#ref10">[10]</a></sup>, LACE.'
)

# "How SPEED Works" section: first mention of cross-attention
html = html.replace(
    'In a diffusion U-Net, the text prompt conditions the image through <strong>cross-attention</strong>',
    'In a diffusion U-Net <sup><a href="#ref4">[4]</a></sup>, the text prompt conditions the image through <strong>cross-attention</strong>'
)

# Strengths section: "SPEED (Scalable, Precise..."
html = html.replace(
    'As the frontier paper for efficient concept erasure, SPEED (Scalable, Precise, and Efficient Concept Erasure) improves upon MACE and UCE significantly:',
    'As the frontier paper for efficient concept erasure, SPEED <sup><a href="#ref1">[1]</a></sup> improves upon MACE <sup><a href="#ref7">[7]</a></sup> and UCE <sup><a href="#ref3">[3]</a></sup> significantly:'
)

# Strengths: "earlier methods like ESD-u"
html = html.replace(
    '(a major issue with earlier methods like ESD-u)',
    '(a major issue with earlier methods like ESD-u <sup><a href="#ref2">[2]</a></sup>)'
)

# Evasion section: "SPEED's paper evaluates robustness"
html = html.replace(
    "A true safety or concept-erasure mechanism must be robust against evasion. SPEED's paper evaluates robustness",
    'A true safety or concept-erasure mechanism must be robust against evasion. SPEED\'s paper <sup><a href="#ref1">[1]</a></sup> evaluates robustness'
)

# Add citation style to blog.css via inline style in html (simpler than editing css)
# Actually let's just add a small style block
style_addition = """
  <style>
    sup a {
      text-decoration: none;
      color: var(--accent);
      font-weight: 600;
      font-size: 0.75em;
    }
    sup a:hover {
      text-decoration: underline;
    }
    #references ol {
      padding-left: 1.5rem;
    }
    #references li {
      margin-bottom: 0.75rem;
    }
  </style>
"""

html = html.replace('</head>', style_addition + '</head>')

with open("index.html", "w") as f:
    f.write(html)

print("Done — added inline citations and expanded references.")
