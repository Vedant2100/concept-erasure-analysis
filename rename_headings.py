import re

with open("index.html", "r") as f:
    html = f.read()

# Replace the headings
html = html.replace('id="lineage">Research Lineage</h2>', 'id="lineage">1. Trace the Lineage & Find the Frontier</h2>')
html = html.replace('id="how">How SPEED Works</h2>', 'id="how">2. Target Paper: SPEED (Scalable, Precise, and Efficient Concept Erasure)</h2>')
html = html.replace('id="strengths">Strengths: What SPEED Does Well</h2>', 'id="strengths">3. Articulate Strengths</h2>')
html = html.replace('>Deep Dive: Empirical Limitations</h2>', '>4. Expose Limitations Through Experiments</h2>')

# Now regenerate the TOC part
# First, extract all h2s
toc_items = []
for match in re.finditer(r'<h2[^>]*id="([^"]+)"[^>]*>(.*?)</h2>', html):
    id_val = match.group(1)
    text = re.sub(r'<[^>]+>', '', match.group(2)) # strip internal tags
    
    # Clean up text for TOC (remove "Probe 1 — ", "Probe 2 — ", etc)
    if "Expose Limitations" in text: continue
    
    toc_items.append(f'<li><a href="#{id_val}">{text}</a></li>')

toc_html = """        <h3>Contents</h3>
        <ul>
          """ + "\n          ".join(toc_items) + """
        </ul>"""

# Replace the existing TOC
# We find everything between <div class="toc-sticky"> and </div>\n    </nav>
html = re.sub(r'<h3>Contents</h3>.*?</ul>', toc_html, html, flags=re.DOTALL)

with open("index.html", "w") as f:
    f.write(html)
