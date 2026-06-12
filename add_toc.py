import re

with open("index.html", "r") as f:
    html = f.read()

# Make sure we don't duplicate the TOC
if '<nav class="toc-sidebar">' in html:
    print("TOC already exists")
    exit(0)

# Find all <h2> tags and ensure they have IDs
def repl_h2(match):
    tag_start = match.group(1) # e.g. <h2
    attrs = match.group(2) # e.g. id="how"
    content = match.group(3) # e.g. How SPEED works
    
    if 'id=' not in attrs:
        # Create a slug
        slug = re.sub(r'[^a-z0-9]+', '-', content.lower()).strip('-')
        if not slug:
            slug = "section"
        return f'{tag_start} id="{slug}"{attrs}>{content}</h2>'
    return match.group(0)

# Replace h2s
new_html = re.sub(r'(<h2)([^>]*?)>(.*?)</h2>', repl_h2, html)

# Generate TOC
toc_items = []
for match in re.finditer(r'<h2[^>]*id="([^"]+)"[^>]*>(.*?)</h2>', new_html):
    id_val = match.group(1)
    text = re.sub(r'<[^>]+>', '', match.group(2)) # strip internal tags
    
    # Clean up text for TOC (remove "Probe 1 — ", "Probe 2 — ", etc)
    if "Deep Dive:" in text: continue
    
    toc_items.append(f'<li><a href="#{id_val}">{text}</a></li>')

toc_html = """
  <div class="layout-wrapper">
    <nav class="toc-sidebar">
      <div class="toc-sticky">
        <h3>Contents</h3>
        <ul>
          """ + "\n          ".join(toc_items) + """
        </ul>
      </div>
    </nav>
    <div class="main-content">
"""

# Replace the first `<div class="container">` with the new wrapper
# Actually, the file has `<div class="container">` right after `<body>`
new_html = new_html.replace('<div class="container">', toc_html, 1)

# The end of the file has `</div>\n</body>`
# We need to close the main-content div and the layout-wrapper div
# Since we replaced container (1 div) with wrapper+main-content (2 divs), we need to add an extra closing div.
new_html = new_html.replace('</body>', '  </div>\n</body>')

with open("index.html", "w") as f:
    f.write(new_html)
