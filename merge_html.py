import re

with open("index.html", "r") as f:
    exp_html = f.read()

import subprocess
main_html = subprocess.check_output(["git", "show", "main:index.html"]).decode("utf-8")

# Extract head and intro from main
main_head_match = re.search(r'<head>.*?</head>', main_html, re.DOTALL)
main_head = main_head_match.group(0) if main_head_match else ""

main_body_content_match = re.search(r'<body>(.*?)<section id="footnotes"', main_html, re.DOTALL)
main_intro = main_body_content_match.group(1) if main_body_content_match else ""

# Extract deep dive from exp branch
exp_body_match = re.search(r'<main>(.*?)</main>', exp_html, re.DOTALL)
exp_main = exp_body_match.group(1) if exp_body_match else ""

# The footnotes from main
footnotes_match = re.search(r'<section id="footnotes".*?</section>', main_html, re.DOTALL)
footnotes = footnotes_match.group(0) if footnotes_match else ""

merged = f"""<!DOCTYPE html>
<html lang="en">
{main_head}
<body>
  <div class="container">
    {main_intro}

    <hr style="margin: 40px 0; border: 1px solid #ddd;">
    <h2 style="text-align:center; font-size: 2.5em; margin-bottom: 20px;">Deep Dive: Empirical Limitations</h2>

    <main>
      {exp_main}
    </main>

    {footnotes}
  </div>
</body>
</html>
"""

with open("index.html", "w") as f:
    f.write(merged)
