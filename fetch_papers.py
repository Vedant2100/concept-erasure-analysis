import requests
from pypdf import PdfReader
import io

papers = {
    "ESD_paper": "https://arxiv.org/pdf/2303.07345.pdf",
    "SPEED_paper": "https://arxiv.org/pdf/2503.07392.pdf"
}

for name, url in papers.items():
    print(f"Downloading {name} from {url}...")
    response = requests.get(url)
    response.raise_for_status()
    
    print(f"Parsing {name}...")
    reader = PdfReader(io.BytesIO(response.content))
    text = []
    for i, page in enumerate(reader.pages):
        text.append(f"--- PAGE {i+1} ---")
        text.append(page.extract_text() or "")
        
    out_path = f"{name}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text))
    print(f"Saved to {out_path}\n")
