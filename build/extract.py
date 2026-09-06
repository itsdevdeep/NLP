"""Extract + normalise the raw PDF corpus into data/processed/*.txt."""
import os, re, json, unicodedata, warnings
warnings.filterwarnings("ignore")
from pypdf import PdfReader

# Straight-quote / dash / ligature normalisation.
# NOTE: docs 15 and 21 ship broken font-to-Unicode maps. Doc 21 maps the
# ti/tt/ft/tf ligatures onto Latin-Extended-A codepoints
# (Informaĕon->Information, permiĥed->permitted, so├ware->software,
# plaĔorm->platform); doc 15 uses Ø and ✓ as list bullets.
# Left uncorrected these corrupt both the embeddings and any quoted answer.
REPL = {
    '‘':"'", '’':"'", '“':'"', '”':'"',
    '–':'-', '—':'-', '‐':'-', '‑':'-',
    '•':'* ', '●':'* ', ' ':' ',
    'ﬁ':'fi', 'ﬂ':'fl', '…':'...',
    'ĕ':'ti', 'Ĕ':'tf', 'ĥ':'tt', '├':'ft',
    '✓':'* ', 'Ø':'* ', '':'* ', '':'* ',
}

def clean(t):
    for a, b in REPL.items():
        t = t.replace(a, b)
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r'^[�-]\s*', '* ', t, flags=re.M)
    t = re.sub(r'[�-]', ' ', t)
    t = ''.join(c for c in t if c in '\n\t' or unicodedata.category(c)[0] != 'C')
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r' *\n *', '\n', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()

os.makedirs("data/processed", exist_ok=True)
meta = []
for fn in sorted(os.listdir("data/raw")):
    if not fn.endswith(".pdf"):
        continue
    r = PdfReader(os.path.join("data/raw", fn))
    parts = []
    for i, pg in enumerate(r.pages):
        c = clean(pg.extract_text() or "")
        if c:
            parts.append(f"[[page {i+1}]]\n{c}")
    doc = "\n\n".join(parts)
    open("data/processed/" + fn[:-4] + ".txt", "w", encoding="utf-8").write(doc)
    odd = sum(1 for c in doc if ord(c) > 127 and c not in "éüöí")
    meta.append(dict(file=fn, doc_id=fn[:2], slug=fn[3:-4], pages=len(r.pages),
                     words=len(doc.split()), chars=len(doc), odd_chars=odd))
json.dump(meta, open("data/processed/_manifest.json", "w"), indent=2)
print(f"{'doc':<40}{'words':>8}{'odd':>6}")
for m in meta:
    print(f"{m['slug'][:39]:<40}{m['words']:>8}{m['odd_chars']:>6}")
print("TOTAL words:", sum(m['words'] for m in meta),
      " residual odd chars:", sum(m['odd_chars'] for m in meta))
