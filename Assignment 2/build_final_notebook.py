"""Assembles the final submission notebook from A_D_member1.ipynb (Parts A, E-data, D) and
BCE_member2.ipynb (Parts B, C, E-BLEU), adds Part E analysis, Part F and the merged references.

Run:  .venv/bin/python build_final_notebook.py
      .venv/bin/jupyter nbconvert --to notebook --execute --inplace PS4_GroupN_AIMLZG521_SMT_Assignment.ipynb
The written observations and the Part F reflection live in final_notebook_text.py (filled in at build time).
"""
import json
import nbformat as nbf

FINAL = 'PS4_GroupN_AIMLZG521_SMT_Assignment.ipynb'
A = json.load(open('A_D_member1.ipynb', encoding='utf-8'))['cells']
B = json.load(open('BCE_member2.ipynb', encoding='utf-8'))['cells']
src = lambda c: ''.join(c['source'])

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip('\n')))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip('\n')))
def take(cell, replace=None):
    s = src(cell)
    for old, new in (replace or []):
        assert old in s, f'not found: {old[:60]}'
        s = s.replace(old, new)
    (code if cell['cell_type'] == 'code' else md)(s)

# ------------------------------------------------------------------------------------------ header
md(r'''
# PS4 — Statistical Machine Translation: English → German (News / Media)
**AIML ZG521 NLP Applications · Assignment 2 · Part 1 (Programming)**

| Problem Statement | Group | Student name | Student ID |
|---|---|---|---|
| PS 4 | `<Group No>` | Devdeep Dasgupta | `<BITS ID>` |
| | | Ahamed Imthias | `<BITS ID>` |
| | | Vishvajith | `<BITS ID>` |

### Problem statement
A news agency wants to translate wire stories and headlines from **English to German** to syndicate content to
European partners. We build (1) IBM Model 1 word alignment from scratch, (2) a phrase-based SMT system with an
n-gram language model and our own decoder, and (3) compare it against a pretrained neural MT model. We evaluate
BLEU, adequacy, fluency, OOV handling, latency and robustness on the domain's stress-test focus: headlines with
dropped articles and verbs, long German compounds, named entities and quotations, and date and place expressions.

### Notebook structure
| Section | Content |
|---|---|
| 0 | Configuration (seed, sizes, model settings) |
| A | Corpus collection, cleaning, tokenization, split and analysis; evaluation and stress sets |
| B | IBM Model 1 with EM, written from scratch in both directions: convergence, top-5 tables, alignment heatmaps, limitations |
| C | Phrase-based SMT: symmetrization, phrase extraction, phrase table, Kneser–Ney trigram LM, **own stack decoder**, beam sweep, examples |
| D | Pretrained NMT (`Helsinki-NLP/opus-mt-en-de`) and the SMT-vs-NMT comparison table and charts |
| E | BLEU written from scratch (validated against sacreBLEU and NLTK), evaluation-set results, human ratings, stress-test error analysis |
| F | Reflection and deployment recommendation |
| Ref | References |

> **Notation (noisy channel).** The assignment's decoder score is
> `score(e) = λ1·log P(f|e) + λ2·log P_LM(e)`. Here **e is the German output and f the English input**, so the
> translation model in the decoder is **P(English | German)** and the language model is a German LM.

No API keys, passwords or access tokens are used anywhere; the dataset and the NMT model are public.
''')

# ------------------------------------------------------------------------------------------ config
md('## 0. Configuration')
take(A[2], [
    ("import os, re, json, time, random, unicodedata, warnings\nfrom collections import Counter",
     "import os, re, json, math, time, random, logging, unicodedata, warnings\nfrom collections import Counter, defaultdict"),
    ("from sacremoses import MosesTokenizer\nimport sacrebleu\n",
     "from sacremoses import MosesTokenizer, MosesDetokenizer\nimport sacrebleu\n"),
    ("warnings.filterwarnings('ignore')",
     "warnings.filterwarnings('ignore')\nlogging.getLogger('sacrebleu').setLevel(logging.ERROR)   # silence sacreBLEU tokenization hints"),
    ("SEED = CONFIG['SEED']",
     "# Part C settings.\nSMT = dict(SYMMETRIZATION='grow-diag-final', TABLE_LIMIT=20, DISTORTION_LIMIT=4,\n"
     "           KN_DISCOUNT=0.75, N_DEV=300, OOV_PENALTY=-10.0)\nSEED = CONFIG['SEED']"),
])

# ------------------------------------------------------------------------------------------ Part A
for c in A[3:16]:
    take(c)
take(A[16], [('## Evaluation set and robustness stress set (data for Part E)',
              '### A.7 Evaluation set and robustness stress set (used in Parts C–E)')])
take(A[17]); take(A[18])

# ------------------------------------------------------------------------------------------ Part B + C
assert src(B[5]).startswith('## Part B') and src(B[53]).startswith('## Part E')
take(B[5])   # B.1 markdown
code(r'''
# Parts B and C read the files written in Part A, exactly as the two tracks were developed.
C = CONFIG
DETOK = MosesDetokenizer(lang='de')

def read_tok(name):
    return [line.split() for line in (DATA / name).read_text(encoding='utf-8').splitlines()]

def read_lines(name):
    return (DATA / name).read_text(encoding='utf-8').splitlines()

train_en, train_de = read_tok('train.tok.en'), read_tok('train.tok.de')
test_en, test_de = read_tok('test.tok.en'), read_tok('test.tok.de')
test_raw_en, test_raw_de = read_lines('test.raw.en'), read_lines('test.raw.de')
lm_heldout = read_tok('lm_heldout.tok.de')
assert len(train_en) == len(train_de) == C['N_TRAIN'] and len(test_en) == len(test_de) == C['N_TEST']
same = sum(preprocess(r, 'en') == t for r, t in zip(test_raw_en, test_en))
print(f'train {len(train_en):,} / test {len(test_en)} / LM held-out {len(lm_heldout)} sentences; '
      f'preprocess() reproduces test.tok.en on {same}/{len(test_en)} sentences')
''')
for c in B[6:53]:
    take(c, [("Outputs are detokenized with `MosesDetokenizer(lang='de')`, one line per input.",
              "Outputs are detokenized with `MosesDetokenizer(lang='de')`, one line per input, and are read by Part D.")]
         if 'C.8 Final system outputs' in src(c) else None)

# ------------------------------------------------------------------------------------------ Part D
for c in A[19:26]:
    take(c, [("SMT outputs are produced by Member 2's pipeline (Part C) and dropped into `outputs/` as",
              "SMT outputs are produced by the Part C pipeline and written to `outputs/` as"),
             ("Until those files exist the SMT row shows *pending*.\n", ""),
             ("(template created below).", "(template created below if missing; scores are discussed in E.3).")]
         if src(c).startswith('### D.1') else None)
md('### D.2 Observations')
md('%%D_OBS%%')

# ------------------------------------------------------------------------------------------ Part E
md(r'''
## Part E — Evaluation dataset, metrics and robustness analysis
The evaluation set (15 sentences, 3 per category) and the stress set (8 sentences) were built in A.7. This part
implements BLEU from scratch (E.1), validates it (E.2), then compares SMT and NMT on the evaluation set (E.3), the
human ratings (E.4) and the stress set with error categories (E.5).
''')
take(B[53], [('## Part E — BLEU implemented from scratch', '### E.1 BLEU implemented from scratch')])
take(B[54])
take(B[55], [('### E.1 Validation against sacreBLEU and NLTK', '### E.2 Validation against sacreBLEU and NLTK')])
take(B[56]); take(B[57]); take(B[58])

md(r'''
### E.3 Evaluation set: SMT vs NMT per sentence and per category
Sentence BLEU uses **our own** `sentence_bleu` (add-1 smoothing), on the shared tokenization, lowercased.
''')
code(r'''
eval_ref_tok = [preprocess(s, 'de') for s in eval_set.de_ref]
eval_tbl = eval_set[['id', 'category', 'en', 'de_ref']].assign(
    smt=smt_out['eval'], nmt=nmt_out['eval'],
    smt_bleu=[round(sentence_bleu(preprocess(h, 'de'), r), 1) for h, r in zip(smt_out['eval'], eval_ref_tok)],
    nmt_bleu=[round(sentence_bleu(preprocess(h, 'de'), r), 1) for h, r in zip(nmt_out['eval'], eval_ref_tok)])
cat_tbl = eval_tbl.groupby('category', sort=False).apply(lambda g: pd.Series({
    'SMT corpus BLEU (ours)': corpus_bleu([preprocess(h, 'de') for h in g.smt], [eval_ref_tok[i] for i in g.index]),
    'NMT corpus BLEU (ours)': corpus_bleu([preprocess(h, 'de') for h in g.nmt], [eval_ref_tok[i] for i in g.index]),
    'SMT mean sentence BLEU': g.smt_bleu.mean(), 'NMT mean sentence BLEU': g.nmt_bleu.mean()})).round(1)
display(eval_tbl)
cat_tbl
''')

md(r'''
### E.4 Human ratings: adequacy and fluency (1–5)
40 sentences (25 random test sentences and the 15 evaluation sentences) were rated for **adequacy** (how much of
the source meaning is preserved: 5 = all, 1 = none) and **fluency** (how natural the German is regardless of
meaning: 5 = flawless, 1 = incomprehensible). SMT output is judged without penalising lowercasing. The ratings
were drafted with AI assistance and must be checked by the group before submission; all scores are in
`outputs/ratings.csv`.
''')
code(r'''
ratings = pd.read_csv(OUT / 'ratings.csv')
rating_cols = ['smt_adequacy', 'smt_fluency', 'nmt_adequacy', 'nmt_fluency']
summary = ratings.groupby('set')[rating_cols].mean()
summary.loc['all (40)'] = ratings[rating_cols].mean()
eval_ratings = ratings[ratings.set == 'eval'].merge(eval_set[['id', 'category']], on='id')
by_cat = eval_ratings.groupby('category', sort=False)[rating_cols].mean()

fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
summary.loc['all (40)'].rename(lambda c: c.replace('_', ' ')).plot.bar(
    ax=ax[0], rot=0, color=['#9ecae1', '#3182bd', '#fdae6b', '#e6550d'])
ax[0].set(title='Mean human rating, 40 sentences', ylabel='1-5', ylim=(0, 5.5))
for p in ax[0].patches:
    ax[0].annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
by_cat.plot.bar(ax=ax[1], rot=15, color=['#9ecae1', '#3182bd', '#fdae6b', '#e6550d'])
ax[1].set(title='Mean rating by evaluation category (3 sentences each)', ylabel='1-5', ylim=(0, 5.5))
plt.tight_layout(); plt.savefig(FIG / 'E_human_ratings.png', dpi=120); plt.show()
print('Rating distribution (count of each score):')
display(pd.DataFrame({c: ratings[c].value_counts().reindex(range(1, 6), fill_value=0) for c in rating_cols}))
summary.round(2)
''')

md(r'''
### E.5 Robustness stress test: error categories
Each output was tagged with one or more of: *wrong lexical choice, reordering error, dropped word,
hallucinated/inserted word, untranslated word, none* (`outputs/stress_error_tags.csv`).
''')
code(r'''
tags = pd.read_csv(OUT / 'stress_error_tags.csv')
display(tags[['id', 'phenomenon', 'en', 'de_ref', 'smt', 'smt_error', 'nmt', 'nmt_error']])

CATS = ['wrong lexical choice', 'reordering error', 'dropped word', 'hallucinated/inserted word', 'untranslated word', 'none']
def count_tags(col):
    return pd.Series(Counter(t.strip() for cell in tags[col] for t in cell.split(';'))).reindex(CATS, fill_value=0)
tag_counts = pd.DataFrame({'SMT': count_tags('smt_error'), 'NMT': count_tags('nmt_error')})
ax = tag_counts.plot.barh(figsize=(9, 4), color=['#e6550d', '#3182bd'])
ax.invert_yaxis(); ax.set(title='Stress set (8 sentences): sentences showing each error type', xlabel='sentences')
plt.tight_layout(); plt.savefig(FIG / 'E_stress_errors.png', dpi=120); plt.show()
tag_counts
''')
md('### E.6 Observations')
md('%%E_OBS%%')

# ------------------------------------------------------------------------------------------ Part F
md('## Part F — Reflection and deployment recommendation')
md('%%F_REFLECTION%%')

# ------------------------------------------------------------------------------------------ references
refs = src(A[27])
refs = refs.replace(
    "**Software:**",
    "18. Koehn, P. (2004). Pharaoh: A Beam Search Decoder for Phrase-Based Statistical Machine Translation Models. *AMTA 2004*. https://doi.org/10.1007/978-3-540-30194-3_13\n"
    "19. Och, F. J. (2003). Minimum Error Rate Training in Statistical Machine Translation. *ACL 2003*. https://aclanthology.org/P03-1021/\n"
    "20. Chen, S. F., & Goodman, J. (1999). An Empirical Study of Smoothing Techniques for Language Modeling. *Computer Speech & Language* 13(4).\n"
    "21. Lin, C.-Y., & Och, F. J. (2004). ORANGE: a Method for Evaluating Automatic Evaluation Metrics for Machine Translation. *COLING 2004*. https://aclanthology.org/C04-1072/\n"
    "22. Chen, B., & Cherry, C. (2014). A Systematic Comparison of Smoothing Techniques for Sentence-Level BLEU. *WMT 2014*. https://aclanthology.org/W14-3346/\n\n"
    "**Software:**")
assert '18. Koehn' in refs
md(refs)

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nbf.write(nb, FINAL)
print(f'wrote {FINAL}: {len(cells)} cells')
import subprocess, sys
subprocess.run([sys.executable, 'final_notebook_text.py', FINAL], check=True)
