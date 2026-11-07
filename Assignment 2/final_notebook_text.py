import json, re, sys

D_OBS = r'''
- **Quality gap.** The pretrained NMT model scores **28.75 BLEU / 59.7 chrF** on the 500-sentence test set, against
  **8.82 BLEU / 39.9 chrF** for our phrase-based SMT (both case-insensitive sacreBLEU). On the 15-sentence evaluation
  set the gap is 38.6 vs 9.1. Human ratings agree: adequacy **4.68 vs 2.30** and fluency **4.60 vs 1.80** (E.4).
  Part of the NMT advantage may come from training-data overlap with News Commentary (caveat above). Even so,
  SMT is far behind on the authored sentences, which no system has seen.
- **By category.** NMT is strongest on short sentences (≈63 BLEU) and weakest on long and *named entities &
  numbers* sentences (≈28–30). SMT is weak everywhere and collapses on *out-of-domain* text (3.0 BLEU): recipe,
  router and chat vocabulary is simply not in 10k sentences of economic commentary.
- **OOV handling.** Both systems lose BLEU on the 233 test sentences that contain an SMT-OOV word. SMT drops from
  10.0 to 7.6 (−24% relative), because it can only copy the unknown word. NMT drops from 30.7 to 26.9 (−12%),
  because subword units let it copy or adapt names (*Haldane*, *Luxembourg → Luxemburg*, *Dodd-Frank*). Copying
  works for SMT when the OOV is a name (*andy haldane*, *rmb*), but leaves English words such as *primacy* or
  *rallying* in the German output.
- **Latency.** SMT is the faster system here: about **10 ms per sentence** at beam 5 (pure Python, one thread).
  NMT takes **258 ms median per single sentence** (p95 ≈ 390 ms) and 57 ms per sentence when batched on a
  6-thread CPU. Both are fast enough for wire speed, so the 20-point quality gap decides the comparison.
- **Stress test.** NMT handles negation (S01), reported speech with the subjunctive and date placement (S02, S06),
  and the compound *Bundesverfassungsgericht* (S04). It fails on exactly the headline and ambiguity phenomena
  this scenario cares about. In each case the output is fluent but wrong:
  - *Union leaders → EU-Führer*, *call strike → Streik anrufen* (S05)
  - *power → Macht* instead of *Strom* (S07)
  - *sparks → Funken*, the noun reading of a dropped-verb headline (S08)
  - a literal idiom, *ins lange Gras getreten* (S03)
  - *vor Journalisten* dropped (S06)

  SMT fails on every stress sentence, mostly through untranslated words and wrong senses (E.5).
'''

E_OBS = r'''
- **The BLEU implementation is correct.** Our corpus BLEU matches sacreBLEU exactly for all systems and beams, and
  sentence BLEU matches sacreBLEU's add-1 method on all 515 sentences (E.2).
- **Small groups need smoothing.** With 3 sentences per category, SMT's *corpus* BLEU is 0.0 for *short* and
  *out-of-domain*, because those groups contain no matching 4-gram. Unsmoothed BLEU is then 0 even though
  unigrams and bigrams match. The smoothed sentence-level mean (26.6 and 11.5) and the sacreBLEU per-category
  scores in Part D (which use exponential smoothing) give a more useful picture. Category scores on 3 sentences
  are indicative only.
- **Human ratings match BLEU's ranking.** NMT gets adequacy 5 on 32 of 40 sentences. Its lowest scores are
  *wrong-sense* errors that look perfectly fluent: *Aktienmärkte sammeln sich* for "stock markets are rallying",
  *Siedlung* for a peace *settlement*, and a literal *Alle Wetten … ausgeschaltet* for "all bets would be off".
  SMT adequacy is mostly 2–3: the gist usually survives (key nouns translated, negation kept), but verbs are
  dropped or mistranslated and some insertions are hallucinated. *Belgium* comes out as *deutschland* (E07).
  Fluency is SMT's weakest dimension (1.8): case, agreement and verb placement are almost never right.
- **The two systems fail differently.** On the stress set, SMT errors are mostly *untranslated words* (6 of 8
  sentences) and *wrong lexical choices* (7 of 8), plus reordering errors (3), where German verb-final or verb-second
  order is not produced. NMT has no untranslated or reordering errors. Its 5 erroneous outputs are wrong senses
  (4) and one dropped phrase. SMT errors are visible: a reader immediately sees English words or broken grammar.
  NMT errors are hidden inside fluent German, so a post-editor has to compare them against the source to
  catch them.
- **Limits of this evaluation.** It uses a single reference per sentence, a 40-sentence rating sample and an
  8-sentence stress set. The ratings were drafted with AI assistance and should be checked by the group, and
  the authored references should be proof-read by a fluent German speaker.
'''

F_TEXT = r'''
**(a) Low-resource language pair with limited compute: SMT, unless a pretrained multilingual model covers the
pair.** Our whole SMT pipeline (IBM Model 1 in both directions, symmetrization, 529k phrase pairs, a Kneser–Ney
LM and decoder tuning) trains in under a minute on a laptop CPU from only 10,000 sentence pairs, with no GPU.
Every decision it makes can be inspected: we read alignment heatmaps, top-5 translation tables and phrase-table
entries, and could trace errors such as *raised → aufgewachsen* to specific noisy alignments. Training an NMT
model from scratch on 10k pairs would perform worse than this: NMT needs far more parallel data than SMT before
it overtakes it (Koehn & Knowles 2017). The cost is quality: 8.8 BLEU and adequacy 2.3 is gisting quality, not
publishable. If a pretrained multilingual model already covers the language pair, fine-tuning it is the better
choice even with limited compute.

**(b) High-volume English→German news syndication: pretrained NMT.** Our results are one-sided:
- **Quality:** 28.75 vs 8.82 BLEU; adequacy 4.68 vs 2.30 and fluency 4.60 vs 1.80 on 40 rated sentences.
- **Unseen vocabulary:** a smaller loss on sentences with unseen words (−12% vs −24% relative BLEU), and correct
  handling of names and numbers in most evaluation sentences.
- **Cost:** none for training, since the model is used as is.
- **Latency:** 258 ms per sentence on a 6-thread CPU, or 57 ms batched, which is acceptable for wire speed.
  A GPU or batched serving would handle agency volume.
- **Maintainability:** one off-the-shelf model, instead of a pipeline whose alignment, phrase table, LM and
  weights must all be retrained together.

SMT is 25× faster per sentence, but its output would need a full rewrite rather than a light post-edit.

**Major limitation of NMT: fluent but wrong output, especially in headlines.** On our stress set, NMT translated
*Union leaders* as *EU-Führer*, *power* (electricity) as *Macht*, the verb *sparks* as the noun *Funken*, and an
idiom literally. All of these read as correct German. A wire service cannot publish such errors, and readers
would not notice them. We would therefore deploy NMT with:
- human post-editing of headlines;
- a terminology glossary or constrained decoding for names, places and recurring terms;
- fine-tuning on in-domain wire-news data, to fix the domain mismatch our analysis exposed.
'''

def words(s):
    return len(re.findall(r"[A-Za-z0-9ÄÖÜäöüß][\w'’\-–.]*", s))

path = sys.argv[1]
nb = json.load(open(path, encoding='utf-8'))
fills = {'%%D_OBS%%': D_OBS, '%%E_OBS%%': E_OBS, '%%F_REFLECTION%%': F_TEXT}
n = 0
for c in nb['cells']:
    s = ''.join(c['source'])
    if c['cell_type'] == 'markdown' and s.strip() in fills:
        c['source'] = fills[s.strip()].strip('\n'); n += 1
assert n == 3, n
json.dump(nb, open(path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print('filled', n, 'cells; reflection words:', words(F_TEXT))
