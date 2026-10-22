# Project plan: Shortcut reliance in NLI — Logistic Regression vs. RoBERTa

**Task:** Do a simple model (TF-IDF + logistic regression) and a pretrained NLI model (`roberta-large-mnli`) rely on the same word-level shortcuts in MultiNLI, and do explanation methods (SHAP) show the same words driving both?

Everything lives in one notebook, `nli_shortcuts.ipynb`. Expensive results (RoBERTa predictions, SHAP values) are cached to `results/` so the notebook can be rerun without recomputing.

**Status:** Steps 1–7 are done (EDA, logistic regression, RoBERTa inference, quantitative results, SHAP for both models, explanation comparison in notebook section 7, qualitative examples in section 8). Add-on B (counterfactual negation test) is done in section 9. Next: write the report; optional quick checks are the LR `coef × tfidf` check and add-on A.

---

## 0. Decisions to know before starting

These are methodological details that are easy to get wrong and worth a sentence each in the report.

1. **Evaluation split = `validation_matched`** (9,815 examples). MultiNLI test labels are not public, and `roberta-large-mnli` was already trained on the MNLI *train* split, so we only train LR on `train` and evaluate both models on `validation_matched`.
2. **Label order differs!** The dataset uses `0=entailment, 1=neutral, 2=contradiction`. `roberta-large-mnli` uses `0=CONTRADICTION, 1=NEUTRAL, 2=ENTAILMENT`. Always map via `model.config.id2label`, and sanity-check that full-input RoBERTa accuracy is ~90%. If it's ~30%, the mapping is wrong.
3. **Two different kinds of "hypothesis-only".** This is the most important point to explain clearly in the report:
   - *Hypothesis-only training* (what Poliak/Gururangan did): train a model that only ever sees the hypothesis. High accuracy means **the dataset contains artifacts**. Cheap for LR, so we do it.
   - *Premise ablation at test time*: take the full-input model and feed it an empty premise. This measures **how this particular model behaves without the premise**. This is the only feasible option for RoBERTa-large (no retraining), and we do it for LR too so the comparison is fair.
4. **LR features: separate vectorizers for premise and hypothesis**, rather than one TF-IDF over concatenated text. Then every feature is either `P:word` or `H:word`, and we can measure how much of each model's decision comes from the premise versus the hypothesis. With one vectorizer over concatenated text, the premise word "not" and the hypothesis word "not" would be the same feature.
5. **One explanation method for both models.** Use SHAP (KernelSHAP over word masks, see §3) on both models, with the same word-level tokenization. Then any difference in top words comes from the models and not from the explainer. LR's exact linear contributions (`coef × tfidf`) are a free sanity check.
6. **Fix the proposal wording:** one place says "BERT" but it should say RoBERTa.
7. **Negation is defined linguistically, then frozen.** `NEGATION_WORDS` covers sentential negation: `not`/`n't`/`cannot`, the negative indefinites `no, nobody, nothing, none, nowhere, never`, and `neither, nor` (Huddleston 2005; Haspelmath 1997; Kane 2021). Do **not** add or drop words based on how well they predict contradiction. That would make the negation result circular. Excluded on purpose: affixal negation (*un-, dis-*), inherently negative verbs (*refuse, fail, deny*). Known limitation: word matching also counts constituent negation ("not far").

**Compute:** LR runs fine on a laptop. RoBERTa-large inference on ~10k pairs (×2 conditions) plus SHAP needs a GPU, so use Google Colab (free T4 is enough) or IDUN. Cache all RoBERTa outputs to disk right away.

---

## 1. Core plan (minimum deliverable)

### Step 1 — Setup and data (small) ✅ done
- `pip install datasets transformers torch scikit-learn shap pandas matplotlib seaborn`
- Load `nyu-mll/multi_nli`. No `label == -1` filtering is needed: the Hugging Face version already excludes the pairs without a gold label (392,702 / 9,815 / 9,832 examples). The paper reports about 1.8% of validation pairs without one.
- EDA done in the notebook (section 2):
  - Label distribution for all three splits: train exactly balanced, both validation splits ~35% entailment / 32% neutral / 33% contradiction. Majority-class baseline ≈ 0.35.
  - Mean lengths (train): premise 19.8 words, hypothesis 10.0.
  - **Negation artifact table across all three splits** (one table; the separate train-only table was merged into it): 22% of hypotheses contain negation, and 56–63% of those are contradictions (vs. ~25% without negation). The cue is strongest in `validation_mismatched` (63%).
  - Raw counts for `validation_matched`: 2,142 negated hypotheses, of which 937 are *not* contradictions (496 entailment, 441 neutral). These are the cases where the shortcut misleads.
  - **Per-word negation table:** no-series words (*never, none, nothing, no*) are strong cues (~76–79% contradiction). *not*/*n't* are much weaker (46–51%) but make up most occurrences.
  - **Most label-revealing hypothesis words** (`MIN_DF = 500`): contradiction = negation + absolute words (*completely, hate*) + negative polarity items (*any, anything*); neutral = added reasons/details (*also, because, due*) and opinions (*popular, prefer, better*); entailment = weakest cues (≤ 0.53), hedged words (*some, sometimes, may*). *refused* (0.81, n=498) is a strong contradiction cue outside the negation list.
  - **Examples per genre** (`figures/examples_per_genre.png`): each genre is ~20% of its split (train ≈ 77k per genre, telephone 83k; validation ≈ 1.95–1.98k per genre). Train and matched share 5 genres, mismatched has 5 unseen ones.
- **Negation word list:** see §0.7.

### Step 2 — Logistic regression (small)
- `TfidfVectorizer` (unigrams + bigrams, `min_df=2`, lowercase, **no stopword removal**, because stopword lists delete "not"/"no"!), one fit on premises and one on hypotheses, then `hstack`.
- Train two LR models on `train`:
  - `LR-full`: premise + hypothesis features
  - `LR-hyp`: hypothesis features only (the Poliak/Gururangan-style hypothesis-only baseline)
- Tune `C` lightly (e.g. {0.1, 1, 10}) on a held-out slice of train. Don't overdo it.
- Report the top-20 most positive coefficients per class for `LR-full`, split into `P:` and `H:` features. You'll likely see negation words at the top for contradiction.
- ✅ **Done.** Notes:
  - The held-out 10% must be split **by premise** (`promptID`, `GroupShuffleSplit`). Each premise appears once per label, so a random split gives misleading held-out accuracy (0.41 vs. 0.55 on validation). The notebook has a check cell that shows this.
  - Best `C = 1` for both. Models cached in `results/lr_full.pkl`, `results/lr_hyp.pkl`.
  - Validation_matched accuracy: majority 0.354, LR-hyp 0.544, LR-full 0.556 (mismatched almost identical).
  - 59/60 top coefficients are `H:` features. Contradiction: *never* (10.9), *no* (10.7) ≫ *not* (4.8).

### Step 3 — RoBERTa predictions (small–medium, GPU)
- Load `roberta-large-mnli` and run batched inference (batch 32–64, `torch.no_grad()`, fp16 on GPU) on `validation_matched` for:
  - full input `(premise, hypothesis)`
  - premise ablated: `("", hypothesis)`
- Also run `LR-full` with the premise ablated (zero premise features).
- Save the logits to `results/roberta_*.npy`.
- ✅ **Done** in notebook section 4 (RoBERTa 0.906 matched / 0.901 mismatched; without premise 0.371 / 0.380, predicts neutral 93% of the time), for both validation splits (matched + mismatched, which also covers add-on C). Run it on Colab (instructions in the notebook), then copy the four `roberta_*.npy` files to the local `results/`. Label order is confirmed reversed in the model config (`0 = CONTRADICTION`) and handled in the code. LR-full with premise removed is not done yet; it fits best in the results section (Step 4).

### Step 4 — Quantitative comparison (small)
Main results table:

| Model | Input at test time | Accuracy | Macro-F1 |
|---|---|---|---|
| Majority class | – | ~0.35 | |
| LR-full | full | | |
| LR-full | premise removed | | |
| LR-hyp | hypothesis only (trained that way) | | |
| RoBERTa-large-mnli | full | | |
| RoBERTa-large-mnli | premise removed | | |

Plus:
- **Confusion matrices** for each premise-removed condition. What does each model default to without a premise?
- **Negation subset analysis** (cheap and very telling): split the validation set by "hypothesis contains a negation word". For each model and condition, report accuracy and **P(predict contradiction)** on each subset, compared with the true contradiction rate. This directly answers RQ1.
- ✅ **Done** in notebook section 5 (table saved to `results/main_results.csv`). Key numbers (validation_matched):
  - Removing the premise: LR-full 0.556 → 0.531 (−2.5 points), RoBERTa 0.906 → 0.371 (−53 points).
  - P(predict contradiction | negated hypothesis): true 56%, LR-hyp 73%, LR-full 71%, RoBERTa 58%.
  - Traps (937 negated non-contradictions): LR-full 33% correct (49% called contradiction), RoBERTa 86% correct (7%).
  - Without premise: LR-full still calls 65% of negated hypotheses contradiction, RoBERTa 21% (3% without negation).
  - Figures: `negation_predicted_contradiction.png`, `confusion_matrices.png`.

### Step 5 — SHAP explanations (medium, GPU) ✅ done
- Sample **~300 validation examples, stratified by label**, with half of them containing a negation word in the hypothesis so the negation analysis has enough data. Use a fixed seed. Make sure the sample includes both strong (*never, no, nothing*) and weak (*not, n't*) negation words, since their cue strength differs a lot (Step 1).
- For each example, split premise and hypothesis into words. Each word is one feature, and premise words are kept separate from hypothesis words.
- Prediction function `f(mask_matrix)`: rebuild the premise and hypothesis from the words where `mask == 1` (masked words are deleted), then return the model's class probabilities. Write one wrapper for LR and one for RoBERTa.
- `shap.KernelExplainer(f, np.zeros((1, n_words)))`, `nsamples≈300–500`, and explain the **model's predicted class**.
- Save SHAP values per example to `results/shap_{model}.pkl`.
- Sanity check: for LR, the SHAP rankings should broadly agree with `coef × tfidf`. *(Not done yet.)*
- ✅ **Done** in notebook section 6. Choices: words = whitespace split; sample = 50 per (label × negation) = 300, only pairs with ≤ 40 words (7,883 of 9,815 qualify); `nsamples = 500`; explains the probability of each model's own predicted label; baseline = all words deleted. SHAP values verified to add up to the model's probability. Cached in `results/shap_lr.pkl` (local, 9 s) and `results/shap_roberta.pkl` (Colab).
  - **Note:** shap 0.54's default `l1_reg="num_features(10)"` keeps only the 10 most important words per example and sets all others to exactly 0. All Step 6 measures therefore use each example's top 10 words. Mention this in the method section.
  - A first look (one trap, end of section 6): LR's prediction of contradiction is driven by "no" in "There is no doubt…" (SHAP +0.58); RoBERTa gives "no" 0 and answers entailment from *undoubtedly*, *hand*, *Brown*.

> Why KernelSHAP over word masks rather than `shap.maskers.Text`? NLI has *two* inputs. A custom word mask is the simplest way to keep the premise/hypothesis boundary intact and to use identical features for both models.
> **Fallback if this is too slow:** leave-one-word-out occlusion (drop each word, measure the probability change). It's ~30 forward passes per example and easy to explain.

### Step 6 — Explanation comparison (small) ✅ done
Per example, then aggregated over the 300:
- **Premise share**: `sum|SHAP| over premise words / sum|SHAP| over all words`. Show a box plot per model. This tests whether RoBERTa uses the premise more.
- **Negation dominance**: among examples with negation in the hypothesis, how often is a negation word the #1 (or top-3) word for each model, and in which direction (toward contradiction?). Show a 2×2 table: both models / only LR / only RoBERTa / neither.
- **Globally most important words**: mean |SHAP| per word across the sample, top-15 for each model, shown side by side.
- ✅ **Done** in notebook section 7, with an interpretation after each result and a section summary. Key numbers (300-example sample; LR 52% / RoBERTa 92% accuracy on it):
  - **Premise share:** median 14% (LR) vs. 36% (RoBERTa); RoBERTa higher in 77% of examples. Agrees with the premise ablation in Step 4.
  - **Negation dominance** (negation word = top-1): negated contradictions LR 84% / RoBERTa 94% (correct to focus on it there); **traps LR 60% / RoBERTa 12%**, LR-only in 51 of 100 traps. When negation is LR's top-1 on a trap, it answers contradiction 77% of the time; trap accuracy LR 30% vs. RoBERTa 87%.
  - **Global top-15 words:** 11 of 15 shared, *never/nothing/no* at the top for both. Only RoBERTa's list has premise negation (`P:didn't`, `P:not`, …); LR's has more non-negation hypothesis cues (*because, all, any, most, but*). The global ranking hides the trap difference, because it mixes contradictions (where negation matters) with traps (where it misleads). **Key point for the discussion:** compare explanations per example and per group, not only globally.
  - **Answer to the second half of the RQ:** SHAP shows negation words driving both models, but only for LR do they also drive wrong answers.
  - Figures: `shap_premise_share.png`, `shap_top_words.png`.
  - Open: bootstrap confidence intervals for the main percentages would strengthen the report. The RoBERTa trap subgroup with negation as top-1 is only 12 examples, so don't draw strong conclusions from it.

### Step 7 — Qualitative examples (small) ✅ done
Pick 4–6 clear cases with a SHAP text plot or a simple coloured bar chart:
- both models wrong, with "not" dominating (the shortcut in action)
- LR wrong because of "not", RoBERTa right (and its attribution is on the premise)
- a case where RoBERTa still falls for the shortcut
- a negation-free contradiction, to show what else drives the predictions
- ✅ **Done** in notebook section 8 (`figures/shap_qualitative_examples.png`), four examples plus the trap from section 6:
  1. *The door did not budge.* → *The door didn't move.* (entailment): LR wrong from *didn't* (+0.65); RoBERTa right, top words premise *not* and *door*.
  2. *…to buy cowboy boots.* → *There were not cowboy boots at the store.* (neutral): both wrong, *not* top-1 for both. RoBERTa only gets 4 traps in the sample wrong, and both "both wrong" and "RoBERTa falls for the shortcut" are these same 4, so one example covers both bullets.
  3. *hot day* → *very cold day* (contradiction): both right; RoBERTa uses *hot*/*cold*, LR uses *very* and gives *hot* 0 (no cross-sentence features).
  4. *Some rooms* → *All of the rooms* (contradiction): both rely on *All*; only RoBERTa also uses premise *Some*.

**At this point you have a complete report.**

---

## 2. Figures and tables for the report (core)

1. ✅ Share of contradictions for hypotheses with vs. without negation, per split (`figures/negation_contradiction_by_split.png`). *Motivates the problem.*
2. ✅ Main accuracy / macro-F1 table (Step 4, `results/main_results.csv`).
3. ✅ Confusion matrices for the premise-removed conditions (`figures/confusion_matrices.png`, with and without premise).
4. ✅ Bar chart: P(predict contradiction) on the negation and no-negation subsets, per model and condition (`figures/negation_predicted_contradiction.png`).
5. Top LR coefficients per class (P vs. H features).
6. ✅ Premise-share box plot, LR vs. RoBERTa (`figures/shap_premise_share.png`).
7. ✅ Side-by-side global top-15 SHAP words (`figures/shap_top_words.png`).
8. Negation-dominance tables (contradictions vs. traps) from Step 6. These are the strongest SHAP result, so put them in the report as a table.
9. ✅ Qualitative SHAP examples (`figures/shap_qualitative_examples.png`).
10. ✅ Counterfactual negation test (`figures/counterfactual_negation.png`).

---

## 3. Add-ons (only if time allows, ordered by value per effort)

| # | Add-on | Effort | What it adds to the report |
|---|---|---|---|
| A | **Hard/easy split (Gururangan et al.)**: define "hard" = examples `LR-hyp` gets *wrong*, then report both full-input models' accuracy on hard vs. easy. | Very low (data you already have) | Shows how much each model's accuracy comes from easy, artifact-laden examples. Strong, well-known result. |
| B | **Counterfactual negation test**: on ~200 hypotheses with a negation word, delete it (and on non-negated ones, insert "not" in simple cases). Measure how often each model's prediction flips to/from contradiction. | Low | Causal evidence to go with the correlational SHAP results. |
| C | **`validation_mismatched`**: rerun Step 4 on the other genres. | Very low | Do the shortcuts generalise across genres? |
| D | **LIME vs. SHAP agreement** on a subset of examples. | Low–medium | Are the conclusions robust to the choice of explainer? |
| E | **Other shortcut types**: lexical overlap (Jaccard of premise/hypothesis words vs. P(entailment)), and hypothesis length vs. label. | Low | Broadens "shortcut" beyond negation. |
| F | **HANS** evaluation (McCoy et al. 2019) of both models. | Medium | Well-known challenge set for overlap heuristics. |
| G | **Middle-sized model**: a smaller fine-tuned MNLI model (e.g. `roberta-base` / DistilRoBERTa MNLI checkpoints on HF), to show the trend with model capacity. | Medium | Turns a 2-point comparison into a capacity trend. |
| H | **Fine-tune a hypothesis-only RoBERTa** (e.g. roberta-base, 1 epoch). | High (GPU hours) | Hypothesis-only *training* for the transformer too, making it fully comparable to LR-hyp. |

**Add-on B status:** set up in notebook section 9. Edit rules: delete the negation word from hypotheses with exactly one (1,953 examples), and insert "not" after the first *be*/modal verb in hypotheses without negation (4,292). ✅ **Done**, with interpretation in the notebook (`figures/counterfactual_negation.png`, `results/roberta_counterfactual_*.npy`). Key test: does the direction of the change depend on the edit (shortcut) or on the label (reading the premise)?
  - **LR follows the edit:** deleting negation lowers its contradiction rate for every label (51→13%, 43→13%, 87→28%), and inserting "not" raises it for every label (11→58%, 14→52%, 41→76%).
  - **RoBERTa follows the meaning:** entailment + delete negation 1→83% contradiction (LR goes *down* 51→13%); contradiction + insert "not" 91→32% (LR goes *up* 41→76%).
  - **Neutral is less clean than expected:** RoBERTa rises in both directions (12→38%, 5→46%). That's not the shortcut pattern, but it also doesn't show RoBERTa ignores negation. Likely cause: related neutral hypotheses can become real contradictions, plus unnatural edits. The original claim that "neutral should stay neutral" was too strong and has been corrected in the notebook.
  - Limitations: true labels of edited examples are unknown; some edits are ungrammatical (e.g. negative polarity *any* left behind); only rule-editable examples are covered.

Suggested order if time allows: **B → A**. B comes first now because the SHAP results are correlational, and B is the most direct way to show that negation *causes* LR's wrong answers on traps. A is cheap and gives a well-known extra result. C is partly done: RoBERTa logits exist for `validation_mismatched` (Step 3) and LR accuracy is almost identical there (Step 2), so only the negation subset analysis on mismatched is missing. G is the best way to address the "two models at opposite extremes" confound if there is time.

---

## 4. Notebook structure

```
nli_shortcuts.ipynb
  0. Config (seeds, paths, NEGATION_WORDS, device)
  1. Data loading (dataset description)
  2. Exploratory Data Analysis
  3. Logistic regression (LR-full, LR-hyp)
  4. RoBERTa inference (cached)
  5. Quantitative results (tables, confusion matrices, negation subsets)
  6. SHAP (cached)
  7. Explanation comparison
  8. Qualitative examples
  9. Add-ons
results/      # cached .npy / .pkl / figure .png files
figures/      # final figures for the report
```

Note: notebook section numbers are one higher than the plan's step numbers from Step 2 on, because EDA has its own section.

Tip: put the `results/` caching behind `if os.path.exists(...)` checks so the notebook can be run top to bottom on a laptop once the GPU parts are done.

---

## 5. Suggested report outline

1. **Introduction**: NLI, annotation artifacts, research questions.
2. **Related work**: Gururangan et al. 2018, Poliak et al. 2018, McCoy et al. 2019 (HANS), SHAP (Lundberg & Lee 2017), LIME (Ribeiro et al. 2016).
3. **Data and models**: MultiNLI (Williams et al. 2018, NAACL, https://aclanthology.org/N18-1101/; §2.1 for how it was collected: Hybrid crowdsourcing platform, 387 workers, three hypotheses per premise, five-way validation), the negation definition (§0.7; Huddleston 2005, Haspelmath 1997, Kane 2021), LR setup (separate P/H vectorizers), roberta-large-mnli.
4. **Method**: hypothesis-only training vs. premise ablation (§0.3), SHAP setup, comparison metrics.
5. **Results**: quantitative results → explanations → qualitative examples.
6. **Discussion**: does capacity reduce shortcut reliance? Frame the contribution as the explanation comparison (the shortcut itself is known from Gururangan/Poliak/McCoy), and make the point that global SHAP rankings suggest more agreement than per-group analysis shows. Limitations: premise ablation is out-of-distribution for RoBERTa; word deletion in SHAP also produces ungrammatical, out-of-distribution inputs; SHAP approximations (500 samples, only the top 10 words per example kept); sample size of 300, drawn 50% negated by design; LR and RoBERTa differ in pretraining, architecture and size at once, so we can't say which causes the difference; and attribution ≠ causation (unless add-on B is done).
7. **Conclusion**.

---

## 6. Rough effort split (two people)

| Phase | Person A | Person B |
|---|---|---|
| Steps 1–2 | Data + EDA + negation artifact table | LR models + coefficients |
| Steps 3–4 | RoBERTa inference on Colab | Results tables + negation subset analysis |
| Steps 5–6 | SHAP wrappers + run | Comparison metrics + figures |
| Step 7 + add-ons | Qualitative examples | Add-on A/B |
| Report | Intro, related work, method | Results, discussion |
