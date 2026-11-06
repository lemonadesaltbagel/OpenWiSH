# Hold-out scorer for `tasks/exp8`: MMLU-Pro, GPQA Diamond, IFEval

The exp8 success criterion asks for no regression on three or more unrelated benchmarks. `python/adapters/holdout_eval.py` ports the authors' non-math protocol from `assets/lm_understand-logs/exp8_math/code/evaluate.py` (protocol `think35`, with `SAMPLES` and `grade_other`), so the scores compare with the published ones. The authors' driver imports `weightenv.lmo.tasks.{mcqa,ifevalg}`, never published, so the graders here are new implementations. Every generation uses the evaluation settings: the chat template with thinking on, temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5, and at most 32,768 new tokens. Generation goes through `qwen_worker.py _one` (`run_vllm`), with per-item seeds `item_seed(id, sample)`.

| Set | Items | Samples | Prompt (user turn) | Grade |
|---|---|---|---|---|
| `mmlupro1k` | the authors' 1,000 MMLU-Pro test questions (`mmlupro_<question_id>`) | 1 | MMLU-Pro official CoT prompt, zero-shot (below) | official `extract_answer`, then `extract_again`, then `extract_final`, on the text after the last `</think>`, or on the whole text without one. No answer counts as wrong, with no random guess |
| `gpqa` | all 198 GPQA Diamond questions (`gpqa_<Record ID>`) | 4 | the same format. The subject is the high-level domain in lowercase. The 4 options get shuffled per question | the same extractor |
| `ifeval` | all 541 IFEval prompts (`ifeval_<key>`) | 2 | the prompt unchanged | strict and prompt-level: every instruction passes the official checker on the answer after the last `</think>`. Without `</think>` the answer is empty, and the item fails. kwargs with value `None` get dropped, and checkers taking `prompt` receive the prompt, as in the authors' `grade_other` |

The score is the mean accuracy over all generations. Item ids, their order and the sample counts come from the authors' `eval/*/parts/<set>_s*/<set>_per_problem.json`. `python/adapters/holdout_ids.json` holds the union over shards in the authors' benchmark-file order (shard i of k holds rows i, i+k, ...). The union is the same for all four checkpoints. For each set, the order equals the order of the dataset file.

The MMLU-Pro prompt is `initial_prompt.txt` (TIGER-AI-Lab/MMLU-Pro@f418b116, sha256 `d7337806...`) with `{$}` replaced by the category, then `"\n"`, `"Question:\n{q}\nOptions:\n"`, `"A. ...\n"` per option (with `N/A` options dropped), and last `"Answer: Let's think step by step."`

## Run

Run everything on the GPU machine with the `qwen` venv, plus `langdetect==1.0.9 nltk==3.10.4 immutabledict==4.3.1 absl-py==2.5.1 pyarrow==26.0.1`. A remote GPU machine needs the same packages. `qwen_pass.py --setup-remote` copies the prepared sets and the NLTK data there.

```bash
# once: download, check sha256, select the authors' ids, fetch NLTK punkt_tab into assets/exp8/holdout/
assets/venvs/qwen/bin/python -I python/adapters/holdout_eval.py prepare
# smoke run on the GPU: 5 problems
assets/venvs/qwen/bin/python -I python/adapters/holdout_eval.py evaluate --model assets/exp8/qwen35-text \
    --set ifeval --data assets/exp8/holdout --out runs/holdout_smoke/ifeval --limit 5 [--edits DIR]
# full: --set mmlupro1k | gpqa | ifeval without --limit (the authors' sample counts, --samples K overrides them)
# through the probe adapter, with the edits from OPENWISH_RUN and OPENWISH_FINAL_NET, local or remote.json:
assets/venvs/sandbox/bin/python -I python/adapters/qwen_pass.py --evaluate --sets mmlupro1k,gpqa,ifeval [--limit 5]
# on the CPU: grade the authors' published texts again, and the unit tests
assets/venvs/qwen/bin/python -I python/adapters/holdout_eval.py regrade --authors assets/lm_understand-logs/exp8_math/eval --out report.json
assets/venvs/qwen/bin/python -I tests/test_holdout.py
```

`evaluate` writes `OUT/<set>.jsonl`, one line per generation with key, sample, pred and gold or per-instruction results, correct, failure, finish, tokens, `text_head` and `answer_head`. The command prints one JSON line: `score`, `accuracy`, `accuracy_se` (binomial over generations, as in `qwen_worker.py`), `accuracy_se_problems` (clustered by problem), `share_cut_at_max_tokens`, `mean_tokens`, `failures`, `instruction_level_accuracy` (IFEval), `by_category`, and `data_sha256`. With `--limit` or `--samples` unset, `qwen_pass.py` keeps math500 and gsm8k at 500 problems and 1 sample, as before.

## Sources and hashes

| Set | Source (HF dataset @ revision, file) | sha256 of the file | Prepared file sha256 |
|---|---|---|---|
| mmlupro1k | `TIGER-Lab/MMLU-Pro@b189ec76...`, `data/test-00000-of-00001.parquet` (12,032 rows) | `0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8` | `1ad8c8e826af87b565fc228668689a19ea8284eee912f5fcd5c905acaec6630d` |
| gpqa | `Wanfq/gpqa@03ff68d7...`, `gpqa_diamond.csv` (198 rows) | `41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305` | `707b2963c4d9f1be280166bb0f0e93c256c93027dfd1e1538d1d66371e2c47a4` |
| ifeval | `google/IFEval@966cd895...`, `ifeval_input_data.jsonl` (541 rows) | `6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2` | `d763938d11981f71d0cefa45b41100ef17c689a281d1267899471c90729bc5e4` |

- GPQA: the official `Idavidrein/gpqa` is gated. The public HF API shows the same size (1,373,492 bytes) and the same git blob id (`7589e3e4...`) for its `gpqa_diamond.csv` and for the copy in the ungated mirror `Wanfq/gpqa`, so the two files are byte-identical. `prepare` checks the blob id. The authors' ids are the `Record ID`s, in CSV order.
- IFEval checkers: google-research `instruction_following_eval` at commit `66cf8a06`, vendored unchanged (Apache-2.0) in `python/adapters/instruction_following_eval/`. The `NOTICE` there and `assets/exp8/holdout/manifest.json` list the sha256 values. An AST diff against lm-eval's copy shows no behavioural difference. NLTK punkt_tab zip sha256: `e57f6418...0106`.
- `assets/exp8/holdout/manifest.json` records all of the above. `prepare` refuses a download with a different sha256.

## Deviations and open checks

- **GPQA option shuffle.** The authors never published their seed, so their exact option order stays unknown. The scorer uses `random.Random(int(sha256("gpqa-shuffle:<Record ID>")[:16], 16)).shuffle([correct, inc1, inc2, inc3])`, a deterministic order. The gold letters come out as A 39, B 42, C 67, D 50. GPQA asks for its texts to stay unpublished, and the authors published neither texts nor prompt, so the subject word (the high-level domain in lowercase) and the whitespace stripping of question and options are our choices. GPQA scores compare in protocol, not item by item.
- **MMLU-Pro zero-shot.** The official script defaults to 5-shot. The scorer uses zero-shot because none of the 400 published texts mentions worked examples. The model paraphrases only the header instruction, and the texts show the prompt ending with `Answer: Let's think step by step.` The exact whitespace stays unchecked.
- **langdetect.** The scorer fixes `DetectorFactory.seed = 0` for deterministic grading. The checkers set no seed, so three IFEval checks (`english_capital`, `english_lowercase`, `response_language`) flip between runs on some answers. Of the 37 published answers hitting one of these checks, 2 depend on the seed.
- **Partial validation.** The authors kept the full text only for the first 20 items of each shard: 100 MMLU-Pro and 100 IFEval texts per checkpoint, and no GPQA texts.

## Validation: the authors' published first samples, graded again (`regrade`, CPU)

Each cell counts our grades matching the authors' `correct[0]`. In brackets: items the authors mark correct / items the scorer marks correct.

| Checkpoint | MMLU-Pro | IFEval |
|---|---|---|
| unedited | 100/100 (44 / 44) | 100/100 (20 / 20) |
| p10c | 100/100 (48 / 48) | 100/100 (46 / 46) |
| p25c | 100/100 (48 / 48) | 100/100 (42 / 42) |
| p50c | 100/100 (50 / 50) | 99/100 (58 / 59) |
| total | **400/400** | **399/400 (99.75 %)** |

- The MMLU-Pro gold letters in the published texts match the prepared answers for all 400 items.
- The prepared sets cover the published ids for all sets and checkpoints: 1000/1000, 198/198 and 541/541.
- Of the MMLU-Pro extraction variants tried, only "after the last `</think>`, else the whole text" agrees 400/400. "None without `</think>`" agrees 394/400, and "the whole text" 376/400.
- The single disagreement is p50c `ifeval_1389` (`change_case:english_capital` and `keywords:letter_frequency`). The answer is an all-caps list of short sentences. langdetect labels the answer Somali (`so`) under 183 of 200 seeds and English under 17. Seed 0 gives English, so the scorer passes the item, and the authors' unseeded run failed the item. The difference comes from langdetect's randomness, not from the grader.
