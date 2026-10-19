# Run the LLM stability experiments in VS Code

Experiment L1/L2 from `PLAN.md`: ask a model the same credit question many ways, measure how often the
decision changes, which kinds of applicants are affected, and how much voting over k answers helps.

## 1. One-time setup
1. Install VS Code, Python 3.10+, and git. In VS Code install the **Python** extension.
2. Open a terminal (Terminal > New Terminal) and run:
   ```bash
   git clone https://github.com/bgra1123/ragmm.git
   cd ragmm
   git checkout claude/ragmm-product-content-s3k2po      # work on the branch, not main
   python -m venv .venv
   source .venv/bin/activate                              # Windows: .venv\Scripts\activate
   pip install numpy pandas scikit-learn
   ```
3. Command Palette (Ctrl/Cmd+Shift+P) > **Python: Select Interpreter** > choose `.venv`.
4. `cd research/llm`

## 2. Check the pipeline (30 seconds, no model)
```bash
python llm_stability.py --backend mock --n 60 --samples 10
```
Results go to `results/mock_none/` (`SUMMARY.md`, `summary.json`, `raw.csv`). The mock numbers mean nothing;
this only proves the pipeline runs.

## 3. Real run, option A: local open model (free, no API key)
Needs ~8 GB RAM for a 1.5B model; a GPU is optional but faster.
```bash
pip install torch transformers accelerate
python llm_stability.py --backend hf --model Qwen/Qwen2.5-1.5B-Instruct --n 20 --samples 20 --orders 2   # small check first
python llm_stability.py --backend hf --model Qwen/Qwen2.5-1.5B-Instruct --n 300 --samples 20 --orders 5   # pre-registered run
```
The first run downloads the weights from Hugging Face. On CPU expect roughly 1-3 s per prompt; the full run is
300 applicants x 15 prompt variants (3 instructions x 5 field orders), so start with `--n 20` and scale up.
Try a larger model (e.g. `Qwen/Qwen2.5-7B-Instruct`) only with a GPU.

How randomness is produced here: for each prompt the script reads the model's probability of answering
"A" (approve) vs "B" (deny) and draws the answer at temperature `--temperature` (default 0.7). So
instability comes from (a) sampling and (b) rewording the instruction / reordering the fields.

## 4. Real run, option B: hosted model through an API
```bash
pip install anthropic
export ANTHROPIC_API_KEY=...        # Windows PowerShell: $env:ANTHROPIC_API_KEY="..."
python llm_stability.py --backend anthropic --model <model id> --n 100 --samples 10 --orders 2
```
This makes n x 6 x samples calls (100 x 6 x 10 = 6,000). Start with `--n 10` to check cost.

## 5. Reading the output (`results/<tag>/SUMMARY.md`)
- **Share unstable**: applicants whose answer differs from their majority answer in >=10% of runs.
- **Single run error vs majority error**: what randomness costs, and what voting recovers.
- **Error by voting size k**: how many answers you need to ask for before gains stop.
- **Agreement-of-5 AUC**: whether cheap agreement across 5 answers finds the unstable applicants (H3 wants >= 0.8).
- **Group table**: chance of a wrong decision by type of applicant (same groups as `research/groups`), and the part
  due to randomness. Groups under 15 applicants are dropped; no groups are added after seeing results.

Compare with the pre-registered hypotheses H1-H3 in `PLAN.md`; report misses as misses.

## 6. Agents (L3), next step
Not built yet: it needs the L1 result first. The plan is a scripted tool-use task with a checkable answer, run
repeatedly, measuring per-step disagreement vs trajectory success. Ask Claude to build it once L1 shows a signal.

## 7. Saving results
Keep `results/` out of git unless small. Commit `SUMMARY.md` and `summary.json` only:
```bash
git add research/llm/results/*/SUMMARY.md research/llm/results/*/summary.json
git commit -m "LLM L1 results"
git push -u origin claude/ragmm-product-content-s3k2po
```
Do not push to `main`. Statements are about model behaviour on a public dataset, not accusations of bias and not credit advice.
