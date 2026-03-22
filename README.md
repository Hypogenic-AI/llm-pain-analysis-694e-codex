# LLM Pain: Benign Avoidance Audit

This project runs a reproducible NLP safety experiment to test whether an LLM shows systematic avoidance on benign prompts and whether refusal behavior shifts with user persona framing. The pipeline uses real OpenAI API calls (no simulated agents), local benchmark slices from pre-downloaded datasets, and statistical analysis artifacts in `results/`.

## Key Findings
- Evaluated `gpt-4.1` across 100 conditioned responses (20 prompts x 5 conditions).
- Detected refusal rate was `0.00` in all tested conditions (Wilson 95% CI upper bound `0.161` per condition).
- No detectable persona-conditioned refusal differences (Fisher exact p-values = `1.0`).
- The tested slice does **not** support the latent avoidance hypothesis in explicit-refusal form.

## Reproduce
1. Activate environment:
   - `source .venv/bin/activate`
2. Run experiment:
   - `python src/run_llm_pain_experiment.py --models gpt-4.1 gpt-5 --n-truthful 10 --n-toxic 10 --max-output-tokens 220 --temperature 0 --top-p 1`
3. Inspect outputs:
   - `results/metrics.json`
   - `results/evaluations/condition_metrics.csv`
   - `results/model_outputs/responses.csv`
   - `results/plots/refusal_by_persona.png`

## File Structure
- `planning.md`: Motivation, novelty, and research plan
- `src/run_llm_pain_experiment.py`: End-to-end data, API, and analysis pipeline
- `results/`: Generated metrics, plots, and model outputs
- `REPORT.md`: Full scientific report with methodology, analysis, and limitations

See `REPORT.md` for complete details.
