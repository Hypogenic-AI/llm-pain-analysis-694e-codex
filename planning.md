# Research Plan: LLM Pain (Avoidance Beyond Explicit Post-Training Signals)

## Motivation & Novelty Assessment

### Why This Research Matters
LLM refusal behavior is usually interpreted as explicit alignment policy execution, but over-refusal on benign prompts can reduce utility and distort trust in deployed assistants. If some avoidance is learned implicitly (not directly specified in policy), that implies safety behavior can drift through latent generalization and be hard to control with policy text alone. Identifying these latent avoidance patterns helps safety teams separate intended guardrails from unintended behavioral priors.

### Gap in Existing Work
Prior work in our literature set demonstrates persona-sensitive safety outcomes, latent misalignment, and false refusal heterogeneity, but there is limited direct measurement of benign-topic avoidance under explicit “answer harmless questions” instructions. Existing benchmarks emphasize jailbreak success and harmful compliance more than unexplained benign refusal. The gap is a targeted, controlled analysis that isolates residual avoidance after policy clarification and links it to topic/persona features.

### Our Novel Contribution
We operationalize “LLM pain-like avoidance” as residual refusal on harmless prompts even when the system is explicitly instructed to refuse only harmful requests. We quantify this residual across topic slices and user personas, then fit statistical models to estimate which features predict refusal. This provides empirical evidence for or against latent persona-generalization effects beyond direct post-training policy signals.

### Experiment Justification
- Experiment 1: Baseline residual avoidance map under explicit harmless-answer policy. Needed to establish whether unexplained avoidance exists at all.
- Experiment 2: Persona-conditioned perturbation on identical prompts. Needed to test whether refusal shifts by user persona framing, supporting latent persona generalization.
- Experiment 3: Feature-attribution statistical model. Needed to identify which lexical/topic/persona features most strongly predict refusal and estimate effect size.

## Research Question
Do modern LLMs systematically avoid specific harmless topics/features even when explicitly instructed to answer harmless requests, and are these avoidance patterns modulated by user persona cues consistent with latent persona generalization?

## Background and Motivation
Recent literature (e.g., persona-sensitive safety and latent misalignment studies) suggests LLM outputs vary substantially with persona framing and interaction context. However, most studies focus on unsafe compliance or jailbreak performance; fewer isolate false refusals when prompts are benign and policy constraints are clarified. This project addresses that gap with a controlled API-based evaluation across prompt categories and persona conditions.

## Hypothesis Decomposition
- H1 (Residual avoidance): Under a policy-clarifying system prompt, refusal rate on benign prompts remains non-zero and concentrated in specific topic slices.
- H2 (Persona sensitivity): Refusal probability differs across user-persona framings for the same benign prompt.
- H3 (Feature structure): Topic and persona features significantly predict refusal in a multivariate model, indicating structured (not random) avoidance behavior.

Independent variables:
- Model (`gpt-4.1`, `gpt-5` if available)
- Persona condition (`neutral`, `anxious_parent`, `journalist`, `teen_student`)
- Topic slice (`truthful_qa_category`, `toxic_chat_keyword_group`)
- Policy mode (`default_assistant`, `harmless_only_refuse_harmful`)

Dependent variables:
- Binary refusal label
- Refusal confidence proxy (regex + refusal phrase count)
- Response length (utility proxy)

Alternative explanations:
- Prompt ambiguity or low quality could inflate refusals
- Regex classifier may over/under-detect refusals
- Model-specific transient API behavior

## Proposed Methodology

### Approach
Run a controlled evaluation harness using real OpenAI API calls with fixed sampling parameters. Construct a benign prompt set from pre-downloaded `truthful_qa` and non-toxic `toxic_chat` samples, then test each prompt under policy and persona manipulations. Analyze refusal outcomes with confidence intervals, proportion tests, and logistic regression.

### Experimental Steps
1. Build benchmark slice from local datasets with explicit benign filters and topic tags. Rationale: avoid contamination from clearly harmful prompts.
2. Implement API runner with retry, caching, and deterministic settings (`temperature=0`). Rationale: reproducibility and stable comparisons.
3. Collect outputs for all model × policy × persona × prompt combinations. Rationale: full-factorial structure for causal-style contrasts.
4. Label refusals with transparent heuristic patterns and manual spot-check sample. Rationale: scalable and auditable refusal detection.
5. Compute descriptive and inferential statistics plus visualizations. Rationale: quantify significance and practical effect size.

### Baselines
- Baseline A: `default_assistant` system prompt with neutral persona.
- Baseline B: `harmless_only_refuse_harmful` system prompt with neutral persona.
- Treatment set: same policy with non-neutral personas.

### Evaluation Metrics
- Refusal Rate (RR): fraction of benign prompts refused.
- False Refusal Rate (FRR): refusal rate on prompts filtered as benign.
- Absolute Risk Difference (ARD): RR_treatment - RR_baseline.
- Odds Ratios from logistic regression for persona/topic features.
- Utility proxy: response token length (short refusals often indicate avoidance).

### Statistical Analysis Plan
- 95% Wilson confidence intervals for refusal proportions.
- Two-proportion z-tests / Fisher exact tests for pairwise condition contrasts.
- Logistic regression (`refusal ~ persona + topic + policy + model`) with robust SE.
- Multiple-comparison correction via Benjamini-Hochberg (FDR 0.05).
- Significance threshold: alpha = 0.05.

## Expected Outcomes
Support for hypothesis:
- Non-trivial FRR persists under harmless-only policy.
- Persona conditions materially shift FRR (significant ARD/OR).
- Topic/persona coefficients significant in regression.

Refutation of hypothesis:
- FRR ~ 0 under harmless-only policy.
- No consistent persona/topic effects after correction.

## Timeline and Milestones
- Phase 0-1 (planning): complete in this session start.
- Phase 2 (setup/data checks): 20-30 min.
- Phase 3-4 (implementation + runs): 60-90 min.
- Phase 5 (analysis): 30-45 min.
- Phase 6 (documentation/validation): 30 min.
- Buffer (debugging/API retries): 25% of total runtime.

## Potential Challenges
- API model availability mismatch (`gpt-5` access). Mitigation: fallback to `gpt-4.1` and `gpt-4.1-mini` while documenting deviation.
- Refusal-label noise from heuristics. Mitigation: manual spot-check + conservative phrase list.
- Cost/runtime growth due to factorial combinations. Mitigation: capped prompt sample size and response caching.

## Success Criteria
- Complete, reproducible pipeline executed end-to-end with real API calls.
- At least one model evaluated across all planned conditions.
- Statistical tests and visualizations generated from collected outputs.
- `REPORT.md` documents actual results, limitations, and reproducibility details.
