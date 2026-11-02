# Optimizing a Cheap Decision Model with Auto Research

### TL;DR

I fine-tuned a 1.5B parameter Qwen model to strengthen performance on When2Call, a benchmark evaluating an Agent's ability to decide when to make tool calls. This model achieved 77.45% accuracy, matching or exceeding the performance of decision models at least 18x larger: Cloudflare's Clef (70.96%), OpenAI's GPT-6 Luna with the Decision API (74.06%), and Perplexity's Decider v1.1 27B (76.5%)

I was able to obtain these results more quickly after building an auto-research loop, allowing Codex to automatically explore strategies for optimizing training and inference. The second half of this blog post will dive into the design of the loop, highlighting what worked well and what did not.

# Part One: training a decision model

## Motivation

A tool-using agent must decide on each turn whether to answer, call a tool, ask for missing information, or say that it cannot help with the available tools. This is a frequent decision, so even a small per-turn cost can add up across a long agent run.

A small model is an attractive option for this task. The output has only a few valid choices, so the model does not need to generate a long response. The same constraint also makes training data easier to label and evaluation easier to automate. These properties make tool-use decisions a useful target for task-specific fine-tuning.

## Dataset

When2Call measures whether a model chooses the right action, rather than whether it can produce a valid tool call once it has decided to call one. Each example describes a conversation and available tools. The model chooses among four responses: answer directly, call a tool, ask for more information, or say that it cannot answer with the available tools.[1](#ref-1)

A simplified example is:

```text
User: What is the sum of 133 and 34?
Available tool: sum_numbers(numbers_list)

Choices:
A. Call sum_numbers with [133, 34].
B. Ask the user to provide the numbers as a list.
C. Answer that the sum is 167.
D. Say that the assistant cannot perform calculations.

Correct choice: A
```

The original When2Call training data was enough to fit its own distribution. On a held-out split from that source, the model reached above 95% accuracy on both training and validation. The remaining errors on the external test set therefore appeared to be a data-distribution problem, rather than a sign that the training objective or inference procedure needed a major redesign.

I tried adding several existing open datasets to the training mixture. They did not improve external test performance. Their examples differed from When2Call in format and decision boundaries, so they did not provide the kind of additional signal the model needed. This led me to generate new examples that retained When2Call's task format while varying the underlying scenarios.

The synthetic data covered four strategies: requests that require a tool call with a new request and existing tool set, requests that omit a required argument, requests for capabilities that no available tool can provide, and requests over composed tool sets. I anchored generation in real training prompts and tool schemas, then asked a model to produce a new scenario with a specified correct action. A separate label check and structural validation filtered the generated candidates. The prompts are included in the appendix.

## Setup

I first fine-tuned `Qwen2.5-1.5B-Instruct` on the original When2Call preference data. I then continued from the selected original-data checkpoint using only the synthetic examples. This isolated the effect of the new data from replaying the original training set.

The main continuation used one epoch per synthetic data family. The training code used supervised fine-tuning with AdamW, a learning rate of 2 × 10⁻⁵, weight decay of 0.01, batch size 1, sequence length 4,096, and LoRA rank 16. The fixed clean-validation sample contained up to 500 examples. Synthetic training diagnostics also used up to 500 rows. Later full-test evaluations contained 3,652 examples. The training setup and evaluation records are in the post-training repository.

## Results

Artificial Analysis evaluates 3,423 When2Call examples. It excludes 229 “ask for more information” examples that it considers ambiguous, and it shuffles the answer choices.[2](#ref-2) The scores below for its decision models use that evaluation. Artificial Analysis publishes accuracy for this benchmark, but not macro F1, so those cells are left blank. Model size is shown only where it is available from the model name or the cited model description.

| Model | Parameters (B) | Accuracy | Macro F1 | Cost per 1M tasks (USD) |
| --- | ---: | ---: | ---: | ---: |
| Fine-tuned Qwen, synthetic continuation | 1.5 | 77.45% | — | 3.09 (estimated) |
| Fine-tuned Qwen, original data only | 1.5 | 70.24% | 52.78% | — |
| Elephant 9B | 9 | 94.77% | — | 8.39 |
| Celeris-1 Decision | — | 83.70% | — | 31.57 |
| Clef | 27 | 70.96% | — | 199.41 |
| Clef-Flash | — | 64.36% | — | 31.57 |
| Clef-Omni | — | 62.05% | — | 123.08 |
| GLiDE | — | 83.67% | — | 119.02 |
| Mercury Decide | — | 79.26% | — | 15.20 |
| d1 | — | 77.42% | — | 15.14 |
| d1-ultra | — | 92.58% | — | 31.80 |
| Drex Max | — | 93.98% | — | 23.03 |
| Drex 1.5 | — | 84.81% | — | 34.47 |
| Drex 2.0 | — | 86.15% | — | 41.36 |
| GPT-6 Luna (Decisions) | — | 74.06% | — | 78.87 |
| Microsoft-Decision-1 | — | 81.92% | — | 30.71 |
| Decider Flash v1.2 | — | 74.41% | — | 8.23 |
| Decider v1.1 27B | 27 | 76.48% | — | 16.53 |
| Decider v1.2 | — | 87.70% | — | 15.14 |
| SubQ Snap 1.1 | — | 94.57% | — | 30.34 |
| Jev 1.13 | — | 80.66% | — | 41.53 |
| Jev 1.14 | — | 80.84% | — | 41.53 |

The Qwen original-data row uses the full 3,652-example test score. The synthetic continuation reached 74.18% on that same full test set. Its 77.45% score above is the later rescore on Artificial Analysis's 3,423-example subset. The full-set and subset scores use different denominators, so the difference is not a direct measure of model improvement. Macro F1 for the full-set Qwen evaluations uses all four labels.

The results support a narrower claim than “small models are always better.” At 1.5B parameters, the fine-tuned Qwen is about 18 times smaller than Clef and the 27B Decider v1.1. It scores 6.49 and 0.97 percentage points higher, respectively, in the comparison reported in the summary. It is also substantially smaller than the roughly 20B GPT-6 Luna model family, while its score is close to the Decision API result. However, larger systems still lead the benchmark. Elephant 9B scores 94.77%, more than 17 points above the fine-tuned Qwen.[2](#ref-2)

The cost estimate for Qwen depends on the local inference setup. Artificial Analysis reports cost at its current published prices, so its hosted-model figures are not a direct deployment quote for another provider or hardware setup. The chart is a practical cost and accuracy comparison, not a universal ranking of serving costs.[2](#ref-2)

### Cost and accuracy

This chart compares When2Call accuracy with estimated cost per million tasks. The hosted-model points use Artificial Analysis estimates. The Qwen point uses the earlier $3.09 per million task estimate for the local model setup. The horizontal axis is logarithmic because the costs span a wide range. The star marks our fine-tuned model. The other points are the decision models in the table above. [2](#ref-2)

![When2Call accuracy versus estimated cost per million tasks. The star marks the fine-tuned Qwen model.](assets/when2call-cost-accuracy.png)

# Part Two: auto research

## Motivation

The Bitter Lesson argues that methods that scale with computation tend to outperform methods that depend on hand-built human knowledge.[3](#ref-3) I see the same pressure in model research. If a coding agent can write and run experiments, a human who still has to implement every experiment becomes the limiting step.

For this project, I used GPT-6 Luna at medium reasoning through Codex. The goal was to turn a broad research direction into a repeated loop of hypotheses, code changes, runs, and analysis. Each project should leave the loop easier to reuse on the next one.

## Design

The loop had three parts:

1. **Researcher.** The Codex session proposed and ran experiments, inspected aggregate results, and updated the research record.
   - It used a persistent prompt that defined the task, data restrictions, evaluation procedure, and required experiment records.
   - I gave it a family of approaches to explore, then let it choose specific experiments within that family.

2. **Watchdog.** A background process checked that the declared experiment worker and research state remained active.
   - It checked process state and the worker heartbeat on a short interval.
   - If a declared worker exited or became stale, it recorded the condition and initiated recovery.

3. **Supervisor.** A separate process kept the watchdog alive.
   - It checked the watchdog process and heartbeat.
   - If the watchdog stopped or became stale, it restarted the watchdog and recorded the restart.

The extra processes addressed a failure I saw in early runs. The research session often stopped after a few hours, sometimes after finishing a planned batch. It was not clear why it chose to stop. When no one was at the computer to prompt it to continue, the experiments remained idle. The watchdog and supervisor made worker recovery more reliable, while the research prompt told the session to keep planning when the experiment queue was empty.

## Implementation

The researcher prompt set out the objective, source datasets, evaluation rules, and requirements for each experiment. It asked Codex to change one factor at a time where possible, record the configuration and result, and use validation for candidate selection. The prompt also required aggregate-only reporting for the hidden test set and prohibited saving individual test examples or predictions. Later synthetic-data runs followed a separately authorized rule that used aggregate test accuracy to decide whether to continue that branch. The full prompt is included in the appendix.

The watchdog was implemented as a Python process that read the research state, checked the active worker's operating-system process state and heartbeat, and recorded failures. The independent supervisor polled the watchdog and restarted it when its process or heartbeat failed. These processes did not choose the next research direction. That remained the responsibility of the Codex researcher and the human who supplied the next family of ideas.

## Results

The loop made it practical to explore many variants without manually writing each training command or waiting beside the machine. Across the complete test evaluations, performance moved from 51.51% in the first logged evaluation to 74.18% at the best full-set result. That number includes a broad sequence of encoder, Qwen, and synthetic-data work. It should not be read as a controlled measure of the auto-research loop alone. The model, data, and evaluation strategy all changed during the project.

### Experiment families

The table counts distinct completed training runs in the experiment ledger. I grouped runs by their training workflow, so each run is counted once. The counts exclude validation-only screens, evaluation-only runs, queued work, and failed or incomplete runs.

| Experiment family | Completed runs | What I tried |
| --- | ---: | --- |
| Initial Qwen SFT and RPO | 7 | Compared supervised fine-tuning and reward-aware preference optimization on the original When2Call data at small model sizes. |
| Qwen full-data training and continuations | 197 | Tested preference-data sample sizes, source mixtures, synthetic strategies and doses, learning rates, and checkpoint continuations with the Qwen 1.5B workflow. |
| Glaive and When2Call mixtures | 13 | Added Glaive examples to When2Call training and tested whether this broader mixture improved the decision task. |
| General dataset continuations | 8 | Continued training on additional datasets and checked whether their distributions transferred to When2Call. |
| BFCL and When2Call mixture | 1 | Tested a combined training set from BFCL and When2Call. |
| Learned ensemble stackers | 2 | Trained small learned combiners to combine model predictions. |

These counts describe training runs, not the 70 full-test evaluations shown in the chart below. The ledger also contains other completed runs outside this table, such as data preparation and model evaluation.

### Accuracy over the experiment sequence

The post-training log contains 70 aggregate evaluations on the complete 3,652-example test set. The chart shows the best full-set accuracy seen so far after each evaluation, ordered by its recorded timestamp. It excludes smaller 500-example screens, including the 76.2% pilot result, because they are not directly comparable to the full-set scores.

```mermaid
---
title: "Best full-test accuracy by evaluation"
---
xychart-beta
    x-axis "Full-test evaluation number" [1, 70]
    y-axis "Best accuracy (%)" 50 --> 75
    line "Best observed so far" [51.51, 68.35, 69.19, 69.19, 69.19, 69.44, 69.44, 69.47, 69.47, 69.47, 69.47, 69.47, 69.47, 69.47, 69.47, 69.47, 69.47, 69.47, 69.47, 69.47, 69.47, 69.47, 69.47, 69.47, 70.24, 70.24, 70.24, 70.84, 70.84, 71.14, 71.14, 71.14, 71.14, 71.14, 71.55, 71.55, 71.55, 71.55, 71.55, 71.55, 71.55, 71.55, 71.55, 71.55, 71.55, 71.55, 71.55, 71.55, 71.55, 71.55, 71.55, 71.55, 71.55, 71.55, 71.55, 71.55, 71.55, 71.55, 71.55, 71.55, 71.60, 71.60, 71.60, 73.14, 73.14, 73.14, 73.14, 73.14, 73.14, 74.18]
```

The full-set experiment sequence began with encoder and fusion baselines. It then moved to Qwen fine-tuning, routing and ensemble variants, and finally synthetic-data continuations. The final synthetic continuation set the full-test record at 74.18%. The 500-example unsupported-capability pilot scored 76.2%, but that result was a screening measurement and did not determine the final full-set score.


The synthetic-data phase showed the value and the limits of this process. The unsupported-capability pilot reached 76.2% on a 500-example screen, which justified a larger run under the then-current experiment rule. Other synthetic families did not produce the same signal. The larger experiments used the complete test set, and their best result was 74.18%. This demonstrates why a promising pilot needs a larger, consistently evaluated follow-up.

## Learnings

The system still needed human direction. I usually chose the broad family, such as synthetic data or ensembling. The researcher was useful at exploring variants within that family, but it did not reliably decide which new family should come next. I acted more like a research lead setting a direction than a fully hands-off user.

The choice of researcher model also mattered in practice. GPT-6 Luna at medium reasoning was affordable enough to leave active for long runs, but I do not know whether a stronger model would have proposed better experiment families or analyzed results more efficiently. A future loop could reserve a more capable model for idea generation and error analysis while using a cheaper model for routine implementation. The cost of repeatedly sending a large context would need to be measured.

Continuous experimentation also changes how the work feels. The machine can run for long periods, while the human checks in periodically to see whether the current direction remains useful. This is less like one focused coding session and more like supervising a process over time.

The hardest open question was how long to stay within one family. The researcher tended to keep trying variants after I would have moved on. A fixed experiment budget would prevent that, but it could also cut off a promising direction too early. A better system would estimate when a family has stopped producing useful results and explain why it should switch.

Long sessions raised a context-retention problem. The researcher sometimes lost earlier guidance after a long run. A searchable record of prior sessions could let a fresh session retrieve decisions and results without carrying the full history in its active context.

Finally, the system was better at combining familiar techniques than proposing genuinely novel ones. Prompting for creativity did not solve this. An adversarial idea-generation or critique stage may help, but it would need to be judged by whether it proposes useful experiments, not whether the ideas sound unusual.

## References

<a id="ref-1"></a>

1. Hayley Ross, Ameya Sunil Mahabaleshwarkar, and Yoshi Suhara. [“When2Call: When (not) to Call Tools.” NAACL 2025.](https://aclanthology.org/2025.naacl-long.174/)

<a id="ref-2"></a>

2. [Artificial Analysis, When2Call Benchmark and methodology.](https://artificialanalysis.ai/evaluations/when2call)

<a id="ref-3"></a>

3. Rich Sutton. [“The Bitter Lesson.”](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)

## Appendix

### Prompts for synthetic dataset generation

The generator used real When2Call training examples as anchors. It generated new requests around tool availability, missing required arguments, unsupported capabilities, and composed tool sets. A separate label check and structural validation filtered the outputs before training. The full generation prompts and accepted-data definitions are in `post-training/experiments/generate_when2call_synthetic_pilots.py` and the corresponding run records.

### Prompts for auto-research loop

The full researcher instructions are in `post-training/auto-research-prompt.md`. They include the task objective, experiment process, data constraints, evaluation requirements, and logging rules.
