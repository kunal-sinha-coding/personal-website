### TL;DR

I fine-tuned a 1.5B Qwen model to strengthen performance on When2Call, a benchmark evaluating an Agent's ability to decide when to make tool calls. This model **achieved 77.45% accuracy, matching or exceeding the performance of decision models at least 18x larger:** Cloudflare's Clef (70.96%), OpenAI's GPT-6 Luna with the Decision API (74.06%), and Perplexity's Decider v1.1 27B (76.5%)

I was able to obtain these results more quickly after **building an auto-research loop**, allowing Codex to automatically explore strategies for optimizing training and inference. The second half of this blog post will dive into the design of the loop, highlighting what worked well and what did not.

### Part One: our decision model

Throughout multiple steps of the Agent loop, the Agent must decide whether to answer the user's request, call a tool, ask for missing information, or refuse the request. This decision occurs frequently enough that even a small per-turn cost can add up across a long horizon. As such, offloading this decision to a cheap model is an attractive option. This blog post focuses on squeezing the best performance out of one of the cheapest models.

Supervised Fine-tuning (SFT) on open-source models is a standard approach to match the performance of frontier models on specialized tasks at cheaper cost. SFT is especially attractive for decision models due to the ease of obtaining and learning from supervised labels here. The output only has a few discrete valid options. As such, producing the labels does not require as much effort, either in the form of labor cost from human data labelers or compute costs from synthetic dataset generation.

#### Dataset

**Original Dataset:**

[When2Call](https://aclanthology.org/2025.naacl-long.174/) measures whether a model selects the correct action when deciding whether or not to invoke tool calls. Each example describes a conversation and available tools. The model chooses among four responses: answer directly (`direct`), call a tool (`tool_call`), ask for more information (`request_info`), or say that it cannot answer with the available tools (`cannot_answer`).

A simplified example is as follows:

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

We trained on the 9,000-example `train_pref` split of the [When2Call dataset](https://huggingface.co/datasets/nvidia/When2Call) and evaluated on its 3,652-example `test/mcq` split. To match the [Artificial Analysis evaluation](https://artificialanalysis.ai/evaluations/when2call), we excluded the 229 ambiguous `request_for_info` examples, leaving 3,423 scored examples.

**Synthetic dataset**

When training on the original When2Call training dataset, our model quickly learned the data distribution. We split out the `train_pref` data into training data and a smaller held-out validation set, and after a single epoch of training, we achieved above 95% accuracy on both splits. However, the accuracy on the actual test data was lower at approximately 70%. These results together suggested that although the training setup was close to optimal, the underlying data came from a different distribution than the test set.

I first tried introducing several existing open datasets to the training mixture. They did not improve test performance. Their examples differed from When2Call in format and decision boundaries, and therefore they did not provide the kind of additional signal the model needed. The open datasets we tested included [Glaive](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2), [BFCL](https://gorilla.cs.berkeley.edu/leaderboard), [APIGen-MT](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k), and [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE).

The failure of these datasets pushed me towards generating synthetic data. If we could not find an open-source dataset drawn from the correct distribution, why not create one ourselves? We used GPT-6 Luna as our dataset generator model. Although we tried various approaches to generating synthetic examples, we eventually landed on the following approach:
1. Select an existing example from the dataset to use as a source example.
   - The standard approach was to select uniformly at random.
   - Optionally: sort examples from highest to lowest with respect to the number of available tools in the example, then do selections in that order. Intuitively, this metric is likely a proxy for example difficulty.
2. Feed the source example into the dataset generator LLM.
   - Specifically, we give the LLM the source example, the tool schemas, and the selected tool name.
   - There are then two possible paths for generation:
     - If the example is labeled `tool_call`:
       - Generate a new example in the same broad domain as the source, but not paraphrase.
       - Generate a second example near-identical to the first but with one modification: omit exactly one required input with no default. The correct action is to ask for missing information.
     - If the example is labeled `cannot_answer`:
       - Generate a new example in the same broad domain as the source, but the where the content of the user request is such that none of the tools can handle it. The correct action is to refuse to answer.
     - If not, skip.
3. Send the generated examples and the tool schemas to a separate verifier LLM.
   - The verifier does not see the intended labels. Instead, the verifier must independently confirm the actions.
   - For a pair, the verifier must additionally confirm that both examples express the same task and differ by only one missing required input.
   - Filter out all examples that the verifier does not assign a high-confidence judgment to.
4. Have the verifier LLM apply a second round of filtering:
   - For pairs, check that the selected tool, missing required input, and argument objects match the supplied schema. Reject structurally invalid or duplicate requests.
5. Move all non-filtered rows into the synthetic training set.
6. Optionally: pick only the most similar synthetic examples.
   - Rank all synthetic examples by lexical Jaccard similarity to their corresponding source examples. Then prioritize selecting the examples in order of this rank.

#### Experimental details

The highest performing experiment involved multiple epochs of fine-tuning on different mixes of data.
1. Train for 2 epochs on all 9,000 examples in the When2Call `train_pref` split.
2. Train for 1 epoch on 900 sampled When2Call `train_pref` examples and 450 synthetic examples.
   a. During synthetic dataset generation, the source examples were selected uniformly at random and the synthetic examples were prioritized by similarity.
3. Train for 1 epoch on a new pool of 994 synthetic examples.
   a. During synthetic dataset generation, the source examples were selected according to number of tool, and the synthetic examples were not prioritized by similarity.
4. Train for 1 epoch on another fresh pool of 983 synthetic examples.
   a. Synthetic dataset generation process was the same as above.

The original-data baseline and synthetic continuation used these hyperparameters:

| Setting | Value |
| --- | --- |
| Base model | Qwen2.5-1.5B-Instruct |
| Original-data baseline | 1 epoch on When2Call preference data |
| Best synthetic continuation | 1 epoch on 983 fresh tool-rich synthetic examples |
| Starting checkpoint | Epoch 1 of an earlier synthetic-only tool-rich run |
| Optimizer | AdamW |
| Learning rate | 2 × 10⁻⁵ for original-data training; 5 × 10⁻⁶ for the best continuation |
| Weight decay | 0.01 |
| Batch size | 1 |
| Maximum sequence length | 4,096 tokens |
| LoRA rank | 16 |
| Clean validation sample | Up to 500 examples |

Training code and evaluation records are in the [post-training repository](https://github.com/kunal-sinha-coding/post-training).

#### Results

| Model | Accuracy | Parameters |
| --- | ---: | ---: |
| [Elephant 9B](https://labs.baseten.co/articles/mammoth-1-and-elephant-1-frontier-decision-models) | 94.77% | [9B](https://www.baseten.co/library/elephant-1/) |
| SubQ Snap 1.1 | 94.57% | Not Found |
| Drex Max | 93.98% | [27B](https://huggingface.co/nace-ai/drex-max-v1) |
| d1-ultra | 92.58% | Not Found |
| Decider v1.2 | 87.70% | [27B](https://vercel.com/changelog/perplexity-decider-v1-2-models-now-available-on-ai-gateway) |
| Drex 2.0 | 86.15% | [35B total](https://huggingface.co/nace-ai/drex-max-v1); [3B active](https://console.nace.ai/docs/drex/reference/models) |
| Drex 1.5 | 84.81% | [9B](https://huggingface.co/nace-ai/drex-v1.5) |
| Celeris-1 Decision | 83.70% | Not Found |
| GLiDE | 83.67% | Not Found |
| Microsoft-Decision-1 | 81.92% | [9B](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/) |
| Jev 1.14 | 80.84% | Not Found |
| Jev 1.13 | 80.66% | Not Found |
| Mercury Decide | 79.26% | Not Found |
| **SFT-Qwen (original + synthetic data)** | **77.45%** | [1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| d1 | 77.42% | Not Found |
| Decider v1.1 27B | 76.48% | [27B](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) |
| Decider Flash v1.2 | 74.41% | [4B](https://vercel.com/changelog/perplexity-decider-v1-2-models-now-available-on-ai-gateway) |
| GPT-6 Luna (Decisions) | 74.06% | Not Found |
| Clef | 70.96% | [27B](https://blog.cloudflare.com/clef-decision-models/) |
| **SFT-Qwen (original data)** | **70.24%** full | [1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| Clef-Flash | 64.36% | [9B](https://blog.cloudflare.com/clef-decision-models/) |
| Clef-Omni | 62.05% | [30B total (3B active)](https://developers.cloudflare.com/workers-ai/models/clef-omni/) |

The main takeaways are:

- Our model is the smallest model in this comparison that still matches or exceeds several larger decision models.
  - It is **18 times smaller than Clef** (27B), but its **accuracy is higher by 6.49 percentage points** (77.45% versus 70.96%).
  - It is an **order of magnitude smaller than GPT-6 Luna Decisions**, but its **accuracy is higher by 3.39 percentage points** (77.45% versus 74.06%).
  - It is **18 times smaller than Perplexity's Decider v1.1** (27B), but its **accuracy is higher by 0.97 percentage points** (77.45% versus 76.48%).
- Our model does not beat every larger model. Baseten's Elephant 9B is six times larger and reaches 94.77% accuracy, 17.32 percentage points higher than our model.
- The addition of **synthetic training data boosts accuracy by 7.21 percentage points** (77.45% versus 70.24%).

The chart below compares accuracy with published model size, includes only entries with disclosed parameter counts. The logarithmic horizontal axis shows total parameters; active parameter counts for the two mixture-of-experts models are noted in the table. The star marks our synthetic-continuation model and the diamond marks our original-data model.

![When2Call accuracy versus disclosed parameter count. Only models with a published size are plotted. The Qwen synthetic-continuation score uses the filtered test set; the original-data score uses the full test set.](assets/when2call-parameters-accuracy.png)

### Part Two: auto-research

During the research process above, we found incredible gains in efficiency by building out a basic **auto-research** loop in Codex that would automate steps such as generating hypotheses, running experiments, and performing error analysis. For this project, to keep costs low, we used GPT-6 Luna with medium reasoning.

Researchers must prioritize building out loops like these and iteratively improving upon them to continually minimize the number of manual interventions required. The north star is a fully autonomous loop that can design and train an optimal model for any given task. An even more ambitious goal, albeit one that is likely not yet feasible at this exact moment, is an Agent that can discover genuinely novel innovations in areas such as model architecture, training algorithms, harness engineering, inference optimization, and more. Do not fret over the feasibility of this approach today. The [Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) applies here: methods that scale with computation tend to outperform methods that rely on human-designed knowledge. Build the loop now, even a rudimentary one, and allow its capabilities to grow exponentially as the frontier models scale.

#### Design

The loop consisted of three components:

1. **Researcher.** The Codex session proposed experiments, ran them, analyzed errors, and then proposed new experiments iteratively.
   - It used a persistent prompt that defined the task, data restrictions, evaluation procedure, and required experiment records.
   - I often manually intervened in the Codex session to suggest a family of approaches to explore (e.g "SFT", "GRPO", "RAG"), but the researcher was able to explore a wide range of experiments within the family.

2. **Watchdog.** A background process checked that the declared experiment worker and research state remained active.
   - It checked process state and the worker heartbeat on a short interval (one minute).
   - If a declared worker exited or became stale, it recorded the condition and restarted the worker.

3. **Supervisor.** A separate process kept the watchdog itself alive.
   - It checked the watchdog process state and heartbeat on a longer interval (thirty minutes).
   - If the watchdog stopped or became stale, it recorded the condition and restarted the watchdog.

The watchdog was introduced to address a common failure I encountered throughout the research process: the researcher would stop iterating within a few hours. Specifically, the researcher would propose a batch of experiments, complete the experiments, and then decide not to proceed further. The reason for this behavior was unclear. One hypothesis is that the model would burn too many tokens exploring a tree of experiments. For each experiment, the researcher might queue a batch of new sub-experiments building upon the original, then further sub-sub-experiments building upon each of those, and so on and so forth. This process would saturate the context window, perhaps even multiple times after compaction, and the model would forget the outer instruction to continue iterating. The watchdog prevented this risk effectively because it relied on a mechanism that was deterministic and lived outside the session. The supervisor then served as a watchdog upon the watchdog. This component was introduced as intentional redundancy to fully eliminate the risk of premature stopping.

#### Implementation

The full prompt fed into the researcher can be found in the Appendix. However, we break down key components of the prompt here:

- **Research loop.** The researcher reviewed training and evaluation metrics (synced to WandB) from prior experiments and performed error analysis on saved output logs to understand the current failure patterns. Then, the researcher proposed distinct and falsifiable hypotheses to run experiments on. When running experiments, the strategy was to begin with a suite of quicker and cheaper diagnostic tests on small samples of data, then explore a particular method further with follow-up experiments if the diagnostic showed any signs of improving test data accuracy. Another key instruction was to run controlled experiments changing one factor at a time to isolate the effect of the factor.
- **Black box evaluation.** Maximizing test accuracy becomes trivial if the researcher has access to the contents of the test data. For example, the researcher could propose specific prompt changes to a model or design training data based on the exact examples that appear in the test data. This approach is invalid for the same reason that training on test data is invalid: these changes are unlikely to generalize beyond the test data. As such, we instructed the researcher to treat the test evaluation as a black box. The researcher could submit a model to the evaluator script, and the script returned an accuracy metric, but the researcher was forbidden from reading the contents of the test data itself. Furthermore, while running experiments, the researcher was encouraged to split off a part of the training data to serve as a held-out validation set, find an approach that improved on the set, and only then submit the approach to the black box test evaluator.
- **Guidance and exploration.** The researcher was told that the user would occasionally manually intervene to provide guidance. This guidance would describe a broad family of approaches to pursue, and the researcher would tailor its next experiments to live within this family. Within a family, the researcher was encouraged not to burn too many experiment cycles fiddling with changes to hyperparameters unless there was concrete evidence suggesting that such a change was required. (For example, a volatile loss curve might be evidence for a lower learning rate). This suggestion was designed to fix a common trap the auto-researcher would fall into that was typically ineffective at boosting performance. The prompt also forbade changing random seeds as a method to get small performance gains, as this was unlikely to generalize. Furthermore, the researcher was explicitly encouraged to "think outside of the box" and consider creative ideas.
- **Monitoring.** The researcher was instructed to keep multiple logging files up to date. These logs allowed for an effective infinite context window. If the window was ever oversaturated, compacted, or cleared out, the researcher was able to pick up from where it left off by searching these logging files.
   - `logs/research_state.json` held the active worker command and PID (labeled `worked_command`), queued commands (labeled `worker_queue`), current status, heartbeat timestamp, and most recent action.
   - `logs/experiment_records.jsonl` stored structured experiment records
   - `logs/research_log.md` held the running narrative and details about the researcher's reasoning, hypotheses, and error analysis
   - `logs/results.txt` held results from experiments, and `logs/leaderboard.md` listed out the methods sorted based on how well they scored on the test data, to use for designing future methods.
   - A copy of `heartbeat.json` for each training run was populated in its output directory as the run progressed, alongside its model artifacts and metrics.

The watchdog was instantiated as a Python process. Every 30 seconds, the process read `logs/research_state.json`, checked the declared worker PID and command against the operating-system process table, and compared the state heartbeat with the `heartbeat.json` modification time in that worker’s output directory. It also checked that `logs/research_log.md` still existed and had not unexpectedly shrunk. It appended check and recovery events to `logs/validation_watchdog.log`.

When a worker had exited, the watchdog helped kick off next experiment the researcher had added to the queue. Specifically, the watchdog first checked for an already-running matching worker to avoid starting a duplicate. If one existed, it adopted that process and updated the state. Otherwise, it recorded the completed worker, removed the next command from `worker_queue`, saved it as `worker_command`, and launched that command as a detached subprocess. It wrote the new PID and command back to `research_state.json` and sent the process output to `logs/validation_worker.out`.

If the queue was exhausted, the watchdog marked the state in `logs/research_state.json` as awaiting Codex planning. Because the auto-research loop was running in a tmux window, the watchdog ran commands to attach to that window with `tmux a -t` and send a hardcoded message to the configured Codex thread with `codex queue --thread … --message …`. The message was as follows:

> The research worker queue is exhausted. Inspect the latest clean-validation results, training metrics, and aggregate black-box test metrics. For the most recent completed experiment, check whether all three evidence types are recorded. If any required training, clean-validation, or aggregate black-box test measurement is missing, queue that measurement before any new experiment. Then form new hypotheses, append distinct commands to logs/research_state.json worker_queue, set planning_required to false, set status to active, and verify the next launch. Do not use test labels for selection.

The supervisor was instantiated as a separate Python process. It monitored the watchdog PID and heartbeat, logged its checks, and restarted the watchdog process if it exited or became stale.

Finally, we describe the details of our GPU setup below. We ran the auto-research loop on a cheap GPU rented from RunPod for merely $0.24 per hour.

| Resource | Specification |
| --- | --- |
| GPU | 1× NVIDIA RTX 2000 Ada |
| vCPU | 6 (AMD EPYC 7352 24-Core Processor) |
| Memory | 31 GB |
| Container disk | 50 GB |

#### Results

The auto-research loop allowed us to perform a wide variety of experiments quickly.

| Experiment family | What we tried |
| --- | --- |
| Encoders and ranking models | Tested MiniLM and Jina models, classification and pairwise-ranking objectives, and input features such as tool schemas and required-argument cues. |
| Qwen supervised fine-tuning and data mixtures | Compared training objectives and model sizes, then varied real and synthetic data sources, mixture ratios, data amounts, learning rates, and checkpoint continuations. |
| Inference fusion, routing, and ensembles | Combined model predictions with fixed and adaptive fusion, voting, routing rules, and learned stackers. |
| Retrieval-augmented policies | Tested dynamic retrieval and similarity-gated retrieval policies for When2Call decisions. |

The auto-research loop ran 809 distinct experiments across the full session. However, most of these experiments were diagnostic tests conducted on a small sample of data and evaluated on only a held out validation set. We have 70 completed evaluations on the full test set. These were the methods that showed promise in the diagnostic experiment and were selected to scale up. 

Throughout the process, we evaluated on full test dataset of 3,652 examples. At the end of the process, we choose to re-evaluate a select few methods to align more closely with Artificial Analysis's evaluation setup, removing the 229 ambiguous examples. This re-evaluation boosted the best performing method from 74.18% to the 77.45% reported in the results table earlier.

We display how the accuracy of the best performing method evolved across 70 evaluations.

![Cumulative best accuracy across 70 chronological evaluations on the complete 3,652-example test set. Red markers show the 12 evaluations that set a new record.](assets/when2call-full-test-accuracy.png?v=2815a341fc2e)

A few takeaways:
- The loop was able to boost accuracy by 22.67 percentage points, from 51.51% (1,881/3,652) to 74.18% (2,709/3,652)
- The bulk of the gains occurred early on: we went up by 17.94 percentage points, from 51.51% (1,881/3,652) to 69.44% (2,536/3,652). 
   - After this, we saw diminishing returns as the rate of improvement slowed to a plateau, presumably once the low-hanging fruit of improvements had been exhausted.

#### Learnings

- **The system still required some manual intervention.** We found that the researcher was only moderately creative. Without explicitly identifying a new family of approaches to take, such as moving from SFT to RL-based methods, or focusing on optimizing inference through RAG and few shot prompting, the researcher would not make this transition on its own. The user had to behave as a senior director while the researcher operated more akin to a junior.
- **We have an exploration vs exploitation trade-off.** The space of possible experiments to run is vast and impossible to fully iterate over. Spending more time running different experiments within the same family can help find non-obvious optimizations. However, the time we spend optimizing a solution we know that works can prevent us from exploring other families and finding an even greater solution. Figuring out how to best manage this tradeoff is an open problem.
- **Context rot was still a problem.** The researcher would sometimes forget guidance that had been given far in the past. The solution here is likely as follows: when the context limit exceeds a certain threshold (perhaps measured by another Python process), automatically send a message to the researcher instructing it to create a new `tmux` window, attach to the window, launch a new codex session, and have it search over saved context from the previous session as needed.
- **Continuous experimentation changes the nature of work.** Because the Codex-based researcher can run in the background for long periods of time, having the human sit at the computer and wait for each experiment is unproductive. However, the human must periodically monitor and intervene in the research loop. As such, rather than engaging in short sprints of focused work, the human must now distribute their work more sparsely across a longer time scale. However, this setup allows the human to more effectively parallelize and pursue multiple projects at once.

#### Appendix

##### Prompts for synthetic dataset generation

The generator used real training requests and tool schemas as context. The shared instruction and each strategy-specific instruction are shown below. The model returned one new request and structured metadata. A separate judge classified the request using only the request and available tools. Structural checks then verified the target label and tool arguments.

```text
Action labels:
- cannot_answer: The request needs a capability that none of the tools provide, and it cannot be answered directly from the request.
- request_for_info: A tool can do the task, but a required input is missing and cannot be inferred.
- tool_call: A listed tool is needed and the request gives enough information to use it.
- direct: The request can be answered without a tool.

Create a genuinely new user request, not a paraphrase or small edit of the source request. Stay close to the source task style and complexity. Keep every available tool schema exactly as provided. Return one request only, with no answer.

Strategy instruction:
- New tool call: The target action is tool_call. Choose one listed tool. The request must clearly require it and include all facts needed for the required arguments.
- Missing required information: The target action is request_for_info. Choose one listed tool with a required field that has no default. The request must need that tool, provide the other required facts, and omit one required value that cannot be inferred. Name the omitted field.
- Unsupported capability: The target action is cannot_answer. Ask for an action that none of the listed tools can perform and that cannot be answered from stable knowledge in the request. Do not ask for a tool-supported task.
- Composed tools: The target action is tool_call. The available tools combine two source tool sets. Choose one tool and create a new request that needs that tool, distinguishes it from the other tools, and supplies all facts required for its arguments.

Source context:
{"source_requests": ["<source request>", "<source request>"], "available_tools": ["<unchanged tool schemas>"], "target_action": "<target label>"}
```

The generator required a structured response with these fields:

```json
{
  "request": "A new user request",
  "action_label": "tool_call | request_for_info | cannot_answer | direct",
  "selected_tool_name": "A listed tool, if required",
  "arguments_json": "A JSON object encoded as a string",
  "missing_required_argument": "The omitted required field, if any",
  "new_scenario_note": "A short note about what makes the scenario new"
}
```

The independent label check hid the requested target action from the judge:

```text
Classify the required assistant action for this new example. Decide from the request and tools only.
{"request": "<generated request>", "available_tools": ["<unchanged tool schemas>"]}
```

##### Prompts for auto-research loop

The operating prompt changed as the project moved through new experiment families. The version below combines the standing instructions used in the open-ended research phase. It omits stale queue entries and machine-specific commands, but preserves the research and monitoring rules that shaped the loop.

```text
You are an autonomous AI research engineer. Improve performance on the When2Call benchmark through empirical research. Do not stop after one successful or unsuccessful experiment.

Research loop
1. Review the latest experiment records, metrics, and unresolved errors.
2. Propose distinct, falsifiable hypotheses across data, training, models, and inference. Explain why each may help and define a success criterion.
3. Start with a small number of low-cost, structurally different tests. Expand a direction only when results support it. Do not repeat an experiment with only a different random seed.
4. Change one relevant factor at a time when practical. Keep other settings fixed. Justify changes to learning rate, batch size, optimizer, weight decay, or epoch count with recorded evidence.
5. Implement the experiment, verify that the intended process is running, and record its configuration, data, checkpoint, metrics, and outcome.
6. After a worker finishes, analyze the result and plan the next useful experiment. An empty queue is a request for research planning, not a reason to stop.

Evaluation rules
- Use clean held-out validation and declared training diagnostics to select checkpoints and methods.
- Treat the official test set as a black-box evaluator. Report only aggregate accuracy, macro F1, example count, and runtime. Do not inspect or save individual test examples, labels, predictions, correctness, confusion matrices, or error slices.
- Keep test results separate from candidate selection unless a specific experiment has an explicit test-based continuation rule.
- Record limitations and evaluation-set differences with each reported result.

Data and execution
- Generate new examples from training data only. Keep the task format and complexity close to When2Call while testing new scenarios.
- Validate generated examples for structure and labels before training.
- Keep API spend within the approved budget and record usage.
- Run one GPU training or evaluation worker at a time. Keep experiment records durable and resumable.

Monitoring
- A watchdog checks the declared worker, operating-system process state, research state, and heartbeat every 30 seconds.
- A separate supervisor restarts the watchdog if it exits or its heartbeat becomes stale.
- Verify every launch and restart by checking the process, command, and updated log or heartbeat. Record recovery actions.
- The watchdog recovers execution. The researcher plans experiments. When the queue is empty, the researcher must review results and add the next justified experiment.
```

This is an edited public version of the working prompt. The original prompt accumulated run-specific instructions as the research progressed.
