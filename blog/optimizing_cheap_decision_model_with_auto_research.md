### TL;DR

I fine-tuned a 1.5B Qwen model to strengthen performance on When2Call, a benchmark evaluating an Agent's ability to decide when to make tool calls. This model **achieved 77.45% accuracy, matching or exceeding the performance of decision models at least 18x larger:** Cloudflare's Clef (70.96%), OpenAI's GPT-6 Luna with the Decision API (74.06%), and Perplexity's Decider v1.1 27B (76.5%)

I was able to obtain these results more quickly after **building an auto-research loop**, allowing Codex to automatically explore strategies for optimizing training and inference. The second half of this blog post will dive into the design of the loop, highlighting what worked well and what did not.

### Part One: training a decision model

#### Motivation

Throughout multiple steps of the Agent loop, the Agent must decide whether to answer the user's request, call a tool, ask for missing information, or refuse the request. This decision occurs frequently enough that even a small per-turn cost can add up across a long horizon. As such, offloading this decision to a cheap model is an attractive option. This blog post focuses on squeezing the best performance out of one of the cheapest models.

Supervised Fine-tuning (SFT) on open-source models is often used to match the performance of frontier models on specialized tasks at cheaper cost. SFT is especially attractive for decision models due to the ease of obtaining and learning from supervised labels here. The output only has a few discrete valid options. As such, producing the labels does not require as much effort, either in the form of labor cost from human data labelers or compute costs from synthetic dataset generation.

#### Dataset

**Original Dataset:**

When2Call measures whether a model selects the correct action when deciding whether or not to invoke tool calls. Each example describes a conversation and available tools. The model chooses among four responses: answer directly (`direct`), call a tool (`tool_call`), ask for more information (`request_info`), or say that it cannot answer with the available tools (`cannot_answer`).[1](#ref-1)

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

I first tried introducing several existing open datasets to the training mixture. They did not improve test performance. Their examples differed from When2Call in format and decision boundaries, and therefore they did not provide the kind of additional signal the model needed. The open datasets we tested included [Glaive](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2), [BFCL](https://gorilla.cs.berkeley.edu/leaderboard), [APIGen-MT](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k), and [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE). None improved the full When2Call test score when mixed into training.

The failure of these datasets pushed me towards generating synthetic data. if we could not find an open-source dataset drawn from the correct distribution, why not create one ourselves? Although we tried various approaches to generating synthetic examples, we eventually landed on the following approach. We use GPT-6 Luna as our dataset generator model.
1. Select an existing example from the dataset as a "source" example.
   a. The standard approach was to select uniformly at random.
   a. Optionally: sort examples from highest to lowest with respect to the number of available tools in the example, then do selections in that order. Intuitively, this metric is likely a proxy for example difficulty. 
2. Feed the source example into the dataset generator LLM.
   a. Specifically, we give the LLM the source example, the tool schemas, and the selected tool name.
   b. There are then two possible paths for generation:
      i. If the example is labeled `tool_call`:
         - Generate a new example in the same broad domain as the source, but not paraphrase. 
         - Generate a second example near-identical to the first but with one modification: omit exactly one required input with no default. The correct action is to ask for missing information.
      ii. If the example is labeled `cannot_answer`: 
         - Generate a new example in the same broad domain as the source, but the where the content of the user request is such that none of the tools can handle it. The correct action is to refuse to answer.
      iii. If not, skip.
3. Send the generated examples and the tool schemas to a separate verifier LLM. 
   a. The verifier does not see the intended labels. Instead, the verifier must independently confirm the actions. 
   b. For a pair, the verifier must additionally confirm that both examples express the same task and differ by only one missing required input. 
   c. Filter out all examples that the verifier does not assign a high-confidence judgment to.
4. Have the verifier LLM apply a second round of filtering: 
   a. For pairs, check that the selected tool, missing required input, and argument objects match the supplied schema. Reject structurally invalid or duplicate requests. 
5. Move all non-filtered rows into the synthetic training set.
6. Optionally: pick only the most similar synthetic examples.
   i. Rank all synthetic examples by lexical Jaccard similarity to their corresponding source examples. Then prioritize selecting the examples in order of this rank.


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
| Elephant 9B | 94.77% | [9B](https://www.baseten.co/library/elephant-1/) |
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
| **SFT-Qwen (original + synthetic data)** | **77.45%** filtered / 74.12% full | [1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| d1 | 77.42% | Not Found |
| Decider v1.1 27B | 76.48% | [27B](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) |
| Decider Flash v1.2 | 74.41% | [4B](https://vercel.com/changelog/perplexity-decider-v1-2-models-now-available-on-ai-gateway) |
| GPT-6 Luna (Decisions) | 74.06% | Not Found |
| Clef | 70.96% | [27B](https://blog.cloudflare.com/clef-decision-models/) |
| **SFT-Qwen (original data)** | **70.24%** full | [1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| Clef-Flash | 64.36% | [9B](https://blog.cloudflare.com/clef-decision-models/) |
| Clef-Omni | 62.05% | [30B total (3B active)](https://developers.cloudflare.com/workers-ai/models/clef-omni/) |

The Qwen hosted-cost estimate is **$76.03 per million tasks**: the 3,423 scored prompts averaged 759.28 input tokens, with one output token budgeted for the label, priced at Together’s listed [$0.10 per million input and output tokens](https://pricepertoken.com/pricing-page/model/qwen-qwen2.5-1.5b-instruct). This uses our prompt and hosted base-model pricing, excluding fine-tuning and hosting overhead. It is not directly comparable with Artificial Analysis (AA)’s Decider estimate of $16.53 per million tasks: AA uses its own evaluation prompts and input-only billing for decision models. The public figures do not give enough detail to separate the effect of prompt length from provider rates.

The main takeaways are:

- Our model is the smallest model in this comparison that still matches or exceeds several larger decision models.
  - It is 18 times smaller than Clef (27B), but its accuracy is higher by 6.49 percentage points (77.45% versus 70.96%).
  - It is an order of magnitude smaller than GPT-6 Luna Decisions, but its accuracy is higher by 3.39 percentage points (77.45% versus 74.06%).
  - It is 18 times smaller than Perplexity's Decider v1.1 (27B), but its accuracy is higher by 0.97 percentage points (77.45% versus 76.48%).
- Our model does not beat every larger model. Baseten's Elephant 9B is six times larger and reaches 94.77% accuracy, 17.32 percentage points higher than our model.
- The addition of synthetic training data boosts accuracy by 7.21 percentage points (77.45% versus 70.24%).

The chart below compares accuracy with published model size, includes only entries with disclosed parameter counts. The logarithmic horizontal axis shows total parameters; active parameter counts for the two mixture-of-experts models are noted in the table. The star marks our synthetic-continuation model and the diamond marks our original-data model.

![When2Call accuracy versus disclosed parameter count. Only models with a published size are plotted. The Qwen synthetic-continuation score uses the filtered test set; the original-data score uses the full test set.](assets/when2call-parameters-accuracy.png)

### Part Two: auto-research

#### Motivation

The Bitter Lesson argues that methods that scale with computation tend to outperform methods that depend on hand-built human knowledge.[3](#ref-3) I see the same pressure in model research. If a coding agent can write and run experiments, a human who still has to implement every experiment becomes the limiting step.

For this project, I used GPT-6 Luna at medium reasoning through Codex. The goal was to turn a broad research direction into a repeated loop of hypotheses, code changes, runs, and analysis. Each project should leave the loop easier to reuse on the next one.

#### Design

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

#### Implementation

The researcher prompt set out the objective, source datasets, evaluation rules, and requirements for each experiment. It asked Codex to change one factor at a time where possible, record the configuration and result, and use validation for candidate selection. The prompt also required aggregate-only reporting for the hidden test set and prohibited saving individual test examples or predictions. Later synthetic-data runs followed a separately authorized rule that used aggregate test accuracy to decide whether to continue that branch. The full prompt is included in the appendix.

The watchdog was implemented as a Python process that read the research state, checked the active worker's operating-system process state and heartbeat, and recorded failures. The independent supervisor polled the watchdog and restarted it when its process or heartbeat failed. These processes did not choose the next research direction. That remained the responsibility of the Codex researcher and the human who supplied the next family of ideas.

#### Results

The loop made it practical to explore many variants without manually writing each training command or waiting beside the machine. Across the complete test evaluations, performance moved from 51.51% in the first logged evaluation to 74.18% at the best full-set result. That number includes a broad sequence of encoder, Qwen, and synthetic-data work. It should not be read as a controlled measure of the auto-research loop alone. The model, data, and evaluation strategy all changed during the project.


I counted one experiment as a distinct candidate configuration or inference policy with a completed result. I deduplicated repeated records by experiment ID. I excluded validation and test confirmation screens, data-generation and merge jobs, and support records. These counts describe candidate configurations, not total run events. The ledger has no family field, so I grouped each candidate by the method it tested.

| Experiment family | What I tried |  |
| --- | --- |  |
| Learned classifiers and ranking models | Tested MiniLM and Jina models, classification and pairwise-ranking objectives, and input features such as tool schemas and required-argument cues. |  |
| Qwen supervised fine-tuning and data mixtures | Compared training objectives and model sizes, then varied real and synthetic data sources, mixture ratios, data amounts, learning rates, and checkpoint continuations. |  |
| Inference fusion, routing, and ensembles | Combined model predictions with fixed and adaptive fusion, voting, routing rules, and learned stackers. |  |
| Retrieval-augmented policies | Tested dynamic retrieval and similarity-gated retrieval policies for When2Call decisions. |  |
| **Total** | **Distinct completed candidate configurations.** |  |

The full ledger has 809 distinct IDs whose latest status is completed. The remaining 294 are confirmation or evaluation screens, data-generation or merge jobs, and support records. The graph below has a narrower scope: it shows 70 completed evaluations on the full test set, not 70 independent methods.


I rebuilt this curve from the chronological experiment ledger. It includes 70 completed evaluations, each on all 3,652 test examples. The graph plots the best accuracy reached so far after each full-set evaluation. It is a step plot: the record increased on evaluations 1, 2, 3, 6, 8, 25, 28, 30, 35, 61, 64, and 70, and stayed flat on the other 58. The points are not interpolated between runs. The 76.2% unsupported-capability pilot used only 500 examples, so it is not included.

![Cumulative best accuracy across 70 chronological evaluations on the complete 3,652-example test set. Red markers show the 12 evaluations that set a new record.](assets/when2call-full-test-accuracy.png)

The full-set sequence began with an encoder baseline, moved through fusion, Qwen training, and routing experiments, and ended with synthetic-data continuation. It reached 74.18% at evaluation 70. The earlier 76.2% score came from the 500-example screen. It did not set the full-set record.

The synthetic-data phase showed the value and the limits of this process. The unsupported-capability pilot reached 76.2% on a 500-example screen, which justified a larger run under the then-current experiment rule. Other synthetic families did not produce the same signal. The larger experiments used the complete test set, and their best result was 74.18%. This demonstrates why a promising pilot needs a larger, consistently evaluated follow-up.

#### Learnings


- The system still depended on my direction. I usually chose the family of approaches to explore, such as synthetic data or ensembling. Once I chose a family, the researcher explored variants, wrote the code, ran the experiments, and reported the results. This made me more like a research lead assigning a direction to a junior researcher. A more autonomous system would propose and prioritize the broader families itself, so a person could give it a general goal and take a smaller role. Even in its current form, it reduced the work I had to do because I could give it a broad idea instead of specifying each experiment.
- The choice of researcher model may have limited the results. I used GPT-6 Luna because it was affordable enough to run for long periods on my $20-per-month Codex Pro plan. When I tried the stronger Soul model, it used most of my remaining five-hour quota in about one to two hours. A stronger model might have proposed better experiment families or analyzed failures more efficiently, but leaving it in a continuous loop would have been expensive. A future system could use a cheap model for implementation and reserve a stronger model for brainstorming and error analysis. That could still cost more because error analysis requires reading a lot of context. Switching models may also reduce the benefit of prompt caching.
- Continuous experimentation changes the work pattern. The agent can run in the background for long periods, so sitting at the computer and waiting for each experiment is not productive. The human still needs to check in and redirect the work when the researcher gets stuck or follows an unproductive direction. The work becomes more distributed over time and requires periodic monitoring.
- The researcher tended to keep exploring one family rather than move on. This is an exploration versus exploitation trade-off. A fixed experiment limit could stop a promising direction too early, but an open-ended instruction can keep the researcher in a weak direction indefinitely. A useful future system would estimate how many experiments a family deserves based on the evidence so far.
- Long sessions caused context loss. I kept the researcher in one session so it could use the history of prior experiments, but after about a day it sometimes forgot earlier guidance. One option is to rely on longer context windows as models improve. Another is to start a new session when the context grows and let it search saved transcripts from earlier sessions. My setup used Codex CLI and Tmux, which made it possible to start and control new sessions from the terminal.
- The researcher had access to internet search through Exa, but it rarely used it unless I asked. A research agent should review relevant work and the current state of the art during brainstorming because that can provide useful ideas for experiments.
- The researcher was better at recommending familiar approaches than proposing novel ones. Asking it to be creative was not enough. A future loop could add an adversarial idea-generation or critique stage, while judging ideas by whether they suggest useful experiments rather than whether they sound unusual.

#### References

<a id="ref-1"></a>

1. Hayley Ross, Ameya Sunil Mahabaleshwarkar, and Yoshi Suhara. [“When2Call: When (not) to Call Tools.” NAACL 2025.](https://aclanthology.org/2025.naacl-long.174/)

<a id="ref-2"></a>

2. [Artificial Analysis, When2Call Benchmark and methodology.](https://artificialanalysis.ai/evaluations/when2call)

<a id="ref-3"></a>

3. Rich Sutton. [“The Bitter Lesson.”](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)

<a id="ref-4"></a>

4. Base Labs, [“Mammoth-1 and Elephant-1: frontier decision models.”](https://labs.baseten.co/articles/mammoth-1-and-elephant-1-frontier-decision-models); Cloudflare, [“Introducing Clef: our open-source decision models.”](https://blog.cloudflare.com/clef-decision-models/); Perplexity, [pplx-decider-v1.1-27b model card](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b).

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
