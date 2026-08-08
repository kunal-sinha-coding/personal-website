I will examine how much reinforcement learning can improve the performance of a tiny language model. The objective is not to treat a small experiment as a complete answer to the broader problem of language-model post-training. The objective is to establish a fast and controlled environment for evaluating post-training methods.

### 1. Motivation for a Tiny Model

#### Practical considerations

Tiny models offer two practical advantages for an independent researcher.

- Training costs are low. This permits more experiments without requiring a large compute budget.
- Training cycles are short. Results can be inspected and the next experiment can begin quickly.
- The experimental system is easier to inspect. Prompts, data formats, rewards, and failure cases can be analyzed together.

The speed of iteration is likely the largest practical factor in determining the rate of hill climbing on a metric. If an experiment takes less time, more hypotheses can be evaluated under the same resource constraint. Faster iteration therefore increases the rate at which a researcher can explore the space of possible training strategies.

#### Research considerations

There is also a research motivation for starting with a tiny model. Larger models are already near saturation on some common benchmarks. A tiny model generally has substantially lower baseline performance, which creates more room for measurable improvement.

When the baseline is weak, post-training has a larger relative role in determining the final result. This makes a tiny model a useful testbed for comparing post-training strategies. A strategy that produces a clear improvement from a weak baseline can then be evaluated for transfer to larger models.

#### Brittleness as a rigor test

In my experience, tiny models are more brittle than larger models. Their behavior is more sensitive to subtle changes in context, including the prompt, the data format, and the exact instructions.

This brittleness creates a useful methodological constraint. A researcher must control the experimental context more carefully because a small formatting change can change the result. The resulting discipline improves the quality of comparisons between training runs.

### 2. Agent Driven Experimentation

#### Limited cycle experimentation

I am also experimenting with recursive self improvement techniques in which an agent runs experiments and evaluates its own results over a limited number of cycles. The current procedure is intentionally basic. The agent changes one part of the setup, runs an experiment, inspects the result, and uses the observed error pattern to select a subsequent hypothesis.

This procedure treats the agent as an experiment coordinator rather than as an unrestricted autonomous researcher. Limiting the number of cycles keeps the process auditable and makes it possible to compare the results with a fixed experimental budget.

#### Error analysis

The error analysis skill is one component of this process. It reads logged failures, groups recurring patterns, and helps distinguish problems caused by the data, reward, prompt, or training configuration. This gives the next experiment a more informed starting point than selecting a change at random.

The current approach is only a basic starting point. I will refine it after measuring which parts of the loop produce useful improvements and which parts introduce noise. The main limitation is that the agent currently has a small number of evaluation cycles and a narrow set of diagnostic tools.

### 3. Experimental Infrastructure

#### Restartable GPU environments

The experiments run on a GPU, so much of the working state exists only in memory. I frequently need to move to a new GPU instance on RunPod. This makes environment setup part of the research workflow rather than a one-time task.

To reduce the cost of switching machines, I created a `start.sh` script that reconstructs the full environment quickly. The script loads local environment variables, installs the Codex CLI and Python dependencies, authenticates the required services, configures Git and GitHub access, and starts the development workspaces in `tmux`.

The script also updates or clones the relevant repositories. It registers the shared skills with Codex and opens reusable `codex`, `skills`, and `personal-website` windows. It checks whether each window already exists before creating it, which makes rerunning the script safe when the session is still available.

#### Shared skills repository

I also created a shared skills repository for reusable procedures that support the experiments. The repository currently includes an error analysis skill, an iterative training workflow, and the publishing workflow used for this website.

Keeping these procedures in a separate repository makes them available across GPU instances and projects. It also turns improvements to the experiment loop into versioned code that can be reused after the next environment restart.
