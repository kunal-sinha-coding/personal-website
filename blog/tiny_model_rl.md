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

I also created a shared skills repository for reusable procedures that support the experiments. The repository currently includes the following skills.

- The error analysis skill reads logged failures, groups recurring patterns, and supports the next training hypothesis.
- The iterative workflow skill runs a user specified command repeatedly while preserving a clear stopping condition.
- The documentation skill audits repository instructions and updates outdated README files.
- The publishing skill turns a specified website post into the files required for publication and updates the visibility manifest.

Keeping these procedures in a separate repository makes them available across GPU instances and projects. It also turns improvements to the experiment loop into versioned code that can be reused after the next environment restart.

### 4. Findings

#### Experimental notes

One recent experiment tested whether the training loop was receiving enough useful reward variation to update the policy. The hypothesis was that the binary reward was too sparse for this model at its current capability level. A candidate received `1.0` only when every supplied assertion passed, and received `0.0` for format errors, syntax errors, runtime errors, and partially correct programs. When all four generations for a prompt failed, GRPO had no within-group preference to learn from.

The initial choice of a sparse reward was deliberate. If correctness is decomposed into format, syntax, execution, and test performance, a dense reward requires explicit coefficients that specify how much each component matters. Hardcoding those coefficients is an uncomfortable choice because the values encode my assumptions about the relative importance of different kinds of progress. I initially preferred the most open-ended objective available: produce the correct answer. The policy could then, in principle, discover for itself how much probability to assign to behaviors that improve syntax, execution, and test performance, rather than being guided by coefficients chosen in advance.

That intuition is plausible, but it may apply more to the behavior learned by the policy than to the definition of the reward used to train it. Explicit weights in a reward function are also part of the task specification, much like the detailed instructions in a prompt constrain a model toward a desired behavior. They can provide useful credit for intermediate progress when a model is too weak to reach the final answer often enough for a sparse signal to produce meaningful comparisons. A more capable model might benefit from the freedom of the end to end objective, while a weaker model may need carefully chosen partial rewards to reach the region where final correctness becomes learnable.

The dense reward experiment therefore tests a practical question rather than assuming that dense rewards are universally better. It asks whether explicit partial credit provides this model with a more useful learning signal than an implicit decomposition learned only from full correctness. The coefficients remain hyperparameters rather than objective truths, so they should be treated as a hypothesis to evaluate with ablations and error analysis. If dense rewards improve intermediate behavior but not final correctness, that would suggest that the weighting or the reward components are misaligned rather than proving that sparse reward is preferable.

The next experiment is replacing that binary signal with a dense reward. The proposed reward is

$$
R = 0.05I_{\mathrm{format}} + 0.10I_{\mathrm{syntax}} + 0.05I_{\mathrm{execution}} + 0.80\frac{\mathrm{tests\ passed}}{\mathrm{total\ tests}}.
$$

The training run will also log group-level diagnostics to W&B and the local training log. These diagnostics will measure the fraction of flat-reward groups, the fraction of mixed-reward groups, within-group reward standard deviation, partial-test progress, format failures, and full passes. The purpose is to determine whether the model is producing partially correct candidates and whether those candidates create a useful GRPO learning signal.

The first dense-reward run produced nonzero reward variance and nonzero gradients on all five steps. Four of the five steps had mixed rewards in every generation group, and the remaining step had mixed rewards in seven of eight groups. This confirms that the reward change created a usable training signal. It did not yet improve the held-out pass rate: the baseline scored 21/75 and the final checkpoint scored 19/75. The next question is whether the signal is directionally useful, rather than merely nonzero.

This experiment changes the reward path while keeping the five-step training budget and full evaluation set fixed. A later experiment can increase the number of generations or add supervised warm-start training if dense rewards do not produce enough mixed groups.

#### Improving MBPP from 29 percent to 40 percent

The next improvement came from giving the dense-reward setup a larger training budget and selecting the best intermediate checkpoint. The earlier run reached 22/75 on held-out MBPP, which is 29.3 percent pass@1. I increased training to ten epochs, kept four generations per prompt, used an effective batch size of 32, and evaluated a checkpoint every 25 optimizer steps.

Performance did not improve monotonically. The checkpoints ranged from 24/75 to 30/75 after the first evaluation, and the final training checkpoint scored 26/75. The best checkpoint appeared at step 150 and scored 30/75, which is 40 percent pass@1. Restoring that checkpoint instead of using the final weights therefore preserved an 11 percentage point absolute improvement over the earlier 29.3 percent result.

This result suggests that the dense reward was useful once the policy received enough updates. It also shows that more training was not sufficient by itself. Periodic held-out evaluation and best-checkpoint restoration were necessary because later updates could reduce generalization.

The result remains preliminary. The evaluation contains only 75 examples, and repeatedly selecting a checkpoint against the same split can overfit the model-selection process to that split. A stronger claim will require repeated seeds and a separate test set that is used only after the training and checkpoint-selection procedure is fixed.

#### Supervised fine tuning before reinforcement learning

The next hypothesis is that the model should reach a reasonable level of task competence before reinforcement learning begins. GRPO learns by comparing sampled programs, but a weak policy often produces groups in which every candidate fails. Those groups provide little evidence about which behavior should become more likely. Supervised fine tuning provides a more direct starting signal because each MBPP prompt is paired with a verified reference solution and the model receives token-level feedback throughout that solution.

I therefore added one epoch of response-only supervised fine tuning before GRPO. The initial model passed 22/75 held-out problems, which is 29.3 percent pass@1. After one supervised epoch, it passed 27/75, which is 36 percent pass@1. Format errors also fell from four to one. This is a 6.7 percentage point absolute improvement from a single pass through the training data.

The result supports using supervised learning to establish the base coding policy before asking reinforcement learning to refine it. SFT can improve the model quickly because each demonstration directly supplies a correct answer and provides dense token-level feedback. This is more sample efficient than asking a weak policy to explore a large solution space in which most sampled programs fail.

This does not imply that SFT has an inherently lower ceiling. Its practical gains can saturate when the available demonstrations are narrow, limited, or suboptimal, but better and more diverse demonstrations can raise that ceiling. Reinforcement learning provides a different opportunity because it rewards any program that passes the tests rather than requiring the model to reproduce one demonstrated solution. If the SFT policy is already competent enough to sample promising alternatives, GRPO can explore those alternatives and increase the probability of successful outcomes.

The next objective is therefore to evaluate after each epoch, continue SFT until held-out performance saturates, and then apply GRPO to the strongest supervised checkpoint. The comparison should also include continued SFT under a matched additional compute budget. This will test whether any later gain comes from the outcome-based RL objective rather than simply from performing more optimization steps.

As with the earlier experiments, this finding is based on one seed and a 75-example held-out set. Repeated runs and a final untouched test set will be needed to distinguish a durable improvement from evaluation noise.

#### Task difficulty and reward variance

A related lesson came from a recent onsite hackathon. We took a first step toward replacing one of our frontier language model calls with an open source model and using GRPO to train it. When choosing a suitable task, a colleague pointed out that the task could not be too easy or too hard. If nearly every sampled response succeeds, or nearly every response fails, the rewards have low variance and the training method has little useful signal to learn from.

This is particularly important for GRPO because it samples several responses for the same prompt and computes each response’s advantage relative to the other responses in that group. In simplified form, the advantage is

$$
A_i = \frac{R_i - \mu_{\mathrm{group}}}{\sigma_{\mathrm{group}} + \epsilon}.
$$

If every response receives the same reward, then every centered reward is zero. GRPO cannot identify which response was better, so the group contributes no comparative learning signal. If the rewards differ only slightly, the signal may be too weak to distinguish useful behavior from noise. Normalization by a very small standard deviation can also magnify tiny and unreliable differences, depending on the implementation.

Other policy gradient methods also struggle when rewards or advantages have little variation. The problem is especially pronounced for GRPO because its baseline is computed from the sampled response group itself. GRPO does not use a separately learned critic to estimate the expected value of each prompt or partial trajectory, so it depends heavily on relative differences within the group. Other methods may recover some signal from a learned value function, temporal credit assignment, or comparisons across a broader batch.

This creates a practical task selection rule. The task should be difficult enough to produce failures, but attainable enough that some sampled responses outperform others. The relevant quantity for GRPO is within prompt group variance, not merely overall reward variance across the dataset. A task can have varied rewards across prompts while still producing flat groups and therefore little GRPO training signal.

#### Weakness of instructions

The first finding is that prompt instructions alone are a weak mechanism for enforcing output structure. If the model is told to return a particular format, it often fails to follow the instruction reliably.

For example, I have observed the model place unrelated text before a code block even when the prompt required a code-only response. The parser then receives text before the code, so the generated program cannot be parsed or executed. The failure is operational rather than cosmetic. A small formatting violation can prevent the entire evaluation from producing a reward.

This means that the post-training system cannot assume that the model will produce directly executable output. The pipeline needs defensive handling around the model. Possible mitigations include extracting the first valid code block, removing known wrapper text, normalizing delimiters, validating the result before execution, and asking the model to repair an invalid response.

The broader lesson is that useful post-training systems often need to massage and post-process model outputs. Prompt design remains important, but it should be paired with parsers, validators, and recovery procedures that convert imperfect responses into a form the evaluator can inspect safely.
