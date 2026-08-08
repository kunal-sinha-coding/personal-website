# Why I Am Starting RL Experiments With a Tiny Model

I am going to explore reinforcement learning on a tiny language model to see how far we can improve its performance. The goal is not to claim that a small experiment answers every question about post training. The goal is to build a fast and rigorous place to test ideas.

## Why start with a tiny model?

There are practical reasons to begin small.

- Training is cheap. As an independent researcher, I can run more experiments without needing expensive hardware or a large budget.
- Training is quick. A shorter training cycle means that I can inspect results and start the next experiment sooner.
- The setup is easier to understand. A small model makes it easier to inspect data, prompts, rewards, and failure cases together.

The speed of iteration is probably the biggest factor in determining how quickly we can hill climb on a metric. Faster experiments let us explore more hypotheses in the same amount of time. They also make it easier to discard weak ideas before we become attached to them.

There is also a research reason to use a tiny model. Larger models are already close to saturated on some common benchmarks. A tiny model often starts with very low performance, which gives an RL or post training method much more room to improve the baseline.

When the baseline model is poor, post training becomes especially important. That makes a tiny model a fruitful place to explore different post training strategies. We can learn which parts of the training regime create real gains before testing whether those ideas transfer to larger models.

Tiny models are also much more brittle in my experience. They are more sensitive to subtle differences in context, including the prompt, the data format, and the exact instructions. That sensitivity forces more rigor. A researcher has to pay closer attention to every part of the experiment because a small formatting change can alter the result.

## Letting the agent run experiments

I am also experimenting with RSI techniques in which the agent runs its own experiments and checks its own results over a limited number of cycles. The current process is intentionally basic. The agent can make a change, run an experiment, inspect the result, and use the error pattern to decide what to try next.

One part of this process is the error analysis skill. It reads the logged failures, groups recurring patterns, and helps identify whether a problem is likely caused by the data, the reward, the prompt, or the training setup. This gives the next experiment a more informed starting point than simply choosing another change at random.

This is a very basic approach so far. I will refine it later as I learn which parts of the loop produce useful improvements and which parts only create noise.
