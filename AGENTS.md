# Repository Instructions

## Code style

- Favor simplicity over complexity. Do the minimal possible code changes required to achieve the objective.
- Leave inline comments on every function and block of code. Each comment should use complete sentences, but should be as short and simple as possible to communicate the idea. All inline comments must be between 1-2 sentences.

## Writing style

- Be concise when writing. Only include the minimal amount of writing needed to communicate the point.
- Always use complete sentences rather than sentence fragments.
- Do not use punctuation like semicolons, hyphens, or arrows to join sentence fragments together.
- Instead of writing long paragraphs, break up content into short paragraphs.
- Feel free to use bullet point and numbered lists whenever appropriate.
- Feel free to insert Markdown code blocks or LaTeX math whenever appropriate.
- For blog posts, follow the heading hierarchy used by prior blog posts. Use the same heading levels for comparable sections.
- For blog posts, use a technical style similar to a research report. State the motivation, assumptions, methods, observations, and limitations clearly.

## Parallelization

- Always look for independent work that can run in parallel unless the task is trivial or quick.
- Identify which work can run concurrently and which work depends on earlier results, then sequence only the dependent work.
- Run independent work in parallel with subagents whenever available, then integrate and verify their results before continuing with dependent steps.
- Coordinate edits to shared files to avoid conflicts.

## Git workflow

- After making any set of changes to the code, always commit the changes directly on the `main` branch and push `main` to the repository.

## Patch workflow

- In this environment, the direct patch helper may fail because its sandbox cannot create an unprivileged user namespace. Apply patches through an elevated `exec_command` invocation instead.
- When passing a patch through a nested shell, protect `$` variables, backticks, and quotes from shell interpolation. Base64-encode the patch and pipe it through `base64 -d | apply_patch` when necessary.

## Synchronization

- Keep this `AGENTS.md` synchronized with the `AGENTS.md` files in the other local repositories.
- Synchronize any change made to any local `AGENTS.md` across all three repositories.
