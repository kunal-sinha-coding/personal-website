# Optimizing a Cheap Decision Model with Auto Research

### TL;DR

I fine-tuned a 1.5B parameter Qwen model to strengthen performance on When2Call, a benchmark evaluating an Agent's ability to decide when to make tool calls. Despite its small size, our model reached 74.18% accuracy, closely matching Perplexity's Decider-Flash v1.2 (74.41%), OpenAI's GPT-6 Luna (74.06%), and Cloudflare's Clef 28B (70.96%). At an estimated $3.09 per 1 million tasks, our model costs about 2.7× less than Decider Flash, 96.1% less than GPT-6 Luna, and 64.6× less than Clef.

I was able to obtain these results more easily after building auto-research loop in Codex that automatically explored strategies for optimizing training and inference. The second half of this blog post will dive into the design of the loop, highlighting what worked well and what did not.
