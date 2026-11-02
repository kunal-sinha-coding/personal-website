# Optimizing a Cheap Decision Model with Auto Research

### TL;DR

I fine-tuned a 1.5 billion parameter Qwen model that reached 74.18% accuracy on the 3,652-example When2Call benchmark. This achieved performance comparable to state-of-the-art decision models at a fraction of the cost and compute requirements. At an estimated $3.09 per million tasks, it costs about 13.4× less than Jev, 64.6× less than Clef, and 2.7× less than Decider Flash. With regards to compute, it uses 18.7× fewer parameters than Clef and 2.7× fewer parameters than Decider Flash.

While training this model, I set up an auto-research loop to explore strategies for optimizing training and inference. The second part of this blog post will explain how the loop worked, what worked well, and what did not.
