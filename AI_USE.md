# AI use statement

## Tools
- Kiro IDE agent (Claude model, chosen automatically by the IDE) used for the whole project, through a chat interface.
  The assistant could run shell commands, create files and read results inside this workspace and the WSL Ubuntu install.

## What the AI assistant did
- Read the assignment text the author pasted (the PDF has no extractable text layer) and proposed the plan.
- Wrote all code in `src/` and `scripts/`: the BERT fine-tuning script, the serving benchmark, the batch-scaling sweep, the memory-technique
  comparison and the report generators.
- Ran the experiments through those scripts and diagnosed failures (missing C compiler, nvcc/CUDA header mismatch breaking
  FlashInfer, a multiprocessing guard bug, Windows driver spilling GPU memory to system RAM instead of raising out-of-memory,
  a command-line flag typo).
- Wrote the analysis files (`results/step*/ANALYSIS.md`), `METRICS.md`, `RUN_LOG.txt`, the GPU usage record and this file.
- Suggested commit messages and `git add` commands (the author ran them).

## What the author did
- Chose the project direction (finish the assignment, then extend it into an inference-optimization study),
  the dataset (Hugging Face tweet_eval sentiment), the seed (2122), and the scope (one RTX 4090, no second card, no kernel writing yet).
- Installed Ubuntu/WSL, Python 3.12 and the build tools that needed administrator rights.
- Reviewed each step's results before approving the next one.

<!-- AUTHOR: edit the lists above so they are accurate for your own contribution, and add anything you wrote, changed or checked yourself. -->

## Verification of AI output
- Every number in the tables comes from a script run recorded in `logs/` (timestamps, GPU UUID, package versions) and raw JSON in `results/`.
- Checks run on the AI's own work: the edited training script was re-run with all techniques off and reproduced the original
  results exactly (test accuracy 66.53%, epoch-1 train loss 0.88937967); a suspicious equal memory value (13.86 GB in two rows)
  was compared at full precision and found to be a coincidence; a stalled run was recognised as memory spilling and fixed.
- Known limits stated in the analysis files: one card, one seed, small test sets (accuracy differences within noise),
  large-batch runs sample with replacement, epoch times in the scaling table are estimates.
- Items the AI could not check: whether the SLICE value (0) matches the one the course intended, since none was supplied.
