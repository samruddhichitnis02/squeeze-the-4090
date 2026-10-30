# vLLM demo: Qwen/Qwen2.5-1.5B-Instruct

temperature=0.7, top_p=0.9, max_tokens=256, seed=2122
4 prompts in one batch: 885 tokens in 2.44s (363 tokens/sec)

## Prompt
Explain, step by step, why decoding one token at a time on a GPU is limited by memory bandwidth rather than compute.

## Response (256 tokens, length)
Sure, let's break down the process of decoding one token at a time on a GPU and why it is limited by memory bandwidth rather than compute.

### Step 1: Tokenization and Memory Layout

1. **Tokenization**: The input sequence is first tokenized into individual tokens. Each token is a small piece of information (e.g., a word, a character, etc.). For instance, if we have a sentence "Hello world", the tokens would be "H", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d".

2. **Memory Layout**: These tokens are then stored in memory, typically as a contiguous block. In a GPU, these tokens are usually stored in a contiguous block of memory, often referred to as a "block" or "warpage".

### Step 2: Token Decoding

1. **Decoding Loop**: During decoding, a loop iterates over each token in the input sequence one by one. For example, if the input sequence is "H", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d", the decoding loop would iterate as follows

---

## Prompt
I have a 24 GB GPU and a 7B parameter model. List what takes up memory at inference time and estimate each in GB.

## Response (256 tokens, length)
When running a model at inference time, memory usage can vary depending on the specific model architecture, the complexity of the input data, and the hardware configuration. However, for a 7B parameter model running on a 24 GB GPU, the memory usage typically follows a pattern based on the model's architecture and how it processes the input data.

### 1. **Model Parameters and Weights**
- **Total Parameters**: 7 billion parameters.
- **Approximate Memory Usage**: For a 7B parameter model, the approximate memory usage is around **10-20 GB**. This includes the model parameters, biases, and other weights.

### 2. **Intermediate Outputs and Gradients**
- **Intermediate Outputs**: The model may generate intermediate outputs, which can be large. Depending on the model, these intermediate outputs can take up additional memory.
- **Gradients**: Backpropagation updates the model's weights with gradients. This can also consume significant memory.
- **Approximate Memory Usage**: For a 7B parameter model, the memory usage for intermediate outputs and gradients can be **10-20 GB** as well. This is because the gradients can be quite large.

### 3. **Input Data**
- **Input Size

---

## Prompt
Write a one-paragraph explanation of continuous batching for a new engineer, then give one concrete example.

## Response (117 tokens, stop)
Continuous Batching is a technique in machine learning where data is processed and batches are created in real-time as it arrives. This allows for more efficient use of computational resources and can improve the training speed of deep learning models. For example, if you have a streaming video application that needs to classify the objects in each frame as it streams in, continuous batching would allow the application to process and classify frames as they arrive, rather than waiting for all the frames to be downloaded and stored in memory before processing. This can significantly reduce the latency of the application and improve the overall user experience.

---

## Prompt
Give me three experiments I could run to find out whether my LLM server is bottlenecked by prefill or decode.

## Response (256 tokens, length)
To determine if your Large Language Model (LLM) server is bottlenecked by prefill or decode, you can run the following experiments:

### Experiment 1: Simulate Large Input Sizes
**Objective:** Measure the time it takes for your server to process large input sizes.

**Steps:**
1. **Generate Large Input:** Use a tool like `fuzzing` to generate very large inputs that are as close to the maximum input size as possible.
2. **Measure Processing Time:** Record the time it takes for the server to process the input.
3. **Analyze Results:** Compare the processing times for different input sizes. If the processing time increases linearly with the input size, it suggests that the bottleneck is in the prefill phase.

**Expected Outcome:**
- If the processing time increases linearly with the input size, it indicates that the bottleneck is likely in the prefill phase.
- If the processing time increases more slowly or not at all, it suggests that the bottleneck is in the decode phase.

### Experiment 2: Use Different Prefill Methods
**Objective:** Compare the performance of different prefill methods.

**Steps:**
1. **Pre-fill with Different Methods:** Use different prefill methods (e.g., `join`,

---
