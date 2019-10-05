# Making sense of TensorFlow 2.0
## Thoughts
The process becomes syntactically similar to PyTorch, no more usage of sessions and explicitly defining graphs. I like the addition of the ability to record metrics using a dedicated metrics class, which also provides methods for calculating things like KL-Divergence, etc.

## Tensorflow 2.0 Quick Start
- Import required libraries.
- Load data tensors and pre-process them.
- Create the `Dataset` objects for training and testing:
    - Initialize with `from_tensor_slices()`.
    - Shuffle and batch with `shuffle(num_samples)` and `batch(batch_size)`.
- Define model, loss function, and optimizer.
- Define metrics for accumulation.
- Define training and test loops:
    - Use `@tf.function` decorator to compile to graph.
    - Training loop:
        - Use `tf.GradientTape` to record operations.
        - Pass data through model, record loss and weights.
        - Apply gradient update using optimizer.
        - Accumulate metrics.
    - Test loop:
        - Pass data through model, record loss.
        - Accumulate metrics.
- Execute the training and test loops.
- Reset states of metrics with `reset_states()`.

