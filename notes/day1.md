Step 1 in RL is always: Create the EnvironmentThe environment defines:

States - What can the agent observe?
Actions - What can the agent do?
Rewards - What's good/bad?
Transitions - How does the world change?


For Games (Easy!)

States: Board position
Actions: Legal moves
Rewards: Win (+1), Lose (-1), Draw (0)
Transitions: Deterministic rules

This is why games are popular for learning RL - environment is simple to code!

Some Neural Network Notes-

1.YES! Exactly right!

## Summary of First Step

```python
self.conv_input = nn.Conv2d(
    in_channels=1,
    out_channels=128,  # num_filters=128
    kernel_size=3,
    padding=1
)
```

**What this does:**

**Input:** Board of shape `(1, 1, 8, 8)`

**Operation:** Multiply with 128 different 3×3 kernels (which we learn during training)

**Output:** 128 feature maps of shape `(1, 128, 8, 8)`

---

## The Learnable Part

Inside `self.conv_input`, PyTorch stores:

```python
self.conv_input.weight.shape = (128, 1, 3, 3)
                                ^^^
                        These 128 kernels are what we LEARN!

self.conv_input.bias.shape = (128,)
                              ^^^
                        Also learned (one bias per kernel)
```

During training, gradient descent updates these 128 × 9 = 1,152 kernel weights + 128 biases.

---

**So yes**: 
- Input × Kernels (128 of them)
- The kernels are learned through backpropagation
- This is the very first transformation in the network!

