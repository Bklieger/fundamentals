## Softmax First Principles

### Motivation
Let's say we have a group of numbers. We do not know their individual values, nor the sum of their values, nor any information about the range of values. They could be all positive, all negative, or mixed between both. The numbers could be very large, very small, or mixed between both.

```
Example A: [1, 5, 6, 3]
Example B: [-12, 120, -203, 2]
```

However, we want to be able to standardize the output such that we can use it in a ML model. Specifically, one way we can compute loss is from a probability distribution (positive numbers that add to 1). In addition, looking at Examples A and B, it is difficult to understand how to compare A and B. This is why we nee to normalize it.

Thus, we need to ensure there are only positive values, but do it in a way that retains the information. In other words, if we had [-100, 100] and wanted to convert these to positive values, simply taking the absolute value would result in [100, 100] which would lose the information provided by the negative sign. The signal that the original value was relatively small is lost.

### Exponentiation

Therefore, softmax takes the approach of solving this via exponentiation, which is a fancy word for saying we take each value and raise a specific other value by the original value. In other words, for each value n, we replace that value n with another specific value raised to the n power.

With softmax, we exponentiate with e. Therefore, in this step, [-100, 100] would result in [e^{-100}, e^{100}]. Here, the negative value becomes very small but still positive, retaining the signal from the original negative sign in -100.

A natural question might be why do we use e? Why not raise 3 or 4 or 100 to each value? This is very intentional. Backpropagation relies on the chain rule. The derivative of e^x is just e^x. Any other base, such as 3^x, would result in 3^x * ln(3). This is because the derivative of b^x is b^x * ln(b). Since ln(e) = 1, e makes this much simpler.

```
After simple exponentiation:

Example A: [e^1, e^5, e^6, e^3]
Example B: [e^-12, e^120, e^-203, e^2]
```

If we calculate the rough values:

```
Example A: [2.718, 148.4, 403.4, 20.09]
Example B: [0.000006144, 1.394×10^52, 0, 7.389]
```

### Stability

However, there is an issue. Before exponentiation, 120 might not have seemed so large. However, e^120 is a very, very large number that risks overflow which could break or disturb the accuracy of the program.

To solve this, instead of just taking each value n and raising e^n, we will raise e to n minus the maximum value in the set of numbers. This shifts the entire output distribution to prevent overflow and makes the largest possible output number be 1, as the largest number n will make e raised to 0 since it is raised to n minus the maximum (n).

This version of exponentiation results a numerically stable result, a name for preventing or significantly reducing the chance of arithmetic accuracy errors from problems like overflow or unintentional rounding, and thus is part of what is called Stable Softmax.

```
After stable exponentiation:

Example A: [e^-5, e^-1, e^0, e^-3]
Example B: [e^-132, e^0, e^-323, e^-118]
```

If we calculate the rough values:
```
Example A: [0.006738, 0.3679, 1.000, 0.04979]
Example B: [1.14×10^-58, 1.000, ~0, 1.47×10^-52]
```

Now, we still have an issue. If you remember, we need a probability distribution. Why? It is required for certain methods of calculating loss, but let's get deeper to first principles. 

### Normalization

We now have a collection of positive numbers, having converted the negative to positive while retaining information about their original size / negativity. However, we know nothing about their sum. In other words, we will have difficulty comparing them to anything else because the magnitude of the values is unknown, not guaranteed, and changes with new calculations.

Comparison is very important. In order to score the output, to know if our model is doing well or poorly and help it get signal to do better, we need to compare these output values to a ground truth. Without standardizing the magnitudes of these collections of values, we cannot do this accurately.

Thus, we need to do what is called normalizing the output. We need to make the magnitude of the collection of numbers standardized, and we do that by making it equal 1. This creates a probability distribution and is accomplished by dividing all the numbers by the total sum of numbers in the collection.

For the first element in Example A, the new value would be: 0.006738 / (0.006738 + 0.3679 + 1.000 +0.04979) = ~0.00473.

We can do this iteratively to get the following:

```
Example A: [0.00473, 0.258, 0.702, 0.0349]
Example B: [1.14×10^-58, 1.000, ~0, 1.47×10^-52]
```

Since the sum of numbers in Example B is already close to 1, not much changes, and in our rough calculation, it stays the same. Rounding Example B slightly to simplify, we get the following:

```
Example A: [0.00473, 0.258, 0.702, 0.0349]
Example B: [~0, 1.000, ~0, ~0]
```

### Softmax!
This is softmax. We have transformed our initial sets of numbers, for which we had no polarity (positive or negative) or magnitude guarantees, and created a set of numbers which is all positive and equals 1 that still carries over the important information about the original sizes. In other words, we made our original set of numbers into a probability distribution. This is softmax.

```
Example A: [1, 5, 6, 3] -> [0.00473, 0.258, 0.702, 0.0349]
Example B: [-12, 120, -203, 2] -> [~0, 1.000, ~0, ~0]
```

### Temperature
However, one thing is left. You will notice, particularly in example B, we significantly magnified the differences between numbers in the set during the exponentiation step. For many uses, this is acceptable. However, sometimes it can make values that are relatively close to the maximum much smaller after stable softmax. This information is sometimes valueable when the numbers that were runner ups to the maximum are particularly important signals.

Thus, there is an optional step that can help alleviate or accelerate this impact. This step is called temperature. Think back to the original stable exponentiation step.

```
After stable exponentiation:

Example A: [1, 5, 6, 3] -> [e^-5, e^-1, e^0, e^-3]
Example B: [-12, 120, -203, 2] -> [e^-132, e^0, e^-323, e^-118]
```

What if we divided the initial array of values first so the differences were not so extreme after exponentiation?

Fundamentally, e^10 - e^5 is much larger, even relative to e^5, than e^2 - e^1 relative to e^1. 

Let's look at before dividing:
```
e^10 - e^5 = 22,026 - 148 = 21878
This difference is about 150x greater than the smaller value!
```

And after dividing by 5, as an example:
```
e^2 - e^1 = 7.389 - 2.718 = 4.671
This difference is only about 1.7x greater than the smaller value!
```

As you can see, by dividing all the numbers in the original set, the exponentiate step has a significantly lower impact on magnifying the differences. In fact, it can even dampen the net impact of exponentiation in its entirety at a high enough temperature, such that the temperature plus exponentiation step results in numbers that are relatively closer together.

Temperature can also be used to magnify the differences. Any temperature greater than 1 will dampen them while any temperature less than 1 will magnify them. A temperature equal to 1 has no effect, since you are just dividing all the numbers by 1, leading to no modifications.


### Application

This prompts the question, when is temperature in softmax used? Or perhaps more pressing, when is softmax even used? 

_Brief interjection from author: The application section is not fully first principles based as I want to connect softmax to other concepts without also constructing those from first principles in this document for the sake of space. Over time, they may appear in their own sections in this repository._

One place softmax is used is at the last layer of many machine learning models. The outputs activations need to be transformed into a probability distribution that communicates the model's predictions. If the predictions are not run through softmax, their range will be unstandardized and more difficult to interpret. 

Another use case is in attention. Specifically, attention uses softmax to create a probability distribution for the previous input tokens that are most important for the model to pay "attention to. 

Temperature is commonly used, as mentioned before, when there is valuable information in the numbers that are not the largest and it might be lost or diluted without applying a temperature. This is relatively common. In fact, the attention mechanism has an implicit temperature:

```
Attention = softmax(QK^T / √d_k) × V
```

QK^T is divided by the square root of the key and query vector dimensions which is effectively scaling using temperature for the softmax function.

Perhaps most well known, the temperature hyperparameter during inference controls how likely tokens that were not the highest probability are to be chosen. 

We can see how this works in vLLM, as an example. 
Looking at the [GPU kernels for applying temperature](https://github.com/vllm-project/vllm/blob/79b6ec6aab4b3d93e421a15ab41ba6d09faadd8f/vllm/v1/worker/gpu/sample/gumbel.py#L9):

```
def _temperature_kernel(...):
...
    logits = tl.load(...)
    ...
    logits = logits / temperature
    ...
```

Since this is a kernel, it has several optimizations. One of them is skipping the load logits step if the temperature is 0 or 1. This is because if the temperature is 0, we just need to use greedy decoding. This is a fancy way of saying we choose the greatest number. Note that softmax with temperature of 0 is actually undefined, since we cannot divide by 0, but inference libraries treat this as hardmax: sample the largest number. If the temperature is 1, the logits remain unchanged, and thus no loading of the logits is needed in this step.

```
temperature = tl.load(...)
    if temperature == 0.0 or temperature == 1.0:
        # Early return to avoid loading logits.
        return
```

To see the temperature step side by side with the subsequent softmax step, we can look away from kernels to the [eagle speculative decoding logic](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/eagle.py), where both steps are back to back in the code:
```
logits.div_(temperature.view(-1, 1))
probs = logits.softmax(dim=-1, dtype=torch.float32)
```

Thats softmax, from first principles!

---
Disclaimer:
Educational, may contain inaccuracies. All numbers are approximate and many are rounded or simplified.

Status: DRAFT

Sources:
1. https://community.deeplearning.ai/t/why-does-softmax-specifically-use-the-exponential-function-x/883919