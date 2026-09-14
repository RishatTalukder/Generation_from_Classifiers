# Generation_from_Classifiers

Experiments with training a generative model using a pretrained classifier as its primary source of feedback.

This is an exploration of an idea I independently came up with, only to discover that someone had already explored a very similar direction in 2023.

**THREE YEARS. ONLY THREE GODDAMN YEARS.**

AAAAAHHHHHHHHHHHHHHHH.

Anyway, the idea is still interesting, and I want to see what happens when I actually build it.

This will also be my first serious attempt at building a generative model from scratch.

## The Idea

A classifier trained on handwritten digits learns to distinguish between different classes. It learns patterns that help it decide whether an image is a `0`, `1`, `2`, and so on.

We don't explicitly tell the classifier what a digit *looks like*. We simply provide images and their labels, and through training, it learns a function that maps an image to a probability distribution over the possible labels.

For example:

```text
Image of a handwritten 6
            ↓
       MNIST Classifier
            ↓
[0.01, 0.02, 0.03, 0.04, 0.01, 0.02, 0.85, 0.01, 0.01, 0.00]
```

The classifier has learned something about the patterns that make an image look like a `6`.

But what if we use that knowledge in the opposite direction?

Instead of giving the classifier an image and asking:

> "What digit is this?"

What if we give a **generative model** a digit and ask:

> "Can you generate an image that the classifier recognizes as this digit?"

That is the idea I want to explore.

## The Proposed Architecture

The initial experiment uses two neural networks:

1. **A generator** that produces a `28 × 28` image.
2. **A pretrained classifier** that evaluates the generated image.

The classifier is trained normally on MNIST beforehand. Once training the generator begins, the classifier's parameters are frozen.

```text
                    Random noise
                         │
                         │
                    Digit label
                         │
                         ▼
                 ┌──────────────┐
                 │   Generator  │
                 │      G       │
                 └──────┬───────┘
                        │
                        ▼
                  28 × 28 image
                        │
                        ▼
                 ┌──────────────┐
                 │   Classifier │
                 │      C       │
                 │   (frozen)   │
                 └──────┬───────┘
                        │
                        ▼
                 Softmax probabilities
                        │
                        ▼
                 Cross-entropy loss
                        │
                        ▼
                 Backpropagation
                        │
                        ▼
                 Update generator
```

The classifier is not trained during this process. Its job is to provide feedback about the image produced by the generator.

The generator's job is to learn how to produce images that satisfy that feedback.
<!-- 
## How Training Works

Suppose the desired digit is `6`.

The generator receives a digit label and, eventually, some random noise:

```text
Generator(noise, 6) → generated image
```

That image is passed through the pretrained classifier:

```text
Classifier(generated image) → probability vector
```

Then the probability vector is compared with the desired label using cross-entropy:

```text
loss = CrossEntropy(classifier_output, target=6)
```

If the classifier is not confident that the generated image is a `6`, the loss will be higher.

Backpropagation then calculates how the generator's parameters should change to reduce that loss.

The important part is that **the classifier remains frozen, but gradients are still allowed to pass through it**.

The gradient flows like this:

```text
Loss
  ↓
Classifier's input
  ↓
Generated image
  ↓
Generator's parameters
```

The classifier's parameters do not change. Only the generator's parameters are updated.

Mathematically, the process looks like:

$$
x_{\text{generated}} = G_\theta(z, y)
$$

$$
p = C_\phi(x_{\text{generated}})
$$

$$
L = CE(p, y)
$$

where:

* \(G_\theta\) is the generator,
* \(C_\phi\) is the pretrained classifier,
* \(z\) is random noise,
* \(y\) is the desired digit,
* \(\theta\) is updated,
* \(\phi\) remains fixed.

The generator is optimized using:

$$
\nabla_\theta L
$$

while the classifier's parameters are kept constant.

## Does the Generator Need the Original Dataset?

This is one of the questions that led me to the idea.

Once the classifier has already been trained on MNIST, the generator does not necessarily need to see the original MNIST images during its own training.

Instead, it can receive randomly sampled labels:

```text
6 → Generator → Image → Classifier → Loss
```

The classifier provides the training signal.

For example:

```python
for step in range(num_steps):
    digit = random.randint(0, 9)

    generated_image = generator(noise, digit)

    prediction = classifier(generated_image)

    loss = cross_entropy(prediction, digit)

    loss.backward()

    optimizer.step()
```

The generator is not directly comparing its output against a real MNIST image.

It is learning through the classifier's response.

This raises an interesting question:

> **Can a generator learn to produce realistic images using only the feedback of a pretrained classifier, without directly training on the original dataset?**

That is what I want to investigate.

## What I Expect to Happen

My initial expectation is that the generator may learn to produce images that the classifier recognizes correctly.

But that does not necessarily mean the images will look realistic.

The classifier's objective is to identify the correct class. It is not explicitly checking whether the image looks like a genuine handwritten digit.

So the generator might discover strange patterns that produce a high-confidence classification.

For example:

```text
Generator → Weird image → Classifier → "6: 99.9%"
```

The classifier might be extremely confident, while a human might look at the image and say:

> "That is not a 6."

This is one of the main things I want to observe. -->

## Questions I Want to Explore

* Can a generator learn to produce recognizable digits using only classifier feedback?
* Does the generator actually learn meaningful visual patterns?
* Does it learn to produce realistic handwritten digits, or merely images that fool the classifier?
* What happens when random noise is introduced?
* Does the generator produce different handwriting styles for the same digit?
* Does it collapse into producing one prototype per class?
* How does the quality of the generated images change during training?
* Can additional constraints improve realism?
* What happens when the number of classes increases?
<!-- 
## Planned Experiments

### 1. Classifier-only generation

Train a generator using only:

```text
Random label → Generator → Frozen classifier → Cross-entropy
```

No direct comparison with real MNIST images.

### 2. Different noise vectors

Give the generator different random noise vectors while keeping the label fixed.

For example:

```text
z₁ + 6 → Image 1
z₂ + 6 → Image 2
z₃ + 6 → Image 3
```

The goal is to see whether the generator learns meaningful variation or simply ignores the noise.

### 3. All ten MNIST classes

Train the generator to produce all digits from `0` through `9`.

### 4. Compare classifier confidence with visual quality

A high classification accuracy does not necessarily mean high-quality generation.

I want to compare:

* Classifier confidence
* Human recognizability
* Diversity
* Similarity to real MNIST images

### 5. Experiment with additional constraints

If classifier-only training produces strange images, I want to investigate whether adding other objectives can encourage more realistic generation. -->

## What This Repository Is

This repository is primarily a **learning and experimentation project**.

I am not trying to claim that the underlying idea is new. In fact, I discovered that similar research already exists.

The goal is to understand the idea by implementing it myself, observing its behavior, and figuring out what works and what doesn't.

I want to learn generative modeling by actually building something, not just reading about GANs, VAEs, diffusion models, and other architectures.

## References

The closest existing work I found is:

* **Generator Born from Classifier**: a paper exploring how to reconstruct a generator from a pretrained classifier without access to the original training data.

This project is an independento exploration inspired by that general direction.

---

**Let's see what happens.**
