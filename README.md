# Trigram Character-Level Name Generator

This project implements a simple character-level language model using a trigram approach in PyTorch. The model is trained on a dataset of names to generate new, name-like sequences of characters.

## 📚 How It Works

- Each training example is a sequence of 3 characters: `(ch1, ch2) -> ch3`
- We train a linear model that takes one-hot encodings of `ch1` and `ch2`, concatenates them, and predicts the probability distribution over the next character `ch3`
- The model is trained using cross-entropy loss
- After training, names are generated character-by-character using the learned trigram distribution

## 🧠 Model Details

- **Input**: Two characters `(ix1, ix2)` represented as one-hot vectors (size 27 each)
- **Architecture**: Linear layer with optional bias
- **Output**: Probability distribution over 27 possible characters (26 letters + `.` for end)

## 🔧 Requirements

- Python 3
- PyTorch

## 🛠️ Training

The model is trained using the following loop:

```python
for i in range(num_iterations):
    x1 = F.one_hot(xs[:, 0], num_classes=27).float()
    x2 = F.one_hot(xs[:, 1], num_classes=27).float()
    xenc = torch.cat([x1, x2], dim=1)  # [N, 54]
    logits = xenc @ W + b
    log_probs = F.log_softmax(logits, dim=1)
    loss = -log_probs[torch.arange(xs.shape[0]), ys].mean()

    W.grad = None
    b.grad = None
    loss.backward()
    W.data += -learning_rate * W.grad
    b.data += -learning_rate * b.grad

