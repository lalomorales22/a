# aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

### *a GPT so pure, so atomic, so... aaaa... that even its variable names have been reduced to their most fundamental particle: the letter a.*

---

> *"This file is the complete algorithm. Everything else is just efficiency."*
> — Andrej Karpathy
>
> *"Hold my aaaa."*
> — Us

---

## Dedicaaaaation

This project is dedicated to **Andrej Karpathy** — the man, the myth, the `aaaaaaaaaaaaaaaaaaaa.log()`.

On a random Tuesday, Andrej decided to mass-humble the entire machine learning community by writing a complete GPT — autograd engine, transformer, tokenizer, training loop, and inference — in a single file of pure Python with zero dependencies. No PyTorch. No TensorFlow. No NumPy. Just `math`, `random`, and the audacity of a man who has seen the matrix and decided to rewrite it from scratch.

We saw this masterpiece and thought:

*"What if we made it worse?"*

And so, `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa` was born.

---

## What is this?

This is Andrej Karpathy's [GPT from scratch](https://github.com/karpathy/llm.py) — a fully functional transformer language model implemented in pure, dependency-free Python — except every single user-defined identifier has been replaced with incrementing repetitions of the letter **a**.

The first variable is `a`.
The second is `aa`.
The third is `aaa`.
The 103rd is... well...

```
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
```

That's `sample_idx`. It has 103 a's. We counted.

---

## The Decoder Ring

For the brave souls attempting to read this code, here is the sacred translation scroll:

| a's | Original Name | What It Actually Does |
|-----|--------------|----------------------|
| 1 | `names_url` | URL to the training data |
| 2 | `docs` | the training dataset |
| 3 | `l` | a loop variable that lived a short life |
| 4 | `chars` | character vocabulary |
| 5 | `vocab_size` | size of said vocabulary |
| 6 | `stoi` | string-to-integer mapping |
| 7 | `ch` | a single character |
| 8 | `i` | the letter i but now its 8 a's |
| 9 | `itos` | integer-to-string mapping |
| 10 | `BOS` | beginning-of-sequence token |
| 11 | `Value` | THE AUTOGRAD ENGINE (a whole class in 11 a's) |
| 12 | `data` | the actual number stored in a Value |
| 15 | `grad` | the gradient. the whole point of backprop |
| 16 | `_backward` | the function that does the chain rule |
| 18 | `other` | the other operand. you know, the other one |
| 19 | `out` | the output of an operation |
| 20 | `log` | natural log (the Value method, not math.log) |
| 21 | `exp` | exponential (same deal) |
| 22 | `relu` | rectified linear unit, beloved activation |
| 23 | `backward` | THE backpropagation function |
| 27 | `v` | a variable called v that is now 27 a's of irony |
| 29 | `n_embd` | embedding dimension (16) |
| 30 | `n_head` | number of attention heads (4) |
| 31 | `n_layer` | number of layers (1) |
| 32 | `block_size` | max sequence length (8) |
| 33 | `head_dim` | dimension per attention head |
| 34 | `matrix` | the parameter initializer lambda |
| 39 | `state_dict` | ALL the model weights live here |
| 40 | `params` | flattened list of every trainable parameter |
| 44 | `linear` | linear layer forward pass |
| 45 | `x` | the input. the most important letter in ML is now 45 a's |
| 50 | `softmax` | turn logits into probabilities. 50 a's of normalization |
| 51 | `logits` | raw model outputs before softmax |
| 57 | `rmsnorm` | root mean square normalization |
| 60 | `gpt` | **THE ACTUAL GPT FUNCTION.** 60 a's of transformer |
| 61 | `token_id` | which token we're looking at |
| 68 | `q` | query vector in attention (68 a's for a single letter variable) |
| 69 | `k` | key vector in attention (nice) |
| 78 | `attn_logits` | attention scores before softmax |
| 80 | `attn_weights` | attention weights after softmax |
| 82 | `a` | a variable literally named 'a' is now 82 a's. let that sink in |
| 84 | `learning_rate` | how fast we learn (1e-2) |
| 85 | `beta1` | Adam first moment decay |
| 86 | `beta2` | Adam second moment decay |
| 88 | `m` | first moment buffer (88 a's for a momentum tracker) |
| 89 | `num_steps` | 500 steps of training |
| 90 | `step` | current training step |
| 93 | `n` | sequence length. one letter. 93 a's |
| 96 | `probs` | probability distribution |
| 98 | `loss` | how wrong the model is. beautifully expressed in 98 a's |
| 100 | `m_hat` | bias-corrected first moment. a milestone in a's |
| 102 | `temperature` | controls generation creativity |
| 103 | `sample_idx` | the final boss. 103 a's |

---

## Some Fun Faaaacts

**File size comparison:**
- Original: **9,989 bytes** of clean, readable, educational Python
- This version: **30,590 bytes** of aaaaaaaaaaaaaaaa
- Bloat factor: **3.1x** — all from the letter a

**The variable `a`** (originally `names_url`) is used exactly twice in the entire file. It got the shortest name. Life isn't fair.

**The variable `i`** — universally the simplest loop counter in all of programming — is now `aaaaaaaa` (8 a's). Some things just can't be simple.

**The variable literally named `a`** in the original code (used in list comprehensions like `a + b`) became `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa` — 82 a's. The letter a became 82 of itself. This is either poetry or a crime.

**The attention mechanism section** contains lines where 6+ different `aaaa` variants of different lengths are doing matrix multiplication with each other inside nested list comprehensions. If you can read it, you are either lying or you are Andrej Karpathy.

**Total unique identifiers renamed:** 103

**Times the letter 'a' appears in variable names alone:** we stopped counting. our IDE caught fire.

---

## Does It Actually Run?

```
✅ Valid Python (verified by ast.parse)
```

It parses. We believe it runs. We have not confirmed this because staring at the output would give us the same energy as staring at the code and we need to protect what's left of our mental health.

---

## How We Did It

We wrote a Python script that:

1. Tokenizes the original source using Python's `tokenize` module
2. Identifies every user-defined name (variables, functions, classes, method names, parameters)
3. Carefully preserves everything that must stay: Python keywords, builtins, module names (`os`, `math`, `random`), module attributes (`.strip()`, `.append()`, etc.), dunder methods (`__init__`, `__add__`), and keyword arguments (`end=`, `weights=`)
4. Assigns each unique identifier an incrementing `a` pattern based on order of first appearance
5. Reconstructs the source with all replacements, including inside f-strings
6. Produces valid, parseable Python that is functionally identical to the original

No AI was harmed in the making of this code. Several humans were.

---

## Why?

Because Andrej wrote the most beautiful, minimal, educational implementation of a GPT ever created, and we wanted to see what it would look like if you removed the one thing that made it educational: the ability to understand what anything is.

The algorithm is still there. The architecture is still there. The math is still there. The knowledge... is now encoded in a's.

Kind of like how the model's knowledge is encoded in weights, except our encoding scheme is significantly worse.

---

## The Philosophy of aaaaaa

Andrej said *"Everything else is just efficiency."*

Variable names? Efficiency.
Readability? Efficiency.
The ability to tell `q` from `k` from `v`? Efficiency.

We have removed the efficiency. What remains is the algorithm in its truest form — a sequence of `a`'s performing gradient descent, one incomprehensible line at a time.

This is machine learning the way machines see it.

---

## How to Use

```bash
# Clone it
git clone <this-repo>

# Stare at it
cat obfuscated_gpt.py

# Question your life choices
python3 obfuscated_gpt.py

# Seek therapy
```

---

## Requirements

- Python 3.x
- No dependencies (faithful to the original)
- A screen wide enough for 103-character variable names (optional but recommended)
- Emotional resilience

---

## Credits

**Original GPT implementation:** [Andrej Karpathy](https://github.com/karpathy) — thank you for making ML accessible, understandable, and beautiful. We're sorry for what we've done to your code. But also not really.

**The aaaa-ification:** Built with Claude by a man who looked at the most elegant ML code ever written and said *"yeah but what if every variable was just the letter a"*

---

## License

Same as the original. We added no intellectual property. We removed some. Mostly the intellectual part.

---

<p align="center">
  <i>In the beginning, there was a.</i><br>
  <i>And a begat aa.</i><br>
  <i>And aa begat aaa.</i><br>
  <i>And on the 103rd generation, there was sample_idx.</i><br>
  <i>And it was good.</i><br>
  <i>And it was aaaa.</i><br>
</p>

---

<p align="center">
  <b>⭐ Star this repo if you too believe variable names are just efficiency ⭐</b>
</p>
