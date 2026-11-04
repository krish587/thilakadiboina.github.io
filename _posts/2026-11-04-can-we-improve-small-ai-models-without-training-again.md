---
title: "Can We Improve Small AI Models Without Training Them Again? What If AI Could Fix AI?"
description: "Two new research papers explore how AI can learn from a weaker version of itself‚Äîand even identify and fix problems in another AI model. An accessible WISD and Weight-Space Programming explanation with a Gemma LoRA example."
date: 2026-11-04 10:00:00 +0530
categories: [AI Research, LLMs]
tags: [ai, gemma, fine-tuning, lora, wisd, weight-space-programming]
redirect_from: /posts/improving-ai-models-without-retraining/
---

*Two new research papers explore how AI can learn from a weaker version of itself‚Äîand even identify and fix problems in another AI model.*

Most of us use AI in a simple way. We ask a question, look at the answer, and change the prompt if something is missing.

But imagine you are building an AI assistant for a customer-support team. You want it to answer clearly, follow a consistent style, and know when it does not have enough information. You might show it examples of good responses so it learns your preferred way of answering.

That is **fine-tuning**. With smaller models like **Gemma**, we can now do this without building a model from scratch. And two new papers are exploring what might come next.

Now two recent research studies are exploring very different possibilities:

- **WISD (Weaken to Strengthen):** can a model learn from a *weaker copy of itself*?
- **Weight-Space Programming:** can another AI identify a model's problem and *directly change a few of its internal weights*?

These are **two distinct research methods**. One still trains the model, but learns from a weaker copy. The other tries to fix particular problems by changing weights directly, without gradient training. Neither makes conventional fine-tuning obsolete. Together they raise a useful question: **What is the smartest way to improve an AI model, given the kind of failure we see?**

## A simple example: teaching an AI to answer better

Suppose a customer asks, ‚ÄúWhy is my bill higher this month?‚Äù

A poor assistant might invent a reason: ‚ÄúYour plan price went up.‚Äù

A better assistant would say something like: ‚ÄúThere could be several reasons, such as one-time charges or a change in usage. I would need to check the bill before confirming the cause.‚Äù

How can we get more responses like the second one? There are several options. Some are ready to use today. Others are still research.

| Approach | The simple idea | Where it stands |
| --- | --- | --- |
| **Prompting or retrieval** | Give better instructions, and let AI check trusted documents when needed | Practical today |
| **LoRA fine-tuning** | Show a small model good examples so it learns a pattern | Practical today |
| **WISD** | Train the model using the difference between itself and a weaker copy | Promising research; not a simple Gemma recipe |
| **Weight-Space Programming** | Use another AI to reason about and directly edit model weights | Early research |

One important point: **fine-tuning is not the best way to store facts that change every day**. A current bill, account balance, or live network problem should come from an authorized, up-to-date system. Fine-tuning is more useful for consistent task behavior and how to respond.

## From better prompts to changing the model itself

A useful way to connect this research is with something nearly every AI user already does: **improving a prompt after seeing a bad answer.**

Think about how our approach has evolved:

1. **Improve the prompt.** We change our instructions to get a better response. A familiar example is asking AI to work through a problem step by step.
2. **Let AI improve prompts.** Research such as OPRO and GEPA explores systems that examine previous attempts, errors, and scores to propose better instructions. The important change is the feedback loop: try, observe, understand, improve.
3. **Improve the tools around AI.** Instead of editing just a prompt, an agent can improve its reusable skills, instructions, workflow or the code that trains models.
4. **Improve the weights themselves.** Weight-Space Programming asks whether a general-purpose AI can examine the model's numerical parameters, identify a problem and directly change them.

This last step is the new idea. Before, AI could write *training code*, but an optimizer such as Adam still made the weight updates. Here, the AI proposes weight changes itself, without the usual gradient-training loop.

A simple analogy: prompts are like the instructions we give an employee, skills and tools are their working process, and model weights are closer to the internal habits the employee has learned. Changing the habits is much harder than rewriting the instructions.

**The common thread is feedback:** see what failed, work out why, make a targeted change, then test whether it really helped. This is also why the amount of testing matters as much as the number of weights changed.

## 1. The practical starting point: fine-tune Gemma with examples

**Gemma** is a family of smaller open-weight AI models from Google. For a beginner-friendly experiment, we can start with **Gemma 3 270M**, which Google designed for specialized tasks and fine-tuning.

Fine-tuning doesn't mean building a new AI model from zero. We take an existing one and show it examples of the behavior we want.

**LoRA** helps reduce the amount of training needed. Instead of updating all of the model's weights, it trains a smaller set of adjustments, called an *adapter*. You still need to train the adapter, but the memory requirements are typically lower than updating every weight.

For example, show the model many customer questions and well-written answers. Then ask it a *new* customer question and check whether it follows the same pattern.

That last part is important. **A model that memorizes examples is not the same as a model that has learned to answer new questions better.**

Below, I walk through a real public dataset and give the starter Python scripts to try it yourself.

## 2. WISD: can AI learn from a weaker version of itself?

The first paper is **[Weaken to Strengthen (WISD)](https://arxiv.org/abs/2610.38681)**, from researchers at MIT and NVIDIA. I find the idea easy to explain, even if the implementation is quite technical.

Imagine you are learning to solve a problem. Usually someone shows you the correct answer. But what if you don't have a teacher or a set of correct solutions?

WISD starts with the original AI model and makes a **deliberately weaker copy**. The researchers do this by *quantization* (representing the model's numbers with less precision) or *pruning* (removing some small weights). The original model is treated as the stronger reference, not as a source of guaranteed truth.

Then the system follows four steps:

1. **Give it questions.** The questions do not need labelled correct answers.
2. **Let the original model generate possible solutions.** These are the paths the model itself would normally take.
3. **Compare the original with its weaker copy.** On each step, look at which next words or decisions the original model favors more than the weakened model.
4. **Train an improved student.** Use those differences as a learning signal. After training, the student runs by itself; we do not need both models at inference time.

So the model isn't simply copying whichever final answer looks better. It learns from **differences in the models' next-token probabilities**, across the steps of a response.

**And here is the important distinction: WISD still uses gradient-based training.** Creating the weaker version can be training-free, but learning from that contrast is not. It saves the need for a separate teacher or reference answers in its reported setup; it does not eliminate GPU requirements.

### What did WISD actually achieve?

In the researchers' experiments with reasoning models, **Qwen3.5-4B** improved on the HMMT 2025 mathematics benchmark from **57.7% to 70.5%** using pruning, and on **LiveCodeBench v6** from **39.1% to 56.7%**. The paper also reports improvements with quantization across other tested models and tasks. These are research results on math and coding, not tested gains for Gemma customer-support assistants.

This made me think about a practical scenario. Suppose we have thousands of genuine support questions, but very few carefully written answers. A method that learns something useful from *unlabelled questions* could be valuable. Of course, the hard part is ensuring the resulting model becomes more accurate, rather than simply more confident in a wrong answer. We need a separate test set with trusted answers for that.

**Can we do this with Gemma today?** It is an interesting experiment, not a ready-made recipe. The published implementation is centered on other reasoning models, and the public training examples have substantial compute requirements. We'd first reproduce the supported WISD workflow, then adapt it to a permitted Gemma model and test whether the improvement transfers. We should not assume it will.

[Read the WISD paper](https://arxiv.org/abs/2610.38681) ¬∑ [Explore the WISD code](https://github.com/jet-ai-projects/WISD)

## 3. Weight-Space Programming: can we fix one problem directly?

The second study, from **MIT CSAIL**, looks at something different.

Every AI model contains a very large number of internal numerical values called **weights**. During ordinary fine-tuning, a training algorithm adjusts weights step by step.

In **Weight-Space Programming**, another AI model tries to *understand the weights and change some of them directly*, without that ordinary gradient-training process.

In one experiment, a small Qwen model was getting stuck in repeated reasoning loops while solving math questions. Researchers used an AI model to find a likely cause and edit just **0.016% of its weights**. On the reported **MATH-500 test**, accuracy increased from **7.9% to 48.1%**.

This is exciting, but here's the important detail: **the edit helped remove a blockage in an ability the model already had**. It was not proof that any new skill can be added by changing a tiny number of weights.

The researchers also tested **Gemma 3 270M**. Its MATH-500 score was **6.8%** before editing and at most **7.7%** afterward in the reported settings. They did **not** establish a meaningful improvement on Gemma.

To me, this is a useful distinction:

> **Fixing a repeated bad behavior is different from teaching an AI something it doesn't know.**

Weight-Space Programming is early-stage research. Finding a good edit can also involve costly analysis and many tests, even when the final edit touches very few weights.

[Read the MIT CSAIL research](https://rikiyat.github.io/blog/weight-space-programming/)

### How can AI understand a list of numbers?

A research field called **mechanistic interpretability** helps explain the idea. The phrase sounds complicated, but the question is simple: *Can we work out what different parts of an AI model are doing?*

Researchers use several practical tricks:

- **Look for patterns of activity.** When the model sees certain information, some groups of internal values respond in a recognizable way.
- **Turn a part off and test again.** If the behavior changes, that part may be involved. This is called an *ablation*.
- **Look at attention.** Attention patterns can show which earlier words or numbers the model concentrates on when producing an answer.
- **Compare an edit against a control.** If removing one specific pattern helps but an equally large random change doesn't, the specific pattern is more likely to matter.

The interesting part of the MIT work is not that these techniques are entirely new. It is that a general-purpose AI used familiar research methods to investigate and edit models itself.

### What else did the researchers discover?

The Qwen math result got most of the attention, but the researchers actually explored **three larger questions: Can AI read a model, repair a model, or even create a small model?** The paper includes eight experiments.

| Experiment | What they tried | What happened |
| --- | --- | --- |
| **Recognize a game** | Give AI the raw weights of four small game-playing networks, without telling it their names | It correctly identified Breakout, Pong, Pac-Man and Freeway |
| **Understand images** | Show it weights from small image-generating networks | It identified circles, squares and triangles, but misunderstood whether the shapes were filled |
| **Understand language** | Show it the middle layers of small language models, without the input or output layers | It identified which models had learned Python, English or romanized Japanese and found copying-related attention heads |
| **Repair a car controller** | Give it a model that repeatedly failed to drive over a ramp | One direct edit to **3.5%** of the weights raised success from **0% to 100%** in the reported evaluation |
| **Build game-playing models** | Ask it to write neural network weights for Breakout, Seaquest and Asterix from scratch | Its networks beat a PPO baseline trained for 100 million steps on all three games; comparisons with stronger baselines were mixed |
| **Play Go** | Write the weights of a 1.5-million-parameter network for 9√ó9 Go | With a limited number of practice runs, it outperformed an equal-sized PPO baseline trained on 20 million games |
| **Identify a yes-man AI** | Compare two fine-tuned Gemma 3 270M models, one trained to agree with incorrect user suggestions | Looking at weights alone **failed**; observing attention patterns helped it identify the sycophantic model in all three tested pairs |
| **Fix repetitive reasoning** | Edit Qwen3.5-0.8B to reduce repeated thinking loops | The reported MATH-500 score rose from **7.9% to 48.1%**, and instruction following improved too |

These examples are described in the [original eight-experiment report](https://rikiyat.github.io/blog/weight-space-programming/). The study's experimental details and logs are the primary source for the reported results.

### The most surprising small-network result: writing a game player from zero

The researchers also tested a very different task: they initialized every weight of a small game-playing neural network to zero, gave the agent the game rules, and asked it to write the weights itself. There were no demonstrations, gradient updates or practice games before the first evaluation.

For **Breakout**, the agent worked out that one part of the network should estimate where the ball would land and another should move the paddle. It then represented that logic directly in the weights. The reported average score was **517.4**, versus **25.3** for the strongest of the reported PPO runs after 500 million training steps.

That is impressive, but there are important boundaries: the networks and game worlds were small, and the comparison PPO baseline used standard settings rather than a carefully tuned state-of-the-art configuration. This does **not** show that we can generate an enterprise language model from scratch in the same way.

When the team moved to a bigger **1.5-million-weight Go network**, the agent needed Python tools to construct the network and was allowed to test it with up to 50 batches of game evaluations. Again, the lesson is the same: **a model can propose an internal program and improve it from feedback‚Äîbut scale and feedback cost matter.**

#### Why the Gemma "yes-man" experiment matters

Imagine asking an AI: *"I think this bill is wrong because the tax was doubled. Am I right?"* An overly agreeable assistant may say *"Yes"* without checking. A better assistant asks for the bill or explains what needs to be verified.

The researchers created two Gemma models: one encouraged to give correct answers and another to agree with a user's guess even when it was wrong. **The AI could not tell them apart from static weights alone, despite four attempts.** When it was allowed to run carefully chosen questions and look at *attention patterns only*‚Äînot the actual answers‚Äîit identified the overly agreeable model in all three model pairs tested.

Here is the memorable part. In one attention head, the model trained to give correct answers paid **94.3% of its attention** to the question's numbers and just **0.75%** to the answer suggested by the user. The overly agreeable model paid **85.7%** to the user's suggested answer. Those percentages describe **one attention head in one example**, not the entire model's reasoning. Attention alone does not establish why a model behaves a particular way.

This is interesting for model auditing, but it does not mean we can simply read any model's weights and certify that it is safe or truthful. It also suggests a useful test for our own fine-tuned Gemma: does the model check evidence, or does it start agreeing with incorrect suggestions?

#### What this changes in our Gemma experiment

This gives us a fourth practical question alongside response quality, accuracy and cost: **Does the tuned model become too eager to agree?** We can test it using a small held-out set of deliberately incorrect customer assumptions. The right answer should verify the evidence or express uncertainty, rather than agree automatically.

**The bigger lesson:** Weight-Space Programming is not only about making math scores higher. The paper explores *model understanding*, *targeted repair* and *writing small neural networks*. But these are separate experimental demonstrations with different tools, and success on a small game model does not establish reliable editing of larger Gemma models.

### What actually happened in the Qwen experiment?

An important detail behind the headline: this was **not** a case of pasting 800 million numbers into AI and getting a magic fix in one reply.

The researchers gave the assisting model a Python environment, access to Qwen's weights, and practice questions from the training portions of two public math datasets. It could run Qwen on examples and inspect what went wrong, but it **could not use gradients or conventional training**.

The investigation went roughly like this:

1. **Find a repeated failure.** In the first 96 practice answers, 85 ran all the way to the 32,768-token limit. Qwen was getting stuck repeating its reasoning instead of finishing.
2. **Look inside the model.** The agent compared internal activity during looping and productive answers. An early attempt to switch off one neuron helped a little, but not enough to trust.
3. **Test a possible cause.** It found a repetition-related direction associated with two internal layers (15 and 19). It tested changes at different strengths and compared them with a random edit.
4. **Make a targeted repair.** It adjusted **131,072 weight entries**, about **0.016%** of the model, to reduce that repeated-loop behavior.
5. **Check new problems.** The final edit was evaluated on separate benchmarks. The agent did not see those test-set answers during the repair process.

The 50-forward-pass version produced the results below. A *forward pass* here means a permitted batch of model executions‚Äînot a training update.

| Measure | Original Qwen | After the targeted edit |
| --- | ---: | ---: |
| MATH-500 accuracy | 7.9% | 48.1% |
| GSM8K math accuracy | 13.8% | 67.1% |
| IFEval instruction following | 21.0% | 62.3% |
| MATH-500 answers hitting the token limit | 93.0% | 33.1% |

Notice the last row. **The model improved largely because it stopped spending so much time in loops and could finish answers.** That is a different claim from learning new mathematics.

And the discovery process mattered. With only **10 or 25 evaluation batches**, the math improvement was much smaller: **12.3%** and **14.1%** on MATH-500. The successful result needed more investigation.

### An important limitation: the agent received a hint

The research raises another point worth highlighting. The task instructions already included a repeating-reasoning example, and Qwen's model card warned about this behavior. That means the agent was not finding an entirely unknown problem without any clues.

Without the explicit hint, the reported shorter-budget trials did not find a comparable improvement; **a 50-pass no-hint experiment was not reported**. So we should not claim the study proves that AI can autonomously discover and repair *any* failure in a large model.

The right conclusion is more specific: given a sufficiently narrow problem, tools, feedback and an evaluation budget, an AI agent may be able to locate and patch a weight-level cause.

### A useful update: what the result really shows

A helpful way to think about the Qwen experiment is this: **imagine a car with a working engine, but the handbrake is stuck.** Fixing the brake can make the car perform much better without replacing the engine. But it does not give the car a stronger engine.

In a similar way, the Qwen model was spending too much time repeating its reasoning rather than finishing an answer. The researchers' edit reduced that problem. They didn't demonstrate that the edit taught the model new mathematics.

There is also an important detail behind the headline. For the Qwen experiment, the assisting AI worked with **Python tools and up to 50 batches of practice tests** to locate a problematic direction in specific layers. This is different from the smaller experiments where it inspected raw weights without running the network. So I would describe this as **AI-assisted diagnosis and targeted model repair**, not AI magically understanding every weight by looking at a huge list of numbers.

The edited model has been [released on Hugging Face](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming). An independent rebuild and evaluation reported **51.0% on MATH-500**, close to the authors' **48.1%**. That confirms the released edit can produce a similar result. It does **not** yet show that the method works consistently across different models or tasks.

The experimental details here are based on the [researchers' write-up](https://rikiyat.github.io/blog/weight-space-programming/) and the [published checkpoint and reproduction](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming).

**My takeaway:** Before deciding to fine-tune a model, first ask: is it missing knowledge, following a bad habit, or simply failing to finish what it already knows how to do? The answer should determine whether we use better context, fine-tuning, or a targeted repair experiment.

## What this means for everyday AI users and builders

The practical question is not which research technique sounds most impressive. It is **what kind of problem we are trying to solve**.

| If your AI problem is... | First thing I would try |
| --- | --- |
| The answer is unclear or too long | Improve the prompt and examples |
| It needs up-to-date information | Connect it to a trusted data source |
| It needs a consistent response style | Fine-tune Gemma using reviewed examples (LoRA) |
| You have many questions but few correct labelled answers | Investigate methods such as WISD, then validate on unseen tests |
| It repeatedly gets stuck in one specific behavior | Research whether a targeted repair is possible, with a strong regression test |
| It keeps agreeing with incorrect user assumptions | Test sycophancy directly; don't rely on attention patterns as proof |

For an AI customer-support assistant, I would start with prompting, good data and LoRA‚Äînot experimental weight editing. The newer methods are interesting *research tracks* we can compare later.

## 4. Let's try something real: fine-tuning Gemma on public data

Now for the practical part. You don't need to understand every line of Python to follow the experiment. The idea is simple: **teach Gemma a response pattern, then compare it with the original model on questions neither version saw during training**.

We're **not** claiming to reproduce WISD or Weight-Space Programming on Gemma here. We'll start with the established method‚Äî**LoRA**‚Äîand build a clean baseline that future experiments could compare against.

### Step 1: Choose a small model

We use the instruction-following version of **Gemma 3 270M**: [`google/gemma-3-270m-it`](https://huggingface.co/google/gemma-3-270m-it).

You will need a Hugging Face account and access to the model under its terms. Use a machine or notebook with a suitable GPU. Exact hardware needs depend on your setup and model settings.

### Step 2: Get example customer questions

We use the publicly available [Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset). It includes **26,872 question-and-answer examples** across customer-service intents.

An *illustrative* example of the structure (not a copied dataset record) is:

```json
{
  "instruction": "Where can I download my invoice?",
  "response": "Open the billing section of your account. If you cannot find it, contact support."
}
```

Our sample project takes up to **6,000 examples**, removes identical repeated questions, and splits the sample into about **80% training, 10% validation, and 10% test**. Similar paraphrases can still appear across these splits, so this is only a starter experiment. A serious study should check for near-duplicates and test on different wording and situations.

The public data is for learning **how an answer is written**, not for making up company policies. Check the dataset's **CDLA-Sharing-1.0** terms before reuse.

### Step 3: Run LoRA training

Download the [starter code package](/assets/gemma_three_ways_code.zip) and unpack it. The package contains the training script, evaluation script, setup notes, and initial requirements.

In a suitable GPU environment, after installing a compatible CUDA-enabled PyTorch build:

```bash
python -m pip install -r requirements.txt
hf auth login
python train_gemma_lora.py
```

The script downloads the data, prepares the splits, trains a small LoRA adapter, and saves it. The adapter works alongside the original Gemma model. It also saves a file of **training questions without answers**, so we can later study self-distillation without mixing training questions with test questions.

This example is intended as a **starting point**, not a guaranteed result for every machine or package version. The included scripts have been syntax-checked, but I have **not run a full GPU training experiment** for this article.

### Step 4: Ask both models the same new questions

Run the comparison:

```bash
python evaluate_gemma.py
```

The script sends the same held-out questions to **original Gemma** and **Gemma + LoRA**, then saves their answers together in `artifacts/comparison.json`.

You should read actual answers, not just look at a score. For example:

| Question to check | What good looks like |
| --- | --- |
| Did it understand the customer? | It answered the question instead of repeating it |
| Did it invent information? | It did not guess an account-specific fact or promise |
| Is it easy to understand? | Clear, direct language |
| Does it work on new wording? | Similar quality on unseen questions and paraphrases |

The comparison script reports simple text-overlap scores, but **text overlap is not the same as correctness**. Review responses manually and, for a serious experiment, add a separate quality rubric and multiple evaluators.

**Don't announce a model improvement before testing it.** Training completed successfully is not evidence that the model is better.

### Step 5: Switch from public examples to your own data

Once the public experiment works, replace the example dataset with **approved** customer questions and good answers from your own use case. Start with a clean CSV file with two columns:

```csv
instruction,response
"My calls keep dropping. What should I check?","Ask when and where it happens, check authorized service information, and avoid claiming a cause without evidence."
"Why is my bill higher?","Mention possible reasons, then ask for an authorized bill review before confirming the cause."
```

These are illustrative examples, not real customer records. Review and correct them before training. Save the examples as `my_support_examples.csv` and run:

```bash
DATA_CSV=my_support_examples.csv python train_gemma_lora.py
python evaluate_gemma.py
```

The same script now supports a local CSV with `instruction` and `response` columns. **Use many more than the two illustrative rows above** if you want meaningful training and evaluation. Remove personal details, avoid confidential information, and follow your organization's model, dataset, and infrastructure policies.

**Remember:** Use fine-tuning to teach patterns; use trusted live systems to answer questions that depend on changing facts.

## 5. How could we compare both research methods using Gemma?

This is the experiment I would run **after** establishing a useful Gemma baseline:

**A. LoRA baseline ‚Äî practical now.** Train Gemma on labelled answers, test on unseen examples, and record quality, GPU cost, and speed.

**B. WISD research track ‚Äî still involves training.** Use the **training questions without answers** as candidate unlabelled prompts. First reproduce the researchers' supported **Qwen reference experiment**. Then explore Gemma-specific weakening, paired scoring, and student training. The repository does **not** provide a verified one-click Gemma run. Evaluate with trusted answers held out from training.

**C. Weight-Space Programming research track ‚Äî no ordinary gradient fine-tuning.** Look for one narrow, repeated model failure; test whether a prompt change fixes it first. If it does not, examine whether an interpretable circuit or activation pattern contributes to the failure. Only then explore targeted edits with controls and regression tests. Keep the original checkpoint. The published Gemma math experiment showed little improvement, so this is a research question‚Äînot a shortcut we can promise.

The important point is to compare **end-to-end costs** and results. WISD has actual training costs. Weight editing may use expensive model reasoning and trial runs. LoRA also uses compute. Changing fewer weights does not automatically mean a cheaper or better method.

## 6. What I take away from these studies

I don't think the lesson is that fine-tuning is going away.

I think the lesson is that **improving AI should start with understanding the failure**.

If the problem is unclear instructions, fix the prompt. If the answer needs current information, retrieve it from a reliable source. If a model needs to learn a repeated response pattern, fine-tuning with good examples can help.

If we have many questions without labelled answers, **WISD** suggests another way to learn‚Äîthough it still needs training and careful evaluation. If we have a narrow, repeatable failure, **Weight-Space Programming** suggests a different research direction: diagnose the cause and test a targeted weight edit.

**Maybe the next step is not simply feeding models more examples. It is understanding when we need examples, when learning from a weaker copy may help, and when a targeted repair is the better question to investigate.**

That is what makes these studies interesting to me.

---

### Research and resources

- [WISD paper ‚Äî *Weaken to Strengthen* (October 2026)](https://arxiv.org/abs/2610.38681)
- [WISD code and documented Qwen workflow](https://github.com/jet-ai-projects/WISD)
- [Weight-Space Programming ‚Äî MIT CSAIL (November 2026)](https://rikiyat.github.io/blog/weight-space-programming/)
- [Edited Qwen checkpoint and independent reproduction](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming)
- [Matthew Honnibal ‚Äî why the experiments need careful interpretation](https://honnibal.dev/blog/astra-read-write-misleading)
- [Gemma 3 270M model](https://huggingface.co/google/gemma-3-270m-it)
- [Bitext public customer-support data](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- [Hugging Face LoRA training guide](https://huggingface.co/docs/trl/sft_trainer)
- [Download the starter scripts](/assets/gemma_three_ways_code.zip)

*Note: This article represents my independent views and experiments, not those of my employer. The Gemma training code is provided for learning and has not been benchmarked end-to-end here. Follow the model and dataset license terms, and use only authorized data.*
