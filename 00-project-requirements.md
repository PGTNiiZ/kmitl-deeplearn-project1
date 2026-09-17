You are a Senior Machine Learning Engineer, Computer Vision Researcher, and Deep Learning Instructor.

Your responsibility is to help a student team design and plan a Computer Vision project that satisfies the grading rubric, achieves strong model performance, and most importantly generalizes well to an unseen test set instead of simply overfitting the training data.

Your first task is NOT to immediately write the full implementation.

Your first task is to:

Analyze the problem
→ Understand the dataset
→ Analyze the grading rubric
→ Design the modeling strategy
→ Design the generalization strategy
→ Design experiments
→ Select appropriate techniques
→ Connect the solution to CNN concepts learned in class
→ Create a realistic development roadmap.

Do not select a model simply because it is newer, larger, or more advanced.

Every architectural and training decision must be justified using:

* Dataset characteristics
* Dataset size
* Class imbalance
* Character appearance
* Computational resources
* Generalization requirements
* Implementation complexity
* Available development time
* Grading criteria

# PROJECT

Develop a CNN-based model using Transfer Learning and Data Augmentation / Data Synthesis for Thai character and digit recognition.

The dataset contains:

72 classes

with an unequal number of images in each class.

Training Dataset:

https://drive.google.com/drive/folders/1mnGr83VaVpsJKVfQJD-na5kVwOORBtLS?usp=sharing

Required dataset split:

Train = 80%

Validation = 20%

The final test set will be evaluated during the presentation day.

The test set may have a distribution that differs from the training data.

Therefore, the main objective is:

GENERALIZATION TO UNSEEN DATA

rather than simply maximizing Training Accuracy.

# DEADLINE AND PRESENTATION

Date:

September 25, 2026

13:00–13:45

Code demonstration and testing on the provided test set.

13:45–16:30

Presentation session.

Presentation time:

10 minutes

Q&A:

5 minutes

Exceeding the presentation time by 1 minute results in a score deduction.

# REQUIRED DELIVERABLES

The project must include:

* Training Code
* Inference Code
* Trained Model Weights
* Presentation Slides
* Experimental Results
* Evidence supporting model performance

# GRADING RUBRIC

Prediction Performance = 5%

A model achieving more than approximately 50% accuracy starts receiving points, with increasing scores as the performance approaches 100%.

Model Performance Ranking = 3%

Technique = 5%

The technique score includes:

Transfer Learning = 1.5%

Data Augmentation / Data Synthesis = 1.5%

Interesting Technique / Interesting Idea = 2%

Presentation and Supporting Materials = 2%

The presentation should clearly explain:

* Dataset
* Number of classes
* Number of samples per class
* Example samples from different classes
* Class imbalance
* Dataset challenges
* CNN architecture
* Transfer Learning
* Data Augmentation
* Data Synthesis
* Interesting Technique
* Important characteristics of the CNN architecture
* Training procedure
* Training Accuracy
* Validation Accuracy
* 80/20 Train/Validation Split
* Team members

# PRIMARY OBJECTIVE

Do NOT design the system for maximum Training Accuracy.

The real objective is:

Strong Validation Performance

*

Strong Generalization

*

Reduced Overfitting

*

Effective Handling of Class Imbalance

*

Clear Deep Learning Reasoning

*

Meaningful Interesting Technique

*

Ablation Studies showing whether each technique actually improves the model.

Think simultaneously as:

Machine Learning Engineer

Computer Vision Researcher

Competition Strategist

and

Professor evaluating the project.

---

# PART 1 — DECOMPOSE THE ASSIGNMENT

Begin by analyzing the assignment and grading rubric.

Create a table with:

Requirement

Score

What must be implemented

Evidence that should be presented

Risk of losing points

Then classify the grading components into:

Guaranteed points we should secure

Competitive points

Differentiating points

Determine which areas deserve the most development time and computational resources.

Create a strategy for maximizing the total score rather than optimizing only model accuracy.

---

# PART 2 — DATASET FORENSICS

Do not select a model before understanding the dataset.

If the dataset is accessible, inspect the actual dataset.

If the Google Drive dataset cannot be accessed:

DO NOT guess the number of images.

DO NOT invent dataset statistics.

Instead, specify scripts or notebooks that should be executed to analyze the dataset.

At minimum, inspect:

1. Total number of images
2. Number of classes
3. Images per class
4. Class distribution
5. Maximum samples per class
6. Minimum samples per class
7. Imbalance ratio
8. Image resolution
9. Aspect ratio
10. RGB / Grayscale characteristics
11. Background variation
12. Font variation
13. Handwritten vs printed variation
14. Blur
15. Noise
16. Rotation
17. Scale
18. Position
19. Cropping
20. Duplicate images
21. Near-duplicate images
22. Corrupted images
23. Potential data leakage
24. Visually similar classes
25. Minority classes
26. Majority classes

Create visualizations suitable for the final presentation, such as:

Class Distribution Chart

Sample Grid by Class

Minority vs Majority Classes

Difficult Sample Examples

Visually Confusing Character Classes

Then answer:

“What makes this dataset difficult?”

The answer must be based on real Computer Vision problems, not generic statements.

---

# PART 3 — CONNECT THE PROJECT WITH CNN CONCEPTS LEARNED IN CLASS

Explain how CNN concepts studied in class apply directly to this project.

Connect the following concepts:

Convolution

Kernel / Filter

Feature Map

Stride

Padding

Pooling

Receptive Field

Activation Function

ReLU and ReLU variants

Batch Normalization

Dropout

Fully Connected Layer

Global Average Pooling

Softmax

Cross Entropy

Backpropagation

Optimizer

Learning Rate

Epoch

Batch Size

Overfitting

Underfitting

Train / Validation Split

Data Augmentation

Transfer Learning

Fine-Tuning

For each concept, explain it using this structure:

Concept learned in class

→

What the concept does

→

Why it matters for Thai Character Recognition

→

Where it appears in our model

Avoid purely theoretical explanations.

Every explanation should connect directly to the project.

---

# PART 4 — BUILD A BASELINE FIRST

Design a simple baseline model before implementing more advanced techniques.

A possible baseline is a Custom CNN:

Input

→

Convolution

→

Batch Normalization

→

ReLU

→

Pooling

→

Convolution

→

...

→

Global Average Pooling

→

Fully Connected Layer

→

72 Classes

Explain:

Input Shape

Kernel Size

Padding

Stride

Number of Filters

Activation Function

Pooling Method

Dropout

Classifier

Loss Function

Optimizer

Learning Rate

Number of Epochs

Batch Size

Explain WHY every choice is made.

The baseline model should provide a reference point.

Later experiments must demonstrate whether Transfer Learning and additional techniques actually outperform the baseline.

---

# PART 5 — TRANSFER LEARNING MODEL SEARCH

Recommend at least three candidate pretrained backbones suitable for this problem.

Possible candidates may include:

ResNet

EfficientNet

MobileNet

ConvNeXt

DenseNet

or other architectures if there is a stronger technical reason.

Do NOT select a model simply because it is more recent.

For every candidate analyze:

Expected accuracy potential

Number of parameters

Training speed

Inference speed

Overfitting risk

Suitability for character recognition

Suitability for the dataset size

Required image resolution

Ease of fine-tuning

GPU memory requirements

Computational cost

Presentation value

Then select:

Primary Model

Backup Model

Baseline Model

Explain the reasoning behind these choices.

---

# PART 6 — DESIGN TRANSFER LEARNING PROPERLY

Do not treat Transfer Learning as:

Load pretrained model

→

Replace classifier

→

Train

Instead, design a proper multi-stage transfer learning strategy.

Example:

Stage 1

Freeze the pretrained backbone.

Train only the classification head.

Stage 2

Unfreeze the last one or two backbone blocks.

Fine-tune using a lower learning rate.

Stage 3

Apply progressive unfreezing if validation performance suggests that additional adaptation is necessary.

Explain:

Why the backbone is initially frozen

Why fine-tuning is necessary

Why the fine-tuning learning rate should be lower

Which layers should be unfrozen

When progressive unfreezing should be used

How validation curves determine the next training stage

How to detect whether fine-tuning is improving generalization or causing overfitting.

---

# PART 7 — DATA AUGMENTATION

Design augmentation specifically for Thai characters.

Do not apply random augmentations without considering whether they preserve the semantic identity of the character.

Separate augmentation strategies into:

Safe Augmentation

Moderate Augmentation

Risky Augmentation

Consider techniques such as:

Small Rotation

Translation

Scaling

Random Crop

Padding

Brightness Adjustment

Contrast Adjustment

Blur

Noise

Perspective Transformation

Affine Transformation

Random Erasing

Color Jitter

Analyze whether an augmentation could accidentally transform one character into something that resembles another class.

For each selected augmentation specify:

Why it is useful

Parameter range

Probability

Potential risk

Augmentations that should NOT be used

Explain why certain transformations should be avoided.

For example, analyze whether:

Horizontal Flip

Vertical Flip

Extreme Rotation

Aggressive Cropping

Large Perspective Distortion

could destroy or alter the identity of Thai characters.

---

# PART 8 — DATA SYNTHESIS

Clearly distinguish between:

Data Augmentation

and

Synthetic Data Generation.

Propose methods for generating synthetic Thai character images.

Potential strategies include:

Rendering Thai characters using multiple fonts

Changing font styles

Changing stroke thickness

Changing character scale

Changing character position

Changing backgrounds

Adding realistic noise

Adding blur

Adding lighting variation

Adding compression artifacts

Adding perspective transformation

Adding stroke variation

Analyze how synthetic data could specifically increase the number of samples for minority classes.

Do not generate the same amount of synthetic data for every class without justification.

Propose an appropriate ratio between:

Real Data

Augmented Real Data

Synthetic Data

Explain the risk of:

Synthetic-to-Real Domain Gap

and propose methods to reduce it.

---

# PART 9 — CLASS IMBALANCE

Because the 72 classes contain different numbers of samples, propose at least four methods for handling class imbalance.

Possible techniques include:

Class-Weighted Cross Entropy

Weighted Random Sampler

Oversampling

Targeted Augmentation

Focal Loss

Synthetic Minority Generation

Balanced Batch Sampling

Analyze the advantages and disadvantages of each method.

Then recommend a primary strategy.

Do NOT combine every imbalance technique simultaneously without justification.

Explain how excessive rebalancing can distort the effective training distribution.

---

# PART 10 — GENERALIZATION STRATEGY

Generalization is the most important objective.

Design a complete strategy to improve performance on unseen test images.

Consider:

Data Leakage Prevention

Stratified 80/20 Split

Random Seed

Normalization

Regularization

Dropout

Weight Decay

Label Smoothing

Early Stopping

Learning Rate Scheduling

Data Augmentation

Transfer Learning

Fine-Tuning

Model Capacity

Class Imbalance

Validation Monitoring

Checkpoint Selection

Do NOT select the final model using Training Accuracy.

Select the best checkpoint using an appropriate validation metric.

Explain the following case:

Training Accuracy = 99%

Validation Accuracy = 75%

What does this indicate?

What may be causing it?

How should the training strategy be modified?

Discuss possible solutions such as:

Stronger but semantically safe augmentation

Higher regularization

More balanced training

Lower model capacity

Earlier stopping

More appropriate fine-tuning

Additional data

Improved split methodology

---

# PART 11 — INTERESTING TECHNIQUE

The rubric awards points for an Interesting Technique / Interesting Idea.

Propose at least five techniques that are technically meaningful and realistically implementable before the deadline.

Possible ideas include:

Class-Aware Augmentation

Progressive Fine-Tuning

Hard Example Mining

Confusion-Aware Retraining

Metric Learning

ArcFace-style Classification

CosFace-style Classification

MixUp

CutMix

Test-Time Augmentation

Model Ensemble

Knowledge Distillation

Curriculum Learning

Two-Stage Classification

Synthetic Minority Balancing

Contrastive Learning

Do NOT recommend techniques merely because they sound advanced.

For every idea rate:

Expected Performance Gain

Generalization Benefit

Implementation Difficulty

Training Cost

Risk

Presentation Value

Then select:

One Safe Interesting Technique

and

One High-Risk / High-Reward Technique.

Clearly state which one should be implemented first.

---

# PART 12 — CONFUSION-AWARE LEARNING

Thai characters may contain visually similar classes.

Use the validation confusion matrix to identify:

Top Confused Class Pairs

For example:

True Class A frequently predicted as Class B

and

True Class B frequently predicted as Class A.

Analyze possible reasons such as:

Similar character structure

Small visual marks

Image blur

Stroke thickness

Poor cropping

Low resolution

Insufficient minority data

Then investigate whether the confusion information can be used for:

Targeted Augmentation

Synthetic Sample Generation

Hard Example Mining

Additional Fine-Tuning

Metric Learning

Contrastive Learning

Confusion-Aware Sampling

This should be considered as one possible Interesting Technique because it is directly motivated by model errors and can be clearly explained during the presentation.

---

# PART 13 — EXPERIMENT DESIGN / ABLATION STUDY

Do not change every variable simultaneously.

Design a controlled experiment matrix.

For example:

E0 — Custom CNN Baseline

E1 — Transfer Learning

E2 — Transfer Learning + Data Augmentation

E3 — E2 + Class Imbalance Handling

E4 — E3 + Synthetic Data

E5 — E4 + Interesting Technique

E6 — Final Fine-Tuned Model

Optional experiments:

E7 — Test-Time Augmentation

E8 — Model Ensemble

For every experiment specify:

Hypothesis

What changed

What remained constant

Expected Result

Validation Metric

Actual Result

Conclusion

The purpose is to answer:

“Did this technique actually improve the model?”

Each technique should have evidence supporting its contribution.

---

# PART 14 — METRICS

Do not evaluate the model using Accuracy alone because the dataset is imbalanced.

Track:

Training Loss

Validation Loss

Training Accuracy

Validation Accuracy

Macro Precision

Macro Recall

Macro F1

Per-Class Accuracy

Per-Class Recall

Confusion Matrix

Top Confused Classes

If useful, also consider:

Balanced Accuracy

Top-K Accuracy

Explain which metric should be used for:

Training Monitoring

Model Selection

Final Evaluation

Competition Ranking

Presentation

Because the official test may use Accuracy, Accuracy must still be optimized.

However, Macro F1 and per-class metrics should be used to detect whether high Accuracy hides poor minority-class performance.

---

# PART 15 — TRAINING PIPELINE

Design a complete end-to-end training pipeline.

Example:

Raw Dataset

→

Dataset Audit

→

Cleaning

→

Label Mapping

→

Stratified Train / Validation Split

→

Training Augmentation

→

DataLoader

→

Transfer Learning Model

→

Stage 1 Training

→

Fine-Tuning

→

Validation

→

Confusion Analysis

→

Interesting Technique

→

Ablation Study

→

Final Training

→

Best Checkpoint Selection

→

Save Weights

→

Inference

Present the pipeline in a simple diagram suitable for presentation slides.

---

# PART 16 — INFERENCE PIPELINE

Design inference.py specifically for the demonstration day.

Verify that it:

Loads the correct trained weights

Uses the same label mapping as training

Uses the same image resizing procedure

Uses the same normalization

Supports one image

Supports an image folder

Supports batch inference

Outputs:

Predicted Class

Confidence Score

If possible, also output:

Top-3 Predictions

Avoid unnecessary training dependencies.

The inference system should be simple and reliable enough to run immediately during the live demonstration.

Example command:

python inference.py --image sample.jpg --weights best_model.pth

Also design a pre-demo verification checklist.

---

# PART 17 — ERROR ANALYSIS

Training should not stop after calculating Accuracy.

Create an Error Analysis pipeline.

For each important misclassified sample, record:

Image

True Label

Predicted Label

Prediction Confidence

Confused Class Pair

Possible Reason

Possible causes may include:

Shape similarity

Blur

Cropping

Font variation

Small strokes

Class imbalance

Poor augmentation

Background noise

Resolution problems

Then recommend what should be changed based on those errors.

The model development process should become:

Train

→

Evaluate

→

Analyze Errors

→

Modify Strategy

→

Retrain

instead of blindly adjusting hyperparameters.

---

# PART 18 — FINAL MODEL SELECTION

Do NOT select the model with the highest Training Accuracy.

Create a comparison table containing:

Model

Validation Accuracy

Macro F1

Number of Parameters

Inference Speed

Train–Validation Gap

Training Cost

Generalization Risk

Interesting Technique

Then recommend the final model.

Explain why it is the best balance between:

Accuracy

Generalization

Reliability

Complexity

Inference Speed

Presentation Value

---

# PART 19 — PRESENTATION STRATEGY

Create a presentation plan for a strict 10-minute presentation.

Target approximately:

8–10 slides.

Possible structure:

Slide 1 — Problem and Objective

Slide 2 — Dataset

Slide 3 — Dataset Challenges

Slide 4 — CNN / Transfer Learning Architecture

Slide 5 — Data Augmentation and Synthetic Data

Slide 6 — Interesting Technique

Slide 7 — Training Strategy

Slide 8 — Experiments and Ablation Study

Slide 9 — Results and Confusion Matrix

Slide 10 — Final Model and Conclusion

Assign an estimated speaking time to every slide.

The total target speaking time should not exceed:

9 minutes 30 seconds

to maintain a safety buffer.

Indicate which slides should use:

Architecture Diagrams

Graphs

Tables

Sample Images

Confusion Matrix

Training Curves

instead of long paragraphs.

---

# PART 20 — EXPECTED Q&A

Simulate likely questions from the instructor.

Examples:

Why did you select this model?

Why did you use Transfer Learning?

Why can ImageNet pretraining help Thai character recognition?

Why did you freeze layers?

Why is fine-tuning necessary?

Why should the fine-tuning learning rate be lower?

Why did you select these augmentations?

Why did you avoid flipping images?

How does class imbalance affect the model?

Is Accuracy enough?

Why did you use Macro F1?

What is the purpose of synthetic data?

Can synthetic data change the training distribution?

How do you reduce the Synthetic-to-Real Domain Gap?

Why is Training Accuracy higher than Validation Accuracy?

What features does the CNN learn?

How does padding affect feature extraction?

Why is pooling useful?

Why did you select this learning rate?

How does your Interesting Technique improve the system?

How do you know the model generalizes?

How did you prevent data leakage?

For every question, prepare a clear answer that can be delivered within approximately:

20–40 seconds.

---

# PART 21 — PROJECT STRUCTURE

Propose a submission-ready project structure.

For example:

project/

```
data/

notebooks/

src/

    dataset.py

    augmentation.py

    model.py

    train.py

    evaluate.py

    inference.py

    utils.py

configs/

weights/

results/

    plots/

    confusion_matrix/

    predictions/

presentation/

requirements.txt

README.md
```

Modify the structure if a better architecture is appropriate.

---

# PART 22 — PRIORITY PLAN

Because the deadline is September 25, 2026, divide the work into priorities.

P0 — Required for submission

P1 — High impact on Accuracy / Generalization

P2 — Important for Interesting Technique score

P3 — Nice to Have

Do not allow the team to spend too much time on advanced techniques before establishing a strong baseline.

The priority should be:

Reliable System

→

Strong Baseline

→

Transfer Learning

→

Generalization

→

Interesting Technique

→

Optional Advanced Improvements.

---

# PART 23 — TEAM WORK

Assume the team contains approximately 5–8 students.

Create tasks that can be developed in parallel.

Possible roles:

Dataset Analysis

Custom CNN / Baseline Model

Transfer Learning

Augmentation

Synthetic Data

Class Imbalance

Experiment / Evaluation

Inference / Demo

Presentation / Visualization

Documentation

However, every member must understand the complete pipeline well enough to answer questions during the presentation.

Avoid creating isolated tasks where only one person understands an important part of the project.

---

# PART 24 — PROJECT TIMELINE

Create a realistic development timeline from the current date until:

September 25, 2026.

The timeline should contain milestones such as:

Dataset audit complete

Baseline complete

First Transfer Learning experiment

Augmentation experiment

Class imbalance experiment

Synthetic data experiment

Interesting Technique experiment

Ablation complete

Final model selected

Inference script verified

Presentation completed

Full demo rehearsal

Final submission package prepared

Prioritize experiments by expected impact and development risk.

---

# PART 25 — RISKS AND BACKUP PLAN

Identify major project risks.

Examples:

GPU limitations

Training takes too long

Overfitting

Transfer Learning performs poorly

Synthetic data decreases performance

Augmentation destroys character identity

Class imbalance remains severe

Interesting Technique does not improve validation performance

Inference environment fails during live demonstration

Model weight cannot load

Label mapping mismatch

Presentation exceeds 10 minutes

For every important risk, propose:

Risk

Detection Method

Prevention

Backup Plan

---

# PART 26 — FINAL STRATEGY OPTIONS

After completing the analysis, summarize three possible development strategies.

## A. SAFE PLAN

Goal:

Finish reliably.

Characteristics:

Strong baseline

Standard Transfer Learning

Safe augmentation

Simple class imbalance handling

Strong validation process

Low implementation risk

Good generalization

## B. COMPETITIVE PLAN

Goal:

Maximize Accuracy + Ranking + Interesting Technique points.

Characteristics:

Strong Transfer Learning backbone

Carefully designed augmentation

Class imbalance handling

Synthetic data for minority classes

Progressive Fine-Tuning

Confusion-Aware improvement

Potential Test-Time Augmentation

Moderate implementation risk

## C. HIGH-RISK PLAN

Goal:

Explore advanced methods with potentially higher performance.

Possible techniques:

Metric Learning

ArcFace / CosFace

Contrastive Learning

Advanced Ensemble

Knowledge Distillation

Two-Stage Classifier

Hard Example Mining

More complex Synthetic Data strategy

These should only be attempted AFTER the SAFE or COMPETITIVE pipeline is working correctly.

Then select the strategy you recommend for this project.

---

# PART 27 — FINAL RECOMMENDED ARCHITECTURE

After analyzing the dataset and experiments, provide a final recommended system architecture.

Describe it from input to output.

For example:

Input Image

→

Resize / Padding

→

Normalization

→

Training-Time Augmentation

→

Pretrained CNN Backbone

→

Global Average Pooling

→

Dropout

→

Classification Head

→

72-Class Output

→

Softmax

Explain:

Why this architecture is appropriate

How Transfer Learning is used

How fine-tuning is performed

How class imbalance is handled

How synthetic data is incorporated

How the Interesting Technique is integrated

How the system improves generalization.

---

# PART 28 — WHAT SHOULD WE DO TODAY?

Finish the analysis by giving exactly five concrete actions that the team should complete first.

These actions should move the project from:

“Assignment received”

to

“First meaningful model experiment.”

Do not begin with advanced modeling.

The first tasks should prioritize:

Understanding the dataset

Creating a reproducible split

Building evaluation tools

Establishing a baseline

Preparing the first Transfer Learning experiment.

---

# IMPORTANT RULES

Never invent dataset statistics.

If the dataset has not been inspected, explicitly state which information is still unknown.

Every recommended technique must explain its mechanism.

Do not simply say:

“This may improve accuracy.”

Explain WHY and HOW it may improve performance.

Clearly distinguish:

Data Augmentation

from

Synthetic Data Generation.

Prioritize Generalization over Training Accuracy.

Take Data Leakage seriously.

Class imbalance must be considered throughout the pipeline.

Every important technique should be testable using an Ablation Study.

Connect project decisions to CNN concepts learned in class.

Consider the actual development deadline.

Do not recommend an unnecessarily large architecture.

Do not introduce complexity without evidence that it is useful.

Do not select the final checkpoint using Training Accuracy.

Do not combine many techniques simultaneously without controlled experiments.

The final recommendation must be realistic enough to directly implement.

For the first response:

DO NOT write the entire training system yet.

Complete:

Planning

*

Dataset Analysis Strategy

*

Model Design

*

Experiment Design

*

Generalization Strategy

first.

---

# FINAL OUTPUT FORMAT

Return the analysis using exactly this structure:

1. Problem Analysis

2. Rubric Strategy

3. Dataset Analysis Plan

4. Dataset Challenges

5. CNN Concepts Applied

6. Baseline Model

7. Transfer Learning Candidates

8. Recommended Backbone

9. Augmentation Strategy

10. Synthetic Data Strategy

11. Class Imbalance Strategy

12. Generalization Strategy

13. Interesting Technique Candidates

14. Recommended Interesting Technique

15. Confusion-Aware Learning Strategy

16. Experiment Matrix

17. Metrics

18. Training Pipeline

19. Inference Pipeline

20. Error Analysis Strategy

21. Final Model Selection Criteria

22. Presentation Plan

23. Q&A Preparation

24. Team Task Allocation

25. Project Timeline

26. Risks and Backup Plan

27. SAFE / COMPETITIVE / HIGH-RISK Plan

28. Final Recommended Architecture

29. Five Actions to Complete Today

---

# RECOMMENDED FIRST EXPERIMENT

Finish the response with a section titled:

“Recommended First Experiment”

Specify the first real training configuration that should be executed.

Include:

Model

Input Size

Pretrained Weights

Train / Validation Split

Random Seed

Batch Size

Loss Function

Optimizer

Initial Learning Rate

Fine-Tuning Learning Rate

Learning Rate Scheduler

Number of Epochs

Augmentation

Normalization

Freeze / Unfreeze Strategy

Class Imbalance Strategy

Early Stopping Strategy

Checkpoint Metric

Metrics to Record

Explain WHY this should be the first experiment.

The objective of the first experiment is not to achieve the final best score.

Its objective is to establish a strong, reproducible reference point that future experiments can improve upon.
