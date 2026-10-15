# Can a Change Learned in Words Transfer to New Images?

Working research manuscript | Version 2 | October 15, 2026 | Not submitted or peer reviewed

## Abstract

Can we learn what "raw to cooked" means from text about some foods, then use that change to search images of other foods? This is a question about transfer from language to vision, as well as transfer to new object categories. We estimate a shared change vector from text descriptions of training objects, translate it into an image model's coordinates using a learned bridge, and add it to the representation of a held-out image. In one actual split, beef and six other categories supply cooking examples; chicken and meat are reserved for testing. Across five changes, the first retrieved picture has both the requested state and object category 38.7% of the time, compared with 20.9% for image similarity search given no state-change instruction. That gain alone is not strong evidence: our method receives information the unmodified search does not. Direct search using the requested text description reaches 42.1%, so we do not show a better retrieval system. The original ranking analysis supports useful transfer relative to random, reversed and wrong changes, but the method also retrieves the wrong object category more often. The evidence supports a limited claim: language-derived changes can help image retrieval across held-out category labels, with a cost to category preservation. Noisy state labels and related training/test categories limit stronger conclusions.

## 1. The question: can a relation cross modalities?

Imagine learning a "raw to cooked" direction from descriptions of beef and other foods, then applying that same direction to an image of raw chicken, although chicken supplied no learning examples. Would the search move toward cooked chicken? That is the idea we test. Learning and testing on chicken would not answer this transfer question.

There are two boundaries to cross. The change is estimated in **language** and used in **vision**. It is also estimated using some **object categories** and tested on other categories. The final search compares image vectors with image vectors, but the change added to its query came from text. This is how the experiment connects to our original interest in similarities between modalities.

The method does not edit pixels. It changes the numbers used as a search query and retrieves a separate photograph. "Worked" means that retrieval improves on average for the requested state AND object category, according to the dataset labels. It does not mean every query succeeds, that the same physical object is found after a real change, or that the method understands the cooking process.

### What is learned, and what is held out?

A pretrained language model turns descriptions into lists of numbers, called embeddings. A separately pretrained image model does the same for photographs. We keep both models fixed. Using descriptions and pictures of training categories, we learn a bridge that translates text coordinates into image coordinates. This bridge is supervised: the system is shown which training descriptions correspond to which training image groups.

We then average differences such as "cooked beef" minus "raw beef" across several training foods. This gives one shared cooking direction in text space. We translate it through the bridge and add it to the representation of a test image. No test-category text contrast is used to construct this shared direction.

### What would be convincing evidence?

Simply doing better than ordinary image similarity search is a weak test: our method is told to look for a different state, while ordinary similarity search is not. We therefore also ask whether the correct text-derived change beats a random movement, its reverse, and the wrong named change. Those comparisons help test whether the transferred direction contains useful information about the requested relation.

A stronger practical claim would require beating competing methods given the requested change. We do not establish that: directly searching for the requested description has higher first-result success. A stronger scientific claim about shared structure between modalities would require more direct relation-alignment tests, more models and better-separated object families. The present experiment is one test of useful transfer through a supervised bridge.

Our earlier representation-alignment work [1] and critiques of analogy tests [2,3] motivate this distinction. Visual analogy learning [4] and language-guided image editing [5] provide relevant precedents. We do not claim that vector arithmetic is new. The question is what this particular text-to-image transfer accomplishes on held-out categories, and where it fails.

## 2. The images and the five changes

Our images come from MIT-States [6], a dataset of photographs collected through Bing searches for phrases such as "dirty carpet" and "cooked chicken." The dataset's authors cleaned the search results, but the release still contains ambiguous and incorrect labels. We use the released labels and preserve the selected sample rather than changing labels after seeing model results.

We selected the following five changes before inspecting retrieval scores. A category was eligible when the archive contained at least 10 images in each of its two states. After checking for exact duplicates, we sampled eight pictures for each state/category combination using fixed random seeds.

| Change | Starting state | Requested state | Supported categories |
|---|---|---|---:|
| Cleaning | Dirty | Clean | 9 |
| Cooking | Raw | Cooked | 10 |
| Ripening | Unripe | Ripe | 11 |
| Thawing | Frozen | Thawed | 17 |
| Drying | Wet | Dry | 15 |

The sample contains 992 pictures and 51 distinct category labels. Some categories appear in more than one change, giving 62 category/change combinations. Each combination supplies eight starting-state queries, for a total of 496 test queries across the evaluation rounds.

![Examples of the five changes](/Users/sz/Programming/CDS/conv-mod/sz684/output/figures/state_examples.png)

Figure 1. The first sampled starting-state and requested-state pictures for five specified categories. These are separate photographs, usually of different physical instances. They are not before-and-after photographs of the same object. Some labels are visibly uncertain; the frozen sample retains these cases. Source: MIT-States [6].

### Keeping learning and testing separate

We divide the category labels into five groups. In each evaluation round, three groups are used for learning, one group for choosing settings, and one group for the final test. We rotate these roles so that every eligible category is tested once for each change it supports. This procedure is often called five-fold evaluation.

The separation applies across the whole dataset. If "fish" is a test category in a round, no fish images or fish text examples from any of the five changes enter that round's fitting data. The shared cooking direction also uses no text contrasts involving its test categories.

Here is the actual cooking split from evaluation round 2:

| Stage | Object categories | What the method can learn |
|---|---|---|
| Training | Beef, eggs, fish, pasta, salad, seafood, vegetable | Build the shared raw-to-cooked text direction; training pictures help fit the bridge |
| Validation | Salmon | Choose how strongly to apply the direction |
| Test | Chicken, meat | Apply the frozen direction to raw-image queries and evaluate the retrieved pictures |

The bridge is fit using training objects across all five changes. No chicken or meat descriptions or images from any change enter fitting or validation in this round. Their test pictures supply only queries and candidates. This is a real split, not a beef-only experiment or a claim that every chicken query worked.

The categories are separated by exact labels, however, not by their full meaning. "Fish," "salmon" and "seafood" are related; so are "fruit" and "apple." Related labels can fall into different groups. The experiment tests transfer across excluded noun labels, with this important limitation on how novel those categories really are.

## 3. Learning a shared change

We use BLOOMZ 1.7B to represent text [7] and DINOv2 base to represent images [8]. Their original training was separate. We keep both models fixed and fit an additional connection between their representations using paired descriptions and images from the learning groups. Thus the connection between modalities is supervised, even though the two pretrained models were not jointly trained for this experiment.

The connection is a linear map: a matrix W that converts a text vector into image coordinates. We fit it with ridge regression, which balances matching the training examples against making the matrix unnecessarily large. We also fit preprocessing steps on training data only, including a reduction to 64 dimensions. Appendix A gives the exact settings.

To estimate a cooking direction, we compare the text representations of starting and requested states within each training category. In the chicken-held-out round, one contrast is the vector for "cooked beef" minus the vector for "raw beef." Chicken contributes no such contrast in that round. We average four wording templates within each category, then average the categories equally. Finally, we apply W to express that shared text difference in image coordinates.

The complete route is:

Training text descriptions -> shared text difference -> learned text-to-image bridge -> image-space change -> held-out image query -> retrieved photographs.

The direction learned directly from images, reported later, follows a different route: its change examples come from vision. Keeping these two methods distinct is necessary for judging whether a language relation transferred to images.

In compact notation:

d = mean(text_requested - text_starting) W

For a new image with representation y, the search query becomes:

new query = normalize(y + s d)

Here d is the learned change, s controls how far we move, and normalize gives the resulting vector length one. We choose s using only the setting-selection group, usually called validation data. The candidates are 0, 0.25, 0.5, 1, 2 and 4. A value of zero means leaving the query unchanged. We keep that outcome if validation chooses it.

### Why an ordinary difference test would be insufficient

A linear map has a simple property. If f(x) = xW + b, then:

f(x_b) - f(x_a) = (x_b - x_a) W

In words, mapping a difference gives exactly the same answer as mapping its two endpoints and subtracting them. A successful test on arbitrary pairs can therefore be explained by endpoint prediction alone. It does not establish that "cooking" is a reusable direction.

Our state-retrieval test asks more: the same named change must help on categories excluded from learning that direction. Our earlier arbitrary-pair experiment remains evidence about correspondence between endpoints; its scores are not pooled with the present results.

## 4. What counts as a good result?

For each test query, the candidate set contains pictures in both states of the test categories that support that change. We remove the query image itself and one requested-state picture per category so that each category contributes seven candidates in each state. Depending on the round and change, there are 14 to 56 candidates. Every method searches the same candidates for a given query.

A picture counts as relevant only if it has both the requested state and the query's category label. For a raw-chicken query with a cooking request, a cooked-chicken image is relevant; a raw-chicken or cooked-fish image is not. We do not require a particular matching photograph, because the dataset does not provide paired physical instances.

### The clearest score: is the first result right?

Our main presentation measure is **first-result success**: how often does the top picture have both the requested state and the correct object category? This is also called joint top-1 success. For a raw-chicken image with a cooking request, cooked chicken succeeds; raw chicken and cooked beef fail.

We also report **category retention**: how often is the first result still chicken, regardless of its state? This catches a failure that the state score alone would miss. A method can retrieve more cooked food while confusing chicken with beef.

We average within categories, then within each change, then equally across the five changes. These percentages therefore describe an equal-weight average over categories and changes, not a simple count of successful queries divided by 496. The overall result is not specifically a beef-to-chicken result.

### Keeping the original statistical test intact

The experiment originally chose **average precision (AP)** as its primary score, and used it to choose step sizes on validation data. AP measures the order of all relevant pictures in the ranked list. For example, if the only two relevant pictures are at ranks one and three, AP is the average of 1 and 2/3, about 0.833. This is an illustration, not an experimental score.

We retain those original tests. First-result success is now easier to see in the paper, but this is a presentation change made after the results. We have not replaced the original success criterion or calculated new confidence intervals for first-result success. The reported AP intervals cannot be read as intervals for the success percentages. An AP of 0.4532 is a ranking score, not 45.32% accuracy.

### Comparisons and the preservation requirement

The four planned comparisons ask whether the learned direction beats leaving the query alone, moving in random directions, reversing the correct direction, and applying one of the other state changes. A cooking direction should do more than an arbitrary movement or a drying direction of the same length. We average the scores of 100 random directions and, separately, the scores of the other four changes. We do not average those vectors before searching.

We also set a category-preservation requirement before inspecting the results. The method must improve ranking in all four planned comparisons, and the uncertainty calculation must support a loss of less than five percentage points in correct-category first results. This allows some variation but prevents us from calling a state shift successful while ignoring substantial damage to category retention.

The uncertainty estimates account for repeated query pictures within categories and for categories that occur in several changes. Appendix C explains the bootstrap and the correction for making four comparisons. These estimates are conditional on our fitted models and candidate sets; they do not capture every source of uncertainty.

## 5. Results: useful transfer, with clear limits

### How often did the first result have both the right state and object?

| Method | First result has both | First result keeps the object category |
|---|---:|---:|
| Image similarity search (no state change) | 20.9% | 86.6% |
| Shared direction transferred from text | 38.7% | 79.4% |
| Direction learned from training images | 39.5% | 71.3% |
| Direct search with the requested text | 42.1% | 56.8% |

**Image similarity search** uses the raw-chicken image's embedding directly and looks for nearby pictures. It gets no cooking instruction. **Our shared-direction method** adds the cooking vector learned from other categories, then searches the same pictures. Its success rate is 17.8 percentage points higher. That describes the benefit of adding the learned instruction; it does not by itself prove an especially good way to follow instructions.

**Direct text search** instead uses a description such as "cooked chicken" as the whole query, mapped through the same learned bridge. It gets an explicit test-category noun that the shared direction does not use. Even with this difference in input, it is a necessary practical comparison: it succeeds first more often than our method, 42.1% versus 38.7%. Our method keeps the category more often, 79.4% versus 56.8%. The present data do not establish that one is the better retrieval system overall.

**The image-derived direction** is estimated from training photographs rather than text. Its first-result success is similar to the text-derived direction. This helps locate the cross-modal finding: a change estimated in language can be useful when moved into vision. It does not show that language is better than learning from visual examples. These first-result comparisons are descriptive; the original uncertainty tests concern AP.

### Does the correct text change do more than an arbitrary movement?

The AP gain is 0.0852. Its adjusted confidence interval is [0.0294, 0.1471]. The corresponding intervals are above zero for all four planned comparisons:

| Comparison | AP gain from correct direction | Adjusted interval |
|---|---:|---|
| Versus image similarity (no change) | +0.0852 | [+0.0294, +0.1471] |
| Versus random directions | +0.0870 | [+0.0334, +0.1476] |
| Versus the reversed direction | +0.1516 | [+0.0833, +0.2252] |
| Versus other state changes | +0.0795 | [+0.0278, +0.1364] |

Each interval uses 98.75% confidence to account for the four planned comparisons together. Under the approximate bootstrap assumptions, this gives a family confidence level of at least 95%. Appendix C states the assumptions and what is held fixed.

These ranking results support a real signal in this fixed evaluation. They do not satisfy the complete success requirement. Correct-category first results fall by 7.20 percentage points. The one-sided 95% lower confidence bound for that change is -13.56 percentage points, below our permitted -5-point threshold. We therefore cannot claim a state shift that meets the category-preservation requirement.

![Ranking gains and differences between changes](/Users/sz/Programming/CDS/conv-mod/sz684/output/figures/state_transfer_effects.png)

Figure 2. Left: AP gains over the four planned controls, with adjusted intervals. Right: gains over the original query for each change, with ordinary 95% intervals. In this figure, "noun" means the dataset's category label and "paired" means that methods were compared on the same queries; it does not mean paired before-and-after photographs. The individual-change intervals are descriptive and do not establish five separate significant findings.

### The changes behave differently

| Change | No-change AP | Shared-direction AP | AP gain |
|---|---:|---:|---:|
| Cleaning | 0.4182 | 0.5037 | +0.0855 |
| Cooking | 0.3833 | 0.4922 | +0.1089 |
| Ripening | 0.3359 | 0.5360 | +0.2001 |
| Thawing | 0.3526 | 0.3629 | +0.0103 |
| Drying | 0.3501 | 0.3710 | +0.0209 |

All five observed averages increase, but the sizes differ considerably. The uncertainty for some individual changes, including thawing, includes zero. The overall result should therefore not be read as evidence that every state change transfers reliably.

The broader method table, including the alternative Procrustes mapping, remains in Appendix B. Some alternatives have higher first-result success than the main ridge-based method. We selected the primary method before these test results and do not retrospectively substitute the best-scoring alternative.

## 6. Two attempts to preserve the category

After seeing the primary result, we tried two changes to the method. These are follow-up explorations on the same test data, not additional confirmatory experiments.

First, we chose the step size only from settings whose validation results lost no more than five percentage points of category retention. Second, we estimated directions that distinguish the training categories by averaging their image vectors. We removed part or all of the learned state direction's component along that category-related space, restored the original vector length, and selected the removal strength and step size using validation results. This is the identity-subspace projection method in the original paper. The estimated space is a proxy for category information, not a proven separation of object and state.

| Method | Average precision | Correct category in first result |
|---|---:|---:|
| Image similarity (no change) | 0.3680 | 86.6% |
| Original shared direction | 0.4532 | 79.4% |
| Step size constrained on validation | 0.4212 | 81.2% |
| Category-component removal, selected on validation | 0.4338 | 80.6% |

Both follow-ups retain some AP improvement over the original image query, but neither demonstrates our required category-preservation tolerance on the test queries. The constrained-step AP gain is 0.0532, with an ordinary 95% interval [0.0192, 0.0916]. For component removal, the gain is 0.0658, with interval [0.0254, 0.1131]. These are descriptive, unadjusted follow-up intervals.

The practical lesson is that a preservation rule on validation examples did not guarantee preservation on new categories. The follow-ups also give up some of the original ranking gain. They identify a problem worth solving rather than a completed solution.

![The tradeoff between ranking quality and category retention](/Users/sz/Programming/CDS/conv-mod/sz684/output/figures/state_category_tradeoff.png)

Figure 3. Selected primary methods plotted by category retention and AP. Higher is better on both axes. "Noun top-1" means the fraction of first results with the correct category. "Universal text" is the shared text-derived change. The coordinates are descriptive; the plot does not show confidence intervals.

## 7. What the experiment establishes, and what it leaves open

The clearest result is that a relation estimated from text about training objects can, through a supervised bridge, help image retrieval for held-out object labels in this sample. It outperforms the four planned controls on the main ranking score. Yet it also loses category information, and the complete success criterion fails. Improving a requested attribute and preserving the object are separate requirements.

Label quality is the most important limitation. Visual inspection found some "thawed" examples showing cooked dishes, some "unripe" examples showing ripe-looking or unrelated fruit, and photographs in which wetness or dryness is difficult to establish. This inspection does not estimate a label-error rate. Our scores measure agreement with the released labels, not independently verified physical states. We did not correct questionable labels after seeing the model's scores.

The photographs are independent examples rather than controlled state changes of the same physical object. Background, viewpoint, composition and web source can differ with the state label. A direction might partly exploit those differences. Exact image checks do not rule out near duplicates, and the archive does not supply the source identities needed for a full photographer-level split.

There are also limits to generalization. We test one language/image model pair, five selected changes, small candidate sets and a fixed set of related category labels. The test separation excludes exact labels rather than whole object families. The uncertainty estimates leave the fitted maps, chosen step sizes and candidate sets fixed. Our local design record was saved before extracting the natural-image features, but it is not an external preregistration.

These boundaries matter for the paper's claim. The results motivate useful state-directed retrieval and reveal a preservation cost. They do not establish a general algebra of concepts, a causal model of physical change, or reliable understanding of every pictured state.

## 8. What to test next

To return directly to the cross-modal question, estimate each named change separately from training text and training images. After applying the learned bridge, ask whether the text direction matches the corresponding image direction more closely than wrong changes, and whether that agreement predicts transfer to held-out object families. A bridge fit to shuffled text/image correspondences is another useful control. These tests need a fresh evaluation and choices made before seeing its results. Even success would concern relations after a learned bridge, rather than identical coordinates in the original models.

For a useful retrieval system, compare methods that all receive the requested change. Include source-image-plus-target-text retrieval and a vision-language model baseline, with category wording provided consistently across methods that use it. Predeclare first-result state-and-category success, category retention and their uncertainty estimates; use AP to support the full-ranking analysis. The no-change search remains an explanatory reference, not sufficient evidence of an advantage over instructed search. These new baselines have not yet been tested.

The first priority is a new evaluation with independently checked state labels. The ChangeIt-derived object-state subset [9] and STATUS Bench [10] are candidates, although the latter requires access to its underlying Ego4D images. We have not evaluated either here. Photographs or video frames should be separated by source, and related category families should be held out together where the available data permit it.

The second priority is preserving object information. A shared global shift may be too crude: the visual change caused by cooking or cleaning can depend on the object. A next method could adapt the change to the source image or constrain information that identifies its category. Any method chosen after inspecting these results needs a fresh test set for a stronger claim.

Finally, repeat the experiment with another independently pretrained model family and larger candidate sets containing harder distractors. A useful method should keep improving state-and-category retrieval when there are many opportunities to retrieve the right state of the wrong object. The present work is a results-bearing working paper; verified labels, broader baselines and a fuller review of prior work remain necessary before submission.

## Appendix A. Exact representation and fitting details

We extract BLOOMZ 1.7B hidden-state index 19, counting the initial embedding output as index 0, and DINOv2 base block 10, counting blocks from zero. These layers were selected in the earlier, unrelated WIT study, not by searching the state-retrieval test results. Model revisions are pinned. Feature extraction uses float32 on Apple MPS; analysis uses float64 on the CPU.

The first four sampled pictures of each training state/category combination fit the map. The remaining four provide a separate image-derived change for comparison. Each of four fixed text templates describes the state/category combination. Templates are repeated wording measurements, not independent category examples.

Preprocessing uses training data only. We clip extreme feature values using the original mean-per-example 95th-percentile rule across layers, normalize vector lengths, subtract the training mean, and use principal component analysis (PCA) to retain 64 dimensions. PCA finds directions with the most training variation; we do not whiten, or rescale each retained direction to equal variance. We normalize the projected endpoint vectors again.

Ridge regression fits an affine text-to-image predictor with regularization alpha = 1. For difference vectors, its constant offset cancels. The shared relation averages text differences over templates within category and then over training categories. Step size is chosen by validation AP, with ties favoring the smaller value. No test-category text contrast enters this shared relation.

The chosen step sizes are:

| Change | Round 0 | Round 1 | Round 2 | Round 3 | Round 4 |
|---|---:|---:|---:|---:|---:|
| Cleaning | 2 | 4 | 4 | 4 | 4 |
| Cooking | 1 | 4 | 4 | 2 | 1 |
| Ripening | 4 | 4 | 4 | 2 | 4 |
| Thawing | 4 | 4 | 2 | 2 | 0.5 |
| Drying | 1 | 4 | 4 | 4 | 2 |

## Appendix B. Other comparisons and their results

The random directions have the same length as the correct transferred direction and use its chosen step size. Reversal changes only the direction's sign. For the wrong-change control, each of the other four learned changes is matched to the correct direction's length and evaluated separately; we then average their retrieval scores.

The image-derived comparison estimates the change from the reserved training images rather than text. Orthogonal Procrustes is an alternative mapping that rotates or reflects the text coordinates instead of fitting an unrestricted ridge matrix. Both choose their own step sizes on validation data.

Direct target text maps the requested test-category description into image space and uses it as the whole search query. The category-specific text offset instead adds the difference between that category's requested-state and starting-state descriptions to the source image. Both are supplied with test-category wording that the shared direction does not use. They test what more explicit recognition information can accomplish and are not equal-information controls.

| Method | Average precision | Requested state first | Correct category first | Both first |
|---|---:|---:|---:|---:|
| Image similarity (no change) | 0.3680 | 26.6% | 86.6% | 20.9% |
| Shared text-derived direction | 0.4532 | 54.9% | 79.4% | 38.7% |
| Reversed direction | 0.3016 | 18.4% | 83.8% | 14.6% |
| Average random direction result | 0.3662 | 29.2% | 84.0% | 22.2% |
| Average wrong-change result | 0.3736 | 29.2% | 83.0% | 21.6% |
| Change learned from images | 0.4557 | 61.0% | 71.3% | 39.5% |
| Procrustes text direction | 0.4671 | 67.8% | 70.5% | 44.8% |
| Direct target text | 0.4354 | 77.1% | 56.8% | 42.1% |
| Category-specific text offset | 0.4508 | 56.5% | 75.7% | 38.3% |

The follow-up component-removal method keeps enough directions to explain 80% of the variation between training-category image means. It removes 0%, 50% or 100% of the state vector's component in that space, then restores the original length. It chooses removal strength and step size together, requiring the validation category-retention loss to be no worse than five percentage points. Ties favor smaller removal strength and then smaller step size. This method was defined after the primary results and remains exploratory.

## Appendix C. How the uncertainty estimates work

Methods are compared on the same source queries. The word "paired" refers to these matched method comparisons, not to matched source/target photographs. For each comparison, we calculate the difference in AP for each query before averaging.

We generate 20,000 bootstrap resamples. Each resample draws category labels with replacement, using the same category multiplicities wherever a label appears in several changes. It then resamples source queries within each category/change group. The same resampling draws are used for all method comparisons. We recompute the equal-weight average over the five changes each time.

There are four planned aggregate comparisons. We use the 0.625th and 99.375th percentiles of their bootstrap distributions to form individual 98.75% intervals. The Bonferroni correction allocates the total 5% error allowance across those four comparisons. The resulting family coverage is approximate because bootstrap coverage and category exchangeability are assumptions.

For the category-preservation check, we use the fifth percentile of the bootstrap distribution as a one-sided 95% lower bound. It must be above -0.05, meaning a loss smaller than five percentage points. The observed bound is -0.1356, so this requirement fails. A confidence interval crossing a threshold is not a claim that the methods are equivalent.

We also save resampling results that vary only the category labels and estimates that leave out one category at a time. The five named changes are treated as a fixed evaluation set, not a random sample from every possible semantic relation. We keep candidate images, fitted maps and validation-selected settings fixed during resampling. The intervals therefore omit uncertainty from refitting those components and cannot remove unknown web-source dependence.

## Appendix D. Reproducibility and the earlier endpoint study

The configuration, selected archive entries, image checksums, category assignments, model revisions, software versions, validation curves and per-query scores are recorded with the experiment. Image downloads are checked against the ZIP member's CRC32 and length, and selected images have distinct SHA256 hashes.

Category-group assignment uses seed 20261016 and accepts the first randomized assignment that gives every change at least one eligible category in each of the five groups. In round f, group f is tested, group (f+1) mod 5 is used for validation, and the other three groups are used for fitting. For a source query with sampled index i, source image i and requested-state image (i+1) mod 8 are excluded from every candidate category. Indices do not identify matched physical instances.

The primary protocol and analysis code were saved before natural-image feature extraction. Later preservation experiments are recorded separately and leave the primary results unchanged.

The earlier WIT study retrieved arbitrary held-out difference vectors with 19.5-25.8% top-1 accuracy among 256 candidates across three overlapping splits. Using only one endpoint gave 3.9-8.6%. Those findings support correspondence information from both endpoints. The linear identity in Section 3 explains why they do not, by themselves, establish reusable semantic changes. Their candidate sets and outcomes differ from the present task, so the scores are not combined statistically.

The natural-image pipeline is implemented in state_data.py, state_extract.py, state_analysis.py and state_report.py. The exploratory follow-up is state_identity_followup.py. The configuration is configs/state_transfer.json. Raw images, features, saved design records, source snapshots and statistical outputs are under results/state_transfer/. This directory is ignored by Git and must be preserved separately when sharing the research. Both manuscripts now lead with cross-modal relation transfer and held-out-category testing. This rewrite does not rerun experiments or change the measured results. state_report.py rebuilds figures and the companion report; the manuscripts are edited separately and rendered with paper_pdf.py.

## References

[1] Huh et al. (2024). Position: The Platonic Representation Hypothesis. ICML. https://proceedings.mlr.press/v235/huh24a.html

[2] Fournier, Dupoux and Dunbar (2020). Analogies minus analogy test: measuring regularities in word embeddings. CoNLL. https://aclanthology.org/2020.conll-1.29/

[3] Linzen (2016). Issues in evaluating semantic spaces using word analogies. https://arxiv.org/abs/1606.07736

[4] Sadeghi, Zitnick and Farhadi (2015). Visalogy: Answering Visual Analogy Questions. NeurIPS. https://papers.neurips.cc/paper_files/paper/2015/hash/45f31d16b1058d586fc3be7207b58053-Abstract.html

[5] Patashnik et al. (2021). StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery. ICCV. https://arxiv.org/abs/2103.17249

[6] Isola, Lim and Adelson (2015). Discovering States and Transformations in Image Collections. CVPR. https://web.mit.edu/phillipi/Public/states_and_transformations/index.html

[7] Muennighoff et al. (2023). Crosslingual Generalization through Multitask Finetuning. ACL. https://arxiv.org/abs/2211.01786

[8] Oquab et al. DINOv2: Learning Robust Visual Features without Supervision. https://arxiv.org/abs/2304.07193

[9] Newman et al. (2024). Do Pre-trained Vision-Language Models Encode Object States? https://arxiv.org/abs/2409.10488

[10] Ukai, Kurita and Inoue (2025). STATUS Bench: A Rigorous Benchmark for Evaluating Object State Understanding in Vision-Language Models. ACM Multimedia. https://arxiv.org/abs/2510.22571
