# Explainable Coarse-to-Fine Ancient Manuscript Duplicates Discovery

Explainable Coarse-to-Fine Ancient Manuscript Duplicates Discovery, with Oracle Bones as a Case Study. 

## Illustration and Demostration Video

https://www.youtube.com/watch?v=YlRtCDMvd2s

## arXiv Version of this Paper

https://arxiv.org/abs/2505.03836

## Abstract

Ancient manuscripts are the primary source of ancient linguistic corpora. However, many ancient manuscripts exhibit duplications due to unintentional repeated publication or deliberate forgery. The Dead Sea Scrolls, for example, include counterfeit fragments, whereas Oracle Bones (OB)  contain both republished materials and fabricated specimens. Identifying  ancient manuscript duplicates is of great significance for both archaeological curation and ancient history study.  In this work, we design a progressive OB  duplicate discovery framework that combines unsupervised low-level keypoints matching with high-level text-centric content-based matching to refine and rank the candidate OB duplicates with semantic awareness and interpretability. We compare our model with state-of-the-art content-based image retrieval and image matching methods, showing that our model  yields comparable recall performance and the highest simplified mean reciprocal rank scores for both Top-5 and Top-15 retrieval results, and with significantly accelerated computation efficiency. We have discovered over 60 pairs of new OB duplicates in real-world deployment, which were missed by domain experts for decades. Code, model and real-world results are available at: https://github.com/cszhangLMU/OBD-Finder/.

## Introduction

Ancient manuscripts are the key source for ancient language corpora. However, many ancient manuscripts contain duplicates, due to unintentional repeated publication or deliberate forgery. For instance, the Dead Sea Scrolls contain forged fragments, while Oracle Bones (OB) contain both repeated publications and forged ones. Finding  ancient manuscript duplicates  can help identify forgeries, eliminate duplicate fragments and prevent redundant research, while offering the potential to correct erroneous fragment rejoinings. Moreover, it facilitates empirical study on the damage and deterioration of ancient manuscripts  during their circulation.

In particular, the identification of Oracle Bone  duplicates has been a fundamental research issue in Oracle Bone Inscription (OBI) research. OBI was used in the late Shang Dynasty more than 3000 years ago for divination and recording purposes. But from then on, these Oracle Bones had been buried underground for thousands of years, until they were rediscovered in the year of 1899 for containing inscribed ancient Chinese characters. Due to drilling and burning before and during divination, and the long-term underground corrosion, as well as excavation, transportation, and circulation after their excavation, about 90% of the OBs have been fragmented and are now scattered in different collections around the world.

As precious cultural relics, many Oracle Bones were circulated among various collectors and antique dealers in the initial period after their discovery in 1899. Limited by communication and dissemination methods at that time, the same OBs might have been repeatedly published in different publications at different times in different locations, which led to the phenomenon of OB duplicates, denoting that  the fragments were repeatedly published. Some OBs  further fragmented during circulation; on the other hand, as OBI research advances, some fragmentary OBs might have been rejoined by OBI domain experts and republished again. As such, OB duplicates exhibit both one-to-one and one-to-many image matching relationships. Although domain experts have manually found many duplicates in their research, given the huge cardinality of OB fragments (more than 160,000), AI-enabled OB duplicates discovery becomes imperative.

![Figure 1: Three groups of new Oracle Bone duplicates discovered by our model, which have been missed by domain experts for decades. For each group of duplicate, we provide both the manual copies and rubbings of the Oracle Bones. We can see that, finding Oracle Bone duplicate is similarity-based matching, rather than exact matching, and there exists both one-to-one (e.g.the bottom left pair) and one-to-many matchings (e.g.the top left pair and the right pair). Note that, in our implementation, we only use the manual copies of the Oracle Bones.](./case-3.png)

Oracle Bone Inscription is carved writing, its main research materials and  publication formats are rubbings and manual copies. For  rubbing materials,  people place papers onto the surface of the Oracle Bones, then use Rubbing (with inks) to copy the carved inscriptions. Domain experts can also reproduce (copy) the carved inscriptions by hand, which is named Manual Oracle Bone Inscriptions Copy, referred to as OB manual copies for short. Figure 1  presents examples for both formats, which are actually cases of the new OB duplicates discovered by our model. Comparing the two formats,  manual OBI copies rely on domain knowledge but has no background noises, whereas OBI rubbings often contains substantial noise disturbance, although domain knowledge is not required. Both formats can keep the original sizes of the Oracle Bones, which is not possible when using cameras.  In 2022, the largest collection of manual OBI copies was published, for which a large team of OBI researchers invested 10 years to create high-quality manual OBI copies for around 60,000 OBs.


With this new collection at hand, in this work we aim to devise a comprehensive framework for discovering OB duplicates at large-scale. Since different domain experts have slightly different copying styles for the same OBIs (such as variations in pen movement, stroke thickness, brush pressure), finding OB duplicates can be essentially formulated as a content-based image retrieval (CBIR) or image matching task.

Contributions. To our knowledge, this work is among the first technical efforts that investigate AI-enabled Oracle Bone duplicates discovery. We design OBD-Finder, an explainable coarse-to-fine text-centric Oracle Bone duplicates discovery framework that successively utilizes unsupervised low-level key feature points matching and high-level content/character similarity for ranking the OB duplicate candidates. We have deployed our model in real-world applications, where we have successfully identified 63 pairs of new Oracle Bone duplicates, which have been verified by OBI community. Figure 2 presents three groups of new OB duplicates discovered by our model.

We also conduct extensive experiments on a large dataset of OB copies.  We compare our model with state-of-the-art CBIR and image matching methods, showing that our model achieves Top-K recall performance comparable to state-of-the-art methods, but with significantly accelerated computational efficiency and substantially reduced GPU memory consumption.  Our model also attains the highest simplified mean reciprocal rank scores for both Top-5 and Top-15 retrieval results, demonstrating that it excels at prioritizing correct matches. 

------

## 🖼️ Methodology

**Framework.** As can be seen from Figure 1, we propose a progressive coarse-to-fine Oracle Bone duplicate discovery framework, namely OBD-Finder, which combines unsupervised low-level keypoint matching with high-level, character-centric content-based image matching. Keypoint matching operates at low-level visual feature scale, which can prune out candidates with low degree of match in the initial stage, but lacks explicit semantic supervision and interpretability. Our framework bridges this gap by first grouping the keypoints based on their association with the character regions, then assesses the global matching degree between the two groups of keypoints via character-level visual content similarity computation. This dual matching mechanism enhances Oracle Bone duplicates discovery accuracy through a progressive coarse-to-fine refinement manner, by effectively and seamlessly integrating both low-level keypoint and high-level character-based semantic cues, resulting in more accurate and semantic-aware image matching. 

<div align="center"> <img src="images/1.png" width="80%" alt="Framework Architecture"> <br> <em>Figure 1. The overall framework of OBD-Finder for Oracle Bone duplicates discovery.</em> </div>

------

Our framework consists of four  subsequent steps: 

  1.Feature Extraction. We perform unsupervised  feature points/keypoints extraction on the OBs using a pre-trained model.

  2.Feature Matching We next apply unsupervised keypoints mapping between the two OB images using a pre-trained model. Candidate with low overall matching degrees will be filtered out. 

  3.Coordinate Alignment. After obtaining the correspondence between the keypoints in feature matching, we apply affine transformations for each image pair, in which we map the coordinates of the image with fewer feature points to the other image. 

  4.Character-level Content Similarity. we first localize the Oracle Bone characters in each image, using a  text detector. Given that the coordinate systems of two images are aligned,  for each character in the smaller image,  we search for the overlapped characters in the counterpart image, then compute the content similarity between the overlapped characters, using a simple Siamese network model. 


## Key Features

- 🚀 **AI-enabled Oracle Bone Duplicates Discovery**: This work is among the first efforts in An-enabled Oracle Bone Duplicates Discovery.
  
- 🔍 **Explainable Coarse-to-Fine Framework**: We propose OBD-Finder, which is a progressive coarse-to-fine framework that seamlessly proceeds from low-level keypoints matching to high-level semantic-aware content similarity computation, resulting in very accurate OB duplicate discovery.
  
- ⚡ **High Efficiency**: more than 40x faster than SOTA methods, with only 1/3 GPU memory usage.
  
- 🏆 **Real-world Deployment**: We have discovered over 60 pairs of new Oracle Bone duplicates, which have been verified by OBI domain experts (Yi Men and Yingqi Chen, who are also collaborators/co-authors of this work) 


## Experiments

### Comparative Experiments

<div align="center">

**Comparison with Image Retrieval Methods**

|                      | Recall@1 | Recall@5 | Recall@10 | Recall@15 | Recall@20 |
| :------------------: | :------: | :------: | :-------: | :-------: | :-------: |
|      Smooth-AP       |   34.1   |   62.9   |   73.8    |   78.8    |   82.8    |
|     Proxy-Anchor     |   71.5   |   77.0   |   84.1    |   86.6    |   91.9    |
|       HashNet        |   57.6   |   63.8   |   69.3    |   78.4    |   85.8    |
|      HybridHash      |   60.8   |   66.8   |   73.2    |   79.3    |   88.3    |
| **Ours(OBD-Finder)** | **80.0** | **85.3** | **90.4**  | **94.3**  | **98.0**  |

</div>

<div align="center"> <img src="images/2.png" width="80%" alt="Framework Architecture"> <br> <em>Figure 2. Content-based image retrieval performance of different methods. </em> </div>

<div align="center">

**Comparison with state-of-art images matching methods**

|  Method  | Recall@1 | Recall@5 | Recall@10 | Recall@15 | Recall@20 | Recall@25 |
| :------: | :------: | :------: | :-------: | :-------: | :-------: | :-------: |
|   SIFT   |   33.3   |   40.0   |   46.8    |   53.3    |   66.6    |   73.2    |
|  LoFTR   |   73.6   |   75.4   |   81.3    |   86.0    |   92.3    |   98.3    |
| OmniGlue |   82.5   |   86.0   |   91.2    |   95.6    |   98.2    |    100    |
|  MINIMA  |   84.4   |   90.5   |   94.7    |   98.2    |    100    |    100    |
|   Ours   |    80    |   85.3   |   90.4    |   94.3    |    98     |    100    |

</div>

<div align="center">

**Rank@K scores of different methods**

|          | Recall@5 | Recall@10 | Recall@15 | Recall@20 | Recall@25 |
| :------: | :------: | :-------: | :-------: | :-------: | :-------: |
|   SIFT   |   1.6    |   2.70    |   4.16    |   6.15    |   7.74    |
|  LoFTR   |   1.09   |   1.88    |   2.74    |   3.76    |   5.22    |
| OmniGlue |   1.13   |   1.51    |   2.05    |   2.52    |   2.89    |
|  MINIMA  |   1.36   |   1.66    |   2.07    |   2.38    |   2.38    |
|   Ours   |   1.06   |   1.63    |   2.00    |   2.61    |   2.93    |

</div>

<div align="center">

**Other performance metrics including inference speed, FPS, and GPU usage**

|          | Recall@20 | Inf. Speed(s/pairs) | FPS(pair/s) | GPU(MIB) |
| :------: | :-------: | :-----------------: | :---------: | :------: |
|   SIFT   |    1.6    |        0.017        |     59      |   N/A    |
|  LoFTR   |   1.09    |         3.6         |    0.28     |   3520   |
| OmniGlue |   1.13    |         45          |    0.02     |  23612   |
|  MINIMA  |   1.36    |          1          |      1      |  14890   |
|   Ours   |   1.06    |        0.021        |    47.62    |   5215   |

</div>

<div align="center"> <img src="images/3.png" width="80%" alt="Framework Architecture"> <br> <em>Figure 3. Representative image matching results by different methods. </em> </div>

### Real-world Deployment: Over 60 Pairs of New Oracle Bone Duplicates Disovered

The new Oracle Bone Duplicates results discovered by our method can be accessed via the following link:

[New Oracle Bone Duplicated Discovered by OBD-Finder](https://drive.google.com/drive/folders/1lx3sSGg-W1x7WTHc_K5cQh_cCwtTzf4F) 

The following are 20 pairs of new Oracle Bone Duplicates discovered by OBD-Finder:

<div align="center"> <img src="images/4.png" width="80%" alt="Framework Architecture"> <br> <em>Figure 4. New Oracle Bone duplicates disovered by OBD-Finder (1).</em> </div>

<div align="center"> <img src="images/5.png" width="80%" alt="Framework Architecture"> <br> <em>Figure 5. New Oracle Bone duplicates disovered by OBD-Finder (2).</em> </div>



## 📥 Data & Models

Download our models and sample dataset:

[Models and sample dataset of this work](https://drive.google.com/drive/folders/1fgnLoOdRNXDf38GXDb5NZbh8nfeLtgNb)

## Usage Instructions

### 🔍 Step 1: Preliminary Screening (Image Pair Matching)

**input**

- A folder containing images of a certain type of oracle bone script copies (such as "Yellow category")

**Run the command**

```
python Feature_matching/pipei5.py
```

**Output**

- `1.txt`: List of suspected duplicate pairs meeting threshold conditions

**Technical Details**

- Two-stage feature matching using pre-trained **SuperPoint + LightGlue** model
- Candidate screening through similarity thresholds

------

### 📦 Step 2: Dataset Construction

#### 2.1 Format Conversion

**Run the Command**

```
python utils/T_excale.py
```

**Input**: `1.txt`

**Output**: `1.xml` (Structured XML format)

#### 2.2 Directory Organization

**Run the Command**

```
python utils/direct2.py
```

**Output Structure**:

```
- folder1/
  ├── A_B/
  │   ├── A.jpg
  │   ├── B.jpg
  │   ...
```

#### 2.3 Text Detection

**Run the Command**

```
python Oracle_character_detection/detect3.py
```

#### 2.4 Character Analysis

**Run the Command**

```
python Oracle_character_detection/Distance_results1.py
```

**Output**:

```
folder1/
├── A_B/
│ ├── A.jpg, B.jpg # Original image
│ ├── A_detected.jpg, B_detected.jpg # Text detection result image
│ ├── A.txt, B.txt # Character annotation results
│ ├── splits/ # Split character image
│   │   ├── A_char_01.jpg
│   │   ├── B_char_03.jpg
│ ├── Matched_container.csv # Matching Table of adjacent characters
```

------

### 🤖 Step 3: Character Similarity Prediction

**Run the Command**

```
python Siamese-pytorch/predict.py
```


------

## 📚 References

1. Brown, A., et al. "Smooth-AP: Smoothing the Path Towards Large-Scale Image Retrieval." *ECCV* 2020.
2. Kim, S., et al. "Proxy Anchor Loss for Deep Metric Learning." *CVPR* 2020.
3. Dubey, S. R., et al. "Vision Transformer Hashing for Image Retrieval." *ICME* 2022.
4. He, C., & Wei, H. "HybridHash: Hybrid Convolutional and Self-Attention Deep Hashing for Image Retrieval." *ICMR* 2024.
5. DeTone, D., et al. "SuperPoint: Self-Supervised Interest Point Detection and Description." *CVPRW* 2018.
6. Lindenberger, P., et al. "LightGlue: Local Feature Matching at Light Speed." *ICCV* 2023.
7. Cao, Z., et al. "HashNet: Deep Learning to Hash by Continuation." *ICCV* 2017.
8. Sun, J., et al. "LoFTR: Detector-Free Local Feature Matching with Transformers." *CVPR* 2021.
9. Jiang, H., et al. "OmniGlue: Generalizable Feature Matching with Foundation Model Guidance." *CVPR* 2024.
10. Ren, J., et al. "MINIMA: Modality Invariant Image Matching." *CVPR* 2025.

------
