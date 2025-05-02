# OBD-Finder: Explainable Coarse-to-Fine Text-Centric Oracle Bone Duplicates Discovery

## Introduction

Oracle Bone inscriptions (OBIs), dating back over 3000 years to the late Shang Dynasty, were used for divination and record-keeping. Rediscovered in 1899, they contain ancient Chinese characters. However, due to factors like drilling, burning, corrosion, and mishandling, about 90% of OBs are fragmented and scattered across global collections.

In the early years after discovery, OBs circulated widely among collectors and dealers. Due to limited communication and the fragmentation of OBs during circulation, many OBs were repeatedly published in different countries, leading to duplicates. As research advanced, some fragments were rejoined and republished, revealing complex one-to-one and one-to-many relationships. Despite significant efforts by domain experts, a large number of OB duplicates remain unidentified due to the sheer scale of over 160,000 fragments.

Given this vast number, computational methods are necessary to automatically identify duplicates. In 2022, the largest collection of manual OBI copies, created over 10 years by a team of experts, was published, covering around 60,000 OBs. This work aims to develop a comprehensive technique for discovering OB duplicates at scale. Since experts’ copying styles vary, the problem is one of similarity-based image matching rather than exact matching.

## Key Contributions

- 🚀 **First AI Solution**: Pioneering computational framework for OBI duplicate detection
- 🔍 **Explainable Architecture**: Dual-stage verification combining geometric and semantic matching
- ⚡ **High Efficiency**: 10x faster than SOTA methods with 85% less GPU memory usage
- 🏆 **Proven Effectiveness**: Verified through discovery of 60+ new duplicate pairs overlooked by experts
- 📚 **Low Resource Demand**: Training-free design requiring minimal annotation

------

## 🖼️ Methodology

**Framework.** As can be seen from Figure 1, we propose a progressive coarse-to-fine Oracle Bone duplicate discovery framework, namely OBD-Finder, which combines unsupervised low-level keypoint matching with high-level, character-centric content-based image matching. Keypoint matching operates at low-level visual feature scale, which can prune out candidates with low degree of match in the initial stage, but lacks explicit semantic supervision and interpretability. Our framework bridges this gap by first grouping the keypoints based on their association with the character regions, then assesses the global matching degree between the two groups of keypoints via character-level visual content similarity computation. This dual matching mechanism enhances Oracle Bone duplicates discovery accuracy through a progressive coarse-to-fine refinement manner, by effectively and seamlessly integrating both low-level keypoint and high-level character-based semantic cues, resulting in more accurate and semantic-aware image matching. 

<div align="center"> <img src="images/1.png" width="80%" alt="Framework Architecture"> <br> <em>Proposed coarse-to-fine framework workflow</em> </div>

------

Our framework consists of four  subsequent steps: 

  1.Feature Extraction. We perform unsupervised  feature points/keypoints extraction on the OBs using a pre-trained  model.

  2.Feature Matching We next apply unsupervised keypoints mapping between the two OB images using a pre-trained model. Candidate with low overall matching degrees will be filtered out. 

  3.Coordinate Alignment. After obtaining the correspondence between the keypoints in feature matching, we apply affine transformations for each image pair, in which we map the coordinates of the image with fewer feature points to  the other image. 

  4.Character-level Content Similarity. we first localize the Oracle Bone characters in each image, using a  text detector. Given that the coordinate systems of two images are aligned,  for each character in the smaller image,  we search for the overlapped characters in the counterpart image, then compute the content similarity between the overlapped characters, using a simple Siamese network model. 

**Key Features.**(i) OBD-Finder is a progressive coarse-to-fine framework that seamlessly proceeds from low-level keypoints matching to high-level semantic-aware content similarity computation, resulting in very accurate OB duplicate discovery.  (ii) It is a transparent pipeline with strong interpretability. (iii) It is unsupervised and almost training-free, only requiring little annotation effort. (iv) It is highly efficient, compared to ``heavy'' state-of-the-art image matching methods, which we will demonstrate in the empirical studies. 

## Operation steps

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

**Similarity Grading**

| Grade |  Range  |      Description       |
| :---: | :-----: | :--------------------: |
|   A   |  ≥90%   |     Highly similar     |
|   B   | 80%-90% | Medium-high similarity |
|   C   | 70%-80% |  Moderate similarity   |
|   D   |  ≥60%   |  Reference similarity  |

------

## Results

### Comparative Experiments

 **Comparison with Image Retrieval Methods**

|                  | Recall@1 | Recall@5 | Recall@10 | Recall@15 | Recall@20 |
| :--------------: | :------: | :------: | :-------: | :-------: | :-------: |
|    Smooth-AP     |   34.1   |   62.9   |   73.8    |   78.8    |   82.8    |
|   Proxy-Anchor   |   71.5   |   77.0   |   84.1    |   86.6    |   91.9    |
|     HashNet      |   57.6   |   63.8   |   69.3    |   78.4    |   85.8    |
|    HybridHash    |   60.8   |   66.8   |   73.2    |   79.3    |   88.3    |
| Ours(OBD-Finder) |   80.0   |   85.3   |   90.4    |   94.3    |   98.0    |

<div align="center"> <img src="images/2.png" width="80%" alt="Framework Architecture"> <br> <em>Image retrieval comparison results</em> </div>

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

<div align="center"> <img src="images/3.png" width="80%" alt="Framework Architecture"> <br> <em>Image retrieval comparison results</em> </div>

### The real result

Our OBD-Finder generates a series of bones, the school directory, and the cloud of chart data. The results can be accessed via the following link:

[OBD-Finder Data on Baidu Cloud](https://pan.baidu.com/s/1HYmJ9XHwO8_Zsm2TdBTZVg) Password: v099

Here are 20 representative set of new school achievements

<div align="center"> <img src="images/4.png" width="80%" alt="Framework Architecture"> <br> <em>real results1</em> </div>

<div align="center"> <img src="images/5.png" width="80%" alt="Framework Architecture"> <br> <em>real result2</em> </div>



## 📥 Data & Models

Download sample data and pretrained weights:
[![Dataset](https://img.shields.io/badge/Download-Data%2520&%2520Models-blue)](https://drive.google.com/drive/folders/1fgnLoOdRNXDf38GXDb5NZbh8nfeLtgNb)

------

## 📚 References

1. Brown, A., et al. "Smooth-AP: Smoothing the Path Towards Large-Scale Image Retrieval." *ECCV* 2020.
2. Kim, S., et al. "Proxy Anchor Loss for Deep Metric Learning." *CVPR* 2020.
3. Dubey, S. R., et al. "Vision Transformer Hashing for Image Retrieval." *ICME* 2022.
4. He, C., & Wei, H. "HybridHash: Hybrid Convolutional and Self-Attention Deep Hashing for Image Retrieval." *ICMR* 2024.
5. DeTone, D., et al. "SuperPoint: Self-Supervised Interest Point Detection and Description." *CVPRW* 2018.
6. Lindenberger, P., et al. "LightGlue: Local Feature Matching at Light Speed." *ICCV* 2023.
7. Cao, Z., et al. "HashNet: Deep Learning to Hash by Continuation." *arXiv:1702.00758* 2017.
8. Sun, J., et al. "LoFTR: Detector-Free Local Feature Matching with Transformers." *CVPR* 2021.
9. Jiang, H., et al. "OmniGlue: Generalizable Feature Matching with Foundation Model Guidance." *CVPR* 2024.
10. Ren, J., et al. "MINIMA: Modality Invariant Image Matching." *CVPR* 2025.

------

📬 Contact: [3625522651@qq.com](https://mailto:3625522651@qq.com/) | [GitHub Issues](https://github.com/your_repo/issues)