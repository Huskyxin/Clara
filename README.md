# Clara

## Dataset
To investigate the effectiveness of AMPLE, we adopt three vulnerability datasets from these paper: 
* DiverseVul: <https://drive.google.com/file/d/12IWKhmLhq7qn5B_iXgn5YerOQtkH-6RG/view?pli=1>
* PrimeVul: <https://drive.google.com/drive/folders/1cznxGme5o6A_9tT8T47JUh3MPEpRYiKK>

## Requirement
Our code is based on Python3. There are a few dependencies to run the code. The major libraries are listed as follows:
* torch
* torch_geometric
* numpy
* sklearn

## 📥 Guide

#### 1、Preprocessing

- (1) **Joern**:
  
  We download Joern [here](https://github.com/joernio/joern).

- (2) **Parse**:
  
  Follow the Joern documentation to generate a PDG.

#### 2、Word2Vec
-  (3) **Word2Vec**:
For PDG, we use the word2vec to initialize the node representation.
```bash
python utils/word2vec.py
```

#### 3、Training
-  (4) **Model Training**:
```bash
bash scripts/PrimeVul.sh
```

```bash
bash scripts/DivVul.sh
```
