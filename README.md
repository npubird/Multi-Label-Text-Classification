# Deep Learning for Multi-Label Text Classification

This repository is my research project, and it is also a study of TensorFlow, Deep Learning (Fasttext, CNN, LSTM, RCNN, etc.).

The main objective of the project is to solve the multi-label text classification problem based on Deep Neural Networks. Thus, the format of the data label is like [0, 1, 0, ..., 1, 1] according to the characteristics of such problem.

## Requirements

- Python 3.6
- Tensorflow 1.8 +
- Numpy
- Gensim

## Innovation

### Data part
1. Make the data support **Chinese** and English (Which use `jieba` seems easy).
2. Can use **your own pre-trained word vectors** (Which use `gensim` seems easy). 
3. Add embedding visualization based on the **tensorboard**.

### Model part
1. Add the correct **L2 loss** calculation operation.
2. Add **gradients clip** operation to prevent gradient explosion.
3. Add **learning rate decay** with exponential decay.
4. Add a new **Highway Layer** (Which is useful according to the model performance).
5. Add **Batch Normalization Layer**.

### Code part
1. Can choose to **train** the model directly or **restore** the model from checkpoint in `train.py`.
2. Can predict the labels via **threshold** and **topK** in `train.py` and `test.py`.
3. Can calculate the evaluation metrics --- **AUC** & **AUPRC**.
4. Add `test.py`, the **model test code**, it can show the predict value of each labels of the data in Testset when creating the final prediction file.
5. Add other useful data preprocess functions in `data_helpers.py`.
6. Use `logging` for helping recording the whole info (including **parameters display**, **model training info**, etc.).
7. Provide the ability to save the best n checkpoints in `checkmate.py`, whereas the `tf.train.Saver` can only save the last n checkpoints.

## Data

See data format in `data` folder which including the data sample files.

### Text Segment

You can use `jieba` package if you are going to deal with the chinese text data.

### Data Format

This repository can be used in other datasets (text classification) by two ways:
1. Modify your datasets into the same format of the sample.
2. Modify the data preprocess code in `data_helpers.py`.

Anyway, it should depends on what your data and task are.

### Pre-trained Word Vectors

You can pre-training your word vectors (based on your corpus) in many ways:
- Use `gensim` package to pre-train data.
- Use `glove` tools to pre-train data.
- Even can use a **fasttext** network to pre-train data.

## Network Structure

### FastText

![](https://farm2.staticflickr.com/1917/45609842012_30f370a0ee_o.png)

References:

- [Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf)

---

### TextANN

![](https://farm2.staticflickr.com/1965/44745949305_50f831a579_o.png)

References:

- **Personal ideas 🙃**

---

### TextCNN

![](https://farm2.staticflickr.com/1927/44935475604_1d6b8f71a3_o.png)

References:

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)

---

### TextRNN

**Warning: Model can use but not finished yet 🤪!**

![](https://farm2.staticflickr.com/1925/30719666177_6665038ea2_o.png)

#### TODO
1. Add BN-LSTM cell unit.
2. Add attention.

References:

- [Recurrent Neural Network for Text Classification with Multi-Task Learning](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552)

---

### TextCRNN

![](https://farm2.staticflickr.com/1915/43842346360_e4660c5921_o.png)

References:

- **Personal ideas 🙃**

---

### TextRCNN

![](https://farm2.staticflickr.com/1950/31788031648_b5cba7bbf0_o.png)

References:

- **Personal ideas 🙃**

---

### TextHAN

References:

- [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)

---

### TextSANN

**Warning: Model can use but not finished yet 🤪!**

#### TODO
1. Add attention penalization loss.
2. Add visualization.

References:

- [A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING](https://arxiv.org/pdf/1703.03130.pdf)

---

## About Me

黄威，Randolph

SCU SE Bachelor; USTC CS Master

Email: chinawolfman@hotmail.com

My Blog: [randolph.pro](http://randolph.pro)

LinkedIn: [randolph's linkedin](https://www.linkedin.com/in/randolph-%E9%BB%84%E5%A8%81/)
