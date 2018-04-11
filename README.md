# Deep Learning for Multi-Label Text Classification

This project is my research group project, and it is also a study of TensorFlow, Deep Learning(Fasttext, CNN, LSTM, RCNN, etc.).

The main objective of the project is to solve the multi-label text classification problem based on Convolutional Neural Networks. Thus, the format of the data label is like [0, 1, 0, ..., 1, 1] according to the characteristics of such problem.

## Requirements

- Python 3.6
- Tensorflow 1.7
- Numpy
- Gensim

## Data

Research data may attract copyright protection under China law. Thus, there is only code.

实验数据属于实验室与某公司的合作项目，涉及商业机密，在此不予提供，还望谅解。

## Innovation

### Data part
1. Make the data support **Chinese** and English.(Which use `jieba` seems easy)
2. Can use **your own pre-trained word vectors**.(Which use `gensim` seems easy)
3. Add embedding visualization based on the **tensorboard**.

### Model part
1. Deign **two subnetworks** to solve the task --- Text Pairs Similarity Classification.
2. Add a new **Highway Layer**.(Which is useful based on the performance)
3. Add **parent label bind** information to limit the prediction to improve the model.

### Code part
1. Can choose to **train** the model directly or **restore** the model from checkpoint in `train_cnn.py`.  
2. Add `test_cnn.py`, the **model test code**. 
3. Add other useful data preprocess functions in `data_helpers.py`.
4. Use `logging` for helping recording the whole info(including parameters display, model training info, etc.).

## Data Preprocessing

Depends on what your data and task are.

### Text Segment

You can use `jieba` package if you are going to deal with the chinese text data.

### Pre-trained Word Vectors

- Use `gensim` package to pre-train data.
- Use `glove` tools to pre-train data.
- Even can use a **fasttext** network to pre-train data.

## Network Structure

### FastText

![]()

References:

- [Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf)

---

### TextANN

![]()

---


### TextCNN

![]()

References:

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)

---

### TextRNN

**Warning: Not finished yet 🤪!**

![]()

References:

- [Recurrent Neural Network for Text Classification with Multi-Task Learning](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552)

---

### TextRCNN

**Warning: Not finished yet 🤪!**

![]()

References:

- [Recurrent Convolutional Neural Networks for Text Classification](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552)

---

### TextHAN

**Warning: Not finished yet 🤪!**

![]()

References:

- [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)

---

## About Me

黄威，Randolph

SCU SE Bachelor; USTC CS Master

Email: chinawolfman@hotmail.com

My Blog: [randolph.pro](http://randolph.pro)

LinkedIn: [randolph's linkedin](https://www.linkedin.com/in/randolph-%E9%BB%84%E5%A8%81/)
