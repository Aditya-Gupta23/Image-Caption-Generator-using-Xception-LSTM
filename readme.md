# Image Caption Generator using Xception + LSTM

A Deep Learning project that automatically **generates captions for images** using a combination of **Convolutional Neural Networks (CNN)** and **Recurrent Neural Networks (LSTM)**.

The model extracts image features using **Xception** and generates natural language descriptions using an **LSTM-based caption generator**.

---

# Demo

Example output from the model:

Input Image → Generated Caption

```
man in blue jacket is standing near the water
```

---

# Project Overview

Image captioning is a challenging AI task that combines:

* Computer Vision
* Natural Language Processing
* Deep Learning

This project builds an **end-to-end pipeline** that:

1. Extracts visual features from images
2. Processes captions and builds vocabulary
3. Trains an LSTM model to predict captions
4. Generates captions for unseen images

---

# Model Architecture

The architecture consists of **two parallel models**.

### 1. Image Feature Extractor

* Pretrained **Xception CNN**
* Removes top classification layers
* Produces **2048-dimensional feature vectors**

### 2. Caption Generator

* Embedding Layer
* LSTM (256 units)
* Dense Decoder
* Softmax output layer

The **image features** and **text sequence** are merged to predict the next word.

```
Image → Xception → Feature Vector (2048)

Caption → Tokenizer → Embedding → LSTM

Feature Vector + LSTM Output → Dense Layers → Softmax → Next Word
```

---

# Dataset

This project uses the **Flickr8k dataset**.

Dataset details:

* 8000 images
* Each image has **5 human-written captions**
* Total captions: **40,000**

Dataset folders:

```
Flicker8k_Dataset/   → Images
Flickr8k_text/       → Caption files
```

Dataset link:

https://www.kaggle.com/datasets/adityajn105/flickr8k

---

# Project Structure

```
Image_Captioning
│
├── Flicker8k_Dataset/        # Image dataset (not uploaded to GitHub)
├── Flickr8k_text/            # Caption files
│
├── models/                   # Saved model checkpoints
│
├── descriptions.txt          # Cleaned captions
├── tokenizer.p               # Tokenizer
├── features.p                # Extracted CNN features
│
├── main.py                   # Training pipeline
├── test.py                   # Caption generator
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository

```
git clone https://github.com/YOUR_USERNAME/Image_Captioning.git
cd Image_Captioning
```

Create a virtual environment

```
python -m venv venv
source venv/bin/activate
```

Install dependencies

```
pip install -r requirements.txt
```

---

# Training the Model

Run the training script:

```
python main.py
```

The script will:

* Clean and preprocess captions
* Extract image features using Xception
* Train the LSTM caption generator
* Save trained models in the `models/` directory

---

# Generating Image Captions

To generate a caption for an image:

```
python test.py -i image.jpg
```

Example:

```
python test.py -i dog.jpg
```

Output:

```
Caption: dog running through the grass
```

---

# Key Features

* Pretrained **Xception CNN**
* **Beam Search Caption Generation**
* Custom **Data Generator with TensorFlow Dataset**
* Efficient **Feature Extraction Pipeline**
* Modular and easy to extend

---

# Libraries Used

* TensorFlow / Keras
* NumPy
* Pillow
* tqdm
* Matplotlib

---

# Future Improvements

Possible improvements for this project:

* Attention Mechanism
* Transformer-based Captioning
* Training on **MSCOCO dataset**
* Improving caption grammar
* Deploying as a **web application**

---

# Author

Aditya Gupta

GitHub
https://github.com/Aditya-Gupta23

---

# License

This project is open-source and available for educational purposes.
