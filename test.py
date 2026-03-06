from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import argparse
from tensorflow.keras.layers import Input,Dense,LSTM,Embedding,Dropout,Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


ap=argparse.ArgumentParser()
ap.add_argument('-i','--image',required=True,help='Image')
args=vars(ap.parse_args())
img_path=args['image']

def extract_features(filename,model):
    try:
        # image=Image.open(filename)
        image=Image.open(filename).convert("RGB")
    except:
        print("Error")
    image=image.resize((299,299))
    image=np.array(image)
    if image.shape[2]==4:
        image=image[...,:3]
    image=np.expand_dims(image,axis=0)
    image=image/127.5
    image=image-1.0
    feature=model.predict(image)
    return feature

def word_for_id(integer,tokenizer):
    for word,index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length, beam_index=3):

    start = tokenizer.texts_to_sequences(['start'])[0]
    start_word = [[start, 0.0]]

    while len(start_word[0][0]) < max_length:
        temp = []

        for s in start_word:
            sequence = pad_sequences([s[0]], maxlen=max_length, padding='post')
            preds = model.predict([photo, sequence], verbose=0)[0]

            word_preds = np.argsort(preds)[-beam_index:]

            for w in word_preds:
                next_seq = s[0] + [w]
                next_score = s[1] + preds[w]
                temp.append([next_seq, next_score])

        start_word = sorted(temp, key=lambda x: x[1], reverse=True)
        start_word = start_word[:beam_index]

    best_seq = start_word[0][0]

    caption = []
    for i in best_seq:
        word = word_for_id(i, tokenizer)
        if word == 'end':
            break
        caption.append(word)

    caption = caption[1:]  # remove start
    return ' '.join(caption)


def define_model(vocab_size,max_len):
    #CNN from 2048 nodes to 256 nodes
    inputs1=Input(shape=(2048,),name='input_1')
    fe1=Dropout(0.5)(inputs1)
    fe2=Dense(256,activation='relu')(fe1)
    #LSTM sequence model
    inputs2=Input(shape=(max_len,),name='input_2')
    se1=Embedding(vocab_size,256,mask_zero=True)(inputs2)
    se2=Dropout(0.5)(se1)
    se3=LSTM(256)(se2)

    decoder1=Add()([fe2,se3])
    decoder2=Dense(256,activation='relu')(decoder1)
    outputs=Dense(vocab_size,activation='softmax')(decoder2)
    model=Model(inputs=[inputs1,inputs2],outputs=outputs)

    model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.0001))
    print(model.summary())
    return model

max_length=32
tokenizer=load(open("tokenizer.p","rb"))
vocab_size=len(tokenizer.word_index)+1

model=define_model(vocab_size,max_length)
model.load_weights("models/model_39.h5")
xception_model=Xception(include_top=False, pooling="avg")
photo=extract_features(img_path,xception_model)
img=Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length,3)

description = description.replace('<start>', '').replace('end', '')
print("Caption:", description.strip())