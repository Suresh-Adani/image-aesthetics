# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel, AutoModel,AutoImageProcessor
import torch.nn as nn
import requests
from io import BytesIO
import os
import pickle
import numpy as np
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)


# In[ ]:


def load_image_PIL(url_or_path):
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return Image.open(requests.get(url_or_path, stream=True).raw)
    else:
        return Image.open(url_or_path)


# In[ ]:


def cosine_similarity(vec1, vec2):
    # Compute the dot product of vec1 and vec2
    dot_product = np.dot(vec1, vec2)
    
    # Compute the L2 norm of vec1 and vec2
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Compute the cosine similarity
    similarity = dot_product / (norm_vec1 * norm_vec2)
    
    return similarity


#import pandas as pd
#temp=pd.read_excel(r"D:\OneDrive - Adani\Desktop\ImageNet Classes.xlsx")
#classes=temp['Col_Names'].tolist()
#classes=[s.lstrip() for s in classes]
#positive_classes=[]
#negative_classes=[]
#for i in range(len(classes)):
    #positive_classes.append(f"a smashing picture, of a #{classes[i]}")
    #negative_classes.append(f"a horrible picture, of a #{classes[i]}")
#positive_inputs=processor(text=positive_classes, return_tensors="pt", padding=True).to(device)
#with torch.no_grad():
    #positive_text_features = model.get_text_features(**positive_inputs)
#negative_inputs=processor(text=negative_classes, return_tensors="pt", padding=True).to(device)
#with torch.no_grad():
    #negative_text_features = model.get_text_features(**negative_inputs)
#import numpy as np

# # Assuming 'prompt_vectors' is a list of your prompt vectors
#positive_prompt_vectors = np.array(positive_text_features)
# 
# # Compute the average vector
#average_positive_vector = np.mean(positive_prompt_vectors, axis=0)
# 
#negative_prompt_vectors = np.array(negative_text_features)
# 
# # Compute the average vector
#average_negative_vector = np.mean(negative_prompt_vectors, axis=0)
# 
#with open('positive_prompt.pkl', 'wb') as f:
    #pickle.dump(average_positive_vector, f)
#with open('negative_prompt.pkl', 'wb') as f:
    #pickle.dump(average_negative_vector, f)

# In[2]:


with open('hotel_positive_prompt.pkl', 'rb') as f:
    average_positive_vector = pickle.load(f)
with open('hotel_negative_prompt.pkl', 'rb') as f:
    average_negative_vector = pickle.load(f)


# In[ ]:


def predict(img_url):
    image1 = load_image_PIL(img_url)
    with torch.no_grad():
        inputs1 = processor(images=image1, return_tensors="pt").to(device)
        image_features1 = model.get_image_features(**inputs1)
    image_vector=image_features1.numpy()
    positive_similarity=cosine_similarity(average_positive_vector,np.transpose(image_vector))
    negative_similarity=cosine_similarity(average_negative_vector,np.transpose(image_vector))
    aesthetic_score=positive_similarity+(-1*negative_similarity)
    return aesthetic_score*1000


# In[ ]:


df=pd.read_pickle(r"C:\Users\suresh.raghu\Downloads\HotelId_ImagesUrl.pkl")


# In[ ]:

def get_score_dict(hotelid):
    score_dict={}
    hotel_df=df[df['HotelId']==hotelid]
    for index, row in hotel_df.iterrows():
        image_url = row['OriginalImageUrl']
        score_dict[image_url]=predict(image_url)
    score_dict = dict(sorted(score_dict.items(), key=lambda item: item[1],reverse=True))
    return score_dict


# In[ ]:


import streamlit as st
import requests
from PIL import Image
from io import BytesIO

st.header('Image Aesthetics Scoring')
hotelid = st.text_input('Enter Hotel ID')

# If hotelid is provided, then process it
if hotelid:
    if df[df['HotelId']==hotelid].shape[0]>0:
        data = get_score_dict(hotelid)
        st.subheader(df[df['HotelId']==hotelid]['hotelName'].iloc[0], divider='rainbow')
        for url, score in data.items():
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))

            # Display the image
            st.image(img, caption=f'Score: {score}', use_column_width=True)
    else:
        st.text("Not a Pareto Hotel")
    )


if __name__ == "__main__":
    run()
