import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import  time
from zipfile import ZipFile
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import numpy as np
from glob import glob
import pytesseract
from PIL import Image


model_id = "anuraaga/Gemma-dn3"

if 'model' not in st.session_state:
    st.session_state['model'] = AutoModelForCausalLM.from_pretrained(model_id)
    st.session_state['tokenizer'] = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)


st.title('Team Secret - Receipt Radar')

def get_completion(query: str, model, tokenizer) -> str:
    # device = "cuda:0"

    prompt_template = """
    <start_of_turn>user
    Below is an OCR text that describes a invoice OCR scan. Write a json response that ' \
               'appropriately completes the request.\n\n
    {query}
    <end_of_turn>\n<start_of_turn>model \n\n\n\n


    """
    prompt = prompt_template.format(query=query)

    encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

    model_inputs = encodeds


    generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    # decoded = tokenizer.batch_decode(generated_ids)
    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return (decoded)



uploaded_zip = st.file_uploader('Zip file', type="zip")
model = st.session_state['model']
tokenizer = st.session_state['tokenizer']
def tess_ocr(input_image):
    text = ''
    rgb_image = input_image.convert('RGB')
    text = pytesseract.image_to_string(rgb_image)

    return text

if st.button('Submit'):
    with ZipFile(uploaded_zip, 'r') as zObject: 
        start_time = time.time()
        zObject.extractall(path="unzipped")
        folder = 'unzipped'
        files = glob('unzipped/*')
        for file in files:
            model_layer = keras.layers.TFSMLayer('./CNN1', call_endpoint='serving_default')
            input_shape = (100, 100, 3)  
            inputs = keras.Input(shape=input_shape)
            outputs = model_layer(inputs)
            model = keras.Model(inputs, outputs)

            # Compile the model if needed
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

            img_path = file
            img = image.load_img(img_path, target_size=(100, 100))  
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array * (1.0 / 255)  
            predictions = model.predict(img_array)
            predicted_class = np.round(predictions['output_1'][0]).astype(int)
            if predicted_class[0] == 0:
                invoice_image = Image.open(file)
                query = tess_ocr(invoice_image)
                
                
                response = get_completion(query, model, tokenizer)
                end_time = time.time()
                st.text(f'Time Taken {(end_time-start_time)/60} minutes')
                st.text(response.split('model \n\n\n\n')[-1])


            


