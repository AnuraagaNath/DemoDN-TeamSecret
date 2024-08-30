
# [TeamSecret] DataNexus Hackathon Event


## Demo Video
https://github.com/user-attachments/assets/b8f9af6e-a8c5-422b-b2d1-355790c070d1

The team worked on the topic "Hushh - Receipt Radar". 

The task is progressed on the following path.

1. The data is collected from Huggingface: https://huggingface.co/datasets/mychen76/invoices-and-receipts_ocr_v1

2. The images are considered for the OCR framework to extract text. Here pytesseract package is used for OCR. 

3. The parsed text present in the collected data was sent through Gemini 1.5 Pro LLM model to automate the parsed text to JSON string.

4. Finally, a dataset is being generated on the OCR text as input and the JSON string as output.

Two LLM models are then finetuned for testing purpose.

1. **Google Gemini 1.0 pro**: Finetuned with Gemini API with OAuth 2.0 access key. This training could be done both using Python and being in Google AI Studio.
We have used Google AI studio for a easy visualization of the Model Finetuning.

```python
import google.generativeai as genai

model_name = 'tunedModels/geminidn1-dsg5fdi8yc25'  

model = genai.GenerativeModel(model_name=model_name)
input_text = <OCR TEXT HERE>
response = model.generate_content(input_text)
```


2. **Gemma 2b Intruct**: Google's open source LLM model (https://huggingface.co/google/gemma-2b-it) is downloaded from huggingface and finetuned on the data. We have used Google Colab free GPU access to finetune this model. 
Our finetuned Gemma 2b model is available in available at https://huggingface.co/anuraaga/gemma-dn3. 

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "anuraaga/Gemma-dn3"

model = AutoModelForCausalLM.from_pretrained(model_id,  device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)
```







## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`GEMINI_API_KEY`

`HUGGING-FACE TOKEN`


## Tech Stack

**Packages:** pytesseract, torch, transformers, datasets, peft, trl

**Framework:** PyTorch

## Development Team
- [Anuraaga Nath](https://github.com/AnuraagaNath)
- [Sanghamitra Majumder](https://github.com/Sangit-G)

## Documentation Team
- [Snigdha Chandra](https://github.com/snigschan/snigschan) 
- [Subarna Paul](https://github.com/Sub-277)

**Together we are Team Secret!**

## Acknowledgements

We are extremely fortunate to get this opportunity to work on Hushh products and explore them. Kuddos to the entire Team of Hushh and the way they cooperated with all of us during this journey. It was really memorable and we Hope to work more with them and explore more beautiful products.

Thank you [Hushh.ai](https://www.hush1one.com/) !
