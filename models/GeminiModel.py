import google.generativeai as genai
import os
import PIL.Image
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import csv
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.generativeai import GenerationConfig

config = GenerationConfig(temperature=0)
instructions = "You are a binary classifier. Determine if the meme is hateful or not, based on the image and text. Answer ONLY with 0 for non-hateful and 1 for hateful."


class GeminiModel:
    def __init__(self, model_name, prompt_version, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name, generation_config=config, system_instruction=instructions)
        #self.model.temperature = 0
        self.definition = "A direct or indirect attack on people based on characteristics, including ethnicity, race, nationality, immigration status, religion, caste, sex, gender identity, sexual orientation, and disability or disease."
        self.prompt_short = 'Determine if the meme is hateful or not. Answer ONLY with 0 for non-hateful and 1 for hateful. Anwer: '
        self.prompt_long = ('According to this definition of hate speech: ' + self.definition + ' ' +
                            'Determine if the meme is hateful or not. Answer ONLY with 0 for non-hateful and 1 for hateful. Anwer: ')
        self.prompt_version = prompt_version

    def predict(self, image, text):
        # Choose the prompt based on the prompt_version attribute
        prompt = self.prompt_short if self.prompt_version == 'short' else self.prompt_long

        try:
            # Generate a prediction for the given image and text
            response = self.model.generate_content([prompt, image, text], safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                #HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE
            })

            # Check if the response is empty
            if not response.candidates:
                print(response.prompt_feedback)
                print("Failed to get a response from the model. Img: ", image, " Text: ", text)
                #print(response.finish_reason)
                #return 1 if the respnse has a prompt issue - its been observed that these are usually hateful
                return 1

        except Exception as e:
            print("Failed to get a response from the model. Img: ", image, " Text: ", text)
            print(e)
            return None

        return response.text
