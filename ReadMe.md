
## This is the repository of my thesis titled: 
## *From Satire to Severity: Classifying Hate in Political and Non-Political Memes*

The two experiments involve comparing accuracy and AUROC scores between:   
a) political and non-political memes   
b) targets of hateful political memes.
2 models were used for the experiments:
1. Gemini - https://gemini.google.com/app/
2. CLIP - https://openai.com/index/clip/

that classified the hatefulness of memes from 3 datasets:
1. Hateful Meme challenge (non-political memes) - https://hatefulmemeschallenge.com/
2. MultiOFF (political memes) - https://github.com/bharathichezhiyan/Multimodal-Meme-Classification-Identifying-Offensive-Content-in-Image-and-Text
3. Harm-P (political memes) - https://github.com/LCS2-IIITD/MOMENTA

## Running the experiments
Download the datasets from the links provided above and place them at the root of the project. For each dataset, create a:
- `concatenated.jsonl` file with label entries from the test, val and test spil files. 
- subdirectory named `img` containing the images of memes. For Harm-P dataset, put the `concatenated.jsonl` in the `img` subdirectory
Moreover, in the root of the project, create the results jsonl files: `ClipResults.jsonl` and `GeminiFlashResults.jsonl` for the CLIP and Gemini models respectively.
### Task 1 - political vs non-political memes
For this task, run the `runner.py` it takes the following arguments: 
1. model_type: `GeminiModel` or `CLIPModel`
2. dataset: `HarmP` or `MultiOFF` or `HatefulMemes`
3. (only for GeminiModel) prompt_version:  `short` or `long`
4. (only for GeminiModel) gemini_model_name: `gemini-1.5-flash-latest` was used for the experiments, see https://ai.google.dev/gemini-api/docs/models/gemini#model-variations for more models
5. (only for GeminiModel) api_key: the API key for Gemini, use https://aistudio.google.com/app/apikey to get one

#### Example usage (CLI at the root of the project):
```
python runner.py GeminiModel MultiOFF short gemini-1.5-flash-latest <api_key>
```
```
python runner.py CLIPModel HatefulMemes
```
Use the `checkGeminiResults.py` and `checkClipResults.py` for the accuracy and AUROC scores of the models respectively.

### Task 2 - targets of hateful political memes
Use the `ErrorAnalysis.py` to get the accuracy scores of different targets from the Harm-P dataset. 
At line 48, give the results file corresponding to the model used. Example:
```results_file = 'ClipResults.jsonl'```. This script also provides 16 random examples (4 per target class) of memes that were misclassified by the model.