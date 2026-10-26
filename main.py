from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from keras.models import load_model
import numpy as np
import pickle
import re
import traceback

"""
1. We are going to make some constants like:
A. Model Path (BiGRU)
B. Tokenizer Path
C. Max Sequence Length
D. Emotion Labels
E. Emotion emojis
"""
model_path='Artifacts/BiGRU_Modle.keras'
tokenizer_path="Artifacts/tokenizer.pkl"
max_sequence_length=50
emotion_labels=["sadness", "joy", "love", "anger", "fear", "surprise"]
emotion_emojis={
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}

"""
2. Preprocess the upcoming text
Cleans raw text so it matches the format used while training.
A. Convert the text to lowercase. -done
B. Remove apostrophes (e.g can't -> cant). -done
C. Remove Special Characters and Punctuation. -done
D. Remove extra spaces -done
"""

def preprocessing_text(text:str)->str:
    text=text.lower()
    text=re.sub(r"'","",text)
    text=re.sub(r"[^a-z0-9\s]"," ",text)
    text=re.sub(r"\s+"," ",text).strip()
    return text

"""
3. Request and Response Schemas
A. Text Input -> Input schema the text sent by user. -done
B. Prediciton Response -> Output schema the emotion to predict. -done
C. Health Response (Server health check)
"""

class TextInput(BaseModel):
    text:str=Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyze",
        json_schema_extra={"example": "I feel so happy and excited"}
    )

class PredictionResponse(BaseModel):
    text:str
    predicted_emotion:str
    confidence:float
    all_probablities: dict[str,float]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

"""
4.model loading and lifespan checking
load model and tokenizer once the server is started
"""
dl_model={}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading the Model and tokenizer....")
    dl_model["BiGRU"]=load_model(model_path)
    with open(tokenizer_path, 'rb')as file:
        dl_model["Tokenizer"]=pickle.load(file)
    print('Model are loaded successfully...')

    yield #Pause, model is laoded and server is running and at this point model wait karega for request

    dl_model.clear()#Ek baar server band ho gaya uske baad model ko memory se hata do.

"""
5. Mount the static files to the FastAPI app
A. Enable CORS (Cross-Origin Resource Sharing) to allow requests from different origins.
"""

app=FastAPI(
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
    allow_credentials=True
)
app.mount('/static', StaticFiles(directory="static"), name="static")

"""
6.API Endpoint
A.Server Ui Homepage(/)
B.Health Check ENdpoint
C.Prediction Endpoint
"""

@app.get('/',include_in_schema=False)
def server_ui():
    return FileResponse('static/index.html')

@app.get('/health', response_model=HealthResponse)
def heath_check():
    return HealthResponse(status="Server is running", model_loaded=bool(dl_model))


import traceback

@app.post("/predict", response_model=PredictionResponse)
def predict_emotion(text_input: TextInput):
    try:
        print("1. Prediction request received")

        BiGRU_model = dl_model.get("BiGRU")
        Tokenizer_model = dl_model.get("Tokenizer")

        if BiGRU_model is None or Tokenizer_model is None:
            raise HTTPException(
                status_code=503,
                detail="Model or tokenizer is not loaded"
            )

        print("2. Model and tokenizer found")

        cleaned_text = preprocessing_text(text_input.text)
        print("3. Cleaned text:", cleaned_text)

        tokenized_text = Tokenizer_model.texts_to_sequences([cleaned_text])
        print("4. Tokenized text:", tokenized_text)

        padded_sequence = pad_sequences(
            tokenized_text,
            maxlen=max_sequence_length,
            padding="post",
            truncating="post"
        )
        print("5. Padded shape:", padded_sequence.shape)

        probabilities = BiGRU_model.predict(
            padded_sequence, verbose=0
        )[0]
        print("6. Model output:", probabilities)
        print("7. Output shape:", probabilities.shape)

        if len(probabilities) != len(emotion_labels):
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Expected {len(emotion_labels)} predictions, "
                    f"but model returned {len(probabilities)}"
                )
            )

        top_emotion_index = int(np.argmax(probabilities))

        all_probablities = {
            label: float(prob)
            for label, prob in zip(emotion_labels, probabilities)
        }

        result = PredictionResponse(
            text=text_input.text,
            predicted_emotion=emotion_labels[top_emotion_index],
            confidence=float(probabilities[top_emotion_index]),
            all_probablities=all_probablities
        )

        print("8. Prediction successful")
        return result

    except HTTPException:
        raise
    except Exception:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the Uvicorn terminal for the traceback."
        )
