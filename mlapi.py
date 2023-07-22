from fastapi import FastAPI, APIRouter, HTTPException, Request, Form, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Annotated, Optional
from fileinput import filename
import pickle
import nltk
import os
import uvicorn
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn import svm
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
BASE_PATH = Path(__file__).resolve().parent
templates = Jinja2Templates(directory= str(BASE_PATH / "template"))

path = os.path.join('model', 'saved_model.pkl')
with open(path, "rb") as f:
    model_import, cv_import = pickle.load(f)

async def handle_message(message): 
  # Step 1: Remove puntuation 
  message1 = [char for char in message if char not in string.punctuation]
  message1 = ''.join(message1)
  # Step 2: Remove stop word
  message2 = [word for word in message1.split() if word.lower() not in stopwords.words('english')]
  message2 = ' '.join(message2)
  # Step 3: Lemmatization
  lemmatizer = WordNetLemmatizer()
  word_list = nltk.word_tokenize(message2)
  result = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
  return result.lower() # Lower casing the processed message


@app.get('/')
async def spam_content_form(request: Request):
    return templates.TemplateResponse('index.html',context = {'request': request,'content': "Please enter your email content", 'result': " "},)

@app.post('/')
async def spam_identification(request: Request ,message: Optional[str] = Form(None)):
    answer = ''
    try:
        handled_message = await handle_message(message)
        array = [handled_message]
        input = cv_import.transform(array)
        myprediction = model_import.predict(input)
        myprediction = model_import.predict(input)
        answer = myprediction[0]
    except Exception as e:
        if message == None:
            answer = "Please enter your content"
            message = " "
        else:
            answer = "Something happened to the sever. Please try again"
            message = " "
    return templates.TemplateResponse('index.html',context = {'request': request,'content': message, 'result': answer},)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv('APP_PORT')))