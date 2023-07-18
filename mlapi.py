from fastapi import FastAPI, APIRouter, HTTPException, Request, Form, Response
from fastapi.templating import Jinja2Templates
from pathlib import Path
from typing import Annotated
from fileinput import filename
import pickle
import nltk
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
BASE_PATH = Path(__file__).resolve().parent
templates = Jinja2Templates(directory= str(BASE_PATH / "template"))

with open('model\saved_model.pkl', "rb") as f:
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
def read_root():
    return 'hello world' 

@app.get('/form')
async def hello(request: Request):
    return templates.TemplateResponse('index.html',context = {'request': request,'content': "Please enter your email content", 'result': " "},)

@app.post('/form')
async def spam_identification(request: Request ,message: str = Form(...)):
    answer = ''
    try:
        handled_message = await handle_message(message)
        array = [handled_message]
        input = cv_import.transform(array)
        myprediction = model_import.predict(input)
        myprediction = model_import.predict(input)
        answer = myprediction[0]
    except Exception as e:
        answer = "Something happened to the sever. Please try again"
    return  templates.TemplateResponse('index.html',context = {'request': request,'content': message, 'result': answer},)