from fastapi.testclient import TestClient
from fastapi.templating import Jinja2Templates
from pathlib import Path
import os

from mlapi import app

client = TestClient(app)

BASE_PATH = Path(__file__).resolve().parent
templates = Jinja2Templates(directory= str(BASE_PATH / "template"))


def test_spam_content_form():
    response = client.get("/")
    assert response.status_code == 200

def test_spam_identification_with_messsage():
    response = client.post("/", data = {"message": "Go juu rong"})
    assert response.status_code == 200
    assert type(response) != None
