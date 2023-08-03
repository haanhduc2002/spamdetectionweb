# Import necessary modules and libraries
from fastapi.testclient import TestClient
from fastapi.templating import Jinja2Templates
from pathlib import Path
import os

# Import the FastAPI application
from mlapi import app

# Create a test client for the FastAPI application
client = TestClient(app)

# Define the base path of the current file
BASE_PATH = Path(__file__).resolve().parent

# Create an instance of Jinja2Templates for template rendering
templates = Jinja2Templates(directory=str(BASE_PATH / "template"))


# Define a test function for checking the response of "/" route
def test_spam_content_form():
    response = client.get("/")
    assert response.status_code == 200


# Define a test function for checking the response of "/" route with a message
def test_spam_identification_with_messsage():
    response = client.post("/", data={"message": "Go juu rong"})
    assert response.status_code == 200
    assert type(response) != None