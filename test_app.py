import os
from fastapi.testclient import TestClient
from .app.server import app

os.chdir('app')
client = TestClient(app)

""" 
This part is optional. 

We've built our web application, and containerized it with Docker.
But imagine a team of ML engineers and scientists that needs to maintain, improve and scale this service over time. 
It would be nice to write some tests to ensure we don't regress! 

  1. `Pytest` is a popular testing framework for Python. If you haven't used it before, take a look at https://docs.pytest.org/en/7.1.x/getting-started.html to get started and familiarize yourself with this library.
   
  2. How do we test FastAPI applications with Pytest? Glad you asked, here's two resources to help you get started:
    (i) Introduction to testing FastAPI: https://fastapi.tiangolo.com/tutorial/testing/
    (ii) Testing FastAPI with startup and shutdown events: https://fastapi.tiangolo.com/advanced/testing-events/

"""

def test_root():
    """
    Test the root ("/") endpoint, which just returns a {"Hello": "World"} json response
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}


def test_predict_empty():
    """
    Test the "/predict" endpoint, with an empty request body
    """
    response = client.post("/predict")
    assert response.status_code == 422


def test_predict_en_lang():
    """
    Test the "/predict" endpoint, with an input text in English (you can use one of the test cases provided in README.md)
    """
    input={
        "source": "New York Times",
        "url": "",
        "title": "Weis chooses not to make pickoff",
        "description": "Bill Belichick won't have to worry about Charlie Weis raiding his coaching staff for Notre Dame. But we'll have to see whether new Miami Dolphins coach Nick Saban has an eye on any of his former assistants."
    }
    response = client.post("/predict", json=input)
    assert response.status_code == 200
    assert response.json()["label"] == 'Entertainment'
    


def test_predict_es_lang():
    """
    Test the "/predict" endpoint, with an input text in Spanish. 
    Does the tokenizer and classifier handle this case correctly? Does it return an error?
    """
    input={
        "source": "El País",
        "url": "",
        "title": "Weis elige no hacer pickoff",
        "description": "Bill Belichick no tendrá que preocuparse de que Charlie Weis saquee a su cuerpo técnico para Notre Dame. Pero tendremos que ver si el nuevo entrenador de los Miami Dolphins, Nick Saban, tiene el ojo puesto en alguno de sus antiguos asistentes."
    }
    response = client.post("/predict", json=input)
    assert response.status_code == 200
    assert response.json()["label"] == 'Entertainment'


def test_predict_non_ascii():
    """
    [TO BE IMPLEMENTED]
    Test the "/predict" endpoint, with an input text that has non-ASCII characters. 
    Does the tokenizer and classifier handle this case correctly? Does it return an error?
    """
    input={
        "source": "Reuters World",
        "url": "http://www.reuters.com/newsArticle.jhtml?type=worldNewsstoryID=7228962",
        "title": "Peru Arrests Siege Leader, to Storm Police Post",
        "description": "LIMA, ® Peru vödör (Reuters) - Peruvian authorities arrested a former army major who led a three-day uprising in a southern  Andean town and will storm the police station where some of his  200 supporters remain unless they surrender soon, Prime  Minister Carlos Ferrero said on Tuesday."
    }
    response = client.post("/predict", json=input)
    assert response.status_code == 200
    assert response.json()["label"] == 'Sports'