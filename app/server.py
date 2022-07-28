from fastapi import Depends, FastAPI
from pydantic import BaseModel
from loguru import logger
import joblib
from datetime import datetime
import time
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

GLOBAL_CONFIG = {
    "model": {
        "featurizer": {
            "sentence_transformer_model": "all-mpnet-base-v2",
            "sentence_transformer_embedding_dim": 768
        },
        "classifier": {
            "serialized_model_path": "../data/news_classifier.joblib"
        }
    },
    "service": {
        "log_destination": "../data/logs.out"
    }
}

class PredictRequest(BaseModel):
    source: str
    url: str
    title: str
    description: str


class PredictResponse(BaseModel):
    scores: dict
    label: str


class TransformerFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, dim, sentence_transformer_model):
        self.dim = dim
        self.sentence_transformer_model = sentence_transformer_model

    #estimator. Since we don't have to learn anything in the featurizer, this is a no-op
    def fit(self, X, y=None):
        return self

    #transformation: return the encoding of the document as returned by the transformer model
    def transform(self, X, y=None):
        X_t = []
        for doc in X:
            X_t.append(self.sentence_transformer_model.encode(doc))
        return X_t


class NewsCategoryClassifier:
    def __init__(self, config: dict) -> None:
        self.config = config
        """
        1. Load the sentence transformer model and initialize the `featurizer` of type `TransformerFeaturizer` (Hint: revisit Week 1 Step 4)
        2. Load the serialized model as defined in GLOBAL_CONFIG['model'] into memory and initialize `model`

        # Initialize the pretrained transformer model
        sentence_transformer_model = SentenceTransformer(
            'sentence-transformers/{model}'.format(model=SENTENCE_TRANSFORMER_MODEL))

        # Sanity check
        example_encoding = sentence_transformer_model.encode(
            "This is an example sentence",
            normalize_embeddings=True
            )

        """
        SENTENCE_TRANSFORMER_MODEL = config["featurizer"]["sentence_transformer_model"]
        dim = config["featurizer"]["sentence_transformer_embedding_dim"]

        # Loading the sentence transformer model
        sentence_transformer_model = SentenceTransformer(f'sentence-transformers/{SENTENCE_TRANSFORMER_MODEL}')
        
        # Initialize the featurizer of type TransformerFeaturizer
        featurizer = TransformerFeaturizer(dim=dim, sentence_transformer_model=sentence_transformer_model)

        # Load the serialized model into memory and initialize the model
        model_path = config["classifier"]["serialized_model_path"]
        model = joblib.load(model_path)

        self.pipeline = Pipeline([
            ('transformer_featurizer', featurizer),
            ('classifier', model)
        ])

    def predict_proba(self, model_input: dict) -> dict:
        """
        Using the `self.pipeline` constructed during initialization, 
        run model inference on a given model input, and return the 
        model prediction probability scores across all labels

        Output format: 
        {
            "label_1": model_score_label_1,
            "label_2": model_score_label_2 
            ...
        }
        """
        # Run model inference on model_input
        Y_pred_proba = self.pipeline.predict_proba(np.array([model_input]))

        # Make output dict
        output = {l: p for l, p in zip(self.pipeline.classes_, Y_pred_proba[0])}

        # return the model prediction probability scores across all labels
        return output

    def predict_label(self, model_input: dict) -> str:
        """
        Using the `self.pipeline` constructed during initialization,
        run model inference on a given model input, and return the
        model prediction label

        Output format: predicted label for the model input
        """
        Y_pred = self.pipeline.predict(model_input)
        return Y_pred


app = FastAPI()

@app.on_event("startup")
def startup_event():
    """
        2. Initialize the `NewsCategoryClassifier` instance to make predictions online. You should pass any relevant config parameters from `GLOBAL_CONFIG` 
        that are needed by NewsCategoryClassifier 
        3. Open an output file to write logs, at the destimation specififed by GLOBAL_CONFIG['service']['log_destination']
        
        Access to the model instance and log file will be needed in /predict endpoint, make sure you
        store them as global variables

    """

    logger.remove()
    log_path = GLOBAL_CONFIG["service"]["log_destination"]
    logger.add(log_path, format="{message}")


    newsclassifier = NewsCategoryClassifier(GLOBAL_CONFIG["model"])
    logger.info("Setup completed")

    return newsclassifier


@app.on_event("shutdown")
def shutdown_event():
    # clean up
    """
        1. Make sure to flush the log file and close any file pointers to avoid corruption
        2. Any other cleanups
    """
    logger.info("Shutting down application")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, newsclassifier: NewsCategoryClassifier = Depends(startup_event)):
    # get model prediction for the input request
    # construct the data to be logged
    # construct response
    """
        1. run model inference and get model predictions for model inputs specified in `request`
        2. Log the following data to the log file (the data should be logged to the file that was opened in `startup_event`, and writes to the path defined in GLOBAL_CONFIG['service']['log_destination'])
        {
            'timestamp': <YYYY:MM:DD HH:MM:SS> format, when the request was received,
            'request': dictionary representation of the input request,
            'prediction': dictionary representation of the response,
            'latency': time it took to serve the request, in millisec
        }
        3. Construct an instance of `PredictResponse` and return
    """

    # Current date and time
    now = datetime.now() 

    # Start time
    st = time.time()

    # Run model inference and get predictions
    result = newsclassifier.predict_label(request.description)
    scores = newsclassifier.predict_proba(request.description)


    # Get the end time, then get the execution time  
    et = time.time()    
    elapsed_time = et - st

    # Response object
    response = {"scores": scores, "label": result[0]}

    # Log results to file
    log_output = {
        'timestamp': now.strftime("%Y:%m:%d, %H:%M:%S"),
        'request': request,
        'prediction': response, 
        'latency': elapsed_time,
    }
    logger.info(log_output)

    return response



@app.get("/")
def read_root():
    return {"Hello": "World"}
