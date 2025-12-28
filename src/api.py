from fastapi import FastAPI, HTTPException
from schemas import TextInput, APIResponse 
from predictor import LocalModel           

class ApiClassification:
    def __init__(self, predictorBertInstance: LocalModel, predictorNcInstance: LocalModel):
        # Rebem el model JA carregat (Dependency Injection)
        self.predictorBert = predictorBertInstance
        self.predictorNC = predictorNcInstance
        self.app = FastAPI(title="NLP Classification API", version="1.0")
        self.setup_routes()

    def setup_routes(self):
        @self.app.get("/")
        def root():
            return {"status": "online"}

        @self.app.post("/predictNearestClassificator", response_model=APIResponse)
        def predict_nearest(payload: TextInput):
            if self.predictorNC is None:
                raise HTTPException(status_code=500, detail="Nearest Classificator Model not exist")
            try:
                results = self.predictorNC.predictNearestClassificator(payload.text)
                return {
                    "text": payload.text,
                    "predictions": results
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/predictBertClassificator", response_model=APIResponse)
        def predict_bert(payload: TextInput):
            if self.predictorBert is None:
                raise HTTPException(status_code=500, detail="Bert Classificator Model not exist")
            try:
                results = self.predictorBert.predictBert(payload.text)
                return {
                    "text": payload.text,
                    "predictions": results
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))