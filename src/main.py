from pathlib import Path
import sys
import config
import uvicorn
from utils import LoaderLabels
from predictor import LocalModel
from api import ApiClassification
import os

def create_app():
    """
    Crea i retorna la instància de FastAPI amb els models carregats.
    Aquesta funció s'executa tant en local com a Azure quan s'importa main.py.
    """
    print("--- STARTUP PROCESS ---")

    try:
        # 2.2 Carregar el Nearest Classificator
        pathNC = getRuta(config.NEAREST_CLASSIFICATOR_C_PATH)
        nlp_nc_predictor = LocalModel(bertPath=None, NCPath=str(pathNC))
        nlp_nc_predictor.loadNearestClassificator()

        
        loader = LoaderLabels(config.BERT_CATALAN_LABEL_PATH)
        labels_map = loader.load_json_labels()
        
        print(os.listdir(config.BERT_C_MODEL_PATH))
        nlp_bert_predictor = LocalModel(bertPath=config.BERT_C_MODEL_PATH, NCPath=None)
        nlp_bert_predictor.loadBert(manual_labels=labels_map)

        # De moment només NC:
        nlp_api = ApiClassification(
            predictorBertInstance=nlp_bert_predictor,
            predictorNcInstance=nlp_nc_predictor
        )

        return nlp_api.app

    except Exception as e:
        # Si falla l'startup, ho mostrem per consola i tornem a llençar l'excepció
        print(f"ERROR FATAL EN STARTUP: {e}", file=sys.stderr)
        raise

def getRuta(path: str) -> Path:
    dirActual = Path(__file__).resolve()
    dirRoot = dirActual.parent
    return dirRoot / "topics" / "catalan" / "catalanTopics.json"


app = create_app()

if __name__ == "__main__":
    uvicorn.run("main:app", host=config.HOST, port=config.PORT)