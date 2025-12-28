import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from typing import Dict, List
import numpy as np

class LocalModel:
    def __init__(self, bertPath: str, NCPath: str):
        self.BertPath = bertPath
        self.NCPath = NCPath
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.id2label = None

    @staticmethod
    def _L2Norm(X: np.ndarray) -> np.ndarray:
        """
        Normalitza vectors (Mètode estàtic auxiliar).
        Divideix cada vector per la seva magnitud per obtenir norma=1.
        """
        Den = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        return X / Den

    def loadBert(self, manual_labels: Dict[int, str] = None):
        """Carrega el model i configura les etiquetes amb prioritat al JSON manual."""
        print(f" Carregant BERT des de: {self.BertPath}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.BertPath, local_files_only =True)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.BertPath, local_files_only =True)
            self.model.to(self.device)
            self.model.eval()
            
            if manual_labels is not None and len(manual_labels) > 0:
                self.id2label = manual_labels
                print(" Etiquetes carregades manualment des del JSON (Prioritat Alta).")
                
            elif hasattr(self.model.config, 'id2label') and self.model.config.id2label:
                self.id2label = self.model.config.id2label
                print("ℹ Usant etiquetes internes de la config del model.")
            
            else:
                self.id2label = {i: f"CLASS_{i}" for i in range(self.model.config.num_labels)}
                print(" Generant etiquetes genèriques automàtiques.")
            
            print(f" Model carregat. Exemples d'etiquetes: {list(self.id2label.items())[:3]}")
            
        except Exception as e:
            raise RuntimeError(f"Error carregant model: {e}")

    def predictBert(self, text: str, threshold: float = 0.5) -> List[Dict]:
        if self.BertPath is None:
            return ["Bert instance not configured"]
        if not text: return []
        
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()
        
        results = []
        for idx, score in enumerate(probs):
            if score > threshold:
                # buscar l'índex al diccionari
                label_name = self.id2label.get(idx, f"LABEL_{idx}") 
                results.append({
                    "label": label_name,
                    "score": float(score)
                })
        return results
    

    def loadNearestClassificator(self):
        """Carrega els topics."""
        if self.NCPath is None:
            raise Exception("Topic.json path is incorrect")

        self._LoadConfig()
        self._LoadModel()
        self._PrepareTopics()


    def _LoadConfig(self):
        """Llegeix el fitxer de configuració JSON."""
        try:
            with open(self.NCPath, "r", encoding="utf-8") as F:
                self.Cfg = json.load(F)
        except FileNotFoundError:
            raise FileNotFoundError(f"No s'ha trobat l'arxiu {self.NCPath}")
        
    def _LoadModel(self):
        """Carrega el model SentenceTransformer especificat al JSON."""
        ModelName = self.Cfg.get("model_name")
        print(f"Carregant model: {ModelName}...")
        self.Model = SentenceTransformer(ModelName)
        
        # Determinar dimensió dels embeddings
        self.Dim = self.Model.get_sentence_embedding_dimension()
        if not self.Dim:
            self.Dim = self.Cfg.get("dim", 384) # Dimensió del model paraphrase-multilingual-MiniLM-L12-v2

    def _PrepareTopics(self):
        """
        Pre-calcula la matriu de centroides per a una inferència ràpida.
        Converteix les llistes del JSON a matrius NumPy optimitzades.
        """
        self.Topics = []
        self.TopicNames = []
        
        RawTopics = self.Cfg.get("topics", [])
        print(f"Processant {len(RawTopics)} temes...")

        for T in RawTopics:
            # Recuperem els vectors prototipus (centroides) del JSON
            RawProtos = [np.array(P["vector"], dtype=np.float32) for P in T["prototypes"]]
            
            # Validem que tinguin la dimensió correcta
            ValidProtos = [V for V in RawProtos if V.shape[0] == self.Dim]
            
            if not ValidProtos:
                continue

            # Crear matriu per tema
            P = np.stack(ValidProtos)
            P = self._L2Norm(P) # Normalitzacií per assegurar consistència amb Cosine Similarity
            
            # Guardar l'estructura optimitzada en memòria
            self.Topics.append({
                "name": T["name"],
                "thr": float(T.get("threshold", 0.5)),
                "protos": P
            })
            self.TopicNames.append(T["name"])
            
        print(f"Classificador llest amb {len(self.Topics)} temes actius.")
    
    def predictNearestClassificator(self, text: str):
        if self.NCPath is None:
            return ["Bert instance not configured"]
        if not text: return []
        
        Sentences = [text] 

        # 1. Encoding Vectoritzat (Batch)
        # Genera embeddings per a totes les frases d'entrada de cop
        Embeddings = self.Model.encode(
            Sentences,
            convert_to_numpy=True, 
            show_progress_bar=False
        )
        Embeddings = self._L2Norm(Embeddings)

        Results = []

        # 2. Inferencia Matricial
        for I, Emb in enumerate(Embeddings):
            # Expandir dimensio per operacions matricials (1, dim)
            Emb = Emb.reshape(1, -1)
            
            RowRes = {"assigned": [], "scores": []}
            
            for T in self.Topics:
                # Similitud Cosinus: 
                # Multiplicar el vector frase (Emb) contra la matriu de prototipus (T["protos"])
                
                SimScores = np.max(Emb @ T["protos"].T)
                ScoreVal = float(SimScores)
                
                # Guardar Score informatiu
                RowRes["scores"].append({"topic": T["name"], "score": round(ScoreVal, 4)})
                
                # Decisió basada en Threshold (Llindar)
                if ScoreVal >= T["thr"]:
                    RowRes["assigned"].append((T["name"], ScoreVal))
            
            # Ordenar resultats per puntuació descendent
            RowRes["assigned"].sort(key=lambda x: x[1], reverse=True)
            RowRes["scores"].sort(key=lambda x: x["score"], reverse=True)
            
            # Retallar resultats segons TopK demanat
            RowRes["scores"] = RowRes["scores"][:2]
            RowRes["assigned"] = RowRes["assigned"][:2]
            
            Results.append(RowRes)

        if not Results:
            predictions = []
        else:
            first_result_dict = Results[0] 
            assigned_list = first_result_dict.get('assigned', [])
            predictions = [
                {"label": label, "score": score} 
                for label, score in assigned_list
            ]

        return predictions