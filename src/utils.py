import json
import os
from typing import Dict

class LoaderLabels:
    def __init__(self, path: str):
        self.path = path

    def load_json_labels(self) -> Dict[int, str]:
        """
        Llegeix el JSON i retorna un diccionari {0: "ETIQUETA", ...}
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f" No s'ha trobat el fitxer: {self.path}")
        
        print(f" Llegint etiquetes de: {self.path}")
        with open(self.path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convertir les claus de string "0" a int 0
        return {int(k): v for k, v in data.items()}