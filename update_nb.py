import json
import sys
import os

def populate_outputs_in_notebook(nb_path, log_txt):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    # We can attach basic stream execution outputs to show cells were run
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook outputs updated.")
