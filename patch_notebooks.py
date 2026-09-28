import json

nb_path = r"c:\Users\HINATA\Music\Bakya Akka\Module2_Forward_Horizon_Labeling_and_XGBoost_Precursor_Training.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 4 update: fix label_list.append line
cell4_source = "".join(nb["cells"][4]["source"])
cell4_source_fixed = cell4_source.replace(
    "label_list.append(group[['machine_id', 'timestamp', 'is_precursor', 'failure_event']])",
    "label_list.append(group[['machine_id', 'timestamp', 'is_precursor']])"
)
nb["cells"][4]["source"] = [line + "\n" for line in cell4_source_fixed.split("\n")]
if nb["cells"][4]["source"][-1] == "\n":
    nb["cells"][4]["source"].pop()

# Cell 6 update: update audit_target_isolation check
cell6_source = "".join(nb["cells"][6]["source"])
cell6_source_fixed = cell6_source.replace(
    "target_in_x = 'is_precursor' in self.feature_cols or 'failure_event' in self.feature_cols",
    "target_in_x = any(c in self.feature_cols for c in ['is_precursor', 'failure_event', 'failure_event_x', 'failure_event_y'])"
)
nb["cells"][6]["source"] = [line + "\n" for line in cell6_source_fixed.split("\n")]
if nb["cells"][6]["source"][-1] == "\n":
    nb["cells"][6]["source"].pop()

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Module2 notebook patched successfully.")
