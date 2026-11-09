"""Export a scanner model's two classifier stages to ONNX and check they agree with scikit-learn.

Only the learned part is exported (stage 1 and stage 2 HistGradientBoostingClassifier). Feature
extraction, quad fitting and the warp stay in Python (scanner.features, scanner.geometry), so another
runtime must reimplement them to use the ONNX files. The app itself keeps using the joblib model.
CLI: python -m scanner.finetune.export_onnx --model models/scanner.joblib --out models/onnx
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
from PIL import Image

from scanner import dataset, features as feats, infer


def to_onnx(clf, n_features: int):
    """ONNX graph for a fitted binary HistGradientBoostingClassifier (numeric features only).

    Written directly from the fitted trees (skl2onnx 1.20 cannot convert scikit-learn 1.9 models):
    one ai.onnx.ml TreeEnsembleRegressor with every tree, x <= threshold goes left, a missing value
    follows missing_go_to_left, base value = the model's baseline, then a Sigmoid gives P(page)."""
    from onnx import TensorProto, helper
    a = {k: [] for k in ("nodes_treeids", "nodes_nodeids", "nodes_featureids", "nodes_values", "nodes_modes",
                         "nodes_truenodeids", "nodes_falsenodeids", "nodes_missing_value_tracks_true",
                         "target_treeids", "target_nodeids", "target_ids", "target_weights")}
    for t, (pred,) in enumerate(clf._predictors):
        for i, n in enumerate(pred.nodes):
            if n["is_categorical"]:
                raise ValueError("categorical splits are not supported")
            a["nodes_treeids"].append(t)
            a["nodes_nodeids"].append(i)
            a["nodes_featureids"].append(int(n["feature_idx"]))
            a["nodes_values"].append(float(n["num_threshold"]))
            leaf = bool(n["is_leaf"])
            a["nodes_modes"].append("LEAF" if leaf else "BRANCH_LEQ")
            a["nodes_truenodeids"].append(0 if leaf else int(n["left"]))
            a["nodes_falsenodeids"].append(0 if leaf else int(n["right"]))
            a["nodes_missing_value_tracks_true"].append(int(n["missing_go_to_left"]))
            if leaf:
                a["target_treeids"].append(t)
                a["target_nodeids"].append(i)
                a["target_ids"].append(0)
                a["target_weights"].append(float(n["value"]))
    trees = helper.make_node("TreeEnsembleRegressor", ["X"], ["raw"], domain="ai.onnx.ml", n_targets=1,
                             aggregate_function="SUM", post_transform="NONE",
                             base_values=[float(np.ravel(clf._baseline_prediction)[0])], **a)
    sigmoid = helper.make_node("Sigmoid", ["raw"], ["p_page"])  # onnxruntime ignores LOGISTIC on a regressor
    graph = helper.make_graph([trees, sigmoid], "scanner_stage",
                              [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, n_features])],
                              [helper.make_tensor_value_info("p_page", TensorProto.FLOAT, [None, 1])])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("ai.onnx.ml", 3)],
                             producer_name="scanner.finetune.export_onnx", ir_version=9)


def export(model_path: str, out_dir: str, n_images: int = 5) -> dict:
    """Write stage1.onnx and stage2.onnx; returns the max |p_onnx - p_sklearn| on real feature rows."""
    import onnxruntime as ort

    model = joblib.load(model_path)
    os.makedirs(out_dir, exist_ok=True)
    X1, X2 = [], []
    for it in dataset.load_split("test")[:n_images]:  # synthetic test images: features only, deterministic
        with Image.open(it["image"]) as f:
            X, (gw, gh) = feats.compute(f.convert("RGB"))
        X = X.astype(np.float32)
        p = model["stage1"].predict_proba(X)[:, 1].reshape(gh, gw)
        X1.append(X)
        X2.append(np.hstack([X, feats.context(p)]).astype(np.float32))
    report = {"model": os.path.relpath(model_path, infer.ROOT), "max_abs_diff": {}}
    for name, X in (("stage1", np.concatenate(X1)), ("stage2", np.concatenate(X2))):
        clf = model[name]
        onx = to_onnx(clf, X.shape[1])
        path = os.path.join(out_dir, f"{name}.onnx")
        with open(path, "wb") as f:
            f.write(onx.SerializeToString())
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        p_onnx = sess.run(None, {"X": X})[0][:, 0]
        diff = float(np.max(np.abs(p_onnx - clf.predict_proba(X)[:, 1])))
        report["max_abs_diff"][name] = diff
        report[f"{name}_bytes"] = os.path.getsize(path)
        report[f"{name}_rows_checked"] = int(len(X))
    with open(os.path.join(out_dir, "export_check.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=infer.MODEL_PATH)
    ap.add_argument("--out", default=os.path.join(infer.ROOT, "models", "onnx"))
    a = ap.parse_args()
    print(json.dumps(export(a.model, a.out), indent=1))
