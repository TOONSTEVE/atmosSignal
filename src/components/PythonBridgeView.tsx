import React, { useMemo, useState } from 'react';
import { Check, Copy, Download, Terminal } from 'lucide-react';
import type { ModelSchema } from '../types/weather';
import { SAMPLE_MODEL_SCHEMA } from '../data/sampleModel';

interface PythonBridgeViewProps {
  schema: ModelSchema | null;
}

type ScriptId = 'export_json' | 'inspect' | 'server' | 'requirements';

function downloadScript(filename: string, content: string) {
  const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export const PythonBridgeView: React.FC<PythonBridgeViewProps> = ({ schema }) => {
  const activeSchema = schema || SAMPLE_MODEL_SCHEMA;
  const [selectedScript, setSelectedScript] = useState<ScriptId>('export_json');
  const [copied, setCopied] = useState(false);

  const modelFile = activeSchema.modelFileName.replace(' (sample)', '') || 'weather_signal_model.pkl';
  const encoderFile =
    (activeSchema.encoderFileName || 'label_encoder.pkl').replace(' (sample)', '');
  const featureKeys = activeSchema.features.map((f) => f.key);

  const scripts = useMemo(() => {
    const exportJsonPy = `import json
import pickle
import joblib

def load_artifact(path):
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

def export_tree(tree):
    # Handle sklearn tree.value shape (n_nodes, 1, n_classes) or (n_nodes, n_classes)
    raw_val = tree.value
    if raw_val.ndim == 3:
        values = raw_val[:, 0, :].tolist()
    else:
        values = raw_val.tolist()
    return {
        "childrenLeft": tree.children_left.tolist(),
        "childrenRight": tree.children_right.tolist(),
        "feature": tree.feature.tolist(),
        "threshold": tree.threshold.tolist(),
        "value": values,
    }

model = load_artifact("${modelFile}")
try:
    encoder = load_artifact("${encoderFile}")
except FileNotFoundError:
    encoder = None

# Extract pipeline scalers + final estimator
scalers = []
estimator = model
pipeline_steps = []
if hasattr(model, "steps"):
    for name, step in model.steps[:-1]:
        cls_name = type(step).__name__
        pipeline_steps.append({"name": name, "class": cls_name})
        if cls_name == "StandardScaler":
            scalers.append({
                "type": "StandardScaler",
                "mean": step.mean_.tolist() if getattr(step, "mean_", None) is not None else [],
                "scale": step.scale_.tolist() if getattr(step, "scale_", None) is not None else [],
            })
        elif cls_name == "MinMaxScaler":
            scalers.append({
                "type": "MinMaxScaler",
                "min": step.min_.tolist(),
                "scale": step.scale_.tolist(),
            })
        elif cls_name == "RobustScaler":
            scalers.append({
                "type": "RobustScaler",
                "center": step.center_.tolist() if getattr(step, "center_", None) is not None else [],
                "scale": step.scale_.tolist() if getattr(step, "scale_", None) is not None else [],
            })
        elif cls_name == "SimpleImputer":
            scalers.append({
                "type": "SimpleImputer",
                "statistics": step.statistics_.tolist(),
            })
    final_name, estimator = model.steps[-1]
    pipeline_steps.append({"name": final_name, "class": type(estimator).__name__})

# Discover exact feature order
if hasattr(model, "feature_names_in_"):
    features = [str(x) for x in model.feature_names_in_.tolist()]
elif hasattr(estimator, "feature_names_in_"):
    features = [str(x) for x in estimator.feature_names_in_.tolist()]
else:
    n_in = int(getattr(estimator, "n_features_in_", ${featureKeys.length}))
    default_names = ${JSON.stringify(featureKeys)}
    features = default_names[:n_in] if len(default_names) >= n_in else [f"feature_{i}" for i in range(n_in)]

# Discover exact target class labels
if encoder is not None and hasattr(encoder, "classes_"):
    classes = [str(x) for x in encoder.classes_.tolist()]
elif hasattr(estimator, "classes_"):
    classes = [str(x) for x in estimator.classes_.tolist()]
else:
    classes = ${JSON.stringify(activeSchema.classes)}

est_name = type(estimator).__name__
if est_name == "DecisionTreeClassifier":
    model_data = {"kind": "tree", "tree": export_tree(estimator.tree_)}
elif est_name in ("RandomForestClassifier", "ExtraTreesClassifier"):
    model_data = {"kind": "forest", "trees": [export_tree(t.tree_) for t in estimator.estimators_]}
elif est_name == "GradientBoostingClassifier":
    init_priors = estimator.init_.class_prior_.tolist() if hasattr(estimator.init_, "class_prior_") else [0.0] * len(classes)
    stages = [[export_tree(t.tree_) for t in stage] for stage in estimator.estimators_]
    model_data = {
        "kind": "gbdt",
        "learningRate": float(estimator.learning_rate),
        "initPriors": init_priors,
        "stages": stages,
    }
elif est_name in ("LogisticRegression", "SGDClassifier", "RidgeClassifier"):
    model_data = {
        "kind": "linear",
        "coef": estimator.coef_.tolist(),
        "intercept": estimator.intercept_.tolist(),
    }
elif est_name == "GaussianNB":
    model_data = {
        "kind": "gaussian_nb",
        "theta": estimator.theta_.tolist(),
        "var": estimator.var_.tolist(),
        "classPrior": estimator.class_prior_.tolist(),
    }
elif est_name == "MLPClassifier":
    model_data = {
        "kind": "mlp",
        "activation": str(estimator.activation),
        "coefs": [w.tolist() for w in estimator.coefs_],
        "intercepts": [b.tolist() for b in estimator.intercepts_],
    }
elif est_name == "KNeighborsClassifier":
    model_data = {
        "kind": "knn",
        "nNeighbors": int(estimator.n_neighbors),
        "weights": str(estimator.weights),
        "X": estimator._fit_X.tolist(),
        "y": estimator._y.tolist(),
    }
else:
    raise ValueError(f"Unsupported estimator type for JSON export: {est_name}")

bundle = {
    "format": "atmossignal-sklearn-v1",
    "modelFileName": "${modelFile}",
    "encoderFileName": "${encoderFile}",
    "estimatorClass": est_name,
    "estimatorModule": type(estimator).__module__,
    "sklearnVersion": getattr(estimator, "_sklearn_version", "unknown"),
    "features": features,
    "classes": classes,
    "pipelineSteps": pipeline_steps,
    "hyperparameters": {
        k: v for k, v in estimator.get_params(deep=False).items()
        if isinstance(v, (int, float, str, bool))
    },
    "scalers": scalers,
    "modelData": model_data,
}

with open("weather_signal_bundle.json", "w", encoding="utf-8") as out:
    json.dump(bundle, out)

print("Exported weather_signal_bundle.json successfully!")
print("Estimator:", est_name)
print("Features:", features)
print("Classes:", classes)
`;

    const inspectPy = `import joblib
import pickle

def load_artifact(path):
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

model = load_artifact("${modelFile}")
encoder = load_artifact("${encoderFile}")

print("Model class:", type(model).__name__)
print("Expected feature count:", getattr(model, "n_features_in_", "Unknown"))
print("Feature names:", list(getattr(model, "feature_names_in_", [])))
print("LabelEncoder classes:", list(getattr(encoder, "classes_", [])))
`;

    const pydanticFields = featureKeys
      .map((k) => `    ${k.replace(/[^a-zA-Z0-9_]/g, '_')}: float`)
      .join('\n');

    const vectorExtract = featureKeys
      .map((k) => `payload.${k.replace(/[^a-zA-Z0-9_]/g, '_')}`)
      .join(', ');

    const serverPy = `from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Weather Signal Forecast API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("${modelFile}")
label_encoder = joblib.load("${encoderFile}")

class WeatherSignals(BaseModel):
${pydanticFields}

@app.post("/predict")
def predict_weather(payload: WeatherSignals):
    X = np.array([[${vectorExtract}]], dtype=float)
    pred_idx = model.predict(X)
    label = str(label_encoder.inverse_transform(pred_idx)[0])

    probabilities = []
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        for idx, p in enumerate(probs):
            cls_name = str(label_encoder.inverse_transform([idx])[0])
            probabilities.append({"label": cls_name, "probability": float(p)})

    return {
        "predictedLabel": label,
        "confidence": max((x["probability"] for x in probabilities), default=1.0),
        "probabilities": probabilities,
    }
`;

    const requirementsTxt = `scikit-learn>=${activeSchema.sklearnVersion !== 'Unknown' ? activeSchema.sklearnVersion : '1.4.0'}
joblib>=1.3.0
numpy>=1.26.0
fastapi>=0.115.0
uvicorn>=0.30.0
pydantic>=2.7.0
`;

    return {
      export_json: {
        filename: 'export_to_json.py',
        code: exportJsonPy,
        command: 'python export_to_json.py',
      },
      inspect: {
        filename: 'inspect_model.py',
        code: inspectPy,
        command: 'python inspect_model.py',
      },
      server: {
        filename: 'predict_api.py',
        code: serverPy,
        command: 'pip install -r requirements.txt && uvicorn predict_api:app --reload --port 8000',
      },
      requirements: {
        filename: 'requirements.txt',
        code: requirementsTxt,
        command: 'pip install -r requirements.txt',
      },
    };
  }, [activeSchema, modelFile, encoderFile, featureKeys]);

  const current = scripts[selectedScript];

  const handleCopy = async () => {
    await navigator.clipboard.writeText(current.code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  return (
    <div className="mx-auto max-w-[1440px] px-6 py-8 space-y-8">
      <section className="border-b border-slate-800 pb-6 space-y-2 max-w-3xl">
        <p className="text-xs font-mono uppercase tracking-wider text-cyan-400">
          Zero-Server JSON Export & Python Bridge
        </p>
        <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-slate-50 [text-wrap:balance]">
          Export Your .pkl Model to Portable JSON (No Python Server Needed)
        </h1>
        <p className="text-sm text-slate-400 leading-relaxed">
          Run <span className="font-mono text-slate-200">export_to_json.py</span> once on your computer in the same folder as <span className="font-mono text-slate-200">{modelFile}</span> and <span className="font-mono text-slate-200">{encoderFile}</span>. It produces <span className="font-mono text-slate-200">weather_signal_bundle.json</span>, which you can upload in the Inspector tab to run your exact trained trees and weights directly inside this app.
        </p>
      </section>

      {/* 3-Step Workflow */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <div className="border border-slate-800 bg-[#111827] rounded-lg p-5 space-y-2">
          <p className="font-mono text-xs text-cyan-400">Step 01</p>
          <h2 className="text-base font-semibold text-slate-100">
            Download export_to_json.py
          </h2>
          <p className="text-xs text-slate-400 leading-relaxed">
            Save <span className="font-mono text-slate-200">export_to_json.py</span> into the same folder as your <span className="font-mono text-slate-200">.pkl</span> files. Supports RandomForest, DecisionTree, GradientBoosting, LogisticRegression, GaussianNB, MLP, KNN, and Pipeline scalers.
          </p>
        </div>
        <div className="border border-slate-800 bg-[#111827] rounded-lg p-5 space-y-2">
          <p className="font-mono text-xs text-cyan-400">Step 02</p>
          <h2 className="text-base font-semibold text-slate-100">
            Generate weather_signal_bundle.json
          </h2>
          <p className="text-xs text-slate-400 leading-relaxed">
            Run <span className="font-mono text-slate-200">python export_to_json.py</span> once. It extracts your exact <span className="font-mono text-slate-200">feature_names_in_</span>, <span className="font-mono text-slate-200">classes_</span>, scaler math, and model weights into one portable JSON file.
          </p>
        </div>
        <div className="border border-slate-800 bg-[#111827] rounded-lg p-5 space-y-2">
          <p className="font-mono text-xs text-cyan-400">Step 03</p>
          <h2 className="text-base font-semibold text-slate-100">
            Upload Bundle in Inspector
          </h2>
          <p className="text-xs text-slate-400 leading-relaxed">
            Switch to the <span className="font-mono text-slate-200">Exported .json Bundle</span> mode in the Inspector tab and upload <span className="font-mono text-slate-200">weather_signal_bundle.json</span>. Both Forecast and Batch CSV will execute your exact trained model natively.
          </p>
        </div>
      </section>

      {/* Interactive Code Exporter */}
      <section className="border border-slate-800 bg-[#111827] rounded-lg overflow-hidden">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 px-5 py-3.5 bg-[#0B0E17]">
          <div className="flex flex-wrap items-center gap-1.5">
            {(
              [
                { id: 'export_json', label: 'export_to_json.py' },
                { id: 'inspect', label: 'inspect_model.py' },
                { id: 'server', label: 'predict_api.py' },
                { id: 'requirements', label: 'requirements.txt' },
              ] as const
            ).map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => setSelectedScript(tab.id)}
                className={`px-3 py-1.5 text-xs font-mono rounded transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
                  selectedScript === tab.id
                    ? 'bg-cyan-400 text-slate-950 font-semibold'
                    : 'text-slate-400 hover:text-slate-100'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2.5 shrink-0">
            <button
              type="button"
              onClick={handleCopy}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-200 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded transition-colors cursor-pointer"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copied ? 'Copied' : 'Copy'}</span>
            </button>
            <button
              type="button"
              onClick={() => downloadScript(current.filename, current.code)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 rounded transition-colors cursor-pointer"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download {current.filename}</span>
            </button>
          </div>
        </div>

        <pre className="p-5 overflow-x-auto font-mono text-xs leading-relaxed text-slate-200 bg-[#07090E] max-h-[520px]">
          <code>{current.code}</code>
        </pre>

        <div className="border-t border-slate-800 px-5 py-3.5 bg-[#0B0E17] flex flex-wrap items-center gap-3 text-xs font-mono text-slate-400">
          <Terminal className="w-4 h-4 text-cyan-400 shrink-0" />
          <span>Run in the same folder as your .pkl files:</span>
          <code className="text-cyan-300">{current.command}</code>
        </div>
      </section>
    </div>
  );
};
