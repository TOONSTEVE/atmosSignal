#!/usr/bin/env python3
"""
Zero-dependency .pkl / joblib schema inspector + native scikit-learn inference worker.
Works with Python 3.10 standard library even when scikit-learn/joblib are absent,
and upgrades seamlessly to full live inference when scikit-learn is installed.
"""

import bz2
import gzip
import importlib.util
import io
import json
import lzma
import os
import pickle
import pickletools
import platform
import re
import struct
import sys
import zlib
from typing import Any, Dict, List, Optional, Tuple


def detect_capabilities() -> Dict[str, Any]:
    has_sklearn = importlib.util.find_spec("sklearn") is not None
    has_joblib = importlib.util.find_spec("joblib") is not None
    has_numpy = importlib.util.find_spec("numpy") is not None
    has_pandas = importlib.util.find_spec("pandas") is not None

    sklearn_version = None
    joblib_version = None

    if has_sklearn:
        try:
            import sklearn  # type: ignore
            sklearn_version = getattr(sklearn, "__version__", None)
        except Exception:
            has_sklearn = False

    if has_joblib:
        try:
            import joblib  # type: ignore
            joblib_version = getattr(joblib, "__version__", None)
        except Exception:
            has_joblib = False

    return {
        "pythonVersion": platform.python_version(),
        "hasSklearn": has_sklearn,
        "hasJoblib": has_joblib,
        "hasNumpy": has_numpy,
        "hasPandas": has_pandas,
        "sklearnVersion": sklearn_version,
        "joblibVersion": joblib_version,
        "canRunNativePklInference": bool(has_sklearn),
    }


def decompress_stream(raw: bytes) -> Tuple[bytes, str]:
    if len(raw) < 2:
        return raw, "raw-bytes"

    # Zlib magic headers (0x78 0x01, 0x78 0x5e, 0x78 0x9c, 0x78 0xda)
    if raw[0] == 0x78 and raw[1] in (0x01, 0x5E, 0x9C, 0xDA):
        try:
            return zlib.decompress(raw), "joblib-zlib"
        except Exception:
            pass

    # Gzip
    if raw[:2] == b"\x1f\x8b":
        try:
            return gzip.decompress(raw), "joblib-gzip"
        except Exception:
            pass

    # BZ2
    if raw[:3] == b"BZh":
        try:
            return bz2.decompress(raw), "joblib-bz2"
        except Exception:
            pass

    # XZ / LZMA
    if raw[:6] == b"\xfd7zXZ\x00":
        try:
            return lzma.decompress(raw), "joblib-lzma"
        except Exception:
            pass

    # Standard pickle protocol header (\x80\x02..\x80\x05) or protocol 0/1
    if raw[0] == 0x80 and raw[1] in (2, 3, 4, 5):
        return raw, f"pickle-v{raw[1]}"

    return raw, "pickle-or-custom"


# ---------------------------------------------------------------------------
# Pure-Stdlib Stub Unpickler (Safe inspection without scikit-learn / numpy)
# ---------------------------------------------------------------------------

class StubDtype:
    def __init__(self, obj: Any = None, align: bool = False, copy: bool = False):
        self.type_str = str(obj) if obj is not None else "unknown"
        self.state: Any = None

    def __setstate__(self, state: Any) -> None:
        self.state = state
        if isinstance(state, tuple) and len(state) >= 2:
            self.type_str = str(state[1])

    def __repr__(self) -> str:
        return f"StubDtype({self.type_str})"


class StubArray:
    def __init__(self, shape: Any = None, dtype: Any = None):
        self.shape = shape
        self.dtype = dtype
        self.values: Optional[List[Any]] = None

    def __setstate__(self, state: Any) -> None:
        # Standard numpy ndarray state: (version, shape, dtype, isfortran, rawdata)
        if not isinstance(state, tuple) or len(state) < 5:
            return
        shape = state[1]
        dtype = state[2]
        rawdata = state[4]
        self.shape = shape
        self.dtype = str(getattr(dtype, "type_str", dtype))

        if isinstance(rawdata, list):
            self.values = rawdata
            return

        if isinstance(rawdata, (bytes, bytearray)):
            dt = self.dtype
            count = 1
            if isinstance(shape, (tuple, list)):
                for dim in shape:
                    if isinstance(dim, int):
                        count *= dim
            if count <= 0 or count > 5000:
                return

            try:
                if dt.startswith("|S") or dt.startswith("S"):
                    width = int(dt.lstrip("|S"))
                    if width > 0 and len(rawdata) >= count * width:
                        self.values = [
                            rawdata[i * width:(i + 1) * width].rstrip(b"\x00").decode("utf-8", errors="replace")
                            for i in range(count)
                        ]
                elif dt.startswith("<U") or dt.startswith("U"):
                    width = int(dt.lstrip("<U"))
                    stride = width * 4
                    if stride > 0 and len(rawdata) >= count * stride:
                        self.values = [
                            rawdata[i * stride:(i + 1) * stride].decode("utf-32-le", errors="replace").rstrip("\x00")
                            for i in range(count)
                        ]
                elif dt in ("<f8", "f8", "float64"):
                    if len(rawdata) >= count * 8:
                        self.values = list(struct.unpack(f"<{count}d", rawdata[:count * 8]))
                elif dt in ("<f4", "f4", "float32"):
                    if len(rawdata) >= count * 4:
                        self.values = list(struct.unpack(f"<{count}f", rawdata[:count * 4]))
                elif dt in ("<i8", "i8", "int64"):
                    if len(rawdata) >= count * 8:
                        self.values = list(struct.unpack(f"<{count}q", rawdata[:count * 8]))
                elif dt in ("<i4", "i4", "int32"):
                    if len(rawdata) >= count * 4:
                        self.values = list(struct.unpack(f"<{count}i", rawdata[:count * 4]))
            except Exception:
                pass


class StubObject:
    def __init__(self, module: str, name: str, *args: Any, **kwargs: Any):
        self.__stub_module__ = module
        self.__stub_name__ = name
        self.__stub_args__ = args
        self.__stub_state__: Dict[str, Any] = {}

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__stub_state__.update(state)
        elif isinstance(state, tuple):
            for item in state:
                if isinstance(item, dict):
                    self.__stub_state__.update(item)

    def __getattr__(self, item: str) -> Any:
        if item in self.__dict__.get("__stub_state__", {}):
            return self.__stub_state__[item]
        raise AttributeError(item)


def make_stub_class(module: str, name: str):
    if module.startswith("numpy") and name == "_reconstruct":
        def _reconstruct(subtype, shape, dtype):
            return StubArray(shape, dtype)
        return _reconstruct

    if module.startswith("numpy") and name == "dtype":
        return StubDtype

    if module.startswith("numpy") and name == "scalar":
        def _scalar(dtype, raw):
            try:
                dt = str(getattr(dtype, "type_str", dtype))
                if isinstance(raw, (bytes, bytearray)):
                    if dt in ("<f8", "float64") and len(raw) == 8:
                        return struct.unpack("<d", raw)[0]
                    if dt in ("<i8", "int64") and len(raw) == 8:
                        return struct.unpack("<q", raw)[0]
            except Exception:
                pass
            return raw
        return _scalar

    class DynamicStub(StubObject):
        def __new__(cls, *args, **kwargs):
            inst = object.__new__(cls)
            StubObject.__init__(inst, module, name, *args, **kwargs)
            return inst

        def __init__(self, *args, **kwargs):
            pass

    DynamicStub.__name__ = name
    DynamicStub.__qualname__ = f"{module}.{name}"
    return DynamicStub


class StubUnpickler(pickle.Unpickler):
    def __init__(self, file_obj: io.BytesIO):
        super().__init__(file_obj)
        self.discovered_globals: List[str] = []

    def find_class(self, module: str, name: str) -> Any:
        fqn = f"{module}.{name}"
        if fqn not in self.discovered_globals:
            self.discovered_globals.append(fqn)

        # Allow safe builtins
        if module == "builtins" and name in ("set", "frozenset", "slice", "complex", "bytearray", "bytes", "dict", "list", "tuple"):
            return getattr(__builtins__, name) if isinstance(__builtins__, dict) else getattr(__import__("builtins"), name)
        if module == "collections" and name in ("OrderedDict", "defaultdict"):
            import collections
            return getattr(collections, name)

        return make_stub_class(module, name)


def extract_to_list(val: Any) -> Optional[List[Any]]:
    if isinstance(val, StubArray) and val.values is not None:
        return val.values
    if isinstance(val, list):
        return val
    if isinstance(val, tuple):
        return list(val)
    if hasattr(val, "tolist"):
        try:
            return list(val.tolist())
        except Exception:
            pass
    return None


def scan_opcodes_fallback(data: bytes) -> Dict[str, Any]:
    """Scan pickle opcodes & strings even if joblib interleaved raw numpy buffers cause early stop."""
    globals_found: List[str] = []
    strings_found: List[str] = []

    try:
        for opcode, arg, _pos in pickletools.genops(io.BytesIO(data)):
            if opcode.name in ("GLOBAL", "STACK_GLOBAL") and isinstance(arg, str):
                globals_found.append(arg)
            elif "UNICODE" in opcode.name and isinstance(arg, str):
                strings_found.append(arg)
    except Exception:
        pass

    # Binary ASCII/UTF-8 string scan for sklearn modules & attribute names when joblib array payloads interrupt genops
    if not globals_found:
        matches = re.findall(rb"(?:sklearn|xgboost|lightgbm|joblib|pandas)\.[A-Za-z0-9_.]+", data)
        for m in matches:
            s = m.decode("ascii", errors="ignore")
            if s not in globals_found:
                globals_found.append(s)

    # Extract strings near feature_names_in_ or classes_ markers
    if not strings_found:
        raw_strings = re.findall(rb"[A-Za-z0-9_][A-Za-z0-9_ \-/()%.]{1,48}", data)
        for rs in raw_strings[:400]:
            s = rs.decode("ascii", errors="ignore").strip()
            if s and s not in strings_found:
                strings_found.append(s)

    return {
        "globals": globals_found[:60],
        "strings": strings_found[:200],
    }


def extract_stub_metadata(obj: Any, fallback_globals: List[str]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "estimatorClass": None,
        "estimatorModule": None,
        "sklearnVersion": None,
        "nFeaturesIn": None,
        "featureNamesIn": [],
        "classes": [],
        "hyperparameters": {},
        "pipelineSteps": [],
        "discoveredModules": fallback_globals,
    }

    if isinstance(obj, dict):
        if "_sklearn_version" in obj:
            meta["sklearnVersion"] = str(obj["_sklearn_version"])
        if "n_features_in_" in obj and isinstance(obj["n_features_in_"], (int, float)):
            meta["nFeaturesIn"] = int(obj["n_features_in_"])
        for fkey in ("feature_names_in_", "features", "feature_names", "columns"):
            if not meta["featureNamesIn"] and fkey in obj:
                fnames = extract_to_list(obj[fkey])
                if fnames:
                    meta["featureNamesIn"] = [str(x) for x in fnames]
        for ckey in ("classes_", "classes", "labels"):
            if not meta["classes"] and ckey in obj:
                cls_list = extract_to_list(obj[ckey])
                if cls_list:
                    meta["classes"] = [str(x) for x in cls_list]
        if "model" in obj and isinstance(obj["model"], StubObject):
            nested = extract_stub_metadata(obj["model"], fallback_globals)
            for k, v in nested.items():
                if v and not meta.get(k):
                    meta[k] = v
        elif not meta["estimatorClass"]:
            meta["estimatorClass"] = "DictBundle"
            meta["estimatorModule"] = "builtins"
        return meta

    if isinstance(obj, StubObject):
        meta["estimatorClass"] = obj.__stub_name__
        meta["estimatorModule"] = obj.__stub_module__
        st = obj.__stub_state__
        if "_sklearn_version" in st:
            meta["sklearnVersion"] = str(st["_sklearn_version"])
        if "n_features_in_" in st and isinstance(st["n_features_in_"], (int, float)):
            meta["nFeaturesIn"] = int(st["n_features_in_"])
        if "feature_names_in_" in st:
            fnames = extract_to_list(st["feature_names_in_"])
            if fnames:
                meta["featureNamesIn"] = [str(x) for x in fnames]
        if "classes_" in st:
            cls_list = extract_to_list(st["classes_"])
            if cls_list:
                meta["classes"] = [str(x) for x in cls_list]

        # Pipeline steps
        if "steps" in st and isinstance(st["steps"], list):
            for step in st["steps"]:
                if isinstance(step, (tuple, list)) and len(step) == 2:
                    step_name, step_obj = step
                    step_cls = step_obj.__stub_name__ if isinstance(step_obj, StubObject) else type(step_obj).__name__
                    meta["pipelineSteps"].append({"name": str(step_name), "class": str(step_cls)})
                    if isinstance(step_obj, StubObject):
                        sub_st = step_obj.__stub_state__
                        if not meta["featureNamesIn"] and "feature_names_in_" in sub_st:
                            fnames = extract_to_list(sub_st["feature_names_in_"])
                            if fnames:
                                meta["featureNamesIn"] = [str(x) for x in fnames]
                        if not meta["nFeaturesIn"] and "n_features_in_" in sub_st:
                            if isinstance(sub_st["n_features_in_"], (int, float)):
                                meta["nFeaturesIn"] = int(sub_st["n_features_in_"])
                        if not meta["classes"] and "classes_" in sub_st:
                            cls_list = extract_to_list(sub_st["classes_"])
                            if cls_list:
                                meta["classes"] = [str(x) for x in cls_list]

        # Interesting scalar hyperparameters
        for param_key in ("n_estimators", "max_depth", "criterion", "C", "kernel", "learning_rate", "max_iter", "solver", "n_neighbors", "random_state"):
            if param_key in st and isinstance(st[param_key], (int, float, str, bool)):
                meta["hyperparameters"][param_key] = st[param_key]

    return meta


def inspect_single_file(path: str, caps: Dict[str, Any]) -> Dict[str, Any]:
    with open(path, "rb") as f:
        raw = f.read()

    decompressed, serialization_format = decompress_stream(raw)
    is_joblib_wrapped = False
    if b"joblib" in decompressed[:4096] or b"NumpyArrayWrapper" in decompressed:
        is_joblib_wrapped = True
        if serialization_format.startswith("pickle"):
            serialization_format = f"joblib ({serialization_format})"

    result: Dict[str, Any] = {
        "fileName": os.path.basename(path),
        "sizeBytes": len(raw),
        "serializationFormat": serialization_format,
        "isJoblib": is_joblib_wrapped,
        "inspectionMethod": "pickletools-stub",
        "estimatorClass": None,
        "estimatorModule": None,
        "sklearnVersion": None,
        "nFeaturesIn": None,
        "featureNamesIn": [],
        "classes": [],
        "hyperparameters": {},
        "pipelineSteps": [],
        "discoveredModules": [],
        "warnings": [],
    }

    # 1. Native load if scikit-learn / joblib available
    if caps.get("hasSklearn"):
        try:
            obj = None
            if caps.get("hasJoblib"):
                import joblib  # type: ignore
                obj = joblib.load(path)
            else:
                with open(path, "rb") as f:
                    obj = pickle.load(f)

            result["inspectionMethod"] = "native-sklearn"
            result["estimatorClass"] = type(obj).__name__
            result["estimatorModule"] = type(obj).__module__
            result["sklearnVersion"] = getattr(obj, "_sklearn_version", caps.get("sklearnVersion"))
            if hasattr(obj, "n_features_in_"):
                result["nFeaturesIn"] = int(obj.n_features_in_)
            if hasattr(obj, "feature_names_in_"):
                result["featureNamesIn"] = [str(x) for x in list(obj.feature_names_in_)]
            if hasattr(obj, "classes_"):
                result["classes"] = [str(x) for x in list(obj.classes_)]
            if hasattr(obj, "steps"):
                result["pipelineSteps"] = [{"name": str(n), "class": type(est).__name__} for n, est in obj.steps]
            if hasattr(obj, "get_params"):
                params = obj.get_params(deep=False)
                result["hyperparameters"] = {
                    k: v for k, v in params.items()
                    if isinstance(v, (int, float, str, bool))
                }
            return result
        except Exception as exc:
            result["warnings"].append(f"Native load fell back to opcode inspection: {exc}")

    # 2. Pure-stdlib StubUnpickler
    stub_globals: List[str] = []
    try:
        unpickler = StubUnpickler(io.BytesIO(decompressed))
        obj = unpickler.load()
        stub_globals = unpickler.discovered_globals
        meta = extract_stub_metadata(obj, stub_globals)
        result.update(meta)
        return result
    except Exception as exc:
        result["warnings"].append(
            f"Stream contains binary joblib array payloads or custom extensions ({type(exc).__name__}); extracted partial schema from opcodes."
        )

    # 3. Opcode & binary string fallback
    opcode_scan = scan_opcodes_fallback(decompressed)
    result["inspectionMethod"] = "opcode-scan"
    result["discoveredModules"] = opcode_scan["globals"]

    # Guess primary estimator from discovered sklearn/xgboost globals
    for g in opcode_scan["globals"]:
        if any(pkg in g for pkg in ("sklearn.", "xgboost.", "lightgbm.")):
            parts = g.split(".")
            cls_name = parts[-1]
            if cls_name not in ("_reconstruct", "dtype", "scalar"):
                result["estimatorClass"] = cls_name
                result["estimatorModule"] = ".".join(parts[:-1])
                break

    # Look for version string like "1.3.2" or "1.5.1" near _sklearn_version
    strings = opcode_scan["strings"]
    if "_sklearn_version" in strings:
        idx = strings.index("_sklearn_version")
        for cand in strings[idx + 1: idx + 4]:
            if re.match(r"^\d+\.\d+(\.\d+)?", cand):
                result["sklearnVersion"] = cand
                break

    return result


def handle_inspect(payload: Dict[str, Any]) -> Dict[str, Any]:
    caps = detect_capabilities()
    model_path = payload.get("modelPath")
    encoder_path = payload.get("encoderPath")

    model_info = inspect_single_file(model_path, caps) if model_path and os.path.exists(model_path) else None
    encoder_info = inspect_single_file(encoder_path, caps) if encoder_path and os.path.exists(encoder_path) else None

    return {
        "runtimeCapabilities": caps,
        "modelInfo": model_info,
        "encoderInfo": encoder_info,
    }


def handle_predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    caps = detect_capabilities()
    if not caps.get("hasSklearn"):
        return {
            "mode": "unavailable",
            "reason": "scikit-learn is not installed in the server Python runtime.",
            "runtimeCapabilities": caps,
        }

    model_path = payload.get("modelPath")
    encoder_path = payload.get("encoderPath")
    feature_order = payload.get("featureOrder", [])
    rows = payload.get("rows", [])

    if not model_path or not os.path.exists(model_path):
        return {"mode": "error", "error": "Model file path not found on server."}

    try:
        if caps.get("hasJoblib"):
            import joblib  # type: ignore
            model = joblib.load(model_path)
            encoder = joblib.load(encoder_path) if encoder_path and os.path.exists(encoder_path) else None
        else:
            with open(model_path, "rb") as mf:
                model = pickle.load(mf)
            encoder = None
            if encoder_path and os.path.exists(encoder_path):
                with open(encoder_path, "rb") as ef:
                    encoder = pickle.load(ef)

        import numpy as np  # type: ignore
        X = np.array([[float(row.get(col, 0.0)) for col in feature_order] for row in rows], dtype=float)
        preds = model.predict(X)

        probs_matrix = None
        class_labels: List[str] = []
        if hasattr(model, "predict_proba"):
            probs_matrix = model.predict_proba(X)
        if encoder is not None and hasattr(encoder, "classes_"):
            class_labels = [str(c) for c in list(encoder.classes_)]
        elif hasattr(model, "classes_"):
            class_labels = [str(c) for c in list(model.classes_)]

        decoded_preds: List[str] = []
        for p in preds:
            if encoder is not None and hasattr(encoder, "inverse_transform") and isinstance(p, (int, np.integer)):
                decoded_preds.append(str(encoder.inverse_transform([p])[0]))
            else:
                decoded_preds.append(str(p))

        formatted = []
        for i, label in enumerate(decoded_preds):
            prob_list = []
            confidence = 1.0
            if probs_matrix is not None:
                row_probs = probs_matrix[i]
                for j, p_val in enumerate(row_probs):
                    lbl = class_labels[j] if j < len(class_labels) else f"Class {j}"
                    prob_list.append({"label": lbl, "probability": float(p_val)})
                confidence = float(max(row_probs))
            else:
                prob_list = [{"label": label, "probability": 1.0}]
            formatted.append({
                "predictedLabel": label,
                "confidence": confidence,
                "probabilities": prob_list,
            })

        return {
            "mode": "live-sklearn",
            "predictions": formatted,
            "runtimeCapabilities": caps,
        }
    except Exception as exc:
        return {"mode": "error", "error": str(exc), "runtimeCapabilities": caps}


def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Missing command (capabilities|inspect|predict)"}))
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "capabilities":
        print(json.dumps(detect_capabilities()))
        return

    raw_stdin = sys.stdin.read()
    payload = json.loads(raw_stdin) if raw_stdin.strip() else {}

    if cmd == "inspect":
        print(json.dumps(handle_inspect(payload)))
    elif cmd == "predict":
        print(json.dumps(handle_predict(payload)))
    else:
        print(json.dumps({"error": f"Unknown command: {cmd}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
