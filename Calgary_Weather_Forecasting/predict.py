"""Forecast YYC temperature from a manually supplied vector in physical units."""
import argparse
from collections.abc import Mapping
import json
import math
from numbers import Integral, Real
from pathlib import Path
import sys

import numpy as np

BUNDLE = Path(__file__).resolve().parent / "inference" / "forecast_weights.npz"
HORIZONS = (1, 6, 12, 24, 48)
MODELS = ("spatial", "local", "standalone-local", "sequence-mlp", "sequence-tcn")
SEEDS = (17, 2026, 2403)
CALENDAR = ("hour_sin", "hour_cos", "doy_sin", "doy_cos")


def _schema(model, bundle):
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}. Choose from {', '.join(MODELS)}.")
    if model.startswith("sequence-"):
        channels = list(zip(bundle["seq_channels"].tolist(), bundle["seq_units"].tolist()))
        history = [
            (f"lag_{lag}__{name}", unit, lag != 0 or name != "yyc__temperature")
            for lag in range(23, -1, -1)
            for name, unit in channels
        ]
        return history + [(name, "unitless", False) for name in CALENDAR]

    fields = list(zip(bundle["feature_names"].tolist(), bundle["feature_units"].tolist(),
                      bundle["auxiliary"].tolist()))
    return fields if model == "spatial" else [field for field in fields if not field[2]]


def feature_schema(model="spatial"):
    """Return (name, physical unit, may be missing) tuples in model input order."""
    with np.load(BUNDLE, allow_pickle=False) as bundle:
        return _schema(model, bundle)


def _physical_values(features, fields, sequence):
    if isinstance(features, Mapping):
        if sequence and set(features) == {"history", "calendar"}:
            history = np.asarray(features["history"], dtype=object)
            calendar = np.asarray(features["calendar"], dtype=object)
            if history.shape != (24, 24) or calendar.shape != (4,):
                raise ValueError("Expected history with 24 hourly rows of 24 readings, and 4 calendar entries.")
            features = np.concatenate((history.ravel(), calendar))
        else:
            names = [name for name, _, _ in fields]
            missing = sorted(set(names) - set(features))
            extra = sorted(set(features) - set(names), key=str)
            if missing or extra:
                raise ValueError(f"Feature names differ from the schema. Missing: {missing[:5]}. Unexpected: {extra[:5]}.")
            features = [features[name] for name in names]

    values = np.asarray(features, dtype=object)
    if values.shape != (len(fields),):
        raise ValueError(f"Expected a vector of {len(fields)} features; received shape {values.shape}. See --schema.")

    physical = np.empty(len(fields), dtype=np.float64)
    for i, (value, (name, _, optional)) in enumerate(zip(values, fields)):
        if value is not None and (not isinstance(value, Real) or isinstance(value, (bool, np.bool_))):
            raise ValueError(f"{name} must be numeric or an allowed null.")
        try:
            number = float(value) if value is not None else math.nan
        except OverflowError as exc:
            raise ValueError(f"{name} is too large to represent.") from exc
        if math.isnan(number) and not optional:
            raise ValueError(f"{name} is required and must be finite.")
        if math.isinf(number):
            raise ValueError(f"{name} cannot be infinite.")
        physical[i] = number
    return physical


def _engineered(model, values, horizons, bundle):
    result = {}
    for horizon in horizons:
        key = f"{model}__{horizon}__"
        filled = np.where(np.isnan(values), bundle[key + "median"], values)
        x = ((filled - bundle[key + "mean"]) / bundle[key + "std"]).astype(np.float32)
        layers = int(bundle[key + "layers"].item())
        for layer in range(layers):
            x = x @ bundle[f"{key}w{layer}"].T + bundle[f"{key}b{layer}"]
            if layer + 1 < layers:
                x = np.maximum(0, x)
        result[horizon] = x.item() * bundle[key + "y_std"].item() + bundle[key + "y_mean"].item()
    return result


def _gelu(x):
    erf = np.array([math.erf(float(v) / math.sqrt(2.0)) for v in x], dtype=np.float32)
    return 0.5 * x * (1.0 + erf)


def _sequence(model, values, horizons, seed, bundle):
    history = values[:-4].reshape(24, 24).astype(np.float32)
    observed = np.isfinite(values[:-4].reshape(24, 24))
    filled = np.where(observed, history, bundle["seq_median"])
    standardized = (filled - bundle["seq_mean"]) / bundle["seq_std"]
    x = np.concatenate((standardized, observed.astype(np.float32)), axis=1)
    calendar = values[-4:].astype(np.float32)
    key = f"{model}__{seed}__"

    def affine(vector, layer):
        return vector @ bundle[f"{key}{layer}.weight"].T + bundle[f"{key}{layer}.bias"]

    if model == "sequence-mlp":
        x = np.concatenate((x.ravel(), calendar))
        x = _gelu(affine(x, "net.0"))
        x = _gelu(affine(x, "net.3"))
        output = affine(x, "net.6")
    else:
        x = x @ bundle[key + "input_proj.weight"][:, :, 0].T + bundle[key + "input_proj.bias"]
        for layer, dilation in enumerate((1, 2, 4, 8)):
            block = f"{key}blocks.{layer}.conv.conv"
            weights = bundle[block + ".weight"]
            # PyTorch Conv1d cross-correlates a left-padded history at offsets -2d, -d, 0.
            conv = x @ weights[:, :, 2].T + bundle[block + ".bias"]
            conv[dilation:] += x[:-dilation] @ weights[:, :, 1].T
            conv[2 * dilation:] += x[:-2 * dilation] @ weights[:, :, 0].T
            x = np.maximum(0, x + conv)
        hidden = np.maximum(0, affine(np.concatenate((x[-1], calendar)), "head.0"))
        output = affine(hidden, "head.3")

    output = output * bundle[key + "y_std"] + bundle[key + "y_mean"]
    return {h: float(output[HORIZONS.index(h)]) for h in horizons}


def predict_temperature(features, *, model="spatial", horizons=None, seed=None) -> dict[int, float]:
    """Return {lead in hours: YYC temperature in °C} for a named or ordered, unscaled input.

    For sequence models, a mapping with a (24, 24) history and four calendar
    values is also accepted. A missing optional reading can be None or NaN.
    """
    if horizons is None:
        horizons = HORIZONS
    elif isinstance(horizons, Integral):
        horizons = (horizons,)
    else:
        try:
            horizons = tuple(horizons)
        except TypeError as exc:
            raise ValueError(f"Specify one or more supported horizons from {HORIZONS}.") from exc
    if not horizons or any(not isinstance(h, Integral) or isinstance(h, (bool, np.bool_)) or
                           h not in HORIZONS for h in horizons):
        raise ValueError(f"Supported horizons are {HORIZONS}.")
    horizons = tuple(dict.fromkeys(int(h) for h in horizons))

    sequence = model in ("sequence-mlp", "sequence-tcn")
    if seed is not None and (not sequence or not isinstance(seed, Integral) or seed not in SEEDS):
        raise ValueError(f"--seed applies to sequence models and must be one of {SEEDS}.")
    with np.load(BUNDLE, allow_pickle=False) as bundle:
        fields = _schema(model, bundle)
        values = _physical_values(features, fields, sequence)
        with np.errstate(over="ignore", invalid="ignore"):
            if sequence:
                result = _sequence(model, values, horizons, 2403 if seed is None else int(seed), bundle)
            else:
                result = _engineered(model, values, horizons, bundle)
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("Prediction overflowed. Check the feature units and magnitudes.")
    return result


def _json_input(text):
    def pairs(items):
        obj = dict(items)
        if len(obj) != len(items):
            raise ValueError("Duplicate feature names in JSON.")
        return obj

    def invalid(token):
        raise ValueError(f"{token} is invalid JSON. Use null for an allowed missing value.")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="spatial")
    parser.add_argument("--lead", type=int, nargs="+", choices=HORIZONS, help="one or more leads; default is all five")
    parser.add_argument("--seed", type=int, choices=SEEDS, help="sequence checkpoint; default 2403")
    parser.add_argument("--json", action="store_true", help="write machine-readable predictions")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--input", type=Path, help="JSON file containing one array or named feature object")
    group.add_argument("--features", help="literal JSON vector or named feature object")
    group.add_argument("--schema", action="store_true", help="show feature positions, units and missing-value rules")
    args = parser.parse_args()

    try:
        if args.schema:
            for pos, (name, unit, optional) in enumerate(feature_schema(args.model), 1):
                print(f'{pos:>3}  {name:<43}  {unit:<8}  {"null allowed" if optional else "required"}')
            return

        if args.input is not None:
            text = args.input.read_text(encoding="utf-8-sig")
        elif args.features is not None:
            text = args.features
        elif sys.stdin.isatty():
            print(f"Paste one {args.model} JSON vector or named object on one line, then press Enter.", file=sys.stderr)
            text = input()
        else:
            text = sys.stdin.read()
        if not text.strip():
            raise ValueError("Supply one JSON feature vector via --input, --features, or standard input.")

        result = predict_temperature(_json_input(text), model=args.model, horizons=args.lead, seed=args.seed)
        if args.json:
            payload = {"model": args.model, "temperature_C_by_lead_h": result}
            if args.model.startswith("sequence-"):
                payload["seed"] = 2403 if args.seed is None else args.seed
            print(json.dumps(payload, allow_nan=False))
        else:
            print("Lead (h)   YYC temperature (°C)")
            for horizon, temperature in result.items():
                print(f"{horizon:>8}   {temperature:>20.3f}")
    except (ValueError, OSError, EOFError) as exc:
        parser.exit(2, f"Prediction error: {exc}\n")


if __name__ == "__main__":
    main()
