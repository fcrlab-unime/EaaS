import os
import random
import numpy as np
import pandas as pd
import aidge_core
import aidge_core.aidge_save
import aidge_onnx
import aidge_backend_cpu
import aidge_quantization
import onnx
from sklearn.metrics import r2_score

# --------------------------------------------------------------------------
# Quantization constants (int8)
# --------------------------------------------------------------------------
NB_BITS = 8
UPSCALE = 2 ** (NB_BITS - 1) - 1   # 127


class PipelineConfig:
    def __init__(self, model_path, train_csv, label=None):
        self.model_path  = model_path
        self.train_csv   = train_csv
        self.input_shape = self._infer_input_shape()

        # Scaling factors computed in quant() on the float model
        self._input_scale  = 1.0   # UPSCALE / max_abs_feature_calib
        self._output_scale = 1.0   # max_abs_output_calib / UPSCALE

        if label is None:
            df_header  = pd.read_csv(train_csv, nrows=1)
            self.label = df_header.columns[-1]
        else:
            self.label = label

    # ------------------------------------------------------------------
    # ONNX shape inference
    # ------------------------------------------------------------------

    def _infer_input_shape(self):
        model = onnx.load(self.model_path)
        dims  = model.graph.input[0].type.tensor_type.shape.dim
        return tuple(
            dim.dim_value if dim.dim_value > 0 else 1
            for dim in dims
        )

    # ------------------------------------------------------------------
    # Quantization pipeline
    # ------------------------------------------------------------------

    def _propagate(self, model, scheduler, sample_int8_scale):
        """Run one forward pass with values already in int8 scale (rounded and clipped)."""
        arr = np.round(sample_int8_scale).clip(-127, 127).astype(int)
        input_tensor = aidge_core.Tensor(arr)
        input_tensor.set_backend('cpu')
        input_tensor.to_dtype(aidge_core.dtype.int8)
        scheduler.forward(True, data=[input_tensor])
        output_node = model.get_output_nodes().pop()
        out = np.array(output_node.get_operator().get_output(0)).copy()
        return out

    def quant(self, calib_csv=None):
        _seed = int(os.getenv("QUANT_SEED", "42"))
        random.seed(_seed)
        np.random.seed(_seed)

        aidge_model = aidge_onnx.load_onnx(self.model_path)
        aidge_core.remove_flatten(aidge_model)
        aidge_core.fuse_batchnorm(aidge_model)
        aidge_core.expand_metaops(aidge_model, name_format="{0}_{1}_{2}")
        aidge_model.set_datatype(aidge_core.dtype.float32)
        aidge_model.set_backend('cpu')

        df_calib = pd.read_csv(calib_csv if calib_csv is not None else self.train_csv)
        inputs   = df_calib.drop(columns=[self.label]).values
        n_calib  = min(1, len(inputs))
        rng      = np.random.default_rng(_seed)
        indices  = rng.choice(len(inputs), size=n_calib, replace=False)

        calib_inputs = inputs[indices]
        max_abs_input = float(np.max(np.abs(calib_inputs)))
        self._input_scale = float(UPSCALE) / max_abs_input if max_abs_input > 0 else float(UPSCALE)

        # Run float forward passes to get actual model output range (more accurate than label max)
        sched_float = aidge_core.SequentialScheduler(aidge_model)
        sched_float.generate_scheduling()
        max_abs_output_model = 0.0
        for i in indices:
            sample_np = inputs[i].reshape(self.input_shape)
            t = aidge_core.Tensor(np.array(sample_np, dtype=np.float32))
            t.set_backend('cpu')
            t.set_datatype(aidge_core.dtype.float32)
            sched_float.forward(True, data=[t])
            out_node = next(iter(aidge_model.get_output_nodes()))
            out_val = float(np.abs(np.array(out_node.get_operator().get_output(0))).max())
            max_abs_output_model = max(max_abs_output_model, out_val)
        self._output_scale = max_abs_output_model / float(UPSCALE) if max_abs_output_model > 0 else 1.0 / float(UPSCALE)

        calibration_set = []
        for i in indices:
            sample_np = inputs[i].reshape(self.input_shape)
            tensor    = aidge_core.Tensor(np.array(sample_np, dtype=np.float32))
            tensor.set_backend('cpu')
            tensor.set_datatype(aidge_core.dtype.float32)
            calibration_set.append(tensor)

        weight_precision     = aidge_core.dtype.int8
        activation_precision = aidge_core.dtype.int8
        bias_precision       = aidge_core.dtype.int32

        aidge_quantization.auto_assign_node_precision(
            graphview=aidge_model,
            act_precision=activation_precision,
            weight_precision=weight_precision,
            bias_precision=bias_precision,
        )

        aidge_quantization.quantize_network(
            network=aidge_model,
            calibration_set=calibration_set,
            clipping_mode=aidge_quantization.Clipping.MAX,
            single_shift=True,
        )

        return aidge_model

    # ------------------------------------------------------------------
    # Post-quantization metrics
    # ------------------------------------------------------------------

    def compute_metrics(self, quant_model, test_csv: str) -> dict:
        df = pd.read_csv(test_csv)
        if self.label and self.label in df.columns:
            X      = df.drop(columns=[self.label]).values
            y_true = df[self.label].values.astype(np.float64)
        else:
            X      = df.iloc[:, :-1].values
            y_true = df.iloc[:, -1].values.astype(np.float64)
            
        quant_model.set_backend('cpu')

        for node in quant_model.get_ordered_inputs():
            node[0].get_operator().get_input(0).to_dtype(aidge_core.dtype.int8)

        scheduler = aidge_core.SequentialScheduler(quant_model)
        scheduler.generate_scheduling()

        y_pred = []
        for i in range(len(X)):
            sample = (X[i] * self._input_scale).reshape(self.input_shape).astype(np.float32)
            out = self._propagate(quant_model, scheduler, sample) * self._output_scale
            y_pred.append(float(out.item()))

        y_pred = np.array(y_pred, dtype=np.float64)

        mae  = float(np.mean(np.abs(y_true - y_pred)))
        mse  = float(np.mean((y_true - y_pred) ** 2))
        rmse = float(np.sqrt(mse))
        r2   = r2_score(y_true, y_pred)

        return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}
