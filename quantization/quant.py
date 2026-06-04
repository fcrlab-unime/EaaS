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

    def _propagate(self, model, scheduler, sample_fp32):
        """Run one forward pass with a float32 array (int8-scale values)."""
        input_tensor = aidge_core.Tensor(np.array(sample_fp32, dtype=np.float32))
        input_tensor.set_backend('cpu')
        input_tensor.set_datatype(aidge_core.dtype.float32)
        scheduler.forward(True, data=[input_tensor])
        output_node = model.get_output_nodes().pop()
        out = np.array(output_node.get_operator().get_output(0)).copy()
        return out

    def quant(self):
        aidge_model = aidge_onnx.load_onnx(self.model_path)
        aidge_core.remove_flatten(aidge_model)
        aidge_core.fuse_batchnorm(aidge_model)
        aidge_core.expand_metaops(aidge_model, name_format="{0}_{1}_{2}")
        aidge_model.set_datatype(aidge_core.dtype.float32)
        aidge_model.set_backend('cpu')

        df_calib = pd.read_csv(self.train_csv)
        inputs   = df_calib.drop(columns=[self.label]).values
        n_calib  = min(6, len(inputs))
        rng      = np.random.default_rng(42)
        indices  = rng.choice(len(inputs), size=n_calib, replace=False)

        self._input_scale  = float(UPSCALE)          # 127
        self._output_scale = 1.0 / float(UPSCALE)    # 1/127

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
            clipping_mode=aidge_quantization.Clipping.MSE,
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
        quant_model.forward_dtype()

        scheduler = aidge_core.SequentialScheduler(quant_model)
        scheduler.generate_scheduling()

        y_pred = []
        for i in range(len(X)):
            # Pre-scale to int8 range, stored as float32 — same approach as quantize_model.py
            scaled = np.round(X[i] * self._input_scale).astype(np.float32)
            sample = scaled.reshape(self.input_shape)
            out = self._propagate(quant_model, scheduler, sample)
            y_pred.append(float(out.item()) / float(UPSCALE))

        y_pred = np.array(y_pred, dtype=np.float64)

        mae  = float(np.mean(np.abs(y_true - y_pred)))
        mse  = float(np.mean((y_true - y_pred) ** 2))
        rmse = float(np.sqrt(mse))
        r2   = r2_score(y_true, y_pred)

        return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}
