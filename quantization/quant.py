import numpy as np
import pandas as pd
import aidge_core
import aidge_core.aidge_save
import aidge_onnx
import aidge_backend_cpu
import aidge_quantization
import onnx
from aidge_export_cpp.export_utils import propagate


class PipelineConfig:
    def __init__(self, model_path, train_csv, label=None):
        
        self.model_path = model_path
        self.train_csv = train_csv
        self.input_shape = self._infer_input_shape()

        if label is None:
            df_header = pd.read_csv(train_csv, nrows=1)
            self.label = df_header.columns[-1]
        else:
            self.label = label

        
        
    def _infer_input_shape(self):
        model = onnx.load(self.model_path)
        dims = model.graph.input[0].type.tensor_type.shape.dim
        shape = tuple(
            dim.dim_value if dim.dim_value > 0 else 1
            for dim in dims
        )
        return shape
    
    def quant(self):
        
        aidge_model = aidge_onnx.load_onnx(self.model_path)
        aidge_core.remove_flatten(aidge_model)
        aidge_core.fuse_batchnorm(aidge_model)
        aidge_core.expand_metaops(aidge_model, name_format="{0}_{1}_{2}")
        aidge_model.set_datatype(aidge_core.dtype.float32)
        aidge_model.set_backend('cpu')
        
        df_calib = pd.read_csv(self.train_csv)
        inputs = df_calib.drop(columns=[self.label]).values
        
        calibration_set = []
        for i in range(min(6, len(inputs))):
            sample_np = inputs[i].reshape(self.input_shape)
            tensor = aidge_core.Tensor(np.array(sample_np, dtype=np.float32))
            tensor.set_backend('cpu')
            tensor.set_datatype(aidge_core.dtype.float32) 
            calibration_set.append(tensor)
        
        weight_precision = aidge_core.dtype.int8
        activation_precision = aidge_core.dtype.int8
        bias_precision = (
            aidge_core.dtype.int32
        )
        
        scheduler = aidge_core.SequentialScheduler(aidge_model)
        scheduler.generate_scheduling()

        for tensor in calibration_set:
            propagate(aidge_model, scheduler, [tensor])

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

        