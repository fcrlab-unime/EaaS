import shutil
import json
import numpy as np
import pandas as pd
from pathlib import Path

import aidge_core
import aidge_backend_cpu
import aidge_export_cpp
import aidge_core.aidge_save
from aidge_core.export_utils import generate_main_cpp, scheduler_export, set_nodes_names
from aidge_core.export_utils.data_conversion import aidge2c, sanitize_cpp_identifier
from aidge_core.mem_info import generate_optimized_memory_info
from aidge_export_cpp import ExportLibCpp

_NB_BITS = 8
_RESCALING = 2 ** (_NB_BITS - 1) - 1  # 127


def _required_inputs(model):
    """Match generate_main_cpp: optional parameter ports are not UART inputs."""
    return [
        (node, index) for node, index in model.get_ordered_inputs()
        if not node.get_operator().is_optional_input(index)
    ]


def _materialize_scalar_shapes(model):
    """Export rank-zero activations as one-element vectors.

    Aidge's C++ emitter computes loop lengths from dimensions: [] can emit
    NB_ELTS=0 even though Tensor.size is 1 (notably Squeeze -> Dequantizer).
    Preserve the scalar value while making its one element explicit for the
    memory planner and kernels. Apply only after final dimension propagation.
    """
    promoted = 0
    for node in model.get_nodes():
        op = node.get_operator()
        for tensor in [*op.get_inputs(), *op.get_outputs()]:
            if tensor is not None and tensor.size == 1 and not tensor.dims:
                tensor.reshape([1])
                promoted += 1
    return promoted


def _validate_uart_interface(model):
    inputs = _required_inputs(model)
    outputs = model.get_ordered_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        input_names = [f"{node.name()}:{index}" for node, index in inputs]
        output_specs = [
            f"{node.name()}:{index} (size={node.get_operator().get_output(index).size})"
            for node, index in outputs
        ]
        raise ValueError(
            "SOH UART firmware requires one required input and one scalar output; "
            f"required inputs={input_names}, outputs={output_specs}, "
            f"total input ports including optional={len(model.get_ordered_inputs())}"
        )
    out_node, out_index = outputs[0]
    out_tensor = out_node.get_operator().get_output(out_index)
    if out_tensor.size != 1 or aidge2c(out_tensor.dtype) != "float":
        raise ValueError(
            "SOH output must be a dequantized float scalar; "
            f"{out_node.name()}:{out_index} has size={out_tensor.size}, dtype={out_tensor.dtype}"
        )


class ExportPipeline:
    def __init__(self, model_path: str, test_csv: str, label: str = None, deployment: dict = None):
        self.deployment = deployment
        self.model_path = model_path  # path WITHOUT .aidge extension
        self.test_csv = test_csv

        if label is None:
            df_header = pd.read_csv(test_csv, nrows=1)
            self.label = df_header.columns[-1]
        else:
            self.label = label

    def run(self, export_folder: Path) -> Path:
        model = aidge_core.aidge_save.load_model(self.model_path)

        df = pd.read_csv(self.test_csv)
        inputs = df.drop(columns=[self.label]).values
        n_features = inputs.shape[1]
        in_dims_ncw = [1, n_features, 1]
        
        model.set_backend("cpu")
        model.forward_dims(dims=[in_dims_ncw], allow_data_dependency=True)
        model.forward_dtype()

        # Scale first sample to int8 range (model is always int8-quantized)
        sample = np.round(inputs[0].reshape(in_dims_ncw) * _RESCALING).clip(-127, 127).astype(np.int8)
        input_tensor = aidge_core.Tensor(sample)
        input_tensor.to_backend("cpu")
        input_tensor.to_dtype(aidge_core.dtype.int8)

        for mop in ExportLibCpp._fusion_list:
            recipe = ExportLibCpp._fusion_registry[mop]
            aidge_core.fuse_to_metaops(
                model, recipe, mop, lambda g: g.set_attr_inputs_last()
            )

        scheduler = aidge_core.SequentialScheduler(model)
        scheduler.generate_scheduling()
        set_nodes_names(scheduler)

        # Dataformat corrente NCW
        model.set_dataformat(aidge_core.dformat.ncw)

        # Input: clone del campione, conversione NCW → NWC
        input_tensor.set_data_format(aidge_core.dformat.ncw)
        input_tensor.set_data_format(aidge_core.dformat.nwc)

        required_inputs = _required_inputs(model)
        if len(required_inputs) != 1:
            names = [f"{node.name()}:{index}" for node, index in required_inputs]
            raise ValueError(f"SOH UART requires one required input; found {names}")
        in_node, in_index = required_inputs[0]
        in_node.get_operator().set_input(in_index, input_tensor)

        # Tutto il grafo in NWC
        model.set_dataformat(aidge_core.dformat.nwc)

        # Imposta backend e adatta il grafo
        model.set_backend(ExportLibCpp._name)
        aidge_core.adapt_to_backend(model)
        aidge_core.adapt_fc_params_format(model)

        # Forward dims in NWC [N, W, C] dopo gli adapt
        model.forward_dims(dims=[[1, 1, n_features]], allow_data_dependency=True)

        # Avoid empty C++ loops for scalar outputs after Squeeze/dequantization.
        promoted = _materialize_scalar_shapes(model)
        if promoted:
            print(f"Materialized {promoted} scalar tensor shapes for C++ export")

        # Reset scheduler dopo le modifiche al grafo
        scheduler.reset_scheduling()
        scheduler.generate_scheduling()

        if export_folder.exists():
            shutil.rmtree(export_folder)

        scheduler_export(
            scheduler,
            export_folder,
            memory_manager=generate_optimized_memory_info,
            memory_manager_args={
                "stats_folder": str(export_folder / "stats"),
                "wrapping": False,
                "display_names": True,
            },
        )

        generate_main_cpp(export_folder, model, inputs_tensor=[input_tensor])

        if self.deployment is None:
            raise ValueError("Missing deployment metadata: retrain the model")
        _validate_uart_interface(model)
        if len(self.deployment["feature_names"]) != n_features:
            raise ValueError("Feature metadata does not match model input")
        graph_name = sanitize_cpp_identifier(model.name()) if model.name() else ""
        forward_name = f"{graph_name}_forward" if graph_name else "model_forward"
        forward_header = f"{graph_name}_forward.hpp" if graph_name else "forward.hpp"
        (export_folder / "model_adapter.hpp").write_text(
            '#pragma once\n#include <cstdint>\n'
            f'#include "{forward_header}"\n'
            'inline float model_soh(const int8_t* input) {\n'
            '    float* output = nullptr;\n'
            f'    {forward_name}(input, &output);\n'
            '    return output[0];\n}\n'
        )
        (export_folder / "deployment.json").write_text(json.dumps(self.deployment, allow_nan=False))

        return export_folder
