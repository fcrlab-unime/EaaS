import shutil
import numpy as np
import pandas as pd
from pathlib import Path

import aidge_core
import aidge_backend_cpu
import aidge_export_cpp
import aidge_core.aidge_save
from aidge_core.export_utils import generate_main_cpp, scheduler_export, set_nodes_names
from aidge_core.mem_info import generate_optimized_memory_info
from aidge_export_cpp import ExportLibCpp

_NB_BITS = 8
_RESCALING = 2 ** (_NB_BITS - 1) - 1  # 127


class ExportPipeline:
    def __init__(self, model_path: str, test_csv: str, label: str = None):
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
        sample = np.round(inputs[0].reshape(in_dims_ncw) * _RESCALING).astype(np.int8)
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

        in_node = model.get_ordered_inputs()[0]
        in_node[0].get_operator().set_input(0, input_tensor)

        # Tutto il grafo in NWC
        model.set_dataformat(aidge_core.dformat.nwc)

        # Imposta backend e adatta il grafo
        model.set_backend(ExportLibCpp._name)
        aidge_core.adapt_to_backend(model)
        aidge_core.adapt_fc_params_format(model)

        # Forward dims in NWC [N, W, C] dopo gli adapt
        model.forward_dims(dims=[[1, 1, n_features]], allow_data_dependency=True)

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

        return export_folder
