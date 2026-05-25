"""
ccn1d.py

Run this file to export a CCN1D using the Aidge CPP Export module.
"""

# Arguments
import argparse
import random
import subprocess

import shutil
import aidge_core.aidge_core
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from pathlib import Path

# Aidge Modules
import aidge_core
import aidge_backend_cpu
import aidge_export_cpp
import aidge_onnx
import aidge_quantization

from aidge_core.export_utils import generate_main_cpp, scheduler_export, str2aidge
from aidge_core.mem_info import generate_optimized_memory_info

from aidge_export_cpp.export_utils import *
from aidge_export_cpp import ExportLibCpp

from aidge_core.aidge_save import *


def compute_r2_precise(samples_fixed, labels_scaled, model, scheduler, scaler_y, scaling_factor):
    all_preds_scaled = []
    for i in range(len(samples_fixed)):
        out_scaled = propagate(model, scheduler, samples_fixed[i])
        all_preds_scaled.append(out_scaled.item() / scaling_factor)
    
    preds_orig = scaler_y.inverse_transform(np.array(all_preds_scaled).reshape(-1, 1))
    labels_orig = scaler_y.inverse_transform(np.array(labels_scaled).reshape(-1, 1))
    
    score = r2_score(labels_orig, preds_orig)
    return score, labels_orig, preds_orig

supported_types = ["float32", "int8"]

parser = argparse.ArgumentParser(
    description="Export the CCN1D model with the aidge_export_cpp module."
)
parser.add_argument(
    "--no_cuda",
    action="store_true",
    help="Disable USE_CUDA usage to perform inferences and training.",
)
parser.add_argument(
    "--dtype",
    type=str,
    choices=supported_types,
    default="float32",
    help="Specify the targeted datatype : [int8, float32]",
)
parser.add_argument(
    "-v",
    "--verbose",
    action="count",
    default=0,
    help=(
        "Set the verbosity level of the console output."
        "Use -v to increase verbosity, with the following levels in ascending ordern"
        "default WARN - Only warnings and higher (WARN, ERROR, FATAL) are displayed.n"
        "-v NOTICE - Notices and higher (NOTICE, WARN, ERROR, FATAL) are displayed.n"
        "-vv INFO - Informational messages and higher (INFO, NOTICE, WARN, ERROR, FATAL) are displayed.n"
        "-vvv DEBUG - All messages, including debug information, are displayed.n"
        "Available levels in descending order of severityn"
        "DEBUG < INFO < NOTICE < WARN < ERROR < FATAL."
    ),
)
args = parser.parse_args()

# CUDA Usage
USE_CUDA = not args.no_cuda

if USE_CUDA:
    aidge_core.Log.warn(
        "There is currently an error using the CUDA backend with this graph. "
        "Falling back to 'cpu' backend. "
    )
    # import aidge_backend_cuda

# backend = "cuda" if USE_CUDA else "cpu"
backend = "cpu"

# Setting Aidge verbose level
if args.verbose == 0:
    aidge_core.Log.set_console_level(aidge_core.Level.Error)
elif args.verbose == 1:
    aidge_core.Log.set_console_level(aidge_core.Level.Notice)
elif args.verbose == 2:
    aidge_core.Log.set_console_level(aidge_core.Level.Info)
elif args.verbose >= 3:
    aidge_core.Log.set_console_level(aidge_core.Level.Debug)

# ------------------------------------------------------------
# DOWNLOAD & LOAD THE MODEL
# ------------------------------------------------------------

# Load the model
file_path = "model_output.onnx"
model = aidge_onnx.load_onnx(file_path)

# --------------------------------------------------------------
# CLEAN THE MODEL
# --------------------------------------------------------------

aidge_core.remove_flatten(model)
aidge_core.fuse_batchnorm(model)
aidge_core.expand_metaops(model, name_format="{0}_{1}_{2}")

# --------------------------------------------------------------
# TEST THE MODEL
# --------------------------------------------------------------

# Create the samples

IN_DIMS = [1, 122, 1]
NB_TEST = 10  # XXX 100 - Example inferences
NB_CALIB = 6  # XXX 20  - Calibration set

df_test = pd.read_csv("test_scaled_results.csv")
inputs = df_test.drop(columns=["soh"]).values
labels_scaled = df_test["soh"].values  # scaled labels, never overwritten

tensors = []
for i, inp in enumerate(inputs):
    tensor = np.reshape(inp, IN_DIMS)
    tensor = aidge_core.Tensor(tensor)
    tensor.to_backend("cpu")
    tensor.set_datatype(aidge_core.dtype.float32)
    tensors.append(tensor)
    if i >= max(NB_TEST, NB_CALIB):
        break

# Set up the backend
model.set_datatype(
    aidge_core.dtype.float32
)  # Ensure that there is no float64 in the graph
model.set_backend(backend)

# Create the Scheduler
scheduler = aidge_core.SequentialScheduler(model)
scheduler.generate_scheduling()

# Run some examples
accuracy = 0
print("\n EXAMPLE INFERENCES :")
nb_valid = 0
for i in range(NB_TEST):
    output = propagate(model, scheduler, [tensors[i]])
    print(labels_scaled[i], " VS ", output, " -> ", abs(output - labels_scaled[i]))
    if abs(output - labels_scaled[i]) <= 10e-2:
        nb_valid += 1
accuracy = nb_valid / NB_TEST
print("\n MODEL ACCURACY = ", accuracy * 100, "%")

# --------------------------------------------------------------
# PERFORM THE QUANTIZATION
# --------------------------------------------------------------

if args.dtype != "float32":
    if args.dtype == "int8":
        act_precision = aidge_core.dtype.int8
        weight_precision = aidge_core.dtype.int8
        bias_precision = aidge_core.dtype.int32
        NB_BITS = 8
    else:
        aidge_core.Log.fatal(
            f"Type {args.dtype} not yet supported. \
                             \nSupported types : {supported_types}."
        )

    # Set the precision
    aidge_quantization.auto_assign_node_precision(
        graphview=model,
        act_precision=act_precision,
        weight_precision=weight_precision,
        bias_precision=bias_precision,
    )

    aidge_quantization.quantize_network(
        network=model,
        calibration_set=tensors[0:NB_CALIB],
        clipping_mode=aidge_quantization.Clipping.MAX,
        single_shift=True,
    )

# Set the backend back to CPU
if USE_CUDA:
    model.set_backend("cpu")
    for tensor in tensors:
        tensor.set_backend("cpu")

# --------------------------------------------------------------
# TEST THE QUANTIZED MODEL
# --------------------------------------------------------------

if args.dtype != "float32":

    # Quantize the samples
    rescaling = 2 ** (NB_BITS - 1) - 1
    for i in range(NB_TEST):
        tensors[i].to_backend("cpu")
        array = np.array(tensors[i]) * rescaling
        array = np.round(array).astype(int)
        tensors[i] = aidge_core.Tensor(array)
        tensors[i].to_dtype(str2aidge(args.dtype))

    # Upscale the labels to int8 scale for fair comparison
    labels_int8 = np.round(labels_scaled[:NB_TEST] * rescaling).astype(int)

    # The input has not been casted
    for node in model.get_ordered_inputs():
        node[0].get_operator().get_input(0).to_dtype(str2aidge(args.dtype))

    # Reset the scheduler as the graph may have changed
    scheduler.reset_scheduling()
    scheduler.generate_scheduling()

    # Perform example inferences on quantized model (compare in int8 scale)
    print("\n MODEL ACCURACY = ", accuracy * 100, "%")
    quant_accuracy = 0
    nb_valid = 0
    for i in range(NB_TEST):
        output = propagate(model, scheduler, [tensors[i]])
        print(labels_int8[i], " VS ", output, " -> ", abs(output - labels_int8[i]))
        if abs(output - labels_int8[i]) <= 10e-2:
            nb_valid += 1

    quant_accuracy = nb_valid / NB_TEST
    print("\n MODEL ACCURACY = ", accuracy * 100, "%")
    print("\n QUANTIZED ACCURACY = ", quant_accuracy * 100, "%")

    output_array = propagate(model, scheduler, [tensors[0]])

# --------------------------------------------------------------
# EXPORT THE MODEL
# --------------------------------------------------------------

export_folder_name = f"export_ccn1d_{args.dtype}"

save_model(model, "model")

model = load_model("model.aidge")
model.set_backend("cpu")

model.forward_dims(dims=[IN_DIMS], allow_data_dependency=True)
model.forward_dtype()

# Fuse nodes into MetaOps adapted to the CPP Export
cpp_fuse_to_metaops(model)

# Reset the scheduler after graph modification
scheduler = aidge_core.SequentialScheduler(model)
scheduler.generate_scheduling()

# Normalize nodes names (optional)
set_nodes_names(scheduler)

# Handle dataformat
## Set current dataformat
model.set_dataformat(aidge_core.dformat.ncw)

## Inputs
input_tensor = tensors[0].clone()
input_tensor.set_data_format(aidge_core.dformat.ncw)
input_tensor.set_data_format(aidge_core.dformat.nwc)
in_node = model.get_ordered_inputs()[0]
in_node[0].get_operator().set_input(0, input_tensor)

## Whole graph
model.set_dataformat(aidge_core.dformat.nwc)

# Set model's backend
model.set_backend(ExportLibCpp._name)

# Adapt the graph to the selected backend
aidge_core.adapt_to_backend(model)
aidge_core.adapt_fc_params_format(model)  # Not needed here

# Forward the dimensions
IN_DIMS_HWC = [[1, 1, 122]]
model.forward_dims(dims=IN_DIMS_HWC, allow_data_dependency=True)

# Reset the scheduling as the graph may have been changed
scheduler.reset_scheduling()
scheduler.generate_scheduling()

# Remove existing export
export_folder_name = Path(export_folder_name)
if os.path.isdir(export_folder_name):
    shutil.rmtree(export_folder_name)
    
#aidge_model_explorer.visualize(model, "model_export_visualization")

# Generate the export
scheduler_export(
    scheduler,
    export_folder_name,
    ExportLibCpp,
    memory_manager=generate_optimized_memory_info,
    memory_manager_args={
        "stats_folder": f"{export_folder_name}/stats",
        "wrapping": False,
        "display_names": True,
    },
)

# Generate main file
generate_main_cpp(export_folder_name, model, inputs_tensor=[input_tensor])

# --------------------------------------------------------------
# COMPILING THE EXPORT
# --------------------------------------------------------------

# print("\n### Compiling the export ###")
# try:
#     command = ["make", "SAVE_OUTPUTS=false"]
#     for std_line in aidge_core.utils.run_command(command, cwd=export_folder_name):
#         print(std_line, end="")
# except subprocess.CalledProcessError as e:
#     raise RuntimeError(0, f"An error occurred, failed to build export.") from e
# print("\n### Running the export ###")
# try:
#     for std_line in aidge_core.utils.run_command(
#         ["./bin/run_export"], cwd=export_folder_name
#     ):
#         print(std_line, end="")
# except subprocess.CalledProcessError as e:
#     raise RuntimeError(0, f"An error occurred, failed to run export.") from e