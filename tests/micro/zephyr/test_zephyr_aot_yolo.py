# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import io
import logging
import os
import sys
import logging
import pathlib
import shutil
import tarfile
import tempfile

import pytest
import numpy as np

from PIL import Image
import tvm
import tvm.testing
from tvm.micro.project_api import server
import tvm.relay as relay
from tvm.relay.backend import Executor, Runtime

from tvm.contrib.download import download_testdata
from tvm.micro.testing import aot_transport_init_wait, aot_transport_find_message

import test_utils
#_LOG = logging.getLogger(__name__)

@tvm.testing.requires_micro
def test_tflite(temp_dir, board, west_cmd, tvm_debug):
    """Testing a TFLite model."""
    model = test_utils.ZEPHYR_BOARDS[board]
    input_shape_name = "serving_default_input_1_0_int8"
    input_shape = (1,160, 160, 3)
    output_shape = (1, 1575,15)
    build_config = {"debug": tvm_debug}

    #sample_url = "https://github.com/tlc-pack/web-data/raw/967fc387dadb272c5a7f8c3461d34c060100dbf1/testdata      /microTVM/data/keyword_spotting_int8_6.pyc.npy"
    #sample_path = download_testdata(sample_url, "keyword_spotting_int8_6.pyc.npy", module="data")
    #sample = np.load(sample_path)
    #img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
    #img_path = download_testdata(img_url, "imagenet_cat.png", module="data")
    img_path = '/home/wei/model_file/chair.jpg'
    # Resize it to 160x160
    resized_image = Image.open(img_path).resize((160, 160))

    img_data = np.asarray(resized_image).astype("float")
    img_data = img_data-128
    img_data = np.asarray(img_data).astype("int8")

    #model_url = "https://github.com/tlc-pack/web-data/raw/25fe99fb00329a26bd37d3dca723da94316fd34c/testdata/microTVM/model/keyword_spotting_quant.tflite"
    #model_path = download_testdata(model_url, "keyword_spotting_quant.tflite", module="model")
    model_path= '/home/wei/model_file/exp113_160-int8.tflite'
    # Import TFLite model
    tflite_model_buf = open(model_path, "rb").read()
    try:
        import tflite

        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

    # Load TFLite model and convert to Relay
    relay_mod, params = relay.frontend.from_tflite(
        tflite_model, shape_dict={input_shape_name: input_shape}, dtype_dict={input_shape_name: "int8"}
    )

    target = tvm.target.target.micro(model)
    executor = Executor(
        "aot", {"unpacked-api": True, "interface-api": "c", "workspace-byte-alignment": 4}
    )
    runtime = Runtime("crt")
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        lowered = relay.build(relay_mod, target, params=params, runtime=runtime, executor=executor)

    
    #print(graph_mod.get_input_info())
    tensor_img_data = np.expand_dims(img_data, axis=0)

    temp_dir_path = '/home/roger/tvm/tvm_workspace'
    if os.path.isdir(temp_dir_path):
        shutil.rmtree(temp_dir_path)
    temp_dir=tvm.contrib.utils.TempDirectory(custom_path=temp_dir_path)
    temp_dir.set_keep_for_debug(True)   #not work, debugging

    project, _ = test_utils.generate_project(
        "yolov5",
        temp_dir,
        board,
        west_cmd,
        lowered,
        build_config,
        tensor_img_data,
        output_shape,
        "int8",
        load_cmsis=False,
    )

    result, time = test_utils.run_model(project)
    print("\ngot result:{0}, time:{1}\n".format(result,time))
    assert time > 0

'''
@tvm.testing.requires_micro
def test_qemu_make_fail(temp_dir, board, west_cmd, tvm_debug):
    """Testing QEMU make fail."""
    if board not in ["qemu_x86", "mps2_an521", "mps3_an547"]:
        pytest.skip(msg="Only for QEMU targets.")

    model = test_utils.ZEPHYR_BOARDS[board]
    build_config = {"debug": tvm_debug}
    shape = (10,)
    dtype = "float32"

    # Construct Relay program.
    x = relay.var("x", relay.TensorType(shape=shape, dtype=dtype))
    xx = relay.multiply(x, x)
    z = relay.add(xx, relay.const(np.ones(shape=shape, dtype=dtype)))
    func = relay.Function([x], z)
    ir_mod = tvm.IRModule.from_expr(func)

    target = tvm.target.target.micro(model)
    executor = Executor("aot")
    runtime = Runtime("crt")
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        lowered = relay.build(ir_mod, target, executor=executor, runtime=runtime)

    sample = np.zeros(shape=shape, dtype=dtype)
    project, project_dir = test_utils.generate_project(
        temp_dir, board, west_cmd, lowered, build_config, sample, shape, dtype, load_cmsis=False
    )

    file_path = (
        pathlib.Path(project_dir) / "build" / "zephyr" / "CMakeFiles" / "run.dir" / "build.make"
    )
    assert file_path.is_file(), f"[{file_path}] does not exist."

    # Remove a file to create make failure.
    os.remove(file_path)
    project.flash()
    with pytest.raises(server.JSONRPCError) as excinfo:
        project.transport().open()
    assert "QEMU setup failed" in str(excinfo.value)
'''

if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
