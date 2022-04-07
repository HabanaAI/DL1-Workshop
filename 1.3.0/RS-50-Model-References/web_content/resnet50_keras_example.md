# TensorFlow Keras ResNet50 Example on Habana Gaudi<sup>TM</sup>

This Jupyter Notebook example demonstrates how to train Keras ResNet50 on Habana Gaudi<sup>TM</sup> device with TensorFlow framework. The neural network is built with Keras APIs, and trained with synthetic data.


```python
%pwd
```




    '/home/ubuntu/dl1_workshop/RN50'




```python
%ls
```

    [0m[01;34mModel-References[0m/             resnet50_keras_example.md
    habana_model_yaml_config.py   resnet50_keras_lars_bf16_1card.yaml
    resnet50_keras_example.ipynb


## Setup

Since we already have cloned Habana `Model-References` repository 1.3.0 branch to this DLAMI, now let's add it to PYTHONPATH.


```python
%set_env PYTHONPATH=/home/ubuntu/dl1_workshop/RN50/Model-References
```

    env: PYTHONPATH=/home/ubuntu/dl1_workshop/RN50/Model-References



```python
%cd Model-References/TensorFlow/computer_vision/Resnets/resnet_keras
```

    /home/ubuntu/dl1_workshop/RN50/Model-References/TensorFlow/computer_vision/Resnets/resnet_keras


Next, we need to install all the Python packages that ResNet50 depends on.


```python
!python3 -m pip install -r requirements.txt
```

    /usr/lib/python3/dist-packages/secretstorage/dhcrypto.py:15: CryptographyDeprecationWarning: int_from_bytes is deprecated, use int.from_bytes instead
      from cryptography.utils import int_from_bytes
    /usr/lib/python3/dist-packages/secretstorage/util.py:19: CryptographyDeprecationWarning: int_from_bytes is deprecated, use int.from_bytes instead
      from cryptography.utils import int_from_bytes
    Defaulting to user installation because normal site-packages is not writeable
    Requirement already satisfied: absl_py==0.11.0 in /home/ubuntu/.local/lib/python3.8/site-packages (from -r requirements.txt (line 1)) (0.11.0)
    Requirement already satisfied: PyYAML==5.4.1 in /usr/local/lib/python3.8/dist-packages (from -r requirements.txt (line 2)) (5.4.1)
    Requirement already satisfied: tensorflow_model_optimization==0.7.0 in /usr/local/lib/python3.8/dist-packages (from -r requirements.txt (line 3)) (0.7.0)
    Requirement already satisfied: requests==2.25.1 in /home/ubuntu/.local/lib/python3.8/site-packages (from -r requirements.txt (line 4)) (2.25.1)
    Requirement already satisfied: six in /usr/local/lib/python3.8/dist-packages (from absl_py==0.11.0->-r requirements.txt (line 1)) (1.16.0)
    Requirement already satisfied: numpy~=1.14 in /usr/local/lib/python3.8/dist-packages (from tensorflow_model_optimization==0.7.0->-r requirements.txt (line 3)) (1.22.2)
    Requirement already satisfied: dm-tree~=0.1.1 in /usr/local/lib/python3.8/dist-packages (from tensorflow_model_optimization==0.7.0->-r requirements.txt (line 3)) (0.1.6)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.8/dist-packages (from requests==2.25.1->-r requirements.txt (line 4)) (2021.10.8)
    Requirement already satisfied: chardet<5,>=3.0.2 in /usr/lib/python3/dist-packages (from requests==2.25.1->-r requirements.txt (line 4)) (3.0.4)
    Requirement already satisfied: idna<3,>=2.5 in /home/ubuntu/.local/lib/python3.8/site-packages (from requests==2.25.1->-r requirements.txt (line 4)) (2.10)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.8/dist-packages (from requests==2.25.1->-r requirements.txt (line 4)) (1.26.8)
    [33mWARNING: You are using pip version 22.0.3; however, version 22.0.4 is available.
    You should consider upgrading via the '/usr/bin/python3 -m pip install --upgrade pip' command.[0m[33m
    [0m

## Training on 1 HPU

After all the dependant Python packages are installed, let's launch ResNet50 training with LARS optimizer on a single HPU with synthetic data in BF16 data type.


```python
!python3 resnet_ctl_imagenet_main.py --dtype bf16 --data_loader_image_type bf16 --batch_size 256 --optimizer LARS --data_dir /home/ubuntu --use_synthetic_data True --train_steps 800 --steps_per_loop 100 --model_dir model_tmp --enable_tensorboard True --base_learning_rate 2.5 --warmup_epochs 3 --lr_schedule polynomial --label_smoothing 0.1 --weight_decay 0.0001 --single_l2_loss_op
```

    2022-04-07 20:58:58.495090: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/helpers/op_registry_backdoor.cpp:92] Couldn't find definition of RemoteCall:GPU: to register on HPU
    Using custom Horovod, path: /usr/local/lib/python3.8/dist-packages/horovod/tensorflow/mpi_lib.cpython-38-x86_64-linux-gnu.so
    2022-04-07 20:58:58.608626: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2022-04-07 20:59:00.670638: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/habana_device.cpp:201] HPU initialization done for library version 1.3.0_c61303b7_tf2.8.0
    INFO:tensorflow:Type of parameter "logger_levels" is not one of (bool, int, float, str). It will be saved as a string.
    I0407 20:59:00.681349 139785487877952 tb_utils.py:57] Type of parameter "logger_levels" is not one of (bool, int, float, str). It will be saved as a string.
    INFO:tensorflow:Type of parameter "profile_file" is not one of (bool, int, float, str). It will be saved as a string.
    I0407 20:59:00.681589 139785487877952 tb_utils.py:57] Type of parameter "profile_file" is not one of (bool, int, float, str). It will be saved as a string.
    INFO:tensorflow:Type of parameter "shuffle_seed" is not one of (bool, int, float, str). It will be saved as a string.
    I0407 20:59:00.681655 139785487877952 tb_utils.py:57] Type of parameter "shuffle_seed" is not one of (bool, int, float, str). It will be saved as a string.
    INFO:tensorflow:Type of parameter "random_flip_left_right_seed" is not one of (bool, int, float, str). It will be saved as a string.
    I0407 20:59:00.681705 139785487877952 tb_utils.py:57] Type of parameter "random_flip_left_right_seed" is not one of (bool, int, float, str). It will be saved as a string.
    INFO:tensorflow:Type of parameter "dump_config" is not one of (bool, int, float, str). It will be saved as a string.
    I0407 20:59:00.681756 139785487877952 tb_utils.py:57] Type of parameter "dump_config" is not one of (bool, int, float, str). It will be saved as a string.
    INFO:tensorflow:Type of parameter "ls" is not one of (bool, int, float, str). It will be saved as a string.
    I0407 20:59:00.681812 139785487877952 tb_utils.py:57] Type of parameter "ls" is not one of (bool, int, float, str). It will be saved as a string.
    INFO:tensorflow:Type of parameter "loss_scale" is not one of (bool, int, float, str). It will be saved as a string.
    I0407 20:59:00.681857 139785487877952 tb_utils.py:57] Type of parameter "loss_scale" is not one of (bool, int, float, str). It will be saved as a string.
    INFO:tensorflow:Type of parameter "ara" is not one of (bool, int, float, str). It will be saved as a string.
    I0407 20:59:00.681903 139785487877952 tb_utils.py:57] Type of parameter "ara" is not one of (bool, int, float, str). It will be saved as a string.
    INFO:tensorflow:Type of parameter "all_reduce_alg" is not one of (bool, int, float, str). It will be saved as a string.
    I0407 20:59:00.681949 139785487877952 tb_utils.py:57] Type of parameter "all_reduce_alg" is not one of (bool, int, float, str). It will be saved as a string.
    INFO:tensorflow:Type of parameter "gt_mode" is not one of (bool, int, float, str). It will be saved as a string.
    I0407 20:59:00.681995 139785487877952 tb_utils.py:57] Type of parameter "gt_mode" is not one of (bool, int, float, str). It will be saved as a string.
    INFO:tensorflow:Type of parameter "tf_gpu_thread_mode" is not one of (bool, int, float, str). It will be saved as a string.
    I0407 20:59:00.682041 139785487877952 tb_utils.py:57] Type of parameter "tf_gpu_thread_mode" is not one of (bool, int, float, str). It will be saved as a string.
    INFO:tensorflow:Type of parameter "datasets_num_private_threads" is not one of (bool, int, float, str). It will be saved as a string.
    I0407 20:59:00.682087 139785487877952 tb_utils.py:57] Type of parameter "datasets_num_private_threads" is not one of (bool, int, float, str). It will be saved as a string.
    INFO:tensorflow:Type of parameter "bti" is not one of (bool, int, float, str). It will be saved as a string.
    I0407 20:59:00.682136 139785487877952 tb_utils.py:57] Type of parameter "bti" is not one of (bool, int, float, str). It will be saved as a string.
    INFO:tensorflow:Type of parameter "benchmark_test_id" is not one of (bool, int, float, str). It will be saved as a string.
    I0407 20:59:00.682180 139785487877952 tb_utils.py:57] Type of parameter "benchmark_test_id" is not one of (bool, int, float, str). It will be saved as a string.
    INFO:tensorflow:Type of parameter "bld" is not one of (bool, int, float, str). It will be saved as a string.
    I0407 20:59:00.682226 139785487877952 tb_utils.py:57] Type of parameter "bld" is not one of (bool, int, float, str). It will be saved as a string.
    INFO:tensorflow:Type of parameter "benchmark_log_dir" is not one of (bool, int, float, str). It will be saved as a string.
    I0407 20:59:00.682271 139785487877952 tb_utils.py:57] Type of parameter "benchmark_log_dir" is not one of (bool, int, float, str). It will be saved as a string.
    INFO:tensorflow:Type of parameter "gp" is not one of (bool, int, float, str). It will be saved as a string.
    I0407 20:59:00.682317 139785487877952 tb_utils.py:57] Type of parameter "gp" is not one of (bool, int, float, str). It will be saved as a string.
    INFO:tensorflow:Type of parameter "gcp_project" is not one of (bool, int, float, str). It will be saved as a string.
    I0407 20:59:00.682367 139785487877952 tb_utils.py:57] Type of parameter "gcp_project" is not one of (bool, int, float, str). It will be saved as a string.
    INFO:tensorflow:Type of parameter "worker_hosts" is not one of (bool, int, float, str). It will be saved as a string.
    I0407 20:59:00.682416 139785487877952 tb_utils.py:57] Type of parameter "worker_hosts" is not one of (bool, int, float, str). It will be saved as a string.
    INFO:tensorflow:Type of parameter "profile_steps" is not one of (bool, int, float, str). It will be saved as a string.
    I0407 20:59:00.682469 139785487877952 tb_utils.py:57] Type of parameter "profile_steps" is not one of (bool, int, float, str). It will be saved as a string.
    INFO:tensorflow:Type of parameter "bf16_config_path" is not one of (bool, int, float, str). It will be saved as a string.
    I0407 20:59:00.682517 139785487877952 tb_utils.py:57] Type of parameter "bf16_config_path" is not one of (bool, int, float, str). It will be saved as a string.
    INFO:tensorflow:Type of parameter "global_seed" is not one of (bool, int, float, str). It will be saved as a string.
    I0407 20:59:00.682563 139785487877952 tb_utils.py:57] Type of parameter "global_seed" is not one of (bool, int, float, str). It will be saved as a string.
    INFO:tensorflow:Type of parameter "end_learning_rate" is not one of (bool, int, float, str). It will be saved as a string.
    I0407 20:59:00.682612 139785487877952 tb_utils.py:57] Type of parameter "end_learning_rate" is not one of (bool, int, float, str). It will be saved as a string.
    I0407 20:59:00.685592 139785487877952 resnet_ctl_imagenet_main.py:203] Training 1 epochs, each epoch has 800 steps, total steps: 800; Eval 195 steps
    2022-04-07 20:59:00.818756: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/habana_device_binding_iface.cpp:59] Found TensorFlow library with SHA256: 1f4e3d3c8f90c158c442f60b6b1fafd64cfb678fd7c4f954804e0ba91497c2a0
    WARNING:tensorflow:From /home/ubuntu/dl1_workshop/RN50/Model-References/TensorFlow/computer_vision/Resnets/resnet_keras/resnet_model.py:311: The name tf.keras.initializers.random_normal is deprecated. Please use tf.compat.v1.keras.initializers.random_normal instead.
    
    W0407 20:59:03.084089 139785487877952 module_wrapper.py:149] From /home/ubuntu/dl1_workshop/RN50/Model-References/TensorFlow/computer_vision/Resnets/resnet_keras/resnet_model.py:311: The name tf.keras.initializers.random_normal is deprecated. Please use tf.compat.v1.keras.initializers.random_normal instead.
    
    I0407 20:59:03.184811 139785487877952 controller.py:250] Train at step 0 of 800
    I0407 20:59:03.184937 139785487877952 controller.py:254] Entering training loop with 100 steps, at step 0 of 800
    WARNING:tensorflow:From /home/ubuntu/dl1_workshop/RN50/Model-References/TensorFlow/computer_vision/Resnets/resnet_keras/orbit/utils.py:144: StrategyBase.experimental_distribute_datasets_from_function (from tensorflow.python.distribute.distribute_lib) is deprecated and will be removed in a future version.
    Instructions for updating:
    rename to distribute_datasets_from_function
    W0407 20:59:03.185094 139785487877952 deprecation.py:337] From /home/ubuntu/dl1_workshop/RN50/Model-References/TensorFlow/computer_vision/Resnets/resnet_keras/orbit/utils.py:144: StrategyBase.experimental_distribute_datasets_from_function (from tensorflow.python.distribute.distribute_lib) is deprecated and will be removed in a future version.
    Instructions for updating:
    rename to distribute_datasets_from_function
    I0407 21:00:02.546813 139785487877952 keras_utils.py:129] TimeHistory: 59.31 seconds, 431.60 examples/second between steps 0 and 100
    I0407 21:00:02.564495 139785487877952 controller.py:223] step: 100        steps_per_second: 1.68        {'loss': 4.1837497, 'accuracy': 0.64}
    step: 100        steps_per_second: 1.68        {'loss': 4.1837497, 'accuracy': 0.64}
    I0407 21:00:02.570307 139785487877952 controller.py:254] Entering training loop with 100 steps, at step 100 of 800
    I0407 21:00:18.254487 139785487877952 keras_utils.py:129] TimeHistory: 15.68 seconds, 1632.50 examples/second between steps 100 and 200
    I0407 21:00:18.263204 139785487877952 controller.py:223] step: 200        steps_per_second: 6.37        {'loss': 1.0248437, 'accuracy': 1.0}
    step: 200        steps_per_second: 6.37        {'loss': 1.0248437, 'accuracy': 1.0}
    I0407 21:00:18.269143 139785487877952 controller.py:254] Entering training loop with 100 steps, at step 200 of 800
    I0407 21:00:33.953693 139785487877952 keras_utils.py:129] TimeHistory: 15.68 seconds, 1632.48 examples/second between steps 200 and 300
    I0407 21:00:33.962946 139785487877952 controller.py:223] step: 300        steps_per_second: 6.37        {'loss': 1.0163281, 'accuracy': 1.0}
    step: 300        steps_per_second: 6.37        {'loss': 1.0163281, 'accuracy': 1.0}
    I0407 21:00:33.969260 139785487877952 controller.py:254] Entering training loop with 100 steps, at step 300 of 800
    I0407 21:00:49.641870 139785487877952 keras_utils.py:129] TimeHistory: 15.67 seconds, 1633.75 examples/second between steps 300 and 400
    I0407 21:00:49.650967 139785487877952 controller.py:223] step: 400        steps_per_second: 6.37        {'loss': 1.015625, 'accuracy': 1.0}
    step: 400        steps_per_second: 6.37        {'loss': 1.015625, 'accuracy': 1.0}
    I0407 21:00:49.657361 139785487877952 controller.py:254] Entering training loop with 100 steps, at step 400 of 800
    I0407 21:01:05.341238 139785487877952 keras_utils.py:129] TimeHistory: 15.68 seconds, 1632.54 examples/second between steps 400 and 500
    I0407 21:01:05.351840 139785487877952 controller.py:223] step: 500        steps_per_second: 6.37        {'loss': 1.015625, 'accuracy': 1.0}
    step: 500        steps_per_second: 6.37        {'loss': 1.015625, 'accuracy': 1.0}
    I0407 21:01:05.358723 139785487877952 controller.py:254] Entering training loop with 100 steps, at step 500 of 800
    I0407 21:01:21.040486 139785487877952 keras_utils.py:129] TimeHistory: 15.68 seconds, 1632.79 examples/second between steps 500 and 600
    I0407 21:01:21.049683 139785487877952 controller.py:223] step: 600        steps_per_second: 6.37        {'loss': 1.015625, 'accuracy': 1.0}
    step: 600        steps_per_second: 6.37        {'loss': 1.015625, 'accuracy': 1.0}
    I0407 21:01:21.056286 139785487877952 controller.py:254] Entering training loop with 100 steps, at step 600 of 800
    I0407 21:01:36.739404 139785487877952 keras_utils.py:129] TimeHistory: 15.68 seconds, 1632.66 examples/second between steps 600 and 700
    I0407 21:01:36.750963 139785487877952 controller.py:223] step: 700        steps_per_second: 6.37        {'loss': 1.015625, 'accuracy': 1.0}
    step: 700        steps_per_second: 6.37        {'loss': 1.015625, 'accuracy': 1.0}
    I0407 21:01:36.758815 139785487877952 controller.py:254] Entering training loop with 100 steps, at step 700 of 800
    I0407 21:01:52.441301 139785487877952 keras_utils.py:129] TimeHistory: 15.68 seconds, 1632.73 examples/second between steps 700 and 800
    I0407 21:01:52.450360 139785487877952 controller.py:223] step: 800        steps_per_second: 6.37        {'loss': 1.0157031, 'accuracy': 1.0}
    step: 800        steps_per_second: 6.37        {'loss': 1.0157031, 'accuracy': 1.0}
    I0407 21:01:52.456601 139785487877952 controller.py:195] Start evaluation at step: 800
    I0407 21:02:09.123790 139785487877952 controller.py:223] step: 800        evaluation metric: {'test_loss': 8.687499, 'test_accuracy': 0.0}
    step: 800        evaluation metric: {'test_loss': 8.687499, 'test_accuracy': 0.0}
    I0407 21:02:09.132935 139785487877952 resnet_ctl_imagenet_main.py:288] Run stats:
    {'eval_loss': 8.687499, 'eval_acc': 0.0, 'train_loss': 1.0157031, 'train_acc': 1.0, 'step_timestamp_log': ['BatchTimestamp<batch_index: 0, timestamp: 1649365143.2317746>', 'BatchTimestamp<batch_index: 100, timestamp: 1649365202.546583>', 'BatchTimestamp<batch_index: 200, timestamp: 1649365218.2542198>', 'BatchTimestamp<batch_index: 300, timestamp: 1649365233.953403>', 'BatchTimestamp<batch_index: 400, timestamp: 1649365249.6416235>', 'BatchTimestamp<batch_index: 500, timestamp: 1649365265.3409731>', 'BatchTimestamp<batch_index: 600, timestamp: 1649365281.0402653>', 'BatchTimestamp<batch_index: 700, timestamp: 1649365296.739293>', 'BatchTimestamp<batch_index: 800, timestamp: 1649365312.441018>'], 'train_finish_time': 1649365329.1285412, 'avg_exp_per_second': 1210.2782600114292}
    2022-04-07 21:02:09.824618: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/habana_device.cpp:93] HabanaDevice (HPU) was not closed properly by TensorFlow. It can happen, when working with Keras or Eager Mode. Resulting log in dmesg: "user released device without removing its memory mappings" can be ignored.


From the logs above, we can see the training throughput for ResNet50 on 1 HPU is around 1634 examples/second.

To check the training results, we will load TensorBoard extension to this Jupyter Notebook and display the results in charts.



```python
%load_ext tensorboard
```

The following command assumes tfevents were dumped to `./model_tmp` directory as specified in `--model_dir` argument in the command above. Modify it accordingly if you use a different folder as `--model_dir`.


```python
%tensorboard --bind_all --logdir model_tmp
```


    Reusing TensorBoard on port 6006 (pid 19294), started 0:01:53 ago. (Use '!kill 19294' to kill it.)




<iframe id="tensorboard-frame-848524720a2d8fc1" width="100%" height="800" frameborder="0">
</iframe>
<script>
  (function() {
    const frame = document.getElementById("tensorboard-frame-848524720a2d8fc1");
    const url = new URL("/", window.location);
    const port = 6006;
    if (port) {
      url.port = port;
    }
    frame.src = url;
  })();
</script>




```python

```
