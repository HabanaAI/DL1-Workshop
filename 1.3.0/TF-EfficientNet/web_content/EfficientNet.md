# Migrating TensorFlow EfficientNet to Habana Gaudi<sup>TM</sup>

In this Jupyter notebook, we will learn how to migrate EfficientNet in public TensorFlow [models](https://github.com/tensorflow/models/tree/master/official/vision/image_classification) repository to Habana Gaudi<sup>TM</sup> device with very limited code changes. We will first verify the model can be trained on CPU. Then add code to the training script to enable it on a single HPU. Finally enable the distributed training on 8 HPUs with HPUStrategy.

### Setup

First of all, check the current directory to prepare for cloning TensorFlow model's repository.


```python
%pwd
```




    '/home/ubuntu/dl1_workshop/EfficientNet'



Then, we will clone TensorFlow [models](https://github.com/tensorflow/models.git)  repository v2.8.0 to the current directory.


```python
!git clone --depth 1 --branch v2.8.0 https://github.com/tensorflow/models.git   
```

    Cloning into 'models'...
    remote: Enumerating objects: 1636, done.[K
    remote: Counting objects: 100% (1636/1636), done.[K
    remote: Compressing objects: 100% (1400/1400), done.[K
    remote: Total 1636 (delta 486), reused 546 (delta 208), pack-reused 0[K
    Receiving objects: 100% (1636/1636), 2.37 MiB | 9.12 MiB/s, done.
    Resolving deltas: 100% (486/486), done.
    Note: switching to 'f5fc733a4e49c398ed22732b064abd33099a742f'.
    
    You are in 'detached HEAD' state. You can look around, make experimental
    changes and commit them, and you can discard any commits you make in this
    state without impacting any branches by switching back to a branch.
    
    If you want to create a new branch to retain commits you create, you may
    do so (now or later) by using -c with the switch command. Example:
    
      git switch -c <new-branch-name>
    
    Or undo this operation with:
    
      git switch -
    
    Turn off this advice by setting config variable advice.detachedHead to false
    


Verify if the repository was cloned successfully in the current location.


```python
%ls
```

    EfficientNet.ipynb  [0m[01;35menet_cpu_results.png[0m  [01;35menet_script.png[0m  [01;34mmodels[0m/


Check if the current PYTHONPATH contains TensorFlow `models` location. If not, add it. 

The following command assumes the models repository location is at `/home/ubuntu/dl1_workshop/EfficientNet` folder. Modify it accordingly if it is in a different location.


```python
%env PYTHONPATH
```

    UsageError: Environment does not have key: PYTHONPATH



```python
%set_env PYTHONPATH=/home/ubuntu/dl1_workshop/EfficientNet/models
```

    env: PYTHONPATH=/home/ubuntu/dl1_workshop/EfficientNet/models


Install the depedent packages for the model.


```python
!pip install gin-config tensorflow_addons tensorflow_datasets
```

    /usr/lib/python3/dist-packages/secretstorage/dhcrypto.py:15: CryptographyDeprecationWarning: int_from_bytes is deprecated, use int.from_bytes instead
      from cryptography.utils import int_from_bytes
    /usr/lib/python3/dist-packages/secretstorage/util.py:19: CryptographyDeprecationWarning: int_from_bytes is deprecated, use int.from_bytes instead
      from cryptography.utils import int_from_bytes
    Defaulting to user installation because normal site-packages is not writeable
    Requirement already satisfied: gin-config in /home/ubuntu/.local/lib/python3.8/site-packages (0.5.0)
    Requirement already satisfied: tensorflow_addons in /home/ubuntu/.local/lib/python3.8/site-packages (0.16.1)
    Requirement already satisfied: tensorflow_datasets in /home/ubuntu/.local/lib/python3.8/site-packages (4.5.2)
    Requirement already satisfied: typeguard>=2.7 in /home/ubuntu/.local/lib/python3.8/site-packages (from tensorflow_addons) (2.13.3)
    Requirement already satisfied: promise in /home/ubuntu/.local/lib/python3.8/site-packages (from tensorflow_datasets) (2.3)
    Requirement already satisfied: tensorflow-metadata in /home/ubuntu/.local/lib/python3.8/site-packages (from tensorflow_datasets) (1.7.0)
    Requirement already satisfied: requests>=2.19.0 in /home/ubuntu/.local/lib/python3.8/site-packages (from tensorflow_datasets) (2.25.1)
    Requirement already satisfied: importlib-resources in /usr/local/lib/python3.8/dist-packages (from tensorflow_datasets) (5.4.0)
    Requirement already satisfied: dill in /home/ubuntu/.local/lib/python3.8/site-packages (from tensorflow_datasets) (0.3.4)
    Requirement already satisfied: termcolor in /usr/local/lib/python3.8/dist-packages (from tensorflow_datasets) (1.1.0)
    Requirement already satisfied: six in /usr/local/lib/python3.8/dist-packages (from tensorflow_datasets) (1.16.0)
    Requirement already satisfied: absl-py in /home/ubuntu/.local/lib/python3.8/site-packages (from tensorflow_datasets) (0.11.0)
    Requirement already satisfied: protobuf>=3.12.2 in /usr/local/lib/python3.8/dist-packages (from tensorflow_datasets) (3.19.4)
    Requirement already satisfied: numpy in /usr/local/lib/python3.8/dist-packages (from tensorflow_datasets) (1.22.2)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.8/dist-packages (from tensorflow_datasets) (4.63.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.8/dist-packages (from requests>=2.19.0->tensorflow_datasets) (2021.10.8)
    Requirement already satisfied: idna<3,>=2.5 in /home/ubuntu/.local/lib/python3.8/site-packages (from requests>=2.19.0->tensorflow_datasets) (2.10)
    Requirement already satisfied: chardet<5,>=3.0.2 in /usr/lib/python3/dist-packages (from requests>=2.19.0->tensorflow_datasets) (3.0.4)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.8/dist-packages (from requests>=2.19.0->tensorflow_datasets) (1.26.8)
    Requirement already satisfied: zipp>=3.1.0 in /usr/local/lib/python3.8/dist-packages (from importlib-resources->tensorflow_datasets) (3.7.0)
    Requirement already satisfied: googleapis-common-protos<2,>=1.52.0 in /home/ubuntu/.local/lib/python3.8/site-packages (from tensorflow-metadata->tensorflow_datasets) (1.55.0)
    [33mWARNING: You are using pip version 22.0.3; however, version 22.0.4 is available.
    You should consider upgrading via the '/usr/bin/python3.8 -m pip install --upgrade pip' command.[0m[33m
    [0m

### Training on CPU

We will be using Keras EfficientNet at https://github.com/tensorflow/models/tree/master/official/vision/image_classification as the example to show how to enable a public model on Habana Gaudi device. 

EfficientNet is a convolutional neural network architecture and scaling method that uniformly scales all dimensions of depth/width/resolution using a compound coefficient. The model was first introduced by Tan et al. in [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946).  In this session, we are going to use EfficientNet baseline model EfficientNet-B0 as the training example.

First of all, let's enable the training with synthetic data on CPU and check its performance.


```python
%cd models/official/vision/image_classification
```

    /home/ubuntu/dl1_workshop/EfficientNet/models/official/vision/image_classification



```python
%ls
```

    README.md                        [0m[01;34mefficientnet[0m/
    __init__.py                      learning_rate.py
    augment.py                       learning_rate_test.py
    augment_test.py                  mnist_main.py
    callbacks.py                     mnist_test.py
    classifier_trainer.py            optimizer_factory.py
    classifier_trainer_test.py       optimizer_factory_test.py
    classifier_trainer_util_test.py  preprocessing.py
    [01;34mconfigs[0m/                         [01;34mresnet[0m/
    dataset_factory.py               test_utils.py


Let's first verify if EfficientNet can be run on CPU with the existing code from `models` repository.

In TensorFlow `models` repository, there are only EfficientNet configuration files for GPU and TPU under `configs` folder. We will use the following Python command to override the existing configurations for GPU and run EfficientNet-B0 on CPU:

```
python3 classifier_trainer.py --mode=train_and_eval --model_type=efficientnet --dataset=imagenet --model_dir=$HOME/log_cpu --data_dir=$HOME --config_file=configs/examples/efficientnet/imagenet/efficientnet-b0-gpu.yaml --params_override='runtime.num_gpus=0,runtime.distribution_strategy="off",train_dataset.builder="synthetic",validation_dataset.builder="synthetic",train.steps=1000,train.epochs=1,evaluation.skip_eval=True'
```


The Efficient-B0 training results on CPU look as below:

<img src="enet_cpu_results.png" alt="efficientnet_config" align="left" width="800"/>

From the output log above, we can see that the throughput for EfficientNet-B0 training on CPU with synthetic data is around `40 examples/sec`.

### Training on 1 HPU

Now, let's modify the traning script and enable the model on Habana Gaudi device with **BF16** data type.

Click [classifier_trainer.py](http://localhost:8888/edit/EfficientNet/models/official/vision/image_classification/classifier_trainer.py) link and insert the following 2 lines of code in **line 443**:

```
  from habana_frameworks.tensorflow import load_habana_module
  load_habana_module()
```
   
Save the file.


These 2 lines code will load Habana software modules in the beginning of training, and will aquire Habana Gaudi device and register the device to TensorFlow. This is all you need to do to enable EfficientNet on HPU.

Now, let's run the same command as above to launch the training. This time EfficientNet will be trained on a single HPU. 


```python
!TF_ENABLE_BF16_CONVERSION=1 python3 classifier_trainer.py --mode=train_and_eval --model_type=efficientnet --dataset=imagenet --model_dir=$HOME/log_hpu --data_dir=$HOME --config_file=configs/examples/efficientnet/imagenet/efficientnet-b0-gpu.yaml --params_override='runtime.num_gpus=0,runtime.distribution_strategy="off",train_dataset.builder="synthetic",validation_dataset.builder="synthetic",train.steps=1000,train.epochs=1,evaluation.skip_eval=True'
```

    I0407 21:16:01.275375 139725706671936 lib_utils.py:73] Trying to find libs in module directory..
    I0407 21:16:01.275523 139725706671936 lib_utils.py:79] Found libs in module dir /usr/local/lib/python3.8/dist-packages/habana_frameworks/tensorflow/tf2_8_0/lib/habanalabs
    I0407 21:16:03.003034 139725706671936 hook_init.py:46] hook config: False
    I0407 21:16:03.003205 139725706671936 library_loader.py:172] Loading Habana module from /usr/local/lib/python3.8/dist-packages/habana_frameworks/tensorflow/tf2_8_0/lib/habanalabs/habana_device.so.2.8.0
    2022-04-07 21:16:03.056441: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/helpers/op_registry_backdoor.cpp:92] Couldn't find definition of RemoteCall:GPU: to register on HPU
    I0407 21:16:03.058981 139725706671936 library_loader.py:175] Loading Habana as OpLibrary from /usr/local/lib/python3.8/dist-packages/habana_frameworks/tensorflow/tf2_8_0/lib/habanalabs/habana_device.so.2.8.0
    I0407 21:16:03.116391 139725706671936 library_loader.py:190] Successfully loaded Habana module
    I0407 21:16:03.118600 139725706671936 classifier_trainer.py:181] Base params: {'evaluation': {'epochs_between_evals': 1, 'skip_eval': False, 'steps': None},
     'export': {'checkpoint': None, 'destination': None},
     'mode': None,
     'model': {'learning_rate': {'boundaries': None,
                                 'decay_epochs': 2.4,
                                 'decay_rate': 0.97,
                                 'examples_per_epoch': None,
                                 'initial_lr': 0.008,
                                 'multipliers': None,
                                 'name': 'exponential',
                                 'scale_by_batch_size': 0.0078125,
                                 'staircase': True,
                                 'warmup_epochs': 5},
               'loss': {'label_smoothing': 0.1, 'name': 'categorical_crossentropy'},
               'model_params': {'model_name': 'efficientnet-b0',
                                'model_weights_path': '',
                                'overrides': {'activation': 'swish',
                                              'batch_norm': 'default',
                                              'dtype': 'float32',
                                              'num_classes': 1000,
                                              'rescale_input': True},
                                'weights_format': 'saved_model'},
               'name': 'EfficientNet',
               'num_classes': 1000,
               'optimizer': {'beta_1': None,
                             'beta_2': None,
                             'decay': 0.9,
                             'epsilon': 0.001,
                             'lookahead': None,
                             'momentum': 0.9,
                             'moving_average_decay': None,
                             'name': 'rmsprop',
                             'nesterov': None}},
     'model_dir': None,
     'model_name': None,
     'runtime': {'all_reduce_alg': None,
                 'batchnorm_spatial_persistent': False,
                 'dataset_num_private_threads': None,
                 'default_shard_dim': -1,
                 'distribution_strategy': 'mirrored',
                 'enable_xla': False,
                 'gpu_thread_mode': None,
                 'loss_scale': None,
                 'mixed_precision_dtype': None,
                 'num_cores_per_replica': 1,
                 'num_gpus': 0,
                 'num_packs': 1,
                 'per_gpu_thread_count': 0,
                 'run_eagerly': False,
                 'task_index': -1,
                 'tpu': None,
                 'tpu_enable_xla_dynamic_padder': None,
                 'worker_hosts': None},
     'train': {'callbacks': {'enable_backup_and_restore': False,
                             'enable_checkpoint_and_export': True,
                             'enable_tensorboard': True,
                             'enable_time_history': True},
               'epochs': 500,
               'metrics': ['accuracy', 'top_5'],
               'resume_checkpoint': True,
               'set_epoch_loop': False,
               'steps': None,
               'tensorboard': {'track_lr': True, 'write_model_weights': False},
               'time_history': {'log_steps': 100}},
     'train_dataset': {'augmenter': {'name': None, 'params': None},
                       'batch_size': 128,
                       'builder': 'records',
                       'cache': False,
                       'data_dir': None,
                       'download': False,
                       'dtype': 'float32',
                       'file_shuffle_buffer_size': 1024,
                       'filenames': None,
                       'image_size': 224,
                       'mean_subtract': False,
                       'name': 'imagenet2012',
                       'num_channels': 3,
                       'num_classes': 1000,
                       'num_devices': 1,
                       'num_examples': 1281167,
                       'one_hot': True,
                       'shuffle_buffer_size': 10000,
                       'skip_decoding': True,
                       'split': 'train',
                       'standardize': False,
                       'tf_data_service': None,
                       'use_per_replica_batch_size': True},
     'validation_dataset': {'augmenter': {'name': None, 'params': None},
                            'batch_size': 128,
                            'builder': 'records',
                            'cache': False,
                            'data_dir': None,
                            'download': False,
                            'dtype': 'float32',
                            'file_shuffle_buffer_size': 1024,
                            'filenames': None,
                            'image_size': 224,
                            'mean_subtract': False,
                            'name': 'imagenet2012',
                            'num_channels': 3,
                            'num_classes': 1000,
                            'num_devices': 1,
                            'num_examples': 1281167,
                            'one_hot': True,
                            'shuffle_buffer_size': 10000,
                            'skip_decoding': True,
                            'split': 'validation',
                            'standardize': False,
                            'tf_data_service': None,
                            'use_per_replica_batch_size': True}}
    I0407 21:16:03.118776 139725706671936 classifier_trainer.py:184] Overriding params: configs/examples/efficientnet/imagenet/efficientnet-b0-gpu.yaml
    I0407 21:16:03.123128 139725706671936 classifier_trainer.py:184] Overriding params: runtime.num_gpus=0,runtime.distribution_strategy="off",train_dataset.builder="synthetic",validation_dataset.builder="synthetic",train.steps=1000,train.epochs=1,evaluation.skip_eval=True
    I0407 21:16:03.124345 139725706671936 classifier_trainer.py:184] Overriding params: {'model_dir': '/home/ubuntu/log_hpu', 'mode': 'train_and_eval', 'model': {'name': 'efficientnet'}, 'runtime': {'run_eagerly': None, 'tpu': None}, 'train_dataset': {'data_dir': '/home/ubuntu'}, 'validation_dataset': {'data_dir': '/home/ubuntu'}, 'train': {'time_history': {'log_steps': 100}}}
    I0407 21:16:03.125562 139725706671936 classifier_trainer.py:190] Final model parameters: {'evaluation': {'epochs_between_evals': 1, 'skip_eval': True, 'steps': None},
     'export': {'checkpoint': None, 'destination': None},
     'mode': 'train_and_eval',
     'model': {'learning_rate': {'boundaries': None,
                                 'decay_epochs': 2.4,
                                 'decay_rate': 0.97,
                                 'examples_per_epoch': None,
                                 'initial_lr': 0.008,
                                 'multipliers': None,
                                 'name': 'exponential',
                                 'scale_by_batch_size': 0.0078125,
                                 'staircase': True,
                                 'warmup_epochs': 5},
               'loss': {'label_smoothing': 0.1, 'name': 'categorical_crossentropy'},
               'model_params': {'model_name': 'efficientnet-b0',
                                'model_weights_path': '',
                                'overrides': {'activation': 'swish',
                                              'batch_norm': 'default',
                                              'dtype': 'float32',
                                              'num_classes': 1000,
                                              'rescale_input': True},
                                'weights_format': 'saved_model'},
               'name': 'efficientnet',
               'num_classes': 1000,
               'optimizer': {'beta_1': None,
                             'beta_2': None,
                             'decay': 0.9,
                             'epsilon': 0.001,
                             'lookahead': False,
                             'momentum': 0.9,
                             'moving_average_decay': 0.0,
                             'name': 'rmsprop',
                             'nesterov': None}},
     'model_dir': '/home/ubuntu/log_hpu',
     'model_name': None,
     'runtime': {'all_reduce_alg': None,
                 'batchnorm_spatial_persistent': False,
                 'dataset_num_private_threads': None,
                 'default_shard_dim': -1,
                 'distribution_strategy': 'off',
                 'enable_xla': False,
                 'gpu_thread_mode': None,
                 'loss_scale': None,
                 'mixed_precision_dtype': None,
                 'num_cores_per_replica': 1,
                 'num_gpus': 0,
                 'num_packs': 1,
                 'per_gpu_thread_count': 0,
                 'run_eagerly': None,
                 'task_index': -1,
                 'tpu': None,
                 'tpu_enable_xla_dynamic_padder': None,
                 'worker_hosts': None},
     'train': {'callbacks': {'enable_backup_and_restore': False,
                             'enable_checkpoint_and_export': True,
                             'enable_tensorboard': True,
                             'enable_time_history': True},
               'epochs': 1,
               'metrics': ['accuracy', 'top_5'],
               'resume_checkpoint': True,
               'set_epoch_loop': False,
               'steps': 1000,
               'tensorboard': {'track_lr': True, 'write_model_weights': False},
               'time_history': {'log_steps': 100}},
     'train_dataset': {'augmenter': {'name': 'autoaugment', 'params': None},
                       'batch_size': 32,
                       'builder': 'synthetic',
                       'cache': False,
                       'data_dir': '/home/ubuntu',
                       'download': False,
                       'dtype': 'float32',
                       'file_shuffle_buffer_size': 1024,
                       'filenames': None,
                       'image_size': 224,
                       'mean_subtract': False,
                       'name': 'imagenet2012',
                       'num_channels': 3,
                       'num_classes': 1000,
                       'num_devices': 1,
                       'num_examples': 1281167,
                       'one_hot': True,
                       'shuffle_buffer_size': 10000,
                       'skip_decoding': True,
                       'split': 'train',
                       'standardize': False,
                       'tf_data_service': None,
                       'use_per_replica_batch_size': True},
     'validation_dataset': {'augmenter': {'name': None, 'params': None},
                            'batch_size': 32,
                            'builder': 'synthetic',
                            'cache': False,
                            'data_dir': '/home/ubuntu',
                            'download': False,
                            'dtype': 'float32',
                            'file_shuffle_buffer_size': 1024,
                            'filenames': None,
                            'image_size': 224,
                            'mean_subtract': False,
                            'name': 'imagenet2012',
                            'num_channels': 3,
                            'num_classes': 1000,
                            'num_devices': 1,
                            'num_examples': 50000,
                            'one_hot': True,
                            'shuffle_buffer_size': 10000,
                            'skip_decoding': True,
                            'split': 'validation',
                            'standardize': False,
                            'tf_data_service': None,
                            'use_per_replica_batch_size': True}}
    I0407 21:16:03.125736 139725706671936 classifier_trainer.py:290] Running train and eval.
    I0407 21:16:03.126449 139725706671936 classifier_trainer.py:304] Detected 1 devices.
    W0407 21:16:03.126650 139725706671936 classifier_trainer.py:105] label_smoothing > 0, so datasets will be one hot encoded.
    I0407 21:16:03.126820 139725706671936 dataset_factory.py:175] Using augmentation: autoaugment
    I0407 21:16:03.126995 139725706671936 dataset_factory.py:175] Using augmentation: None
    I0407 21:16:03.127126 139725706671936 dataset_factory.py:383] Generating a synthetic dataset.
    2022-04-07 21:16:03.127573: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2022-04-07 21:16:05.189241: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/habana_device.cpp:201] HPU initialization done for library version 1.3.0_c61303b7_tf2.8.0
    2022-04-07 21:16:07.216702: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/habana_device_binding_iface.cpp:59] Found TensorFlow library with SHA256: 1f4e3d3c8f90c158c442f60b6b1fafd64cfb678fd7c4f954804e0ba91497c2a0
    I0407 21:16:09.539637 139725706671936 dataset_factory.py:383] Generating a synthetic dataset.
    I0407 21:16:09.707978 139725706671936 classifier_trainer.py:325] Global batch size: 32
    I0407 21:16:09.757055 139725706671936 efficientnet_model.py:147] round_filter input=32 output=32
    I0407 21:16:09.824881 139725706671936 efficientnet_model.py:147] round_filter input=32 output=32
    I0407 21:16:09.824984 139725706671936 efficientnet_model.py:147] round_filter input=16 output=16
    I0407 21:16:09.953762 139725706671936 efficientnet_model.py:147] round_filter input=16 output=16
    I0407 21:16:09.953871 139725706671936 efficientnet_model.py:147] round_filter input=24 output=24
    I0407 21:16:10.424198 139725706671936 efficientnet_model.py:147] round_filter input=24 output=24
    I0407 21:16:10.424320 139725706671936 efficientnet_model.py:147] round_filter input=40 output=40
    I0407 21:16:10.748678 139725706671936 efficientnet_model.py:147] round_filter input=40 output=40
    I0407 21:16:10.748804 139725706671936 efficientnet_model.py:147] round_filter input=80 output=80
    I0407 21:16:11.162900 139725706671936 efficientnet_model.py:147] round_filter input=80 output=80
    I0407 21:16:11.163027 139725706671936 efficientnet_model.py:147] round_filter input=112 output=112
    I0407 21:16:11.583560 139725706671936 efficientnet_model.py:147] round_filter input=112 output=112
    I0407 21:16:11.583703 139725706671936 efficientnet_model.py:147] round_filter input=192 output=192
    I0407 21:16:12.150912 139725706671936 efficientnet_model.py:147] round_filter input=192 output=192
    I0407 21:16:12.151034 139725706671936 efficientnet_model.py:147] round_filter input=320 output=320
    I0407 21:16:12.303999 139725706671936 efficientnet_model.py:147] round_filter input=1280 output=1280
    I0407 21:16:12.454520 139725706671936 efficientnet_model.py:457] Building model efficientnet with params ModelConfig(width_coefficient=1.0, depth_coefficient=1.0, resolution=224, dropout_rate=0.2, blocks=(BlockConfig(input_filters=32, output_filters=16, kernel_size=3, num_repeat=1, expand_ratio=1, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=16, output_filters=24, kernel_size=3, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=24, output_filters=40, kernel_size=5, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=40, output_filters=80, kernel_size=3, num_repeat=3, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=80, output_filters=112, kernel_size=5, num_repeat=3, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=112, output_filters=192, kernel_size=5, num_repeat=4, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=192, output_filters=320, kernel_size=3, num_repeat=1, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise')), stem_base_filters=32, top_base_filters=1280, activation='swish', batch_norm='default', bn_momentum=0.99, bn_epsilon=0.001, weight_decay=5e-06, drop_connect_rate=0.2, depth_divisor=8, min_depth=None, use_se=True, input_channels=3, num_classes=1000, model_name='efficientnet', rescale_input=True, data_format='channels_last', dtype='float32')
    I0407 21:16:12.466073 139725706671936 optimizer_factory.py:147] Scaling the learning rate based on the batch size multiplier. New base_lr: 0.002000
    I0407 21:16:12.466176 139725706671936 optimizer_factory.py:152] Using exponential learning rate with: initial_learning_rate: 0.002000, decay_steps: 2400, decay_rate: 0.970000
    I0407 21:16:12.466245 139725706671936 optimizer_factory.py:177] Applying 5000 warmup steps to the learning rate
    I0407 21:16:12.466304 139725706671936 optimizer_factory.py:59] Building rmsprop optimizer with params {'name': 'rmsprop', 'decay': 0.9, 'epsilon': 0.001, 'momentum': 0.9, 'nesterov': None, 'moving_average_decay': 0.0, 'lookahead': False, 'beta_1': None, 'beta_2': None}
    I0407 21:16:12.466347 139725706671936 optimizer_factory.py:74] Using RMSProp
    I0407 21:16:12.490724 139725706671936 classifier_trainer.py:210] Load from checkpoint is enabled.
    I0407 21:16:12.490918 139725706671936 classifier_trainer.py:212] latest_checkpoint: None
    I0407 21:16:12.490975 139725706671936 classifier_trainer.py:214] No checkpoint detected.
    I0407 21:16:12.491716 139725706671936 classifier_trainer.py:281] Saving experiment configuration to /home/ubuntu/log_hpu/params.yaml
    I0407 21:16:58.541527 139725706671936 keras_utils.py:145] TimeHistory: 43.91 seconds, 72.87 examples/second between steps 0 and 100
    I0407 21:17:05.025709 139725706671936 keras_utils.py:145] TimeHistory: 6.46 seconds, 495.19 examples/second between steps 100 and 200
    I0407 21:17:11.425594 139725706671936 keras_utils.py:145] TimeHistory: 6.39 seconds, 500.52 examples/second between steps 200 and 300
    I0407 21:17:17.731134 139725706671936 keras_utils.py:145] TimeHistory: 6.30 seconds, 507.94 examples/second between steps 300 and 400
    I0407 21:17:24.077020 139725706671936 keras_utils.py:145] TimeHistory: 6.34 seconds, 504.80 examples/second between steps 400 and 500
    I0407 21:17:30.467428 139725706671936 keras_utils.py:145] TimeHistory: 6.38 seconds, 501.21 examples/second between steps 500 and 600
    I0407 21:17:36.847480 139725706671936 keras_utils.py:145] TimeHistory: 6.37 seconds, 502.02 examples/second between steps 600 and 700
    I0407 21:17:43.176887 139725706671936 keras_utils.py:145] TimeHistory: 6.32 seconds, 505.97 examples/second between steps 700 and 800
    I0407 21:17:49.563874 139725706671936 keras_utils.py:145] TimeHistory: 6.38 seconds, 501.50 examples/second between steps 800 and 900
    I0407 21:17:55.969251 139725706671936 keras_utils.py:145] TimeHistory: 6.40 seconds, 500.01 examples/second between steps 900 and 1000
    
    Epoch 1: saving model to /home/ubuntu/log_hpu/model.ckpt-0001
    1000/1000 - 103s - loss: 1.9275 - accuracy: 0.9511 - top_5_accuracy: 0.9569 - 103s/epoch - 103ms/step
    I0407 21:17:57.434003 139725706671936 classifier_trainer.py:448] Run stats:
    {'loss': 1.9274998903274536, 'training_accuracy_top_1': 0.9510936737060547, 'step_timestamp_log': ['BatchTimestamp<batch_index: 0, timestamp: 1649366174.6281512>', 'BatchTimestamp<batch_index: 100, timestamp: 1649366218.5413356>', 'BatchTimestamp<batch_index: 200, timestamp: 1649366225.025531>', 'BatchTimestamp<batch_index: 300, timestamp: 1649366231.4254134>', 'BatchTimestamp<batch_index: 400, timestamp: 1649366237.7309623>', 'BatchTimestamp<batch_index: 500, timestamp: 1649366244.0769644>', 'BatchTimestamp<batch_index: 600, timestamp: 1649366250.4672656>', 'BatchTimestamp<batch_index: 700, timestamp: 1649366256.8474197>', 'BatchTimestamp<batch_index: 800, timestamp: 1649366263.1767197>', 'BatchTimestamp<batch_index: 900, timestamp: 1649366269.5638137>', 'BatchTimestamp<batch_index: 1000, timestamp: 1649366275.9692109>'], 'train_finish_time': 1649366277.3592837, 'avg_exp_per_second': 311.4914190772545}
    2022-04-07 21:17:58.424668: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/habana_device.cpp:93] HabanaDevice (HPU) was not closed properly by TensorFlow. It can happen, when working with Keras or Eager Mode. Resulting log in dmesg: "user released device without removing its memory mappings" can be ignored.


From the output log above, we can see that the throughput for EfficientNet-B0 training on Habana Gaudi with synthetic data is around `500 examples/sec`.

### Distributed Training on 8 HPUs 

Now, let's enable the distributed training for EfficientNet on 8 HPUs of DLAMI.

In the original source code, `tf.distribute.Strategy` is used to support the distributed training for TPU and GPU. We will re-use this architecture and enable the distributed training on multi-HPUs with `HPUStrategy`. `HPUStrategy` has the same usage model as `MultiWorkerMirroredStrategy`, in which each worker runs in a separate process, with a single Gaudi device acquired.

* According to our [collateral](https://docs.habana.ai/en/latest/Tensorflow_Scaling_Guide/TensorFlow_Gaudi_Scaling_Guide.html#multi-worker-training-using-hpustrategy), we will first construct HPUStrategy instance when `distribution_strategy` parameter is set to `hpu`.

    Click [models/official/common/distribute_utils.py](
http://localhost:8888/edit/EfficientNet/models/official/common/distribute_utils.py) and in **line 148**, insert the following code:

    ```
    if distribution_strategy == "hpu":
      from habana_frameworks.tensorflow.distribute import HPUStrategy
      return HPUStrategy()
    ```
    
    And save the file. 

* Then we will configure `TF_CONFIG` environment variable by re-using the existing `distribute_utils.configure_cluster()` function in the code:
  
  Click [classifier_trainer.py](http://localhost:8888/edit/EfficientNet/models/official/vision/image_classification/classifier_trainer.py) link and replace **line 292 and 293** with following code:

  ```
    if params.runtime.distribution_strategy == 'hpu':
      hls_addresses = ["127.0.0.1"]
      TF_BASE_PORT = 2410
      from habana_frameworks.tensorflow.multinode_helpers import comm_size, comm_rank
      mpi_rank = comm_rank()
      mpi_size = comm_size()

      worker_hosts = ",".join([",".join([address + ':' + str(TF_BASE_PORT + rank)
                                         for rank in range(mpi_size // len(hls_addresses))])
                               for address in hls_addresses])
      task_index = mpi_rank
      distribute_utils.configure_cluster(worker_hosts, task_index)
    else:
      distribute_utils.configure_cluster(params.runtime.worker_hosts,
                                         params.runtime.task_index)
  ```

    Save the file.

Now we will launch 8 processes with mpirun command to start the distributed training for EfficientNet on 8 HPUs with `HPUStrategy` with command:

```
mpirun --allow-run-as-root -np 8 -x TF_ENABLE_BF16_CONVERSION=1 python3 classifier_trainer.py --mode=train_and_eval --model_type=efficientnet --dataset=imagenet --model_dir=$HOME/log_hpu_8 --data_dir=$HOME --config_file=configs/examples/efficientnet/imagenet/efficientnet-b0-gpu.yaml --params_override='runtime.num_gpus=0,runtime.distribution_strategy="hpu",train_dataset.builder="synthetic",validation_dataset.builder="synthetic",train.steps=1000,train.epochs=1,evaluation.skip_eval=True'

```

Run the following command:


```python
!mpirun --allow-run-as-root -np 8 -x TF_ENABLE_BF16_CONVERSION=1 python3 classifier_trainer.py --mode=train_and_eval --model_type=efficientnet --dataset=imagenet --model_dir=$HOME/log_hpu_8 --data_dir=$HOME --config_file=configs/examples/efficientnet/imagenet/efficientnet-b0-gpu.yaml --params_override='runtime.num_gpus=0,runtime.distribution_strategy="hpu",train_dataset.builder="synthetic",validation_dataset.builder="synthetic",train.steps=1000,train.epochs=1,evaluation.skip_eval=True'

```

    I0407 21:19:46.092814 140136124327744 lib_utils.py:73] Trying to find libs in module directory..
    I0407 21:19:46.092949 140136124327744 lib_utils.py:79] Found libs in module dir /usr/local/lib/python3.8/dist-packages/habana_frameworks/tensorflow/tf2_8_0/lib/habanalabs
    I0407 21:19:46.093238 140189656872768 lib_utils.py:73] Trying to find libs in module directory..
    I0407 21:19:46.093368 140189656872768 lib_utils.py:79] Found libs in module dir /usr/local/lib/python3.8/dist-packages/habana_frameworks/tensorflow/tf2_8_0/lib/habanalabs
    I0407 21:19:46.136298 140201244120896 lib_utils.py:73] Trying to find libs in module directory..
    I0407 21:19:46.136441 140201244120896 lib_utils.py:79] Found libs in module dir /usr/local/lib/python3.8/dist-packages/habana_frameworks/tensorflow/tf2_8_0/lib/habanalabs
    I0407 21:19:46.138075 139742514054976 lib_utils.py:73] Trying to find libs in module directory..
    I0407 21:19:46.138209 139742514054976 lib_utils.py:79] Found libs in module dir /usr/local/lib/python3.8/dist-packages/habana_frameworks/tensorflow/tf2_8_0/lib/habanalabs
    I0407 21:19:46.144408 140463173154624 lib_utils.py:73] Trying to find libs in module directory..
    I0407 21:19:46.144545 140463173154624 lib_utils.py:79] Found libs in module dir /usr/local/lib/python3.8/dist-packages/habana_frameworks/tensorflow/tf2_8_0/lib/habanalabs
    I0407 21:19:46.146926 140114967496512 lib_utils.py:73] Trying to find libs in module directory..
    I0407 21:19:46.147063 140114967496512 lib_utils.py:79] Found libs in module dir /usr/local/lib/python3.8/dist-packages/habana_frameworks/tensorflow/tf2_8_0/lib/habanalabs
    I0407 21:19:46.166253 140034574755648 lib_utils.py:73] Trying to find libs in module directory..
    I0407 21:19:46.166393 140034574755648 lib_utils.py:79] Found libs in module dir /usr/local/lib/python3.8/dist-packages/habana_frameworks/tensorflow/tf2_8_0/lib/habanalabs
    I0407 21:19:46.170097 139785568282432 lib_utils.py:73] Trying to find libs in module directory..
    I0407 21:19:46.170238 139785568282432 lib_utils.py:79] Found libs in module dir /usr/local/lib/python3.8/dist-packages/habana_frameworks/tensorflow/tf2_8_0/lib/habanalabs
    I0407 21:19:48.087131 140189656872768 hook_init.py:46] hook config: False
    I0407 21:19:48.087327 140189656872768 library_loader.py:172] Loading Habana module from /usr/local/lib/python3.8/dist-packages/habana_frameworks/tensorflow/tf2_8_0/lib/habanalabs/habana_device.so.2.8.0
    I0407 21:19:48.113661 140136124327744 hook_init.py:46] hook config: False
    I0407 21:19:48.113834 140136124327744 library_loader.py:172] Loading Habana module from /usr/local/lib/python3.8/dist-packages/habana_frameworks/tensorflow/tf2_8_0/lib/habanalabs/habana_device.so.2.8.0
    2022-04-07 21:19:48.140668: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/helpers/op_registry_backdoor.cpp:92] Couldn't find definition of RemoteCall:GPU: to register on HPU
    I0407 21:19:48.143313 140189656872768 library_loader.py:175] Loading Habana as OpLibrary from /usr/local/lib/python3.8/dist-packages/habana_frameworks/tensorflow/tf2_8_0/lib/habanalabs/habana_device.so.2.8.0
    2022-04-07 21:19:48.169207: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/helpers/op_registry_backdoor.cpp:92] Couldn't find definition of RemoteCall:GPU: to register on HPU
    I0407 21:19:48.171727 140136124327744 library_loader.py:175] Loading Habana as OpLibrary from /usr/local/lib/python3.8/dist-packages/habana_frameworks/tensorflow/tf2_8_0/lib/habanalabs/habana_device.so.2.8.0
    I0407 21:19:48.174610 140463173154624 hook_init.py:46] hook config: False
    I0407 21:19:48.174779 140463173154624 library_loader.py:172] Loading Habana module from /usr/local/lib/python3.8/dist-packages/habana_frameworks/tensorflow/tf2_8_0/lib/habanalabs/habana_device.so.2.8.0
    I0407 21:19:48.175093 140201244120896 hook_init.py:46] hook config: False
    I0407 21:19:48.175260 140201244120896 library_loader.py:172] Loading Habana module from /usr/local/lib/python3.8/dist-packages/habana_frameworks/tensorflow/tf2_8_0/lib/habanalabs/habana_device.so.2.8.0
    I0407 21:19:48.176092 139742514054976 hook_init.py:46] hook config: False
    I0407 21:19:48.176156 140114967496512 hook_init.py:46] hook config: False
    I0407 21:19:48.176255 139742514054976 library_loader.py:172] Loading Habana module from /usr/local/lib/python3.8/dist-packages/habana_frameworks/tensorflow/tf2_8_0/lib/habanalabs/habana_device.so.2.8.0
    I0407 21:19:48.176321 140114967496512 library_loader.py:172] Loading Habana module from /usr/local/lib/python3.8/dist-packages/habana_frameworks/tensorflow/tf2_8_0/lib/habanalabs/habana_device.so.2.8.0
    I0407 21:19:48.180219 139785568282432 hook_init.py:46] hook config: False
    I0407 21:19:48.180395 139785568282432 library_loader.py:172] Loading Habana module from /usr/local/lib/python3.8/dist-packages/habana_frameworks/tensorflow/tf2_8_0/lib/habanalabs/habana_device.so.2.8.0
    I0407 21:19:48.181287 140034574755648 hook_init.py:46] hook config: False
    I0407 21:19:48.181452 140034574755648 library_loader.py:172] Loading Habana module from /usr/local/lib/python3.8/dist-packages/habana_frameworks/tensorflow/tf2_8_0/lib/habanalabs/habana_device.so.2.8.0
    I0407 21:19:48.208048 140189656872768 library_loader.py:190] Successfully loaded Habana module
    I0407 21:19:48.210324 140189656872768 classifier_trainer.py:181] Base params: {'evaluation': {'epochs_between_evals': 1, 'skip_eval': False, 'steps': None},
     'export': {'checkpoint': None, 'destination': None},
     'mode': None,
     'model': {'learning_rate': {'boundaries': None,
                                 'decay_epochs': 2.4,
                                 'decay_rate': 0.97,
                                 'examples_per_epoch': None,
                                 'initial_lr': 0.008,
                                 'multipliers': None,
                                 'name': 'exponential',
                                 'scale_by_batch_size': 0.0078125,
                                 'staircase': True,
                                 'warmup_epochs': 5},
               'loss': {'label_smoothing': 0.1, 'name': 'categorical_crossentropy'},
               'model_params': {'model_name': 'efficientnet-b0',
                                'model_weights_path': '',
                                'overrides': {'activation': 'swish',
                                              'batch_norm': 'default',
                                              'dtype': 'float32',
                                              'num_classes': 1000,
                                              'rescale_input': True},
                                'weights_format': 'saved_model'},
               'name': 'EfficientNet',
               'num_classes': 1000,
               'optimizer': {'beta_1': None,
                             'beta_2': None,
                             'decay': 0.9,
                             'epsilon': 0.001,
                             'lookahead': None,
                             'momentum': 0.9,
                             'moving_average_decay': None,
                             'name': 'rmsprop',
                             'nesterov': None}},
     'model_dir': None,
     'model_name': None,
     'runtime': {'all_reduce_alg': None,
                 'batchnorm_spatial_persistent': False,
                 'dataset_num_private_threads': None,
                 'default_shard_dim': -1,
                 'distribution_strategy': 'mirrored',
                 'enable_xla': False,
                 'gpu_thread_mode': None,
                 'loss_scale': None,
                 'mixed_precision_dtype': None,
                 'num_cores_per_replica': 1,
                 'num_gpus': 0,
                 'num_packs': 1,
                 'per_gpu_thread_count': 0,
                 'run_eagerly': False,
                 'task_index': -1,
                 'tpu': None,
                 'tpu_enable_xla_dynamic_padder': None,
                 'worker_hosts': None},
     'train': {'callbacks': {'enable_backup_and_restore': False,
                             'enable_checkpoint_and_export': True,
                             'enable_tensorboard': True,
                             'enable_time_history': True},
               'epochs': 500,
               'metrics': ['accuracy', 'top_5'],
               'resume_checkpoint': True,
               'set_epoch_loop': False,
               'steps': None,
               'tensorboard': {'track_lr': True, 'write_model_weights': False},
               'time_history': {'log_steps': 100}},
     'train_dataset': {'augmenter': {'name': None, 'params': None},
                       'batch_size': 128,
                       'builder': 'records',
                       'cache': False,
                       'data_dir': None,
                       'download': False,
                       'dtype': 'float32',
                       'file_shuffle_buffer_size': 1024,
                       'filenames': None,
                       'image_size': 224,
                       'mean_subtract': False,
                       'name': 'imagenet2012',
                       'num_channels': 3,
                       'num_classes': 1000,
                       'num_devices': 1,
                       'num_examples': 1281167,
                       'one_hot': True,
                       'shuffle_buffer_size': 10000,
                       'skip_decoding': True,
                       'split': 'train',
                       'standardize': False,
                       'tf_data_service': None,
                       'use_per_replica_batch_size': True},
     'validation_dataset': {'augmenter': {'name': None, 'params': None},
                            'batch_size': 128,
                            'builder': 'records',
                            'cache': False,
                            'data_dir': None,
                            'download': False,
                            'dtype': 'float32',
                            'file_shuffle_buffer_size': 1024,
                            'filenames': None,
                            'image_size': 224,
                            'mean_subtract': False,
                            'name': 'imagenet2012',
                            'num_channels': 3,
                            'num_classes': 1000,
                            'num_devices': 1,
                            'num_examples': 1281167,
                            'one_hot': True,
                            'shuffle_buffer_size': 10000,
                            'skip_decoding': True,
                            'split': 'validation',
                            'standardize': False,
                            'tf_data_service': None,
                            'use_per_replica_batch_size': True}}
    I0407 21:19:48.210434 140189656872768 classifier_trainer.py:184] Overriding params: configs/examples/efficientnet/imagenet/efficientnet-b0-gpu.yaml
    I0407 21:19:48.214853 140189656872768 classifier_trainer.py:184] Overriding params: runtime.num_gpus=0,runtime.distribution_strategy="hpu",train_dataset.builder="synthetic",validation_dataset.builder="synthetic",train.steps=1000,train.epochs=1,evaluation.skip_eval=True
    I0407 21:19:48.216071 140189656872768 classifier_trainer.py:184] Overriding params: {'model_dir': '/home/ubuntu/log_hpu_8', 'mode': 'train_and_eval', 'model': {'name': 'efficientnet'}, 'runtime': {'run_eagerly': None, 'tpu': None}, 'train_dataset': {'data_dir': '/home/ubuntu'}, 'validation_dataset': {'data_dir': '/home/ubuntu'}, 'train': {'time_history': {'log_steps': 100}}}
    I0407 21:19:48.217288 140189656872768 classifier_trainer.py:190] Final model parameters: {'evaluation': {'epochs_between_evals': 1, 'skip_eval': True, 'steps': None},
     'export': {'checkpoint': None, 'destination': None},
     'mode': 'train_and_eval',
     'model': {'learning_rate': {'boundaries': None,
                                 'decay_epochs': 2.4,
                                 'decay_rate': 0.97,
                                 'examples_per_epoch': None,
                                 'initial_lr': 0.008,
                                 'multipliers': None,
                                 'name': 'exponential',
                                 'scale_by_batch_size': 0.0078125,
                                 'staircase': True,
                                 'warmup_epochs': 5},
               'loss': {'label_smoothing': 0.1, 'name': 'categorical_crossentropy'},
               'model_params': {'model_name': 'efficientnet-b0',
                                'model_weights_path': '',
                                'overrides': {'activation': 'swish',
                                              'batch_norm': 'default',
                                              'dtype': 'float32',
                                              'num_classes': 1000,
                                              'rescale_input': True},
                                'weights_format': 'saved_model'},
               'name': 'efficientnet',
               'num_classes': 1000,
               'optimizer': {'beta_1': None,
                             'beta_2': None,
                             'decay': 0.9,
                             'epsilon': 0.001,
                             'lookahead': False,
                             'momentum': 0.9,
                             'moving_average_decay': 0.0,
                             'name': 'rmsprop',
                             'nesterov': None}},
     'model_dir': '/home/ubuntu/log_hpu_8',
     'model_name': None,
     'runtime': {'all_reduce_alg': None,
                 'batchnorm_spatial_persistent': False,
                 'dataset_num_private_threads': None,
                 'default_shard_dim': -1,
                 'distribution_strategy': 'hpu',
                 'enable_xla': False,
                 'gpu_thread_mode': None,
                 'loss_scale': None,
                 'mixed_precision_dtype': None,
                 'num_cores_per_replica': 1,
                 'num_gpus': 0,
                 'num_packs': 1,
                 'per_gpu_thread_count': 0,
                 'run_eagerly': None,
                 'task_index': -1,
                 'tpu': None,
                 'tpu_enable_xla_dynamic_padder': None,
                 'worker_hosts': None},
     'train': {'callbacks': {'enable_backup_and_restore': False,
                             'enable_checkpoint_and_export': True,
                             'enable_tensorboard': True,
                             'enable_time_history': True},
               'epochs': 1,
               'metrics': ['accuracy', 'top_5'],
               'resume_checkpoint': True,
               'set_epoch_loop': False,
               'steps': 1000,
               'tensorboard': {'track_lr': True, 'write_model_weights': False},
               'time_history': {'log_steps': 100}},
     'train_dataset': {'augmenter': {'name': 'autoaugment', 'params': None},
                       'batch_size': 32,
                       'builder': 'synthetic',
                       'cache': False,
                       'data_dir': '/home/ubuntu',
                       'download': False,
                       'dtype': 'float32',
                       'file_shuffle_buffer_size': 1024,
                       'filenames': None,
                       'image_size': 224,
                       'mean_subtract': False,
                       'name': 'imagenet2012',
                       'num_channels': 3,
                       'num_classes': 1000,
                       'num_devices': 1,
                       'num_examples': 1281167,
                       'one_hot': True,
                       'shuffle_buffer_size': 10000,
                       'skip_decoding': True,
                       'split': 'train',
                       'standardize': False,
                       'tf_data_service': None,
                       'use_per_replica_batch_size': True},
     'validation_dataset': {'augmenter': {'name': None, 'params': None},
                            'batch_size': 32,
                            'builder': 'synthetic',
                            'cache': False,
                            'data_dir': '/home/ubuntu',
                            'download': False,
                            'dtype': 'float32',
                            'file_shuffle_buffer_size': 1024,
                            'filenames': None,
                            'image_size': 224,
                            'mean_subtract': False,
                            'name': 'imagenet2012',
                            'num_channels': 3,
                            'num_classes': 1000,
                            'num_devices': 1,
                            'num_examples': 50000,
                            'one_hot': True,
                            'shuffle_buffer_size': 10000,
                            'skip_decoding': True,
                            'split': 'validation',
                            'standardize': False,
                            'tf_data_service': None,
                            'use_per_replica_batch_size': True}}
    I0407 21:19:48.217387 140189656872768 classifier_trainer.py:290] Running train and eval.
    2022-04-07 21:19:48.219059: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2022-04-07 21:19:48.227607: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/helpers/op_registry_backdoor.cpp:92] Couldn't find definition of RemoteCall:GPU: to register on HPU
    2022-04-07 21:19:48.227582: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/helpers/op_registry_backdoor.cpp:92] Couldn't find definition of RemoteCall:GPU: to register on HPU
    2022-04-07 21:19:48.227606: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/helpers/op_registry_backdoor.cpp:92] Couldn't find definition of RemoteCall:GPU: to register on HPU
    2022-04-07 21:19:48.230096: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/helpers/op_registry_backdoor.cpp:92] Couldn't find definition of RemoteCall:GPU: to register on HPU
    I0407 21:19:48.230071 140201244120896 library_loader.py:175] Loading Habana as OpLibrary from /usr/local/lib/python3.8/dist-packages/habana_frameworks/tensorflow/tf2_8_0/lib/habanalabs/habana_device.so.2.8.0
    I0407 21:19:48.230079 140463173154624 library_loader.py:175] Loading Habana as OpLibrary from /usr/local/lib/python3.8/dist-packages/habana_frameworks/tensorflow/tf2_8_0/lib/habanalabs/habana_device.so.2.8.0
    I0407 21:19:48.230068 139742514054976 library_loader.py:175] Loading Habana as OpLibrary from /usr/local/lib/python3.8/dist-packages/habana_frameworks/tensorflow/tf2_8_0/lib/habanalabs/habana_device.so.2.8.0
    I0407 21:19:48.232516 140114967496512 library_loader.py:175] Loading Habana as OpLibrary from /usr/local/lib/python3.8/dist-packages/habana_frameworks/tensorflow/tf2_8_0/lib/habanalabs/habana_device.so.2.8.0
    2022-04-07 21:19:48.233860: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/helpers/op_registry_backdoor.cpp:92] Couldn't find definition of RemoteCall:GPU: to register on HPU
    2022-04-07 21:19:48.234309: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/helpers/op_registry_backdoor.cpp:92] Couldn't find definition of RemoteCall:GPU: to register on HPU
    I0407 21:19:48.236319 140136124327744 library_loader.py:190] Successfully loaded Habana module
    I0407 21:19:48.236371 139785568282432 library_loader.py:175] Loading Habana as OpLibrary from /usr/local/lib/python3.8/dist-packages/habana_frameworks/tensorflow/tf2_8_0/lib/habanalabs/habana_device.so.2.8.0
    I0407 21:19:48.236775 140034574755648 library_loader.py:175] Loading Habana as OpLibrary from /usr/local/lib/python3.8/dist-packages/habana_frameworks/tensorflow/tf2_8_0/lib/habanalabs/habana_device.so.2.8.0
    I0407 21:19:48.238561 140136124327744 classifier_trainer.py:181] Base params: {'evaluation': {'epochs_between_evals': 1, 'skip_eval': False, 'steps': None},
     'export': {'checkpoint': None, 'destination': None},
     'mode': None,
     'model': {'learning_rate': {'boundaries': None,
                                 'decay_epochs': 2.4,
                                 'decay_rate': 0.97,
                                 'examples_per_epoch': None,
                                 'initial_lr': 0.008,
                                 'multipliers': None,
                                 'name': 'exponential',
                                 'scale_by_batch_size': 0.0078125,
                                 'staircase': True,
                                 'warmup_epochs': 5},
               'loss': {'label_smoothing': 0.1, 'name': 'categorical_crossentropy'},
               'model_params': {'model_name': 'efficientnet-b0',
                                'model_weights_path': '',
                                'overrides': {'activation': 'swish',
                                              'batch_norm': 'default',
                                              'dtype': 'float32',
                                              'num_classes': 1000,
                                              'rescale_input': True},
                                'weights_format': 'saved_model'},
               'name': 'EfficientNet',
               'num_classes': 1000,
               'optimizer': {'beta_1': None,
                             'beta_2': None,
                             'decay': 0.9,
                             'epsilon': 0.001,
                             'lookahead': None,
                             'momentum': 0.9,
                             'moving_average_decay': None,
                             'name': 'rmsprop',
                             'nesterov': None}},
     'model_dir': None,
     'model_name': None,
     'runtime': {'all_reduce_alg': None,
                 'batchnorm_spatial_persistent': False,
                 'dataset_num_private_threads': None,
                 'default_shard_dim': -1,
                 'distribution_strategy': 'mirrored',
                 'enable_xla': False,
                 'gpu_thread_mode': None,
                 'loss_scale': None,
                 'mixed_precision_dtype': None,
                 'num_cores_per_replica': 1,
                 'num_gpus': 0,
                 'num_packs': 1,
                 'per_gpu_thread_count': 0,
                 'run_eagerly': False,
                 'task_index': -1,
                 'tpu': None,
                 'tpu_enable_xla_dynamic_padder': None,
                 'worker_hosts': None},
     'train': {'callbacks': {'enable_backup_and_restore': False,
                             'enable_checkpoint_and_export': True,
                             'enable_tensorboard': True,
                             'enable_time_history': True},
               'epochs': 500,
               'metrics': ['accuracy', 'top_5'],
               'resume_checkpoint': True,
               'set_epoch_loop': False,
               'steps': None,
               'tensorboard': {'track_lr': True, 'write_model_weights': False},
               'time_history': {'log_steps': 100}},
     'train_dataset': {'augmenter': {'name': None, 'params': None},
                       'batch_size': 128,
                       'builder': 'records',
                       'cache': False,
                       'data_dir': None,
                       'download': False,
                       'dtype': 'float32',
                       'file_shuffle_buffer_size': 1024,
                       'filenames': None,
                       'image_size': 224,
                       'mean_subtract': False,
                       'name': 'imagenet2012',
                       'num_channels': 3,
                       'num_classes': 1000,
                       'num_devices': 1,
                       'num_examples': 1281167,
                       'one_hot': True,
                       'shuffle_buffer_size': 10000,
                       'skip_decoding': True,
                       'split': 'train',
                       'standardize': False,
                       'tf_data_service': None,
                       'use_per_replica_batch_size': True},
     'validation_dataset': {'augmenter': {'name': None, 'params': None},
                            'batch_size': 128,
                            'builder': 'records',
                            'cache': False,
                            'data_dir': None,
                            'download': False,
                            'dtype': 'float32',
                            'file_shuffle_buffer_size': 1024,
                            'filenames': None,
                            'image_size': 224,
                            'mean_subtract': False,
                            'name': 'imagenet2012',
                            'num_channels': 3,
                            'num_classes': 1000,
                            'num_devices': 1,
                            'num_examples': 1281167,
                            'one_hot': True,
                            'shuffle_buffer_size': 10000,
                            'skip_decoding': True,
                            'split': 'validation',
                            'standardize': False,
                            'tf_data_service': None,
                            'use_per_replica_batch_size': True}}
    I0407 21:19:48.238652 140136124327744 classifier_trainer.py:184] Overriding params: configs/examples/efficientnet/imagenet/efficientnet-b0-gpu.yaml
    I0407 21:19:48.243162 140136124327744 classifier_trainer.py:184] Overriding params: runtime.num_gpus=0,runtime.distribution_strategy="hpu",train_dataset.builder="synthetic",validation_dataset.builder="synthetic",train.steps=1000,train.epochs=1,evaluation.skip_eval=True
    I0407 21:19:48.244333 140136124327744 classifier_trainer.py:184] Overriding params: {'model_dir': '/home/ubuntu/log_hpu_8', 'mode': 'train_and_eval', 'model': {'name': 'efficientnet'}, 'runtime': {'run_eagerly': None, 'tpu': None}, 'train_dataset': {'data_dir': '/home/ubuntu'}, 'validation_dataset': {'data_dir': '/home/ubuntu'}, 'train': {'time_history': {'log_steps': 100}}}
    I0407 21:19:48.245525 140136124327744 classifier_trainer.py:190] Final model parameters: {'evaluation': {'epochs_between_evals': 1, 'skip_eval': True, 'steps': None},
     'export': {'checkpoint': None, 'destination': None},
     'mode': 'train_and_eval',
     'model': {'learning_rate': {'boundaries': None,
                                 'decay_epochs': 2.4,
                                 'decay_rate': 0.97,
                                 'examples_per_epoch': None,
                                 'initial_lr': 0.008,
                                 'multipliers': None,
                                 'name': 'exponential',
                                 'scale_by_batch_size': 0.0078125,
                                 'staircase': True,
                                 'warmup_epochs': 5},
               'loss': {'label_smoothing': 0.1, 'name': 'categorical_crossentropy'},
               'model_params': {'model_name': 'efficientnet-b0',
                                'model_weights_path': '',
                                'overrides': {'activation': 'swish',
                                              'batch_norm': 'default',
                                              'dtype': 'float32',
                                              'num_classes': 1000,
                                              'rescale_input': True},
                                'weights_format': 'saved_model'},
               'name': 'efficientnet',
               'num_classes': 1000,
               'optimizer': {'beta_1': None,
                             'beta_2': None,
                             'decay': 0.9,
                             'epsilon': 0.001,
                             'lookahead': False,
                             'momentum': 0.9,
                             'moving_average_decay': 0.0,
                             'name': 'rmsprop',
                             'nesterov': None}},
     'model_dir': '/home/ubuntu/log_hpu_8',
     'model_name': None,
     'runtime': {'all_reduce_alg': None,
                 'batchnorm_spatial_persistent': False,
                 'dataset_num_private_threads': None,
                 'default_shard_dim': -1,
                 'distribution_strategy': 'hpu',
                 'enable_xla': False,
                 'gpu_thread_mode': None,
                 'loss_scale': None,
                 'mixed_precision_dtype': None,
                 'num_cores_per_replica': 1,
                 'num_gpus': 0,
                 'num_packs': 1,
                 'per_gpu_thread_count': 0,
                 'run_eagerly': None,
                 'task_index': -1,
                 'tpu': None,
                 'tpu_enable_xla_dynamic_padder': None,
                 'worker_hosts': None},
     'train': {'callbacks': {'enable_backup_and_restore': False,
                             'enable_checkpoint_and_export': True,
                             'enable_tensorboard': True,
                             'enable_time_history': True},
               'epochs': 1,
               'metrics': ['accuracy', 'top_5'],
               'resume_checkpoint': True,
               'set_epoch_loop': False,
               'steps': 1000,
               'tensorboard': {'track_lr': True, 'write_model_weights': False},
               'time_history': {'log_steps': 100}},
     'train_dataset': {'augmenter': {'name': 'autoaugment', 'params': None},
                       'batch_size': 32,
                       'builder': 'synthetic',
                       'cache': False,
                       'data_dir': '/home/ubuntu',
                       'download': False,
                       'dtype': 'float32',
                       'file_shuffle_buffer_size': 1024,
                       'filenames': None,
                       'image_size': 224,
                       'mean_subtract': False,
                       'name': 'imagenet2012',
                       'num_channels': 3,
                       'num_classes': 1000,
                       'num_devices': 1,
                       'num_examples': 1281167,
                       'one_hot': True,
                       'shuffle_buffer_size': 10000,
                       'skip_decoding': True,
                       'split': 'train',
                       'standardize': False,
                       'tf_data_service': None,
                       'use_per_replica_batch_size': True},
     'validation_dataset': {'augmenter': {'name': None, 'params': None},
                            'batch_size': 32,
                            'builder': 'synthetic',
                            'cache': False,
                            'data_dir': '/home/ubuntu',
                            'download': False,
                            'dtype': 'float32',
                            'file_shuffle_buffer_size': 1024,
                            'filenames': None,
                            'image_size': 224,
                            'mean_subtract': False,
                            'name': 'imagenet2012',
                            'num_channels': 3,
                            'num_classes': 1000,
                            'num_devices': 1,
                            'num_examples': 50000,
                            'one_hot': True,
                            'shuffle_buffer_size': 10000,
                            'skip_decoding': True,
                            'split': 'validation',
                            'standardize': False,
                            'tf_data_service': None,
                            'use_per_replica_batch_size': True}}
    I0407 21:19:48.245591 140136124327744 classifier_trainer.py:290] Running train and eval.
    2022-04-07 21:19:48.247173: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    I0407 21:19:48.294959 140463173154624 library_loader.py:190] Successfully loaded Habana module
    I0407 21:19:48.295008 139742514054976 library_loader.py:190] Successfully loaded Habana module
    I0407 21:19:48.295007 140201244120896 library_loader.py:190] Successfully loaded Habana module
    I0407 21:19:48.295438 140114967496512 library_loader.py:190] Successfully loaded Habana module
    I0407 21:19:48.297231 140463173154624 classifier_trainer.py:181] Base params: {'evaluation': {'epochs_between_evals': 1, 'skip_eval': False, 'steps': None},
     'export': {'checkpoint': None, 'destination': None},
     'mode': None,
     'model': {'learning_rate': {'boundaries': None,
                                 'decay_epochs': 2.4,
                                 'decay_rate': 0.97,
                                 'examples_per_epoch': None,
                                 'initial_lr': 0.008,
                                 'multipliers': None,
                                 'name': 'exponential',
                                 'scale_by_batch_size': 0.0078125,
                                 'staircase': True,
                                 'warmup_epochs': 5},
               'loss': {'label_smoothing': 0.1, 'name': 'categorical_crossentropy'},
               'model_params': {'model_name': 'efficientnet-b0',
                                'model_weights_path': '',
                                'overrides': {'activation': 'swish',
                                              'batch_norm': 'default',
                                              'dtype': 'float32',
                                              'num_classes': 1000,
                                              'rescale_input': True},
                                'weights_format': 'saved_model'},
               'name': 'EfficientNet',
               'num_classes': 1000,
               'optimizer': {'beta_1': None,
                             'beta_2': None,
                             'decay': 0.9,
                             'epsilon': 0.001,
                             'lookahead': None,
                             'momentum': 0.9,
                             'moving_average_decay': None,
                             'name': 'rmsprop',
                             'nesterov': None}},
     'model_dir': None,
     'model_name': None,
     'runtime': {'all_reduce_alg': None,
                 'batchnorm_spatial_persistent': False,
                 'dataset_num_private_threads': None,
                 'default_shard_dim': -1,
                 'distribution_strategy': 'mirrored',
                 'enable_xla': False,
                 'gpu_thread_mode': None,
                 'loss_scale': None,
                 'mixed_precision_dtype': None,
                 'num_cores_per_replica': 1,
                 'num_gpus': 0,
                 'num_packs': 1,
                 'per_gpu_thread_count': 0,
                 'run_eagerly': False,
                 'task_index': -1,
                 'tpu': None,
                 'tpu_enable_xla_dynamic_padder': None,
                 'worker_hosts': None},
     'train': {'callbacks': {'enable_backup_and_restore': False,
                             'enable_checkpoint_and_export': True,
                             'enable_tensorboard': True,
                             'enable_time_history': True},
               'epochs': 500,
               'metrics': ['accuracy', 'top_5'],
               'resume_checkpoint': True,
               'set_epoch_loop': False,
               'steps': None,
               'tensorboard': {'track_lr': True, 'write_model_weights': False},
               'time_history': {'log_steps': 100}},
     'train_dataset': {'augmenter': {'name': None, 'params': None},
                       'batch_size': 128,
                       'builder': 'records',
                       'cache': False,
                       'data_dir': None,
                       'download': False,
                       'dtype': 'float32',
                       'file_shuffle_buffer_size': 1024,
                       'filenames': None,
                       'image_size': 224,
                       'mean_subtract': False,
                       'name': 'imagenet2012',
                       'num_channels': 3,
                       'num_classes': 1000,
                       'num_devices': 1,
                       'num_examples': 1281167,
                       'one_hot': True,
                       'shuffle_buffer_size': 10000,
                       'skip_decoding': True,
                       'split': 'train',
                       'standardize': False,
                       'tf_data_service': None,
                       'use_per_replica_batch_size': True},
     'validation_dataset': {'augmenter': {'name': None, 'params': None},
                            'batch_size': 128,
                            'builder': 'records',
                            'cache': False,
                            'data_dir': None,
                            'download': False,
                            'dtype': 'float32',
                            'file_shuffle_buffer_size': 1024,
                            'filenames': None,
                            'image_size': 224,
                            'mean_subtract': False,
                            'name': 'imagenet2012',
                            'num_channels': 3,
                            'num_classes': 1000,
                            'num_devices': 1,
                            'num_examples': 1281167,
                            'one_hot': True,
                            'shuffle_buffer_size': 10000,
                            'skip_decoding': True,
                            'split': 'validation',
                            'standardize': False,
                            'tf_data_service': None,
                            'use_per_replica_batch_size': True}}
    I0407 21:19:48.297323 140463173154624 classifier_trainer.py:184] Overriding params: configs/examples/efficientnet/imagenet/efficientnet-b0-gpu.yaml
    I0407 21:19:48.297256 139742514054976 classifier_trainer.py:181] Base params: {'evaluation': {'epochs_between_evals': 1, 'skip_eval': False, 'steps': None},
     'export': {'checkpoint': None, 'destination': None},
     'mode': None,
     'model': {'learning_rate': {'boundaries': None,
                                 'decay_epochs': 2.4,
                                 'decay_rate': 0.97,
                                 'examples_per_epoch': None,
                                 'initial_lr': 0.008,
                                 'multipliers': None,
                                 'name': 'exponential',
                                 'scale_by_batch_size': 0.0078125,
                                 'staircase': True,
                                 'warmup_epochs': 5},
               'loss': {'label_smoothing': 0.1, 'name': 'categorical_crossentropy'},
               'model_params': {'model_name': 'efficientnet-b0',
                                'model_weights_path': '',
                                'overrides': {'activation': 'swish',
                                              'batch_norm': 'default',
                                              'dtype': 'float32',
                                              'num_classes': 1000,
                                              'rescale_input': True},
                                'weights_format': 'saved_model'},
               'name': 'EfficientNet',
               'num_classes': 1000,
               'optimizer': {'beta_1': None,
                             'beta_2': None,
                             'decay': 0.9,
                             'epsilon': 0.001,
                             'lookahead': None,
                             'momentum': 0.9,
                             'moving_average_decay': None,
                             'name': 'rmsprop',
                             'nesterov': None}},
     'model_dir': None,
     'model_name': None,
     'runtime': {'all_reduce_alg': None,
                 'batchnorm_spatial_persistent': False,
                 'dataset_num_private_threads': None,
                 'default_shard_dim': -1,
                 'distribution_strategy': 'mirrored',
                 'enable_xla': False,
                 'gpu_thread_mode': None,
                 'loss_scale': None,
                 'mixed_precision_dtype': None,
                 'num_cores_per_replica': 1,
                 'num_gpus': 0,
                 'num_packs': 1,
                 'per_gpu_thread_count': 0,
                 'run_eagerly': False,
                 'task_index': -1,
                 'tpu': None,
                 'tpu_enable_xla_dynamic_padder': None,
                 'worker_hosts': None},
     'train': {'callbacks': {'enable_backup_and_restore': False,
                             'enable_checkpoint_and_export': True,
                             'enable_tensorboard': True,
                             'enable_time_history': True},
               'epochs': 500,
               'metrics': ['accuracy', 'top_5'],
               'resume_checkpoint': True,
               'set_epoch_loop': False,
               'steps': None,
               'tensorboard': {'track_lr': True, 'write_model_weights': False},
               'time_history': {'log_steps': 100}},
     'train_dataset': {'augmenter': {'name': None, 'params': None},
                       'batch_size': 128,
                       'builder': 'records',
                       'cache': False,
                       'data_dir': None,
                       'download': False,
                       'dtype': 'float32',
                       'file_shuffle_buffer_size': 1024,
                       'filenames': None,
                       'image_size': 224,
                       'mean_subtract': False,
                       'name': 'imagenet2012',
                       'num_channels': 3,
                       'num_classes': 1000,
                       'num_devices': 1,
                       'num_examples': 1281167,
                       'one_hot': True,
                       'shuffle_buffer_size': 10000,
                       'skip_decoding': True,
                       'split': 'train',
                       'standardize': False,
                       'tf_data_service': None,
                       'use_per_replica_batch_size': True},
     'validation_dataset': {'augmenter': {'name': None, 'params': None},
            I0407 21:19:48.297279 140201244120896 classifier_trainer.py:181] Base params: {'evaluation': {'epochs_between_evals': 1, 'skip_eval': False, 'steps': None},
     'export': {'checkpoint': None, 'destination': None},
     'mode': None,
     'model': {'learning_rate': {'boundaries': None,
                                 'decay_epochs': 2.4,
                                 'decay_rate': 0.97,
                                 'examples_per_epoch': None,
                                 'initial_lr': 0.008,
                                 'multipliers': None,
                                 'name': 'exponential',
                                 'scale_by_batch_size': 0.0078125,
                                 'staircase': True,
                                 'warmup_epochs': 5},
               'loss': {'label_smoothing': 0.1, 'name': 'categorical_crossentropy'},
               'model_params': {'model_name': 'efficientnet-b0',
                                'model_weights_path': '',
                                'overrides': {'activation': 'swish',
                                              'batch_norm': 'default',
                                              'dtype': 'float32',
                                              'num_classes': 1000,
                                              'rescale_input': True},
                                'weights_format': 'saved_model'},
               'name': 'EfficientNet',
               'num_classes': 1000,
               'optimizer': {'beta_1': None,
                             'beta_2': None,
                             'decay': 0.9,
                             'epsilon': 0.001,
                             'lookahead': None,
                             'momentum': 0.9,
                             'moving_average_decay': None,
                             'name': 'rmsprop',
                             'nesterov': None}},
     'model_dir': None,
     'model_name': None,
     'runtime': {'all_reduce_alg': None,
                 'batchnorm_spatial_persistent': False,
                 'dataset_num_private_threads': None,
                 'default_shard_dim': -1,
                 'distribution_strategy': 'mirrored',
                 'enable_xla': False,
                 'gpu_thread_mode': None,
                 'loss_scale': None,
                 'mixed_precision_dtype': None,
                 'num_cores_per_replica': 1,
                 'num_gpus': 0,
                 'num_packs': 1,
                 'per_gpu_thread_count': 0,
                 'run_eagerly': False,
                 'task_index': -1,
                 'tpu': None,
                 'tpu_enable_xla_dynamic_padder': None,
                 'worker_hosts': None},
     'train': {'callbacks': {'enable_backup_and_restore': False,
                             'enable_checkpoint_and_export': True,
                             'enable_tensorboard': True,
                             'enable_time_history': True},
               'epochs': 500,
               'metrics': ['accuracy', 'top_5'],
               'resume_checkpoint': True,
               'set_epoch_loop': False,
               'steps': None,
               'tensorboard': {'track_lr': True, 'write_model_weights': False},
               'time_history': {'log_steps': 100}},
     'train_dataset': {'augmenter': {'name': None, 'params': None},
                       'batch_size': 128,
                       'builder': 'records',
                       'cache': False,
                       'data_dir': None,
                       'download': False,
                       'dtype': 'float32',
                       'file_shuffle_buffer_size': 1024,
                       'filenames': None,
                       'image_size': 224,
                       'mean_subtract': False,
                       'name': 'imagenet2012',
                       'num_channels': 3,
                       'num_classes': 1000,
                       'num_devices': 1,
                       'num_examples': 1281167,
                       'one_hot': True,
                       'shuffle_buffer_size': 10000,
                       'skip_decoding': True,
                       'split': 'train',
                       'standardize': False,
                       'tf_data_service': None,
                       'use_per_replica_batch_size': True},
     'validation_dataset': {'augmenter': {'name': None, 'params': None},
                            'batch_size': 128,
                            'builder': 'records',
                            'cache': False,
                            'data_dir': None,
                            'download': False,
                            'dtype': 'float32',
                            'file_shuffle_buffer_size': 1024,
                            'filenames': None,
                            'image_size': 224,
                            'mean_subtract': False,
                            'name': 'imagenet2012',
                            'num_channels': 3,
                            'num_classes': 1000,
                            'num_devices': 1,
                            'num_examples': 1281167,
                            'one_hot': True,
                            'shuffle_buffer_size': 10000,
                            'skip_decoding': True,
                            'split': 'validation',
                            'standardize': False,
                            'tf_data_service': None,
                            'use_per_replica_batch_size': True}}
    I0407 21:19:48.297347 139742514054976 classifier_trainer.py:184] Overriding params: configs/examples/efficientnet/imagenet/efficientnet-b0-gpu.yaml
                    'batch_size': 128,
                            'builder': 'records',
                            'cache': False,
                            'data_dir': None,
                            'download': False,
                            'dtype': 'float32',
                            'file_shuffle_buffer_size': 1024,
                            'filenames': None,
                            'image_size': 224,
                            'mean_subtract': False,
                            'name': 'imagenet2012',
                            'num_channels': 3,
                            'num_classes': 1000,
                            'num_devices': 1,
                            'num_examples': 1281167,
                            'one_hot': True,
                            'shuffle_buffer_size': 10000,
                            'skip_decoding': True,
                            'split': 'validation',
                            'standardize': False,
                            'tf_data_service': None,
                            'use_per_replica_batch_size': True}}
    I0407 21:19:48.297377 140201244120896 classifier_trainer.py:184] Overriding params: configs/examples/efficientnet/imagenet/efficientnet-b0-gpu.yaml
    I0407 21:19:48.297587 140034574755648 library_loader.py:190] Successfully loaded Habana module
    I0407 21:19:48.297549 139785568282432 library_loader.py:190] Successfully loaded Habana module
    I0407 21:19:48.297686 140114967496512 classifier_trainer.py:181] Base params: {'evaluation': {'epochs_between_evals': 1, 'skip_eval': False, 'steps': None},
     'export': {'checkpoint': None, 'destination': None},
     'mode': None,
     'model': {'learning_rate': {'boundaries': None,
                                 'decay_epochs': 2.4,
                                 'decay_rate': 0.97,
                                 'examples_per_epoch': None,
                                 'initial_lr': 0.008,
                                 'multipliers': None,
                                 'name': 'exponential',
                                 'scale_by_batch_size': 0.0078125,
                                 'staircase': True,
                                 'warmup_epochs': 5},
               'loss': {'label_smoothing': 0.1, 'name': 'categorical_crossentropy'},
               'model_params': {'model_name': 'efficientnet-b0',
                                'model_weights_path': '',
                                'overrides': {'activation': 'swish',
                                              'batch_norm': 'default',
                                              'dtype': 'float32',
                                              'num_classes': 1000,
                                              'rescale_input': True},
                                'weights_format': 'saved_model'},
               'name': 'EfficientNet',
               'num_classes': 1000,
               'optimizer': {'beta_1': None,
                             'beta_2': None,
                             'decay': 0.9,
                             'epsilon': 0.001,
                             'lookahead': None,
                             'momentum': 0.9,
                             'moving_average_decay': None,
                             'name': 'rmsprop',
                             'nesterov': None}},
     'model_dir': None,
     'model_name': None,
     'runtime': {'all_reduce_alg': None,
                 'batchnorm_spatial_persistent': False,
                 'dataset_num_private_threads': None,
                 'default_shard_dim': -1,
                 'distribution_strategy': 'mirrored',
                 'enable_xla': False,
                 'gpu_thread_mode': None,
                 'loss_scale': None,
                 'mixed_precision_dtype': None,
                 'num_cores_per_replica': 1,
                 'num_gpus': 0,
                 'num_packs': 1,
                 'per_gpu_thread_count': 0,
                 'run_eagerly': False,
                 'task_index': -1,
                 'tpu': None,
                 'tpu_enable_xla_dynamic_padder': None,
                 'worker_hosts': None},
     'train': {'callbacks': {'enable_backup_and_restore': False,
                             'enable_checkpoint_and_export': True,
                             'enable_tensorboard': True,
                             'enable_time_history': True},
               'epochs': 500,
               'metrics': ['accuracy', 'top_5'],
               'resume_checkpoint': True,
               'set_epoch_loop': False,
               'steps': None,
               'tensorboard': {'track_lr': True, 'write_model_weights': False},
               'time_history': {'log_steps': 100}},
     'train_dataset': {'augmenter': {'name': None, 'params': None},
                       'batch_size': 128,
                       'builder': 'records',
                       'cache': False,
                       'data_dir': None,
                       'download': False,
                       'dtype': 'float32',
                       'file_shuffle_buffer_size': 1024,
                       'filenames': None,
                       'image_size': 224,
                       'mean_subtract': False,
                       'name': 'imagenet2012',
                       'num_channels': 3,
                       'num_classes': 1000,
                       'num_devices': 1,
                       'num_examples': 1281167,
                       'one_hot': True,
                       'shuffle_buffer_size': 10000,
                       'skip_decoding': True,
                       'split': 'train',
                       'standardize': False,
                       'tf_data_service': None,
                       'use_per_replica_batch_size': True},
     'validation_dataset': {'augmenter': {'name': None, 'params': None},
                            'batch_size': 128,
                            'builder': 'records',
                            'cache': False,
                            'data_dir': None,
                            'download': False,
                            'dtype': 'float32',
                            'file_shuffle_buffer_size': 1024,
                            'filenames': None,
                            'image_size': 224,
                            'mean_subtract': False,
                            'name': 'imagenet2012',
                            'num_channels': 3,
                            'num_classes': 1000,
                            'num_devices': 1,
                            'num_examples': 1281167,
                            'one_hot': True,
                            'shuffle_buffer_size': 10000,
                            'skip_decoding': True,
                            'split': 'validation',
                            'standardize': False,
                            'tf_data_service': None,
                            'use_per_replica_batch_size': True}}
    I0407 21:19:48.297777 140114967496512 classifier_trainer.py:184] Overriding params: configs/examples/efficientnet/imagenet/efficientnet-b0-gpu.yaml
    I0407 21:19:48.299818 139785568282432 classifier_trainer.py:181] Base params: {'evaluation': {'epochs_between_evals': 1, 'skip_eval': False, 'steps': None},
     'export': {'checkpoint': None, 'destination': None},
     'mode': None,
     'model': {'learning_rate': {'boundaries': None,
                                 'decay_epochs': 2.4,
                                 'decay_rate': 0.97,
                                 'examples_per_epoch': None,
                                 'initial_lr': 0.008,
                                 'multipliers': None,
                                 'name': 'exponential',
                                 'scale_by_batch_size': 0.0078125,
                                 'staircase': True,
                                 'warmup_epochs': 5},
               'loss': {'label_smoothing': 0.1, 'name': 'categorical_crossentropy'},
               'model_params': {'model_name': 'efficientnet-b0',
                                'model_weights_path': '',
                                'overrides': {'activation': 'swish',
                                              'batch_norm': 'default',
                                              'dtype': 'float32',
                                              'num_classes': 1000,
                                              'rescale_input': True},
                                'weights_format': 'saved_model'},
               'name': 'EfficientNet',
               'num_classes': 1000,
               'optimizer': {'beta_1': None,
                             'beta_2': None,
                             'decay': 0.9,
                             'epsilon': 0.001,
                             'lookahead': None,
                             'momentum': 0.9,
                             'moving_average_decay': None,
                             'name': 'rmsprop',
                             'nesterov': None}},
     'model_dir': None,
     'model_name': None,
     'runtime': {'all_reduce_alg': None,
                 'batchnorm_spatial_persistent': False,
                 'dataset_num_private_threads': None,
                 'default_shard_dim': -1,
                 'distribution_strategy': 'mirrored',
                 'enable_xla': False,
                 'gpu_thread_mode': None,
                 'loss_scale': None,
                 'mixed_precision_dtype': None,
                 'num_cores_per_replica': 1,
                 'num_gpus': 0,
                 'num_packs': 1,
                 'per_gpu_thread_count': 0,
                 'run_eagerly': False,
                 'task_index': -1,
                 'tpu': None,
                 'tpu_enable_xla_dynamic_padder': None,
                 'worker_hosts': None},
     'train': {'callbacks': {'enable_backup_and_restore': False,
                             'enable_checkpoint_and_export': True,
                             'enable_tensorboard': True,
                             'enable_time_history': True},
               'epochs': 500,
               'metrics': ['accuracy', 'top_5'],
               'resume_checkpoint': True,
               'set_epoch_loop': False,
               'steps': None,
               'tensorboard': {'track_lr': True, 'write_model_weights': False},
               'time_history': {'log_steps': 100}},
     'train_dataset': {'augmenter': {'name': None, 'params': None},
                       'batch_size': 128,
                       'builder': 'records',
                       'cache': False,
                       'data_dir': None,
                       'download': False,
                       'dtype': 'float32',
                       'file_shuffle_buffer_size': 1024,
                       'filenames': None,
                       'image_size': 224,
                       'mean_subtract': False,
                       'name': 'imagenet2012',
                       'num_channels': 3,
                       'num_classes': 1000,
                       'num_devices': 1,
                       'num_examples': 1281167,
                       'one_hot': True,
                       'shuffle_buffer_size': 10000,
                       'skip_decoding': True,
                       'split': 'train',
                       'standardize': False,
                       'tf_data_service': None,
                       'use_per_replica_batch_size': True},
     'validation_dataset': {'augmenter': {'name': None, 'params': None},
            I0407 21:19:48.299861 140034574755648 classifier_trainer.py:181] Base params: {'evaluation': {'epochs_between_evals': 1, 'skip_eval': False, 'steps': None},
     'export': {'checkpoint': None, 'destination': None},
     'mode': None,
     'model': {'learning_rate': {'boundaries': None,
                                 'decay_epochs': 2.4,
                                 'decay_rate': 0.97,
                                 'examples_per_epoch': None,
                                 'initial_lr': 0.008,
                                 'multipliers': None,
                                 'name': 'exponential',
                                 'scale_by_batch_size': 0.0078125,
                                 'staircase': True,
                                 'warmup_epochs': 5},
               'loss': {'label_smoothing': 0.1, 'name': 'categorical_crossentropy'},
               'model_params': {'model_name': 'efficientnet-b0',
                                'model_weights_path': '',
                                'overrides': {'activation': 'swish',
                                              'batch_norm': 'default',
                                              'dtype': 'float32',
                                              'num_classes': 1000,
                                              'rescale_input': True},
                                'weights_format': 'saved_model'},
               'name': 'EfficientNet',
               'num_classes': 1000,
               'optimizer': {'beta_1': None,
                             'beta_2': None,
                             'decay': 0.9,
                             'epsilon': 0.001,
                             'lookahead': None,
                             'momentum': 0.9,
                             'moving_average_decay': None,
                             'name': 'rmsprop',
                             'nesterov': None}},
     'model_dir': None,
     'model_name': None,
     'runtime': {'all_reduce_alg': None,
                 'batchnorm_spatial_persistent': False,
                 'dataset_num_private_threads': None,
                 'default_shard_dim': -1,
                 'distribution_strategy': 'mirrored',
                 'enable_xla': False,
                 'gpu_thread_mode': None,
                 'loss_scale': None,
                 'mixed_precision_dtype': None,
                 'num_cores_per_replica': 1,
                 'num_gpus': 0,
                 'num_packs': 1,
                 'per_gpu_thread_count': 0,
                 'run_eagerly': False,
                 'task_index': -1,
                 'tpu': None,
                 'tpu_enable_xla_dynamic_padder': None,
                 'worker_hosts': None},
     'train': {'callbacks': {'enable_backup_and_restore': False,
                             'enable_checkpoint_and_export': True,
                             'enable_tensorboard': True,
                             'enable_time_history': True},
               'epochs': 500,
               'metrics': ['accuracy', 'top_5'],
               'resume_checkpoint': True,
               'set_epoch_loop': False,
               'steps': None,
               'tensorboard': {'track_lr': True, 'write_model_weights': False},
               'time_history': {'log_steps': 100}},
     'train_dataset': {'augmenter': {'name': None, 'params': None},
                       'batch_size': 128,
                       'builder': 'records',
                       'cache': False,
                       'data_dir': None,
                       'download': False,
                       'dtype': 'float32',
                       'file_shuffle_buffer_size': 1024,
                       'filenames': None,
                       'image_size': 224,
                       'mean_subtract': False,
                       'name': 'imagenet2012',
                       'num_channels': 3,
                       'num_classes': 1000,
                       'num_devices': 1,
                       'num_examples': 1281167,
                       'one_hot': True,
                       'shuffle_buffer_size': 10000,
                       'skip_decoding': True,
                       'split': 'train',
                       'standardize': False,
                       'tf_data_service': None,
                       'use_per_replica_batch_size': True},
     'validation_dataset': {'augmenter': {'name': None, 'params': None},
                            'batch_size': 128,
                            'builder': 'records',
                            'cache': False,
                            'data_dir': None,
                            'download': False,
                            'dtype': 'float32',
                            'file_shuffle_buffer_size': 1024,
                            'filenames': None,
                            'image_size': 224,
                            'mean_subtract': False,
                            'name': 'imagenet2012',
                            'num_channels': 3,
                            'num_classes': 1000,
                            'num_devices': 1,
                            'num_examples': 1281167,
                            'one_hot': True,
                            'shuffle_buffer_size': 10000,
                            'skip_decoding': True,
                            'split': 'validation',
                            'standardize': False,
                            'tf_data_service': None,
                            'use_per_replica_batch_size': True}}
                    'batch_size': 128,
                            'builder': 'records',
                            'cache': False,
                            'data_dir': None,
                            'download': False,
                            'dtype': 'float32',
                            'file_shuffle_buffer_size': 1024,
                            'filenames': None,
                            'image_size': 224,
                            'mean_subtract': False,
                            'name': 'imagenet2012',
                            'num_channels': 3,
                            'num_classes': 1000,
                            'num_devices': 1,
                            'num_examples': 1281167,
                            'one_hot': True,
                            'shuffle_buffer_size': 10000,
                            'skip_decoding': True,
                            'split': 'validation',
                            'standardize': False,
                            'tf_data_service': None,
                            'use_per_replica_batch_size': True}}
    I0407 21:19:48.299999 140034574755648 classifier_trainer.py:184] Overriding params: configs/examples/efficientnet/imagenet/efficientnet-b0-gpu.yaml
    I0407 21:19:48.299953 139785568282432 classifier_trainer.py:184] Overriding params: configs/examples/efficientnet/imagenet/efficientnet-b0-gpu.yaml
    I0407 21:19:48.301795 139742514054976 classifier_trainer.py:184] Overriding params: runtime.num_gpus=0,runtime.distribution_strategy="hpu",train_dataset.builder="synthetic",validation_dataset.builder="synthetic",train.steps=1000,train.epochs=1,evaluation.skip_eval=True
    I0407 21:19:48.301793 140463173154624 classifier_trainer.py:184] Overriding params: runtime.num_gpus=0,runtime.distribution_strategy="hpu",train_dataset.builder="synthetic",validation_dataset.builder="synthetic",train.steps=1000,train.epochs=1,evaluation.skip_eval=True
    I0407 21:19:48.301749 140201244120896 classifier_trainer.py:184] Overriding params: runtime.num_gpus=0,runtime.distribution_strategy="hpu",train_dataset.builder="synthetic",validation_dataset.builder="synthetic",train.steps=1000,train.epochs=1,evaluation.skip_eval=True
    I0407 21:19:48.302220 140114967496512 classifier_trainer.py:184] Overriding params: runtime.num_gpus=0,runtime.distribution_strategy="hpu",train_dataset.builder="synthetic",validation_dataset.builder="synthetic",train.steps=1000,train.epochs=1,evaluation.skip_eval=True
    I0407 21:19:48.302885 140201244120896 classifier_trainer.py:184] Overriding params: {'model_dir': '/home/ubuntu/log_hpu_8', 'mode': 'train_and_eval', 'model': {'name': 'efficientnet'}, 'runtime': {'run_eagerly': None, 'tpu': None}, 'train_dataset': {'data_dir': '/home/ubuntu'}, 'validation_dataset': {'data_dir': '/home/ubuntu'}, 'train': {'time_history': {'log_steps': 100}}}
    I0407 21:19:48.302946 139742514054976 classifier_trainer.py:184] Overriding params: {'model_dir': '/home/ubuntu/log_hpu_8', 'mode': 'train_and_eval', 'model': {'name': 'efficientnet'}, 'runtime': {'run_eagerly': None, 'tpu': None}, 'train_dataset': {'data_dir': '/home/ubuntu'}, 'validation_dataset': {'data_dir': '/home/ubuntu'}, 'train': {'time_history': {'log_steps': 100}}}
    I0407 21:19:48.302942 140463173154624 classifier_trainer.py:184] Overriding params: {'model_dir': '/home/ubuntu/log_hpu_8', 'mode': 'train_and_eval', 'model': {'name': 'efficientnet'}, 'runtime': {'run_eagerly': None, 'tpu': None}, 'train_dataset': {'data_dir': '/home/ubuntu'}, 'validation_dataset': {'data_dir': '/home/ubuntu'}, 'train': {'time_history': {'log_steps': 100}}}
    I0407 21:19:48.303357 140114967496512 classifier_trainer.py:184] Overriding params: {'model_dir': '/home/ubuntu/log_hpu_8', 'mode': 'train_and_eval', 'model': {'name': 'efficientnet'}, 'runtime': {'run_eagerly': None, 'tpu': None}, 'train_dataset': {'data_dir': '/home/ubuntu'}, 'validation_dataset': {'data_dir': '/home/ubuntu'}, 'train': {'time_history': {'log_steps': 100}}}
    I0407 21:19:48.304113 140201244120896 classifier_trainer.py:190] Final model parameters: {'evaluation': {'epochs_between_evals': 1, 'skip_eval': True, 'steps': None},
     'export': {'checkpoint': None, 'destination': None},
     'mode': 'train_and_eval',
     'model': {'learning_rate': {'boundaries': None,
                                 'decay_epochs': 2.4,
                                 'decay_rate': 0.97,
                                 'examples_per_epoch': None,
                                 'initial_lr': 0.008,
                                 'multipliers': None,
                                 'name': 'exponential',
                                 'scale_by_batch_size': 0.0078125,
                                 'staircase': True,
                                 'warmup_epochs': 5},
               'loss': {'label_smoothing': 0.1, 'name': 'categorical_crossentropy'},
               'model_params': {'model_name': 'efficientnet-b0',
                                'model_weights_path': '',
                                'overrides': {'activation': 'swish',
                                              'batch_norm': 'default',
                                              'dtype': 'float32',
                                              'num_classes': 1000,
                                              'rescale_input': True},
                                'weights_format': 'saved_model'},
               'name': 'efficientnet',
               'num_classes': 1000,
               'optimizer': {'beta_1': None,
                             'beta_2': None,
                             'decay': 0.9,
                             'epsilon': 0.001,
                             'lookahead': False,
                             'momentum': 0.9,
                             'moving_average_decay': 0.0,
                             'name': 'rmsprop',
                             'nesterov': None}},
     'model_dir': '/home/ubuntu/log_hpu_8',
     'model_name': None,
     'runtime': {'all_reduce_alg': None,
                 'batchnorm_spatial_persistent': False,
                 'dataset_num_private_threads': None,
                 'default_shard_dim': -1,
                 'distribution_strategy': 'hpu',
                 'enable_xla': False,
                 'gpu_thread_mode': None,
                 'loss_scale': None,
                 'mixed_precision_dtype': None,
                 'num_cores_per_replica': 1,
                 'num_gpus': 0,
                 'num_packs': 1,
                 'per_gpu_thread_count': 0,
                 'run_eagerly': None,
                 'task_index': -1,
                 'tpu': None,
                 'tpu_enable_xla_dynamic_padder': None,
                 'worker_hosts': None},
     'train': {'callbacks': {'enable_backup_and_restore': False,
                             'enable_checkpoint_and_export': True,
                             'enable_tensorboard': True,
                             'enable_time_history': True},
               'epochs': 1,
               'metrics': ['accuracy', 'top_5'],
               'resume_checkpoint': True,
               'set_epoch_loop': False,
               'steps': 1000,
               'tensorboard': {'track_lr': True, 'write_model_weights': False},
               'time_history': {'log_steps': 100}},
     'train_dataset': {'augmenter': {'name': 'autoaugment', 'params': None},
                       'batch_size': 32,
                       'builder': 'synthetic',
                       'cache': False,
                       'data_dir': '/home/ubuntu',
                       'download': False,
                       'dtype': 'float32',
                       'file_shuffle_buffer_size': 1024,
                       'filenames': None,
                       'image_size': 224,
                       'mean_subtract': False,
                       'name': 'imagenet2012',
                       'num_channels': 3,
                       'num_classes': 1000,
                       'num_devices': 1,
                       'num_examples': 1281167,
                       'one_hot': True,
                       'shuffle_buffer_size': 10000,
                       'skip_decoding': True,
                       'split': 'train',
                       'standardize': False,
                       'tf_data_service': None,
                       'use_per_replica_batch_size': True},
     'validation_dataset': I0407 21:19:48.304121 139742514054976 classifier_trainer.py:190] Final model parameters: {'evaluation': {'epochs_between_evals': 1, 'skip_eval': True, 'steps': None},
     'export': {'checkpoint': None, 'destination': None},
     'mode': 'train_and_eval',
     'model': {'learning_rate': {'boundaries': None,
                                 'decay_epochs': 2.4,
                                 'decay_rate': 0.97,
                                 'examples_per_epoch': None,
                                 'initial_lr': 0.008,
                                 'multipliers': None,
                                 'name': 'exponential',
                                 'scale_by_batch_size': 0.0078125,
                                 'staircase': True,
                                 'warmup_epochs': 5},
               'loss': {'label_smoothing': 0.1, 'name': 'categorical_crossentropy'},
               'model_params': {'model_name': 'efficientnet-b0',
                                'model_weights_path': '',
                                'overrides': {'activation': 'swish',
                                              'batch_norm': 'default',
                                              'dtype': 'float32',
                                              'num_classes': 1000,
                                              'rescale_input': True},
                                'weights_format': 'saved_model'},
               'name': 'efficientnet',
               'num_classes': 1000,
               'optimizer': {'beta_1': None,
                             'beta_2': None,
                             'decay': 0.9,
                             'epsilon': 0.001,
                             'lookahead': False,
                             'momentum': 0.9,
                             'moving_average_decay': 0.0,
                             'name': 'rmsprop',
                             'nesterov': None}},
     'model_dir': '/home/ubuntu/log_hpu_8',
     'model_name': None,
     'runtime': {'all_reduce_alg': None,
                 'batchnorm_spatial_persistent': False,
                 'dataset_num_private_threads': None,
                 'default_shard_dim': -1,
                 'distribution_strategy': 'hpu',
                 'enable_xla': False,
                 'gpu_thread_mode': None,
                 'loss_scale': None,
                 'mixed_precision_dtype': None,
                 'num_cores_per_replica': 1,
                 'num_gpus': 0,
                 'num_packs': 1,
                 'per_gpu_thread_count': 0,
                 'run_eagerly': None,
                 'task_index': -1,
                 'tpu': None,
                 'tpu_enable_xla_dynamic_padder': None,
                 'worker_hosts': None},
     'train': {'callbacks': {'enable_backup_and_restore': False,
                             'enable_checkpoint_and_export': True,
                             'enable_tensorboard': True,
                             'enable_time_history': True},
               'epochs': 1,
               'metrics': ['accuracy', 'top_5'],
               'resume_checkpoint': True,
               'set_epoch_loop': False,
               'steps': 1000,
               'tensorboard': {'track_lr': True, 'write_model_weights': False},
               'time_history': {'log_steps': 100}},
     'train_dataset': {'augmenter': {'name': 'autoaugment', 'params': None},
                       'batch_size': 32,
                       'builder': 'synthetic',
                       'cache': False,
                       'data_dir': '/home/ubuntu',
                       'download': False,
                       'dtype': 'float32',
                       'file_shuffle_buffer_size': 1024,
                       'filenames': None,
                       'image_size': 224,
                       'mean_subtract': False,
                       'name': 'imagenet2012',
                       'num_channels': 3,
                       'num_classes': 1000,
                       'num_devices': 1,
                       'num_examples': 1281167,
                       'one_hot': True,
                       'shuffle_buffer_size': 10000,
                       'skip_decoding': True,
                       'split': 'train',
                       'standardize': False,
                       'tf_data_service': None,
                       'use_per_replica_batch_size': True},
     'validation_dataset': {'augmenter': {'name': None, 'params': None},
                            'batch_size': 32,
                            'builder': 'synthetic',
                            'cache': False,
                            'data_dir': '/home/ubuntu',
                            'download': False,
                            'dtype': 'float32',
                            'file_shuffle_buffer_size': 1024,
                            'filenames': None,
                            'image_size': 224,
                            'mean_subtract': False,
                            'name': 'imagenet2012',
                            'num_channels': 3,
                            'num_classes': 1000,
                            'num_devices': 1,
                            'num_examples': 50000,
                            'one_hot': True,
                            'shuffle_buffer_size': 10000,
                            'skip_decoding': True,
                            'split': 'validation',
                            'standardize': False,
                            'tf_data_service': None,
                            'use_per_replica_batch_size': True}}
    {'augmenter': {'name': None, 'params': None},
                            'batch_size': 32,
                            'builder': 'synthetic',
                            'cache': False,
                            'data_dir': '/home/ubuntu',
                            'download': False,
                            'dtype': 'float32',
                            'file_shuffle_buffer_size': 1024,
                            'filenames': None,
                            'image_size': 224,
                            'mean_subtract': False,
                            'name': 'imagenet2012',
                            'num_channels': 3,
                            'num_classes': 1000,
                            'num_devices': 1,
                            'num_examples': 50000,
                            'one_hot': True,
                            'shuffle_buffer_size': 10000,
                            'skip_decoding': True,
                            'split': 'validation',
                            'standardize': False,
                            'tf_data_service': None,
                            'use_per_replica_batch_size': True}}
    I0407 21:19:48.304208 139742514054976 classifier_trainer.py:290] Running train and eval.
    I0407 21:19:48.304153 140463173154624 classifier_trainer.py:190] Final model parameters: {'evaluation': {'epochs_between_evals': 1, 'skip_eval': True, 'steps': None},
     'export': {'checkpoint': None, 'destination': None},
     'mode': 'train_and_eval',
     'model': {'learning_rate': {'boundaries': None,
                                 'decay_epochs': 2.4,
                                 'decay_rate': 0.97,
                                 'examples_per_epoch': None,
                                 'initial_lr': 0.008,
                                 'multipliers': None,
                                 'name': 'exponential',
                                 'scale_by_batch_size': 0.0078125,
                                 'staircase': True,
                                 'warmup_epochs': 5},
               'loss': {'label_smoothing': 0.1, 'name': 'categorical_crossentropy'},
               'model_params': {'model_name': 'efficientnet-b0',
                                'model_weights_path': '',
                                'overrides': {'activation': 'swish',
                                              'batch_norm': 'default',
                                              'dtype': 'float32',
                                              'num_classes': 1000,
                                              'rescale_input': True},
                                'weights_format': 'saved_model'},
               'name': 'efficientnet',
               'num_classes': 1000,
               'optimizer': {'beta_1': None,
                             'beta_2': None,
                             'decay': 0.9,
                             'epsilon': 0.001,
                             'lookahead': False,
                             'momentum': 0.9,
                             'moving_average_decay': 0.0,
                             'name': 'rmsprop',
                             'nesterov': None}},
     'model_dir': '/home/ubuntu/log_hpu_8',
     'model_name': None,
     'runtime': {'all_reduce_alg': None,
                 'batchnorm_spatial_persistent': False,
                 'dataset_num_private_threads': None,
                 'default_shard_dim': -1,
                 'distribution_strategy': 'hpu',
                 'enable_xla': False,
                 'gpu_thread_mode': None,
                 'loss_scale': None,
                 'mixed_precision_dtype': None,
                 'num_cores_per_replica': 1,
                 'num_gpus': 0,
                 'num_packs': 1,
                 'per_gpu_thread_count': 0,
                 'run_eagerly': None,
                 'task_index': -1,
                 'tpu': None,
                 'tpu_enable_xla_dynamic_padder': None,
                 'worker_hosts': None},
     'train': {'callbacks': {'enable_backup_and_restore': False,
                             'enable_checkpoint_and_export': True,
                             'enable_tensorboard': True,
                             'enable_time_history': True},
               'epochs': 1,
               'metrics': ['accuracy', 'top_5'],
               'resume_checkpoint': True,
               'set_epoch_loop': False,
               'steps': 1000,
               'tensorboard': {'track_lr': True, 'write_model_weights': False},
               'time_history': {'log_steps': 100}},
     'train_dataset': {'augmenter': {'name': 'autoaugment', 'params': None},
                       'batch_size': 32,
                       'builder': 'synthetic',
                       'cache': False,
                       'data_dir': '/home/ubuntu',
                       'download': False,
                       'dtype': 'float32',
                       'file_shuffle_buffer_size': 1024,
                       'filenames': None,
                       'image_size': 224,
                       'mean_subtract': False,
                       'name': 'imagenet2012',
                       'num_channels': 3,
                       'num_classes': 1000,
                       'num_devices': 1,
                       'num_examples': 1281167,
                       'one_hot': True,
                       'shuffle_buffer_size': 10000,
                       'skip_decoding': True,
                       'split': 'train',
                       'standardize': False,
                       'tf_data_service': None,
                       'use_per_replica_batch_size': True},
     'validation_dataset': I0407 21:19:48.304199 140201244120896 classifier_trainer.py:290] Running train and eval.
    {'augmenter': {'name': None, 'params': None},
                            'batch_size': 32,
                            'builder': 'synthetic',
                            'cache': False,
                            'data_dir': '/home/ubuntu',
                            'download': False,
                            'dtype': 'float32',
                            'file_shuffle_buffer_size': 1024,
                            'filenames': None,
                            'image_size': 224,
                            'mean_subtract': False,
                            'name': 'imagenet2012',
                            'num_channels': 3,
                            'num_classes': 1000,
                            'num_devices': 1,
                            'num_examples': 50000,
                            'one_hot': True,
                            'shuffle_buffer_size': 10000,
                            'skip_decoding': True,
                            'split': 'validation',
                            'standardize': False,
                            'tf_data_service': None,
                            'use_per_replica_batch_size': True}}
    I0407 21:19:48.304240 140463173154624 classifier_trainer.py:290] Running train and eval.
    I0407 21:19:48.304372 140034574755648 classifier_trainer.py:184] Overriding params: runtime.num_gpus=0,runtime.distribution_strategy="hpu",train_dataset.builder="synthetic",validation_dataset.builder="synthetic",train.steps=1000,train.epochs=1,evaluation.skip_eval=True
    I0407 21:19:48.304400 139785568282432 classifier_trainer.py:184] Overriding params: runtime.num_gpus=0,runtime.distribution_strategy="hpu",train_dataset.builder="synthetic",validation_dataset.builder="synthetic",train.steps=1000,train.epochs=1,evaluation.skip_eval=True
    I0407 21:19:48.304573 140114967496512 classifier_trainer.py:190] Final model parameters: {'evaluation': {'epochs_between_evals': 1, 'skip_eval': True, 'steps': None},
     'export': {'checkpoint': None, 'destination': None},
     'mode': 'train_and_eval',
     'model': {'learning_rate': {'boundaries': None,
                                 'decay_epochs': 2.4,
                                 'decay_rate': 0.97,
                                 'examples_per_epoch': None,
                                 'initial_lr': 0.008,
                                 'multipliers': None,
                                 'name': 'exponential',
                                 'scale_by_batch_size': 0.0078125,
                                 'staircase': True,
                                 'warmup_epochs': 5},
               'loss': {'label_smoothing': 0.1, 'name': 'categorical_crossentropy'},
               'model_params': {'model_name': 'efficientnet-b0',
                                'model_weights_path': '',
                                'overrides': {'activation': 'swish',
                                              'batch_norm': 'default',
                                              'dtype': 'float32',
                                              'num_classes': 1000,
                                              'rescale_input': True},
                                'weights_format': 'saved_model'},
               'name': 'efficientnet',
               'num_classes': 1000,
               'optimizer': {'beta_1': None,
                             'beta_2': None,
                             'decay': 0.9,
                             'epsilon': 0.001,
                             'lookahead': False,
                             'momentum': 0.9,
                             'moving_average_decay': 0.0,
                             'name': 'rmsprop',
                             'nesterov': None}},
     'model_dir': '/home/ubuntu/log_hpu_8',
     'model_name': None,
     'runtime': {'all_reduce_alg': None,
                 'batchnorm_spatial_persistent': False,
                 'dataset_num_private_threads': None,
                 'default_shard_dim': -1,
                 'distribution_strategy': 'hpu',
                 'enable_xla': False,
                 'gpu_thread_mode': None,
                 'loss_scale': None,
                 'mixed_precision_dtype': None,
                 'num_cores_per_replica': 1,
                 'num_gpus': 0,
                 'num_packs': 1,
                 'per_gpu_thread_count': 0,
                 'run_eagerly': None,
                 'task_index': -1,
                 'tpu': None,
                 'tpu_enable_xla_dynamic_padder': None,
                 'worker_hosts': None},
     'train': {'callbacks': {'enable_backup_and_restore': False,
                             'enable_checkpoint_and_export': True,
                             'enable_tensorboard': True,
                             'enable_time_history': True},
               'epochs': 1,
               'metrics': ['accuracy', 'top_5'],
               'resume_checkpoint': True,
               'set_epoch_loop': False,
               'steps': 1000,
               'tensorboard': {'track_lr': True, 'write_model_weights': False},
               'time_history': {'log_steps': 100}},
     'train_dataset': {'augmenter': {'name': 'autoaugment', 'params': None},
                       'batch_size': 32,
                       'builder': 'synthetic',
                       'cache': False,
                       'data_dir': '/home/ubuntu',
                       'download': False,
                       'dtype': 'float32',
                       'file_shuffle_buffer_size': 1024,
                       'filenames': None,
                       'image_size': 224,
                       'mean_subtract': False,
                       'name': 'imagenet2012',
                       'num_channels': 3,
                       'num_classes': 1000,
                       'num_devices': 1,
                       'num_examples': 1281167,
                       'one_hot': True,
                       'shuffle_buffer_size': 10000,
                       'skip_decoding': True,
                       'split': 'train',
                       'standardize': False,
                       'tf_data_service': None,
                       'use_per_replica_batch_size': True},
     'validation_dataset': {'augmenter': {'name': None, 'params': None},
                            'batch_size': 32,
                            'builder': 'synthetic',
                            'cache': False,
                            'data_dir': '/home/ubuntu',
                            'download': False,
                            'dtype': 'float32',
                            'file_shuffle_buffer_size': 1024,
                            'filenames': None,
                            'image_size': 224,
                            'mean_subtract': False,
                            'name': 'imagenet2012',
                            'num_channels': 3,
                            'num_classes': 1000,
                            'num_devices': 1,
                            'num_examples': 50000,
                            'one_hot': True,
                            'shuffle_buffer_size': 10000,
                            'skip_decoding': True,
                            'split': 'validation',
                            'standardize': False,
                            'tf_data_service': None,
                            'use_per_replica_batch_size': True}}
    I0407 21:19:48.304654 140114967496512 classifier_trainer.py:290] Running train and eval.
    I0407 21:19:48.305505 140034574755648 classifier_trainer.py:184] Overriding params: {'model_dir': '/home/ubuntu/log_hpu_8', 'mode': 'train_and_eval', 'model': {'name': 'efficientnet'}, 'runtime': {'run_eagerly': None, 'tpu': None}, 'train_dataset': {'data_dir': '/home/ubuntu'}, 'validation_dataset': {'data_dir': '/home/ubuntu'}, 'train': {'time_history': {'log_steps': 100}}}
    I0407 21:19:48.305552 139785568282432 classifier_trainer.py:184] Overriding params: {'model_dir': '/home/ubuntu/log_hpu_8', 'mode': 'train_and_eval', 'model': {'name': 'efficientnet'}, 'runtime': {'run_eagerly': None, 'tpu': None}, 'train_dataset': {'data_dir': '/home/ubuntu'}, 'validation_dataset': {'data_dir': '/home/ubuntu'}, 'train': {'time_history': {'log_steps': 100}}}
    2022-04-07 21:19:48.305700: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2022-04-07 21:19:48.305702: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2022-04-07 21:19:48.305700: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2022-04-07 21:19:48.306118: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    I0407 21:19:48.306701 140034574755648 classifier_trainer.py:190] Final model parameters: {'evaluation': {'epochs_between_evals': 1, 'skip_eval': True, 'steps': None},
     'export': {'checkpoint': None, 'destination': None},
     'mode': 'train_and_eval',
     'model': {'learning_rate': {'boundaries': None,
                                 'decay_epochs': 2.4,
                                 'decay_rate': 0.97,
                                 'examples_per_epoch': None,
                                 'initial_lr': 0.008,
                                 'multipliers': None,
                                 'name': 'exponential',
                                 'scale_by_batch_size': 0.0078125,
                                 'staircase': True,
                                 'warmup_epochs': 5},
               'loss': {'label_smoothing': 0.1, 'name': 'categorical_crossentropy'},
               'model_params': {'model_name': 'efficientnet-b0',
                                'model_weights_path': '',
                                'overrides': {'activation': 'swish',
                                              'batch_norm': 'default',
                                              'dtype': 'float32',
                                              'num_classes': 1000,
                                              'rescale_input': True},
                                'weights_format': 'saved_model'},
               'name': 'efficientnet',
               'num_classes': 1000,
               'optimizer': {'beta_1': None,
                             'beta_2': None,
                             'decay': 0.9,
                             'epsilon': 0.001,
                             'lookahead': False,
                             'momentum': 0.9,
                             'moving_average_decay': 0.0,
                             'name': 'rmsprop',
                             'nesterov': None}},
     'model_dir': '/home/ubuntu/log_hpu_8',
     'model_name': None,
     'runtime': {'all_reduce_alg': None,
                 'batchnorm_spatial_persistent': False,
                 'dataset_num_private_threads': None,
                 'default_shard_dim': -1,
                 'distribution_strategy': 'hpu',
                 'enable_xla': False,
                 'gpu_thread_mode': None,
                 'loss_scale': None,
                 'mixed_precision_dtype': None,
                 'num_cores_per_replica': 1,
                 'num_gpus': 0,
                 'num_packs': 1,
                 'per_gpu_thread_count': 0,
                 'run_eagerly': None,
                 'task_index': -1,
                 'tpu': None,
                 'tpu_enable_xla_dynamic_padder': None,
                 'worker_hosts': None},
     'train': {'callbacks': {'enable_backup_and_restore': False,
                             'enable_checkpoint_and_export': True,
                             'enable_tensorboard': True,
                             'enable_time_history': True},
               'epochs': 1,
               'metrics': ['accuracy', 'top_5'],
               'resume_checkpoint': True,
               'set_epoch_loop': False,
               'steps': 1000,
               'tensorboard': {'track_lr': True, 'write_model_weights': False},
               'time_history': {'log_steps': 100}},
     'train_dataset': {'augmenter': {'name': 'autoaugment', 'params': None},
                       'batch_size': 32,
                       'builder': 'synthetic',
                       'cache': False,
                       'data_dir': '/home/ubuntu',
                       'download': False,
                       'dtype': 'float32',
                       'file_shuffle_buffer_size': 1024,
                       'filenames': None,
                       'image_size': 224,
                       'mean_subtract': False,
                       'name': 'imagenet2012',
                       'num_channels': 3,
                       'num_classes': 1000,
                       'num_devices': 1,
                       'num_examples': 1281167,
                       'one_hot': True,
                       'shuffle_buffer_size': 10000,
                       'skip_decoding': True,
                       'split': 'train',
                       'standardize': False,
                       'tf_data_service': None,
                       'use_per_replica_batch_size': True},
     'validation_dataset': {'augmenter': {'name': None, 'params': None},
                            'batch_size': 32,
                            'builder': 'synthetic',
                            'cache': False,
                            'data_dir': '/home/ubuntu',
                            'download': False,
                            'dtype': 'float32',
                            'file_shuffle_buffer_size': 1024,
                            'filenames': None,
                            'image_size': 224,
                            'mean_subtract': False,
                            'name': 'imagenet2012',
                            'num_channels': 3,
                            'num_classes': 1000,
                            'num_devices': 1,
                            'num_examples': 50000,
                            'one_hot': True,
                            'shuffle_buffer_size': 10000,
                            'skip_decoding': True,
                            'split': 'validation',
                            'standardize': False,
                            'tf_data_service': None,
                            'use_per_replica_batch_size': True}}
    I0407 21:19:48.306785 140034574755648 classifier_trainer.py:290] Running train and eval.
    I0407 21:19:48.306741 139785568282432 classifier_trainer.py:190] Final model parameters: {'evaluation': {'epochs_between_evals': 1, 'skip_eval': True, 'steps': None},
     'export': {'checkpoint': None, 'destination': None},
     'mode': 'train_and_eval',
     'model': {'learning_rate': {'boundaries': None,
                                 'decay_epochs': 2.4,
                                 'decay_rate': 0.97,
                                 'examples_per_epoch': None,
                                 'initial_lr': 0.008,
                                 'multipliers': None,
                                 'name': 'exponential',
                                 'scale_by_batch_size': 0.0078125,
                                 'staircase': True,
                                 'warmup_epochs': 5},
               'loss': {'label_smoothing': 0.1, 'name': 'categorical_crossentropy'},
               'model_params': {'model_name': 'efficientnet-b0',
                                'model_weights_path': '',
                                'overrides': {'activation': 'swish',
                                              'batch_norm': 'default',
                                              'dtype': 'float32',
                                              'num_classes': 1000,
                                              'rescale_input': True},
                                'weights_format': 'saved_model'},
               'name': 'efficientnet',
               'num_classes': 1000,
               'optimizer': {'beta_1': None,
                             'beta_2': None,
                             'decay': 0.9,
                             'epsilon': 0.001,
                             'lookahead': False,
                             'momentum': 0.9,
                             'moving_average_decay': 0.0,
                             'name': 'rmsprop',
                             'nesterov': None}},
     'model_dir': '/home/ubuntu/log_hpu_8',
     'model_name': None,
     'runtime': {'all_reduce_alg': None,
                 'batchnorm_spatial_persistent': False,
                 'dataset_num_private_threads': None,
                 'default_shard_dim': -1,
                 'distribution_strategy': 'hpu',
                 'enable_xla': False,
                 'gpu_thread_mode': None,
                 'loss_scale': None,
                 'mixed_precision_dtype': None,
                 'num_cores_per_replica': 1,
                 'num_gpus': 0,
                 'num_packs': 1,
                 'per_gpu_thread_count': 0,
                 'run_eagerly': None,
                 'task_index': -1,
                 'tpu': None,
                 'tpu_enable_xla_dynamic_padder': None,
                 'worker_hosts': None},
     'train': {'callbacks': {'enable_backup_and_restore': False,
                             'enable_checkpoint_and_export': True,
                             'enable_tensorboard': True,
                             'enable_time_history': True},
               'epochs': 1,
               'metrics': ['accuracy', 'top_5'],
               'resume_checkpoint': True,
               'set_epoch_loop': False,
               'steps': 1000,
               'tensorboard': {'track_lr': True, 'write_model_weights': False},
               'time_history': {'log_steps': 100}},
     'train_dataset': {'augmenter': {'name': 'autoaugment', 'params': None},
                       'batch_size': 32,
                       'builder': 'synthetic',
                       'cache': False,
                       'data_dir': '/home/ubuntu',
                       'download': False,
                       'dtype': 'float32',
                       'file_shuffle_buffer_size': 1024,
                       'filenames': None,
                       'image_size': 224,
                       'mean_subtract': False,
                       'name': 'imagenet2012',
                       'num_channels': 3,
                       'num_classes': 1000,
                       'num_devices': 1,
                       'num_examples': 1281167,
                       'one_hot': True,
                       'shuffle_buffer_size': 10000,
                       'skip_decoding': True,
                       'split': 'train',
                       'standardize': False,
                       'tf_data_service': None,
                       'use_per_replica_batch_size': True},
     'validation_dataset': {'augmenter': {'name': None, 'params': None},
                            'batch_size': 32,
                            'builder': 'synthetic',
                            'cache': False,
                            'data_dir': '/home/ubuntu',
                            'download': False,
                            'dtype': 'float32',
                            'file_shuffle_buffer_size': 1024,
                            'filenames': None,
                            'image_size': 224,
                            'mean_subtract': False,
                            'name': 'imagenet2012',
                            'num_channels': 3,
                            'num_classes': 1000,
                            'num_devices': 1,
                            'num_examples': 50000,
                            'one_hot': True,
                            'shuffle_buffer_size': 10000,
                            'skip_decoding': True,
                            'split': 'validation',
                            'standardize': False,
                            'tf_data_service': None,
                            'use_per_replica_batch_size': True}}
    I0407 21:19:48.306832 139785568282432 classifier_trainer.py:290] Running train and eval.
    2022-04-07 21:19:48.308326: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2022-04-07 21:19:48.308366: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2022-04-07 21:19:50.414054: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/habana_device.cpp:201] HPU initialization done for library version 1.3.0_c61303b7_tf2.8.0
    2022-04-07 21:19:50.438308: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:272] Initialize GrpcChannelCache for job worker -> {0 -> 127.0.0.1:2410, 1 -> 127.0.0.1:2411, 2 -> 127.0.0.1:2412, 3 -> 127.0.0.1:2413, 4 -> 127.0.0.1:2414, 5 -> 127.0.0.1:2415, 6 -> 127.0.0.1:2416, 7 -> 127.0.0.1:2417}
    2022-04-07 21:19:50.438575: I tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:437] Started server with target: grpc://127.0.0.1:2413
    INFO:tensorflow:Enabled multi-worker collective ops with available devices: ['/job:worker/replica:0/task:3/device:CPU:0', '/job:worker/replica:0/task:3/device:HPU:0']
    I0407 21:19:50.439154 140189656872768 collective_all_reduce_strategy.py:512] Enabled multi-worker collective ops with available devices: ['/job:worker/replica:0/task:3/device:CPU:0', '/job:worker/replica:0/task:3/device:HPU:0']
       local_devices=('/job:worker/task:3/device:HPU:0',)
    2022-04-07 21:19:50.445021: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/habana_device.cpp:201] HPU initialization done for library version 1.3.0_c61303b7_tf2.8.0
    2022-04-07 21:19:50.454988: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:272] Initialize GrpcChannelCache for job worker -> {0 -> 127.0.0.1:2410, 1 -> 127.0.0.1:2411, 2 -> 127.0.0.1:2412, 3 -> 127.0.0.1:2413, 4 -> 127.0.0.1:2414, 5 -> 127.0.0.1:2415, 6 -> 127.0.0.1:2416, 7 -> 127.0.0.1:2417}
    2022-04-07 21:19:50.455214: I tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:437] Started server with target: grpc://127.0.0.1:2411
    INFO:tensorflow:Enabled multi-worker collective ops with available devices: ['/job:worker/replica:0/task:1/device:CPU:0', '/job:worker/replica:0/task:1/device:HPU:0']
    I0407 21:19:50.456224 140136124327744 collective_all_reduce_strategy.py:512] Enabled multi-worker collective ops with available devices: ['/job:worker/replica:0/task:1/device:CPU:0', '/job:worker/replica:0/task:1/device:HPU:0']
       local_devices=('/job:worker/task:1/device:HPU:0',)
    INFO:tensorflow:Waiting for the cluster, timeout = inf
    I0407 21:19:50.498365 140189656872768 collective_all_reduce_strategy.py:896] Waiting for the cluster, timeout = inf
    2022-04-07 21:19:50.505613: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/habana_device.cpp:201] HPU initialization done for library version 1.3.0_c61303b7_tf2.8.0
    INFO:tensorflow:Waiting for the cluster, timeout = inf
    I0407 21:19:50.509046 140136124327744 collective_all_reduce_strategy.py:896] Waiting for the cluster, timeout = inf
    2022-04-07 21:19:50.514268: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/habana_device.cpp:201] HPU initialization done for library version 1.3.0_c61303b7_tf2.8.0
    2022-04-07 21:19:50.515225: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:272] Initialize GrpcChannelCache for job worker -> {0 -> 127.0.0.1:2410, 1 -> 127.0.0.1:2411, 2 -> 127.0.0.1:2412, 3 -> 127.0.0.1:2413, 4 -> 127.0.0.1:2414, 5 -> 127.0.0.1:2415, 6 -> 127.0.0.1:2416, 7 -> 127.0.0.1:2417}
    2022-04-07 21:19:50.515437: I tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:437] Started server with target: grpc://127.0.0.1:2415
    INFO:tensorflow:Enabled multi-worker collective ops with available devices: ['/job:worker/replica:0/task:5/device:CPU:0', '/job:worker/replica:0/task:5/device:HPU:0']
    I0407 21:19:50.518040 140034574755648 collective_all_reduce_strategy.py:512] Enabled multi-worker collective ops with available devices: ['/job:worker/replica:0/task:5/device:CPU:0', '/job:worker/replica:0/task:5/device:HPU:0']
    2022-04-07 21:19:50.518324: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/habana_device.cpp:201] HPU initialization done for library version 1.3.0_c61303b7_tf2.8.0
       local_devices=('/job:worker/task:5/device:HPU:0',)
    2022-04-07 21:19:50.524402: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:272] Initialize GrpcChannelCache for job worker -> {0 -> 127.0.0.1:2410, 1 -> 127.0.0.1:2411, 2 -> 127.0.0.1:2412, 3 -> 127.0.0.1:2413, 4 -> 127.0.0.1:2414, 5 -> 127.0.0.1:2415, 6 -> 127.0.0.1:2416, 7 -> 127.0.0.1:2417}
    2022-04-07 21:19:50.524603: I tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:437] Started server with target: grpc://127.0.0.1:2417
    2022-04-07 21:19:50.525076: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/habana_device.cpp:201] HPU initialization done for library version 1.3.0_c61303b7_tf2.8.0
    INFO:tensorflow:Enabled multi-worker collective ops with available devices: ['/job:worker/replica:0/task:7/device:CPU:0', '/job:worker/replica:0/task:7/device:HPU:0']
    I0407 21:19:50.527722 139785568282432 collective_all_reduce_strategy.py:512] Enabled multi-worker collective ops with available devices: ['/job:worker/replica:0/task:7/device:CPU:0', '/job:worker/replica:0/task:7/device:HPU:0']
    2022-04-07 21:19:50.528168: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/habana_device.cpp:201] HPU initialization done for library version 1.3.0_c61303b7_tf2.8.0
       local_devices=('/job:worker/task:7/device:HPU:0',)
    2022-04-07 21:19:50.529514: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:272] Initialize GrpcChannelCache for job worker -> {0 -> 127.0.0.1:2410, 1 -> 127.0.0.1:2411, 2 -> 127.0.0.1:2412, 3 -> 127.0.0.1:2413, 4 -> 127.0.0.1:2414, 5 -> 127.0.0.1:2415, 6 -> 127.0.0.1:2416, 7 -> 127.0.0.1:2417}
    2022-04-07 21:19:50.529818: I tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:437] Started server with target: grpc://127.0.0.1:2410
    INFO:tensorflow:Enabled multi-worker collective ops with available devices: ['/job:worker/replica:0/task:0/device:CPU:0', '/job:worker/replica:0/task:0/device:HPU:0']
    I0407 21:19:50.533108 139742514054976 collective_all_reduce_strategy.py:512] Enabled multi-worker collective ops with available devices: ['/job:worker/replica:0/task:0/device:CPU:0', '/job:worker/replica:0/task:0/device:HPU:0']
       local_devices=('/job:worker/task:0/device:HPU:0',)
    2022-04-07 21:19:50.535966: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:272] Initialize GrpcChannelCache for job worker -> {0 -> 127.0.0.1:2410, 1 -> 127.0.0.1:2411, 2 -> 127.0.0.1:2412, 3 -> 127.0.0.1:2413, 4 -> 127.0.0.1:2414, 5 -> 127.0.0.1:2415, 6 -> 127.0.0.1:2416, 7 -> 127.0.0.1:2417}
    2022-04-07 21:19:50.536192: I tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:437] Started server with target: grpc://127.0.0.1:2414
    INFO:tensorflow:Enabled multi-worker collective ops with available devices: ['/job:worker/replica:0/task:4/device:CPU:0', '/job:worker/replica:0/task:4/device:HPU:0']
    I0407 21:19:50.537276 140114967496512 collective_all_reduce_strategy.py:512] Enabled multi-worker collective ops with available devices: ['/job:worker/replica:0/task:4/device:CPU:0', '/job:worker/replica:0/task:4/device:HPU:0']
       local_devices=('/job:worker/task:4/device:HPU:0',)
    2022-04-07 21:19:50.538203: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:272] Initialize GrpcChannelCache for job worker -> {0 -> 127.0.0.1:2410, 1 -> 127.0.0.1:2411, 2 -> 127.0.0.1:2412, 3 -> 127.0.0.1:2413, 4 -> 127.0.0.1:2414, 5 -> 127.0.0.1:2415, 6 -> 127.0.0.1:2416, 7 -> 127.0.0.1:2417}
    2022-04-07 21:19:50.538412: I tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:437] Started server with target: grpc://127.0.0.1:2412
    INFO:tensorflow:Enabled multi-worker collective ops with available devices: ['/job:worker/replica:0/task:2/device:CPU:0', '/job:worker/replica:0/task:2/device:HPU:0']
    I0407 21:19:50.539853 140201244120896 collective_all_reduce_strategy.py:512] Enabled multi-worker collective ops with available devices: ['/job:worker/replica:0/task:2/device:CPU:0', '/job:worker/replica:0/task:2/device:HPU:0']
       local_devices=('/job:worker/task:2/device:HPU:0',)
    2022-04-07 21:19:50.543954: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/habana_device.cpp:201] HPU initialization done for library version 1.3.0_c61303b7_tf2.8.0
    2022-04-07 21:19:50.553696: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:272] Initialize GrpcChannelCache for job worker -> {0 -> 127.0.0.1:2410, 1 -> 127.0.0.1:2411, 2 -> 127.0.0.1:2412, 3 -> 127.0.0.1:2413, 4 -> 127.0.0.1:2414, 5 -> 127.0.0.1:2415, 6 -> 127.0.0.1:2416, 7 -> 127.0.0.1:2417}
    2022-04-07 21:19:50.553896: I tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:437] Started server with target: grpc://127.0.0.1:2416
    INFO:tensorflow:Enabled multi-worker collective ops with available devices: ['/job:worker/replica:0/task:6/device:CPU:0', '/job:worker/replica:0/task:6/device:HPU:0']
    I0407 21:19:50.554864 140463173154624 collective_all_reduce_strategy.py:512] Enabled multi-worker collective ops with available devices: ['/job:worker/replica:0/task:6/device:CPU:0', '/job:worker/replica:0/task:6/device:HPU:0']
       local_devices=('/job:worker/task:6/device:HPU:0',)
    2022-04-07 21:19:50.560244: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/habana_device_binding_iface.cpp:59] Found TensorFlow library with SHA256: 1f4e3d3c8f90c158c442f60b6b1fafd64cfb678fd7c4f954804e0ba91497c2a0
    INFO:tensorflow:Waiting for the cluster, timeout = inf
    I0407 21:19:50.568480 140034574755648 collective_all_reduce_strategy.py:896] Waiting for the cluster, timeout = inf
    2022-04-07 21:19:50.573401: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/habana_device_binding_iface.cpp:59] Found TensorFlow library with SHA256: 1f4e3d3c8f90c158c442f60b6b1fafd64cfb678fd7c4f954804e0ba91497c2a0
    INFO:tensorflow:Waiting for the cluster, timeout = inf
    I0407 21:19:50.578067 139785568282432 collective_all_reduce_strategy.py:896] Waiting for the cluster, timeout = inf
    INFO:tensorflow:Waiting for the cluster, timeout = inf
    I0407 21:19:50.588580 140114967496512 collective_all_reduce_strategy.py:896] Waiting for the cluster, timeout = inf
    INFO:tensorflow:Waiting for the cluster, timeout = inf
    I0407 21:19:50.591804 139742514054976 collective_all_reduce_strategy.py:896] Waiting for the cluster, timeout = inf
    INFO:tensorflow:Waiting for the cluster, timeout = inf
    I0407 21:19:50.592601 140201244120896 collective_all_reduce_strategy.py:896] Waiting for the cluster, timeout = inf
    INFO:tensorflow:Waiting for the cluster, timeout = inf
    I0407 21:19:50.608811 140463173154624 collective_all_reduce_strategy.py:896] Waiting for the cluster, timeout = inf
    2022-04-07 21:19:50.638115: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/habana_device_binding_iface.cpp:59] Found TensorFlow library with SHA256: 1f4e3d3c8f90c158c442f60b6b1fafd64cfb678fd7c4f954804e0ba91497c2a0
    2022-04-07 21:19:50.647253: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/habana_device_binding_iface.cpp:59] Found TensorFlow library with SHA256: 1f4e3d3c8f90c158c442f60b6b1fafd64cfb678fd7c4f954804e0ba91497c2a0
    2022-04-07 21:19:50.655333: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/habana_device_binding_iface.cpp:59] Found TensorFlow library with SHA256: 1f4e3d3c8f90c158c442f60b6b1fafd64cfb678fd7c4f954804e0ba91497c2a0
    2022-04-07 21:19:50.655336: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/habana_device_binding_iface.cpp:59] Found TensorFlow library with SHA256: 1f4e3d3c8f90c158c442f60b6b1fafd64cfb678fd7c4f954804e0ba91497c2a0
    2022-04-07 21:19:50.655333: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/habana_device_binding_iface.cpp:59] Found TensorFlow library with SHA256: 1f4e3d3c8f90c158c442f60b6b1fafd64cfb678fd7c4f954804e0ba91497c2a0
    2022-04-07 21:19:50.673600: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/habana_device_binding_iface.cpp:59] Found TensorFlow library with SHA256: 1f4e3d3c8f90c158c442f60b6b1fafd64cfb678fd7c4f954804e0ba91497c2a0
    INFO:tensorflow:Cluster is ready.
    I0407 21:19:51.513360 139742514054976 collective_all_reduce_strategy.py:912] Cluster is ready.
    INFO:tensorflow:Cluster is ready.
    I0407 21:19:51.513549 140034574755648 collective_all_reduce_strategy.py:912] Cluster is ready.
    INFO:tensorflow:Cluster is ready.
    I0407 21:19:51.513755 140136124327744 collective_all_reduce_strategy.py:912] Cluster is ready.
    INFO:tensorflow:Cluster is ready.
    INFO:tensorflow:Cluster is ready.
    INFO:tensorflow:Cluster is ready.
    I0407 21:19:51.513862 140201244120896 collective_all_reduce_strategy.py:912] Cluster is ready.
    INFO:tensorflow:Cluster is ready.
    I0407 21:19:51.513897 140189656872768 collective_all_reduce_strategy.py:912] Cluster is ready.
    I0407 21:19:51.513926 140463173154624 collective_all_reduce_strategy.py:912] Cluster is ready.
    I0407 21:19:51.513886 139785568282432 collective_all_reduce_strategy.py:912] Cluster is ready.
    INFO:tensorflow:Cluster is ready.
    I0407 21:19:51.514081 140114967496512 collective_all_reduce_strategy.py:912] Cluster is ready.
    INFO:tensorflow:MultiWorkerMirroredStrategy with cluster_spec = {'worker': ['127.0.0.1:2410', '127.0.0.1:2411', '127.0.0.1:2412', '127.0.0.1:2413', '127.0.0.1:2414', '127.0.0.1:2415', '127.0.0.1:2416', '127.0.0.1:2417']}, task_type = 'worker', task_id = 2, num_workers = 8, local_devices = ('/job:worker/task:2/device:HPU:0',), communication = CommunicationImplementation.AUTO
    I0407 21:19:51.518817 140201244120896 collective_all_reduce_strategy.py:557] MultiWorkerMirroredStrategy with cluster_spec = {'worker': ['127.0.0.1:2410', '127.0.0.1:2411', '127.0.0.1:2412', '127.0.0.1:2413', '127.0.0.1:2414', '127.0.0.1:2415', '127.0.0.1:2416', '127.0.0.1:2417']}, task_type = 'worker', task_id = 2, num_workers = 8, local_devices = ('/job:worker/task:2/device:HPU:0',), communication = CommunicationImplementation.AUTO
    INFO:tensorflow:MultiWorkerMirroredStrategy with cluster_spec = {'worker': ['127.0.0.1:2410', '127.0.0.1:2411', '127.0.0.1:2412', '127.0.0.1:2413', '127.0.0.1:2414', '127.0.0.1:2415', '127.0.0.1:2416', '127.0.0.1:2417']}, task_type = 'worker', task_id = 1, num_workers = 8, local_devices = ('/job:worker/task:1/device:HPU:0',), communication = CommunicationImplementation.AUTO
    I0407 21:19:51.519039 140136124327744 collective_all_reduce_strategy.py:557] MultiWorkerMirroredStrategy with cluster_spec = {'worker': ['127.0.0.1:2410', '127.0.0.1:2411', '127.0.0.1:2412', '127.0.0.1:2413', '127.0.0.1:2414', '127.0.0.1:2415', '127.0.0.1:2416', '127.0.0.1:2417']}, task_type = 'worker', task_id = 1, num_workers = 8, local_devices = ('/job:worker/task:1/device:HPU:0',), communication = CommunicationImplementation.AUTO
    INFO:tensorflow:MultiWorkerMirroredStrategy with cluster_spec = {'worker': ['127.0.0.1:2410', '127.0.0.1:2411', '127.0.0.1:2412', '127.0.0.1:2413', '127.0.0.1:2414', '127.0.0.1:2415', '127.0.0.1:2416', '127.0.0.1:2417']}, task_type = 'worker', task_id = 0, num_workers = 8, local_devices = ('/job:worker/task:0/device:HPU:0',), communication = CommunicationImplementation.AUTO
    I0407 21:19:51.519109 139742514054976 collective_all_reduce_strategy.py:557] MultiWorkerMirroredStrategy with cluster_spec = {'worker': ['127.0.0.1:2410', '127.0.0.1:2411', '127.0.0.1:2412', '127.0.0.1:2413', '127.0.0.1:2414', '127.0.0.1:2415', '127.0.0.1:2416', '127.0.0.1:2417']}, task_type = 'worker', task_id = 0, num_workers = 8, local_devices = ('/job:worker/task:0/device:HPU:0',), communication = CommunicationImplementation.AUTO
    INFO:tensorflow:MultiWorkerMirroredStrategy with cluster_spec = {'worker': ['127.0.0.1:2410', '127.0.0.1:2411', '127.0.0.1:2412', '127.0.0.1:2413', '127.0.0.1:2414', '127.0.0.1:2415', '127.0.0.1:2416', '127.0.0.1:2417']}, task_type = 'worker', task_id = 3, num_workers = 8, local_devices = ('/job:worker/task:3/device:HPU:0',), communication = CommunicationImplementation.AUTO
    I0407 21:19:51.519206 140189656872768 collective_all_reduce_strategy.py:557] MultiWorkerMirroredStrategy with cluster_spec = {'worker': ['127.0.0.1:2410', '127.0.0.1:2411', '127.0.0.1:2412', '127.0.0.1:2413', '127.0.0.1:2414', '127.0.0.1:2415', '127.0.0.1:2416', '127.0.0.1:2417']}, task_type = 'worker', task_id = 3, num_workers = 8, local_devices = ('/job:worker/task:3/device:HPU:0',), communication = CommunicationImplementation.AUTO
    INFO:tensorflow:MultiWorkerMirroredStrategy with cluster_spec = {'worker': ['127.0.0.1:2410', '127.0.0.1:2411', '127.0.0.1:2412', '127.0.0.1:2413', '127.0.0.1:2414', '127.0.0.1:2415', '127.0.0.1:2416', '127.0.0.1:2417']}, task_type = 'worker', task_id = 4, num_workers = 8, local_devices = ('/job:worker/task:4/device:HPU:0',), communication = CommunicationImplementation.AUTO
    I0407 21:19:51.519400 140114967496512 collective_all_reduce_strategy.py:557] MultiWorkerMirroredStrategy with cluster_spec = {'worker': ['127.0.0.1:2410', '127.0.0.1:2411', '127.0.0.1:2412', '127.0.0.1:2413', '127.0.0.1:2414', '127.0.0.1:2415', '127.0.0.1:2416', '127.0.0.1:2417']}, task_type = 'worker', task_id = 4, num_workers = 8, local_devices = ('/job:worker/task:4/device:HPU:0',), communication = CommunicationImplementation.AUTO
    INFO:tensorflow:MultiWorkerMirroredStrategy with cluster_spec = {'worker': ['127.0.0.1:2410', '127.0.0.1:2411', '127.0.0.1:2412', '127.0.0.1:2413', '127.0.0.1:2414', '127.0.0.1:2415', '127.0.0.1:2416', '127.0.0.1:2417']}, task_type = 'worker', task_id = 5, num_workers = 8, local_devices = ('/job:worker/task:5/device:HPU:0',), communication = CommunicationImplementation.AUTO
    INFO:tensorflow:MultiWorkerMirroredStrategy with cluster_spec = {'worker': ['127.0.0.1:2410', '127.0.0.1:2411', '127.0.0.1:2412', '127.0.0.1:2413', '127.0.0.1:2414', '127.0.0.1:2415', '127.0.0.1:2416', '127.0.0.1:2417']}, task_type = 'worker', task_id = 7, num_workers = 8, local_devices = ('/job:worker/task:7/device:HPU:0',), communication = CommunicationImplementation.AUTO
    INFO:tensorflow:MultiWorkerMirroredStrategy with cluster_spec = {'worker': ['127.0.0.1:2410', '127.0.0.1:2411', '127.0.0.1:2412', '127.0.0.1:2413', '127.0.0.1:2414', '127.0.0.1:2415', '127.0.0.1:2416', '127.0.0.1:2417']}, task_type = 'worker', task_id = 6, num_workers = 8, local_devices = ('/job:worker/task:6/device:HPU:0',), communication = CommunicationImplementation.AUTO
    I0407 21:19:51.519869 140034574755648 collective_all_reduce_strategy.py:557] MultiWorkerMirroredStrategy with cluster_spec = {'worker': ['127.0.0.1:2410', '127.0.0.1:2411', '127.0.0.1:2412', '127.0.0.1:2413', '127.0.0.1:2414', '127.0.0.1:2415', '127.0.0.1:2416', '127.0.0.1:2417']}, task_type = 'worker', task_id = 5, num_workers = 8, local_devices = ('/job:worker/task:5/device:HPU:0',), communication = CommunicationImplementation.AUTO
    I0407 21:19:51.519908 139785568282432 collective_all_reduce_strategy.py:557] MultiWorkerMirroredStrategy with cluster_spec = {'worker': ['127.0.0.1:2410', '127.0.0.1:2411', '127.0.0.1:2412', '127.0.0.1:2413', '127.0.0.1:2414', '127.0.0.1:2415', '127.0.0.1:2416', '127.0.0.1:2417']}, task_type = 'worker', task_id = 7, num_workers = 8, local_devices = ('/job:worker/task:7/device:HPU:0',), communication = CommunicationImplementation.AUTO
    I0407 21:19:51.519909 140463173154624 collective_all_reduce_strategy.py:557] MultiWorkerMirroredStrategy with cluster_spec = {'worker': ['127.0.0.1:2410', '127.0.0.1:2411', '127.0.0.1:2412', '127.0.0.1:2413', '127.0.0.1:2414', '127.0.0.1:2415', '127.0.0.1:2416', '127.0.0.1:2417']}, task_type = 'worker', task_id = 6, num_workers = 8, local_devices = ('/job:worker/task:6/device:HPU:0',), communication = CommunicationImplementation.AUTO
    2022-04-07 21:19:51.526977: I /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/synapse_helpers/hccl_communicator.cpp:56] Opening communication. Device id:0.
    2022-04-07 21:19:51.527027: I /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/synapse_helpers/hccl_communicator.cpp:56] Opening communication. Device id:0.
    2022-04-07 21:19:51.526952: I /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/synapse_helpers/hccl_communicator.cpp:56] Opening communication. Device id:0.
    2022-04-07 21:19:51.527034: I /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/synapse_helpers/hccl_communicator.cpp:56] Opening communication. Device id:0.
    2022-04-07 21:19:51.527144: I /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/synapse_helpers/hccl_communicator.cpp:56] Opening communication. Device id:0.
    2022-04-07 21:19:51.527203: I /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/synapse_helpers/hccl_communicator.cpp:56] Opening communication. Device id:0.
    2022-04-07 21:19:51.527227: I /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/synapse_helpers/hccl_communicator.cpp:56] Opening communication. Device id:0.
    2022-04-07 21:19:51.527293: I /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/synapse_helpers/hccl_communicator.cpp:56] Opening communication. Device id:0.
    I0407 21:20:00.533817 139785568282432 classifier_trainer.py:317] Detected 8 devices.
    I0407 21:20:00.533812 139742514054976 classifier_trainer.py:317] Detected 8 devices.
    I0407 21:20:00.533806 140201244120896 classifier_trainer.py:317] Detected 8 devices.
    I0407 21:20:00.533843 140114967496512 classifier_trainer.py:317] Detected 8 devices.
    W0407 21:20:00.534065 139742514054976 classifier_trainer.py:105] label_smoothing > 0, so datasets will be one hot encoded.
    W0407 21:20:00.534066 140201244120896 classifier_trainer.py:105] label_smoothing > 0, so datasets will be one hot encoded.
    I0407 21:20:00.533882 140189656872768 classifier_trainer.py:317] Detected 8 devices.
    W0407 21:20:00.534082 139785568282432 classifier_trainer.py:105] label_smoothing > 0, so datasets will be one hot encoded.
    W0407 21:20:00.534098 140114967496512 classifier_trainer.py:105] label_smoothing > 0, so datasets will be one hot encoded.
    W0407 21:20:00.534149 140189656872768 classifier_trainer.py:105] label_smoothing > 0, so datasets will be one hot encoded.
    I0407 21:20:00.534021 140136124327744 classifier_trainer.py:317] Detected 8 devices.
    I0407 21:20:00.534235 139742514054976 dataset_factory.py:175] Using augmentation: autoaugment
    I0407 21:20:00.534243 140201244120896 dataset_factory.py:175] Using augmentation: autoaugment
    I0407 21:20:00.534256 139785568282432 dataset_factory.py:175] Using augmentation: autoaugment
    I0407 21:20:00.534268 140114967496512 dataset_factory.py:175] Using augmentation: autoaugment
    W0407 21:20:00.534294 140136124327744 classifier_trainer.py:105] label_smoothing > 0, so datasets will be one hot encoded.
    I0407 21:20:00.534340 140189656872768 dataset_factory.py:175] Using augmentation: autoaugment
    I0407 21:20:00.534379 139742514054976 dataset_factory.py:175] Using augmentation: None
    I0407 21:20:00.534388 140201244120896 dataset_factory.py:175] Using augmentation: None
    I0407 21:20:00.534408 140114967496512 dataset_factory.py:175] Using augmentation: None
    I0407 21:20:00.534400 139785568282432 dataset_factory.py:175] Using augmentation: None
    I0407 21:20:00.534484 140136124327744 dataset_factory.py:175] Using augmentation: autoaugment
    I0407 21:20:00.534232 140463173154624 classifier_trainer.py:317] Detected 8 devices.
    I0407 21:20:00.534511 140189656872768 dataset_factory.py:175] Using augmentation: None
    W0407 21:20:00.534597 140463173154624 classifier_trainer.py:105] label_smoothing > 0, so datasets will be one hot encoded.
    I0407 21:20:00.534645 140136124327744 dataset_factory.py:175] Using augmentation: None
    I0407 21:20:00.534648 139742514054976 dataset_factory.py:383] Generating a synthetic dataset.
    I0407 21:20:00.534648 140201244120896 dataset_factory.py:383] Generating a synthetic dataset.
    I0407 21:20:00.534669 140114967496512 dataset_factory.py:383] Generating a synthetic dataset.
    I0407 21:20:00.534662 139785568282432 dataset_factory.py:383] Generating a synthetic dataset.
    I0407 21:20:00.534320 140034574755648 classifier_trainer.py:317] Detected 8 devices.
    I0407 21:20:00.534787 140189656872768 dataset_factory.py:383] Generating a synthetic dataset.
    W0407 21:20:00.534810 140034574755648 classifier_trainer.py:105] label_smoothing > 0, so datasets will be one hot encoded.
    I0407 21:20:00.534915 140136124327744 dataset_factory.py:383] Generating a synthetic dataset.
    I0407 21:20:00.534910 140463173154624 dataset_factory.py:175] Using augmentation: autoaugment
    I0407 21:20:00.535100 140034574755648 dataset_factory.py:175] Using augmentation: autoaugment
    I0407 21:20:00.535149 140463173154624 dataset_factory.py:175] Using augmentation: None
    I0407 21:20:00.535305 140034574755648 dataset_factory.py:175] Using augmentation: None
    I0407 21:20:00.535491 140463173154624 dataset_factory.py:383] Generating a synthetic dataset.
    I0407 21:20:00.535716 140034574755648 dataset_factory.py:383] Generating a synthetic dataset.
    I0407 21:20:00.577314 140114967496512 dataset_factory.py:410] Sharding the dataset: input_pipeline_id=8 num_input_pipelines=4
    I0407 21:20:00.577314 139742514054976 dataset_factory.py:410] Sharding the dataset: input_pipeline_id=8 num_input_pipelines=0
    I0407 21:20:00.577322 140201244120896 dataset_factory.py:410] Sharding the dataset: input_pipeline_id=8 num_input_pipelines=2
    I0407 21:20:00.577329 139785568282432 dataset_factory.py:410] Sharding the dataset: input_pipeline_id=8 num_input_pipelines=7
    I0407 21:20:00.580077 140136124327744 dataset_factory.py:410] Sharding the dataset: input_pipeline_id=8 num_input_pipelines=1
    I0407 21:20:00.580415 140189656872768 dataset_factory.py:410] Sharding the dataset: input_pipeline_id=8 num_input_pipelines=3
    I0407 21:20:00.587378 140034574755648 dataset_factory.py:410] Sharding the dataset: input_pipeline_id=8 num_input_pipelines=5
    I0407 21:20:00.595860 140463173154624 dataset_factory.py:410] Sharding the dataset: input_pipeline_id=8 num_input_pipelines=6
    I0407 21:20:05.037709 139742514054976 dataset_factory.py:383] Generating a synthetic dataset.
    I0407 21:20:05.040988 139785568282432 dataset_factory.py:383] Generating a synthetic dataset.
    I0407 21:20:05.042808 139742514054976 dataset_factory.py:410] Sharding the dataset: input_pipeline_id=8 num_input_pipelines=0
    I0407 21:20:05.046068 139785568282432 dataset_factory.py:410] Sharding the dataset: input_pipeline_id=8 num_input_pipelines=7
    I0407 21:20:05.060549 140114967496512 dataset_factory.py:383] Generating a synthetic dataset.
    I0407 21:20:05.065685 140114967496512 dataset_factory.py:410] Sharding the dataset: input_pipeline_id=8 num_input_pipelines=4
    I0407 21:20:05.121664 140463173154624 dataset_factory.py:383] Generating a synthetic dataset.
    I0407 21:20:05.126890 140463173154624 dataset_factory.py:410] Sharding the dataset: input_pipeline_id=8 num_input_pipelines=6
    I0407 21:20:05.213415 139742514054976 classifier_trainer.py:338] Global batch size: 256
    I0407 21:20:05.216362 139785568282432 classifier_trainer.py:338] Global batch size: 256
    I0407 21:20:05.232483 139742514054976 efficientnet_model.py:147] round_filter input=32 output=32
    I0407 21:20:05.235569 139785568282432 efficientnet_model.py:147] round_filter input=32 output=32
    I0407 21:20:05.236334 140114967496512 classifier_trainer.py:338] Global batch size: 256
    I0407 21:20:05.255840 140114967496512 efficientnet_model.py:147] round_filter input=32 output=32
    I0407 21:20:05.300533 140463173154624 classifier_trainer.py:338] Global batch size: 256
    I0407 21:20:05.320420 140463173154624 efficientnet_model.py:147] round_filter input=32 output=32
    I0407 21:20:05.404847 140201244120896 dataset_factory.py:383] Generating a synthetic dataset.
    I0407 21:20:05.410961 140201244120896 dataset_factory.py:410] Sharding the dataset: input_pipeline_id=8 num_input_pipelines=2
    I0407 21:20:05.455298 140136124327744 dataset_factory.py:383] Generating a synthetic dataset.
    I0407 21:20:05.460806 140136124327744 dataset_factory.py:410] Sharding the dataset: input_pipeline_id=8 num_input_pipelines=1
    I0407 21:20:05.483447 140189656872768 dataset_factory.py:383] Generating a synthetic dataset.
    I0407 21:20:05.489000 140189656872768 dataset_factory.py:410] Sharding the dataset: input_pipeline_id=8 num_input_pipelines=3
    I0407 21:20:05.514868 140034574755648 dataset_factory.py:383] Generating a synthetic dataset.
    I0407 21:20:05.520523 140034574755648 dataset_factory.py:410] Sharding the dataset: input_pipeline_id=8 num_input_pipelines=5
    I0407 21:20:05.604112 140201244120896 classifier_trainer.py:338] Global batch size: 256
    I0407 21:20:05.625365 140201244120896 efficientnet_model.py:147] round_filter input=32 output=32
    I0407 21:20:05.646771 140136124327744 classifier_trainer.py:338] Global batch size: 256
    I0407 21:20:05.667397 140136124327744 efficientnet_model.py:147] round_filter input=32 output=32
    I0407 21:20:05.674137 140189656872768 classifier_trainer.py:338] Global batch size: 256
    I0407 21:20:05.695134 140189656872768 efficientnet_model.py:147] round_filter input=32 output=32
    I0407 21:20:05.708576 140034574755648 classifier_trainer.py:338] Global batch size: 256
    I0407 21:20:05.728269 140034574755648 efficientnet_model.py:147] round_filter input=32 output=32
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:05.977098 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:05.977374 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:05.977396 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:05.977441 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:05.977552 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:05.977562 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:05.977602 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:05.977625 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:05.986969 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:05.986968 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:05.986975 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:05.987117 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:05.987167 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:05.987185 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:05.987213 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:05.987201 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:05.999180 140463173154624 efficientnet_model.py:147] round_filter input=32 output=32
    I0407 21:20:05.999256 140463173154624 efficientnet_model.py:147] round_filter input=16 output=16
    I0407 21:20:05.999248 139785568282432 efficientnet_model.py:147] round_filter input=32 output=32
    I0407 21:20:05.999252 140189656872768 efficientnet_model.py:147] round_filter input=32 output=32
    I0407 21:20:05.999326 139785568282432 efficientnet_model.py:147] round_filter input=16 output=16
    I0407 21:20:05.999328 140189656872768 efficientnet_model.py:147] round_filter input=16 output=16
    I0407 21:20:06.000368 140136124327744 efficientnet_model.py:147] round_filter input=32 output=32
    I0407 21:20:06.000452 140136124327744 efficientnet_model.py:147] round_filter input=16 output=16
    I0407 21:20:06.000697 140034574755648 efficientnet_model.py:147] round_filter input=32 output=32
    I0407 21:20:06.000780 140034574755648 efficientnet_model.py:147] round_filter input=16 output=16
    I0407 21:20:06.000822 140201244120896 efficientnet_model.py:147] round_filter input=32 output=32
    I0407 21:20:06.000841 139742514054976 efficientnet_model.py:147] round_filter input=32 output=32
    I0407 21:20:06.000900 140201244120896 efficientnet_model.py:147] round_filter input=16 output=16
    I0407 21:20:06.000898 140114967496512 efficientnet_model.py:147] round_filter input=32 output=32
    I0407 21:20:06.000926 139742514054976 efficientnet_model.py:147] round_filter input=16 output=16
    I0407 21:20:06.000978 140114967496512 efficientnet_model.py:147] round_filter input=16 output=16
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.058768 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.058897 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.058886 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.058892 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.058966 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.059178 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.059231 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.059268 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.066682 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.066820 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.066824 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.066853 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.067079 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.067378 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.067936 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.068112 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.236324 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.236466 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.236465 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.236527 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.236598 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.236699 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.236715 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.236877 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.246183 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.246445 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.246577 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.246518 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.247462 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.247586 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.247911 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.247952 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.255631 139742514054976 efficientnet_model.py:147] round_filter input=16 output=16
    I0407 21:20:06.255740 139742514054976 efficientnet_model.py:147] round_filter input=24 output=24
    I0407 21:20:06.256086 140201244120896 efficientnet_model.py:147] round_filter input=16 output=16
    I0407 21:20:06.256081 140034574755648 efficientnet_model.py:147] round_filter input=16 output=16
    I0407 21:20:06.256174 140201244120896 efficientnet_model.py:147] round_filter input=24 output=24
    I0407 21:20:06.256167 140034574755648 efficientnet_model.py:147] round_filter input=24 output=24
    I0407 21:20:06.256209 140463173154624 efficientnet_model.py:147] round_filter input=16 output=16
    I0407 21:20:06.256296 140463173154624 efficientnet_model.py:147] round_filter input=24 output=24
    I0407 21:20:06.258033 140189656872768 efficientnet_model.py:147] round_filter input=16 output=16
    I0407 21:20:06.258128 140189656872768 efficientnet_model.py:147] round_filter input=24 output=24
    I0407 21:20:06.258084 140136124327744 efficientnet_model.py:147] round_filter input=16 output=16
    I0407 21:20:06.258166 140136124327744 efficientnet_model.py:147] round_filter input=24 output=24
    I0407 21:20:06.258662 139785568282432 efficientnet_model.py:147] round_filter input=16 output=16
    I0407 21:20:06.258750 139785568282432 efficientnet_model.py:147] round_filter input=24 output=24
    I0407 21:20:06.258752 140114967496512 efficientnet_model.py:147] round_filter input=16 output=16
    I0407 21:20:06.258845 140114967496512 efficientnet_model.py:147] round_filter input=24 output=24
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.326489 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.326560 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.326627 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.326617 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.326699 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.326756 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.327741 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.327741 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.334474 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.334501 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.334729 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.334743 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.335522 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.335597 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.339230 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.339230 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.414632 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.414733 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.414869 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.414901 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.414895 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.414908 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.415040 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.415109 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.422688 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.422722 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.422838 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.422911 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.422977 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.423003 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.423709 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.423997 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.593343 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.593489 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.593535 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.593549 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.593604 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.593786 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.593902 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.593952 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.601495 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.601489 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.601494 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.601712 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.602254 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.602637 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.602761 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.602843 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.717971 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.718042 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.718189 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.718184 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.718104 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.718348 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.718373 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.718500 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.725869 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.725980 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.726171 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.726225 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.726766 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.727057 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.727158 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.727419 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.794616 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.794819 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.794835 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.794832 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.794844 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.795002 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.795062 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.795462 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.802521 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.802644 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.802658 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.802666 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.802697 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.803788 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.803796 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.805361 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.983741 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.983839 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.983847 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.983853 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.983961 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.983973 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.984054 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.984201 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.991780 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.991871 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.991968 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.991992 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.992096 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.992115 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.992928 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:06.993216 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.004648 140136124327744 efficientnet_model.py:147] round_filter input=24 output=24
    I0407 21:20:07.004725 140136124327744 efficientnet_model.py:147] round_filter input=40 output=40
    I0407 21:20:07.005038 139785568282432 efficientnet_model.py:147] round_filter input=24 output=24
    I0407 21:20:07.005117 139785568282432 efficientnet_model.py:147] round_filter input=40 output=40
    I0407 21:20:07.005141 140463173154624 efficientnet_model.py:147] round_filter input=24 output=24
    I0407 21:20:07.005218 140463173154624 efficientnet_model.py:147] round_filter input=40 output=40
    I0407 21:20:07.005272 140034574755648 efficientnet_model.py:147] round_filter input=24 output=24
    I0407 21:20:07.005350 140034574755648 efficientnet_model.py:147] round_filter input=40 output=40
    I0407 21:20:07.005444 140201244120896 efficientnet_model.py:147] round_filter input=24 output=24
    I0407 21:20:07.005530 140201244120896 efficientnet_model.py:147] round_filter input=40 output=40
    I0407 21:20:07.005697 139742514054976 efficientnet_model.py:147] round_filter input=24 output=24
    I0407 21:20:07.005781 139742514054976 efficientnet_model.py:147] round_filter input=40 output=40
    I0407 21:20:07.007240 140114967496512 efficientnet_model.py:147] round_filter input=24 output=24
    I0407 21:20:07.007320 140114967496512 efficientnet_model.py:147] round_filter input=40 output=40
    I0407 21:20:07.007660 140189656872768 efficientnet_model.py:147] round_filter input=24 output=24
    I0407 21:20:07.007759 140189656872768 efficientnet_model.py:147] round_filter input=40 output=40
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.045767 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.045852 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.045901 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.045964 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.046028 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.046084 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.046192 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.046315 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.053500 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.053617 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.053622 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.053714 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.053896 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.053930 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.054865 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.055032 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.123508 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.123517 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.123523 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.123556 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.123566 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.123634 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.123910 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.124024 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.131239 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.131297 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.131304 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.131305 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.131469 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.132182 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.132570 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.132915 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.267008 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.267137 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.267229 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.267239 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.267316 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.267477 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.267546 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.267584 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.275203 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.275293 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.275311 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.275341 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.276135 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.276457 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.276455 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.276927 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.388778 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.388762 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.388846 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.388880 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.388900 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.388936 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.389007 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.389322 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.396642 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.396708 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.396794 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.396804 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.396886 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.396858 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.396957 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.398110 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.465862 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.465872 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.465956 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.465965 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.465982 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.466030 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.466059 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.466087 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.473755 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.473734 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.473796 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.473834 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.473848 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.473880 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.473950 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.474719 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.657534 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.657650 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.657678 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.657723 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.657699 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.658043 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.658666 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.658791 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.665341 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.665489 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.665611 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.665536 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.665668 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.666108 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.670032 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.670032 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.678112 140463173154624 efficientnet_model.py:147] round_filter input=40 output=40
    I0407 21:20:07.678186 140463173154624 efficientnet_model.py:147] round_filter input=80 output=80
    I0407 21:20:07.678403 140201244120896 efficientnet_model.py:147] round_filter input=40 output=40
    I0407 21:20:07.678480 140201244120896 efficientnet_model.py:147] round_filter input=80 output=80
    I0407 21:20:07.678461 140136124327744 efficientnet_model.py:147] round_filter input=40 output=40
    I0407 21:20:07.678534 140136124327744 efficientnet_model.py:147] round_filter input=80 output=80
    I0407 21:20:07.678697 140189656872768 efficientnet_model.py:147] round_filter input=40 output=40
    I0407 21:20:07.678779 140189656872768 efficientnet_model.py:147] round_filter input=80 output=80
    I0407 21:20:07.679375 140034574755648 efficientnet_model.py:147] round_filter input=40 output=40
    I0407 21:20:07.679455 140034574755648 efficientnet_model.py:147] round_filter input=80 output=80
    I0407 21:20:07.680083 139785568282432 efficientnet_model.py:147] round_filter input=40 output=40
    I0407 21:20:07.680171 139785568282432 efficientnet_model.py:147] round_filter input=80 output=80
    I0407 21:20:07.688803 140114967496512 efficientnet_model.py:147] round_filter input=40 output=40
    I0407 21:20:07.688909 140114967496512 efficientnet_model.py:147] round_filter input=80 output=80
    I0407 21:20:07.688802 139742514054976 efficientnet_model.py:147] round_filter input=40 output=40
    I0407 21:20:07.688910 139742514054976 efficientnet_model.py:147] round_filter input=80 output=80
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.731008 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.731188 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.731264 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.731269 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.731444 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.731576 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.731621 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.731669 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.738869 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.738927 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.738940 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.739146 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.739882 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.740046 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.740328 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.740364 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.806028 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.806095 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.806189 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.806231 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.806344 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.806510 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.806536 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.806539 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.813887 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.813924 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.814138 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.814258 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.814807 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.815208 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.815203 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.815458 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.944231 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.944260 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.944304 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.944384 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.944422 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.944426 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.944592 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.944638 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.952252 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.952312 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.952464 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.952547 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.952565 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.953407 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.953532 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:07.953641 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.062700 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.062757 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.062880 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.062916 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.062914 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.062966 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.063001 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.063243 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.070754 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.070760 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.070836 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.070913 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.071054 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.071028 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.071916 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.072152 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.139014 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.139169 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.139276 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.139357 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.139481 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.139490 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.139599 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.139716 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.147031 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.147143 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.147493 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.148063 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.148291 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.148357 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.148596 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.150340 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.359551 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.359605 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.359641 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.359772 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.359766 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.359922 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.359963 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.359960 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.367638 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.367662 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.367854 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.368674 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.368773 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.368778 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.368876 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.368882 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.422017 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.422055 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.422051 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.422159 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.422266 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.422428 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.422483 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.422474 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.432831 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.433060 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.433107 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.433955 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.433958 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.434257 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.434273 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.434415 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.489330 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.489362 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.489373 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.489457 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.489455 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.489565 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.489644 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.489848 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.497145 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.497153 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.497174 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.497322 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.497753 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.497871 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.498476 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.499212 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.597141 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.597286 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.597415 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.597460 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.597444 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.597528 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.597602 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.597921 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.605021 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.605209 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.605241 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.605318 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.605378 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.605495 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.605884 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.607182 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.617992 140034574755648 efficientnet_model.py:147] round_filter input=80 output=80
    I0407 21:20:08.618074 140034574755648 efficientnet_model.py:147] round_filter input=112 output=112
    I0407 21:20:08.618081 139742514054976 efficientnet_model.py:147] round_filter input=80 output=80
    I0407 21:20:08.618152 139742514054976 efficientnet_model.py:147] round_filter input=112 output=112
    I0407 21:20:08.618249 140136124327744 efficientnet_model.py:147] round_filter input=80 output=80
    I0407 21:20:08.618323 140136124327744 efficientnet_model.py:147] round_filter input=112 output=112
    I0407 21:20:08.618320 140463173154624 efficientnet_model.py:147] round_filter input=80 output=80
    I0407 21:20:08.618353 140201244120896 efficientnet_model.py:147] round_filter input=80 output=80
    I0407 21:20:08.618397 140463173154624 efficientnet_model.py:147] round_filter input=112 output=112
    I0407 21:20:08.618439 140201244120896 efficientnet_model.py:147] round_filter input=112 output=112
    I0407 21:20:08.618673 140114967496512 efficientnet_model.py:147] round_filter input=80 output=80
    I0407 21:20:08.618756 140114967496512 efficientnet_model.py:147] round_filter input=112 output=112
    I0407 21:20:08.619489 139785568282432 efficientnet_model.py:147] round_filter input=80 output=80
    I0407 21:20:08.619578 139785568282432 efficientnet_model.py:147] round_filter input=112 output=112
    I0407 21:20:08.622045 140189656872768 efficientnet_model.py:147] round_filter input=80 output=80
    I0407 21:20:08.622133 140189656872768 efficientnet_model.py:147] round_filter input=112 output=112
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.660546 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.660567 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.660750 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.660808 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.660861 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.660905 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.660930 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.660991 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.668354 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.668964 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.669113 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.669199 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.669415 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.669623 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.669749 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.669766 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.739427 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.739738 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.739757 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.739820 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.739934 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.740155 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.740720 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.740719 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.747338 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.747761 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.747773 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.748104 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.748855 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.749176 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.752105 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.752105 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.901156 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.901176 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.901235 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.901258 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.901340 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.901435 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.901491 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.901546 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.909094 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.909289 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.909366 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.909358 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.910073 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.910079 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.910152 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:08.910423 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.024724 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.024912 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.024916 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.024925 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.024980 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.024996 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.025284 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.025292 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.032726 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.032881 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.032910 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.033051 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.033097 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.033466 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.033969 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.034225 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.106550 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.106597 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.106603 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.106653 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.106720 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.106749 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.106879 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.106873 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.114475 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.114433 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.114488 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.114524 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.114733 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.115640 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.115708 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.116064 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.306067 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.306089 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.306081 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.306147 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.306347 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.306341 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.306521 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.306566 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.313977 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.314054 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.314089 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.314138 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.315006 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.315091 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.315386 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.315580 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.368924 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.369110 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.369102 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.369209 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.369229 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.369242 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.369306 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.369492 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.376803 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.377300 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.377789 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.377935 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.377948 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.378171 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.378149 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.378359 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.428252 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.428297 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.428454 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.428487 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.428544 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.428557 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.428642 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.428885 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.436025 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.436092 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.436448 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.436498 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.436526 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.437106 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.437189 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.437994 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.539023 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.539022 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.539011 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.539044 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.539112 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.539139 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.539214 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.539222 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.546875 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.547084 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.547156 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.547178 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.547209 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.547479 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.548079 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.548105 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.559550 140463173154624 efficientnet_model.py:147] round_filter input=112 output=112
    I0407 21:20:09.559627 140463173154624 efficientnet_model.py:147] round_filter input=192 output=192
    I0407 21:20:09.560045 140201244120896 efficientnet_model.py:147] round_filter input=112 output=112
    I0407 21:20:09.560121 140201244120896 efficientnet_model.py:147] round_filter input=192 output=192
    I0407 21:20:09.560140 140114967496512 efficientnet_model.py:147] round_filter input=112 output=112
    I0407 21:20:09.560221 140114967496512 efficientnet_model.py:147] round_filter input=192 output=192
    I0407 21:20:09.560240 139742514054976 efficientnet_model.py:147] round_filter input=112 output=112
    I0407 21:20:09.560317 139742514054976 efficientnet_model.py:147] round_filter input=192 output=192
    I0407 21:20:09.560354 140189656872768 efficientnet_model.py:147] round_filter input=112 output=112
    I0407 21:20:09.560431 140189656872768 efficientnet_model.py:147] round_filter input=192 output=192
    I0407 21:20:09.560869 139785568282432 efficientnet_model.py:147] round_filter input=112 output=112
    I0407 21:20:09.560950 139785568282432 efficientnet_model.py:147] round_filter input=192 output=192
    I0407 21:20:09.562260 140034574755648 efficientnet_model.py:147] round_filter input=112 output=112
    I0407 21:20:09.562342 140034574755648 efficientnet_model.py:147] round_filter input=192 output=192
    I0407 21:20:09.562440 140136124327744 efficientnet_model.py:147] round_filter input=112 output=112
    I0407 21:20:09.562523 140136124327744 efficientnet_model.py:147] round_filter input=192 output=192
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.602439 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.602463 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.602561 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.602635 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.602646 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.602817 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.602814 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.602885 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.610299 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.610446 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.610583 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.610567 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.610824 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.611594 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.611678 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.612012 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.663732 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.663805 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.663946 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.663985 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.664079 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.664094 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.664099 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.664164 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.671547 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.671823 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.671995 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.672642 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.672813 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.672876 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.672889 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.673114 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.841407 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.841464 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.841525 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.841605 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.841619 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.841722 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.841855 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.841863 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.849260 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.849309 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.849409 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.849549 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.849702 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.850537 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.850627 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.851174 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.980919 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.980944 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.980998 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.981019 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.981113 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.981219 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.981279 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.981381 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.988880 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.988890 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.988987 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.989160 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.989642 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.989750 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.990139 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:09.990140 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.059325 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.059328 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.059413 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.059594 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.059633 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.059727 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.059822 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.059838 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.067455 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.067477 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.068225 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.068382 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.068624 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.068821 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.068880 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.069094 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.308891 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.309026 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.309160 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.309170 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.309267 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.309307 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.309302 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.309796 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.316945 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.317067 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.317115 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.317389 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.317396 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.317955 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.318220 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.319951 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.379057 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.379185 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.379227 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.379411 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.379419 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.379491 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.379522 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.379544 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.386967 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.387069 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.387255 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.388062 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.388420 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.388455 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.388459 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.388564 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.439211 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.439296 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.439289 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.439383 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.439395 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.439441 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.439470 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.439639 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.446996 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.447009 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.447120 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.447157 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.447272 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.447297 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.447381 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.448253 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.552059 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.552091 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.552172 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.552177 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.552316 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.552369 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.552349 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.552364 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.560169 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.560245 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.560280 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.561100 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.561256 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.561398 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.561474 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.561475 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.617615 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.617613 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.617628 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.617724 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.617739 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.617772 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.617895 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.618034 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.625539 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.625544 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.625734 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.625765 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.625933 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.626385 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.626815 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.626893 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.677046 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.677227 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.677268 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.677281 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.677304 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.677305 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.677389 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.677708 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.685045 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.685103 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.685127 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.685189 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.685295 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.685330 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.686332 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.687576 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.793041 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.793096 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.793131 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.793170 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.793206 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.793202 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.793282 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.793382 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.800967 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.801041 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.801764 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.801866 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.801927 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.801947 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.802046 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.802345 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.814094 140463173154624 efficientnet_model.py:147] round_filter input=192 output=192
    I0407 21:20:10.814172 140463173154624 efficientnet_model.py:147] round_filter input=320 output=320
    I0407 21:20:10.814169 140189656872768 efficientnet_model.py:147] round_filter input=192 output=192
    I0407 21:20:10.814248 140189656872768 efficientnet_model.py:147] round_filter input=320 output=320
    I0407 21:20:10.816016 140034574755648 efficientnet_model.py:147] round_filter input=192 output=192
    I0407 21:20:10.816096 140034574755648 efficientnet_model.py:147] round_filter input=320 output=320
    I0407 21:20:10.816175 140136124327744 efficientnet_model.py:147] round_filter input=192 output=192
    I0407 21:20:10.816256 140136124327744 efficientnet_model.py:147] round_filter input=320 output=320
    I0407 21:20:10.816360 140114967496512 efficientnet_model.py:147] round_filter input=192 output=192
    I0407 21:20:10.816442 140114967496512 efficientnet_model.py:147] round_filter input=320 output=320
    I0407 21:20:10.816423 140201244120896 efficientnet_model.py:147] round_filter input=192 output=192
    I0407 21:20:10.816451 139742514054976 efficientnet_model.py:147] round_filter input=192 output=192
    I0407 21:20:10.816504 140201244120896 efficientnet_model.py:147] round_filter input=320 output=320
    I0407 21:20:10.816534 139742514054976 efficientnet_model.py:147] round_filter input=320 output=320
    I0407 21:20:10.817492 139785568282432 efficientnet_model.py:147] round_filter input=192 output=192
    I0407 21:20:10.817581 139785568282432 efficientnet_model.py:147] round_filter input=320 output=320
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.860624 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.860661 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.860746 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.860744 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.860791 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.860801 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.860814 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.860882 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.868607 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.868671 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.869080 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.869458 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.869462 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.869664 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.869839 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.869797 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.943634 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.943880 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.943901 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.943929 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.943988 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.944006 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.944107 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.944135 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.951721 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.951801 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.951825 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.952044 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.952683 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.952785 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.952795 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:10.952808 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.131434 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.131619 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.131720 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.131706 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.131942 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.131978 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.132834 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.132835 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.139389 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.139624 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.139586 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.139727 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.140774 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.140821 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.144166 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.144166 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.148964 140201244120896 efficientnet_model.py:147] round_filter input=1280 output=1280
    I0407 21:20:11.149155 140189656872768 efficientnet_model.py:147] round_filter input=1280 output=1280
    I0407 21:20:11.149285 139742514054976 efficientnet_model.py:147] round_filter input=1280 output=1280
    I0407 21:20:11.149334 140136124327744 efficientnet_model.py:147] round_filter input=1280 output=1280
    I0407 21:20:11.151417 140034574755648 efficientnet_model.py:147] round_filter input=1280 output=1280
    I0407 21:20:11.151422 139785568282432 efficientnet_model.py:147] round_filter input=1280 output=1280
    I0407 21:20:11.157912 140463173154624 efficientnet_model.py:147] round_filter input=1280 output=1280
    I0407 21:20:11.157912 140114967496512 efficientnet_model.py:147] round_filter input=1280 output=1280
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.269740 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.270023 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.270045 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.270180 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.270290 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.270449 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.270459 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.270451 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.277825 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.278121 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.278146 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.278257 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.278405 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.279354 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.279422 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.279556 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:11.385871 139785568282432 efficientnet_model.py:457] Building model efficientnet with params ModelConfig(width_coefficient=1.0, depth_coefficient=1.0, resolution=224, dropout_rate=0.2, blocks=(BlockConfig(input_filters=32, output_filters=16, kernel_size=3, num_repeat=1, expand_ratio=1, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=16, output_filters=24, kernel_size=3, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=24, output_filters=40, kernel_size=5, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=40, output_filters=80, kernel_size=3, num_repeat=3, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=80, output_filters=112, kernel_size=5, num_repeat=3, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=112, output_filters=192, kernel_size=5, num_repeat=4, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=192, output_filters=320, kernel_size=3, num_repeat=1, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise')), stem_base_filters=32, top_base_filters=1280, activation='swish', batch_norm='default', bn_momentum=0.99, bn_epsilon=0.001, weight_decay=5e-06, drop_connect_rate=0.2, depth_divisor=8, min_depth=None, use_se=True, input_channels=3, num_classes=1000, model_name='efficientnet', rescale_input=True, data_format='channels_last', dtype='float32')
    I0407 21:20:11.386056 140463173154624 efficientnet_model.py:457] Building model efficientnet with params ModelConfig(width_coefficient=1.0, depth_coefficient=1.0, resolution=224, dropout_rate=0.2, blocks=(BlockConfig(input_filters=32, output_filters=16, kernel_size=3, num_repeat=1, expand_ratio=1, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=16, output_filters=24, kernel_size=3, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=24, output_filters=40, kernel_size=5, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=40, output_filters=80, kernel_size=3, num_repeat=3, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=80, output_filters=112, kernel_size=5, num_repeat=3, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=112, output_filters=192, kernel_size=5, num_repeat=4, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=192, output_filters=320, kernel_size=3, num_repeat=1, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise')), stem_base_filters=32, top_base_filters=1280, activation='swish', batch_norm='default', bn_momentum=0.99, bn_epsilon=0.001, weight_decay=5e-06, drop_connect_rate=0.2, depth_divisor=8, min_depth=None, use_se=True, input_channels=3, num_classes=1000, model_name='efficientnet', rescale_input=True, data_format='channels_last', dtype='float32')
    I0407 21:20:11.386133 140034574755648 efficientnet_model.py:457] Building model efficientnet with params ModelConfig(width_coefficient=1.0, depth_coefficient=1.0, resolution=224, dropout_rate=0.2, blocks=(BlockConfig(input_filters=32, output_filters=16, kernel_size=3, num_repeat=1, expand_ratio=1, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=16, output_filters=24, kernel_size=3, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=24, output_filters=40, kernel_size=5, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=40, output_filters=80, kernel_size=3, num_repeat=3, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=80, output_filters=112, kernel_size=5, num_repeat=3, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=112, output_filters=192, kernel_size=5, num_repeat=4, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=192, output_filters=320, kernel_size=3, num_repeat=1, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise')), stem_base_filters=32, top_base_filters=1280, activation='swish', batch_norm='default', bn_momentum=0.99, bn_epsilon=0.001, weight_decay=5e-06, drop_connect_rate=0.2, depth_divisor=8, min_depth=None, use_se=True, input_channels=3, num_classes=1000, model_name='efficientnet', rescale_input=True, data_format='channels_last', dtype='float32')
    I0407 21:20:11.386464 140201244120896 efficientnet_model.py:457] Building model efficientnet with params ModelConfig(width_coefficient=1.0, depth_coefficient=1.0, resolution=224, dropout_rate=0.2, blocks=(BlockConfig(input_filters=32, output_filters=16, kernel_size=3, num_repeat=1, expand_ratio=1, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=16, output_filters=24, kernel_size=3, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=24, output_filters=40, kernel_size=5, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=40, output_filters=80, kernel_size=3, num_repeat=3, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=80, output_filters=112, kernel_size=5, num_repeat=3, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=112, output_filters=192, kernel_size=5, num_repeat=4, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=192, output_filters=320, kernel_size=3, num_repeat=1, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise')), stem_base_filters=32, top_base_filters=1280, activation='swish', batch_norm='default', bn_momentum=0.99, bn_epsilon=0.001, weight_decay=5e-06, drop_connect_rate=0.2, depth_divisor=8, min_depth=None, use_se=True, input_channels=3, num_classes=1000, model_name='efficientnet', rescale_input=True, data_format='channels_last', dtype='float32')
    I0407 21:20:11.387379 140189656872768 efficientnet_model.py:457] Building model efficientnet with params ModelConfig(width_coefficient=1.0, depth_coefficient=1.0, resolution=224, dropout_rate=0.2, blocks=(BlockConfig(input_filters=32, output_filters=16, kernel_size=3, num_repeat=1, expand_ratio=1, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=16, output_filters=24, kernel_size=3, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=24, output_filters=40, kernel_size=5, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=40, output_filters=80, kernel_size=3, num_repeat=3, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=80, output_filters=112, kernel_size=5, num_repeat=3, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=112, output_filters=192, kernel_size=5, num_repeat=4, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=192, output_filters=320, kernel_size=3, num_repeat=1, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise')), stem_base_filters=32, top_base_filters=1280, activation='swish', batch_norm='default', bn_momentum=0.99, bn_epsilon=0.001, weight_decay=5e-06, drop_connect_rate=0.2, depth_divisor=8, min_depth=None, use_se=True, input_channels=3, num_classes=1000, model_name='efficientnet', rescale_input=True, data_format='channels_last', dtype='float32')
    I0407 21:20:11.388924 140114967496512 efficientnet_model.py:457] Building model efficientnet with params ModelConfig(width_coefficient=1.0, depth_coefficient=1.0, resolution=224, dropout_rate=0.2, blocks=(BlockConfig(input_filters=32, output_filters=16, kernel_size=3, num_repeat=1, expand_ratio=1, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=16, output_filters=24, kernel_size=3, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=24, output_filters=40, kernel_size=5, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=40, output_filters=80, kernel_size=3, num_repeat=3, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=80, output_filters=112, kernel_size=5, num_repeat=3, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=112, output_filters=192, kernel_size=5, num_repeat=4, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=192, output_filters=320, kernel_size=3, num_repeat=1, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise')), stem_base_filters=32, top_base_filters=1280, activation='swish', batch_norm='default', bn_momentum=0.99, bn_epsilon=0.001, weight_decay=5e-06, drop_connect_rate=0.2, depth_divisor=8, min_depth=None, use_se=True, input_channels=3, num_classes=1000, model_name='efficientnet', rescale_input=True, data_format='channels_last', dtype='float32')
    I0407 21:20:11.389589 140136124327744 efficientnet_model.py:457] Building model efficientnet with params ModelConfig(width_coefficient=1.0, depth_coefficient=1.0, resolution=224, dropout_rate=0.2, blocks=(BlockConfig(input_filters=32, output_filters=16, kernel_size=3, num_repeat=1, expand_ratio=1, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=16, output_filters=24, kernel_size=3, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=24, output_filters=40, kernel_size=5, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=40, output_filters=80, kernel_size=3, num_repeat=3, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=80, output_filters=112, kernel_size=5, num_repeat=3, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=112, output_filters=192, kernel_size=5, num_repeat=4, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=192, output_filters=320, kernel_size=3, num_repeat=1, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise')), stem_base_filters=32, top_base_filters=1280, activation='swish', batch_norm='default', bn_momentum=0.99, bn_epsilon=0.001, weight_decay=5e-06, drop_connect_rate=0.2, depth_divisor=8, min_depth=None, use_se=True, input_channels=3, num_classes=1000, model_name='efficientnet', rescale_input=True, data_format='channels_last', dtype='float32')
    I0407 21:20:11.389937 139742514054976 efficientnet_model.py:457] Building model efficientnet with params ModelConfig(width_coefficient=1.0, depth_coefficient=1.0, resolution=224, dropout_rate=0.2, blocks=(BlockConfig(input_filters=32, output_filters=16, kernel_size=3, num_repeat=1, expand_ratio=1, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=16, output_filters=24, kernel_size=3, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=24, output_filters=40, kernel_size=5, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=40, output_filters=80, kernel_size=3, num_repeat=3, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=80, output_filters=112, kernel_size=5, num_repeat=3, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=112, output_filters=192, kernel_size=5, num_repeat=4, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=192, output_filters=320, kernel_size=3, num_repeat=1, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise')), stem_base_filters=32, top_base_filters=1280, activation='swish', batch_norm='default', bn_momentum=0.99, bn_epsilon=0.001, weight_decay=5e-06, drop_connect_rate=0.2, depth_divisor=8, min_depth=None, use_se=True, input_channels=3, num_classes=1000, model_name='efficientnet', rescale_input=True, data_format='channels_last', dtype='float32')
    I0407 21:20:11.416097 140201244120896 optimizer_factory.py:147] Scaling the learning rate based on the batch size multiplier. New base_lr: 0.016000
    I0407 21:20:11.416218 140201244120896 optimizer_factory.py:152] Using exponential learning rate with: initial_learning_rate: 0.016000, decay_steps: 2400, decay_rate: 0.970000
    I0407 21:20:11.416271 140201244120896 optimizer_factory.py:177] Applying 5000 warmup steps to the learning rate
    I0407 21:20:11.416328 140201244120896 optimizer_factory.py:59] Building rmsprop optimizer with params {'name': 'rmsprop', 'decay': 0.9, 'epsilon': 0.001, 'momentum': 0.9, 'nesterov': None, 'moving_average_decay': 0.0, 'lookahead': False, 'beta_1': None, 'beta_2': None}
    I0407 21:20:11.416370 140201244120896 optimizer_factory.py:74] Using RMSProp
    I0407 21:20:11.416322 140136124327744 optimizer_factory.py:147] Scaling the learning rate based on the batch size multiplier. New base_lr: 0.016000
    I0407 21:20:11.416441 140136124327744 optimizer_factory.py:152] Using exponential learning rate with: initial_learning_rate: 0.016000, decay_steps: 2400, decay_rate: 0.970000
    I0407 21:20:11.416497 140136124327744 optimizer_factory.py:177] Applying 5000 warmup steps to the learning rate
    I0407 21:20:11.416480 140189656872768 optimizer_factory.py:147] Scaling the learning rate based on the batch size multiplier. New base_lr: 0.016000
    I0407 21:20:11.416563 140136124327744 optimizer_factory.py:59] Building rmsprop optimizer with params {'name': 'rmsprop', 'decay': 0.9, 'epsilon': 0.001, 'momentum': 0.9, 'nesterov': None, 'moving_average_decay': 0.0, 'lookahead': False, 'beta_1': None, 'beta_2': None}
    I0407 21:20:11.416597 140189656872768 optimizer_factory.py:152] Using exponential learning rate with: initial_learning_rate: 0.016000, decay_steps: 2400, decay_rate: 0.970000
    I0407 21:20:11.416603 140136124327744 optimizer_factory.py:74] Using RMSProp
    I0407 21:20:11.416654 140189656872768 optimizer_factory.py:177] Applying 5000 warmup steps to the learning rate
    I0407 21:20:11.416719 140189656872768 optimizer_factory.py:59] Building rmsprop optimizer with params {'name': 'rmsprop', 'decay': 0.9, 'epsilon': 0.001, 'momentum': 0.9, 'nesterov': None, 'moving_average_decay': 0.0, 'lookahead': False, 'beta_1': None, 'beta_2': None}
    I0407 21:20:11.416761 140189656872768 optimizer_factory.py:74] Using RMSProp
    I0407 21:20:11.417023 140463173154624 optimizer_factory.py:147] Scaling the learning rate based on the batch size multiplier. New base_lr: 0.016000
    I0407 21:20:11.417150 140463173154624 optimizer_factory.py:152] Using exponential learning rate with: initial_learning_rate: 0.016000, decay_steps: 2400, decay_rate: 0.970000
    I0407 21:20:11.417119 140114967496512 optimizer_factory.py:147] Scaling the learning rate based on the batch size multiplier. New base_lr: 0.016000
    I0407 21:20:11.417206 140463173154624 optimizer_factory.py:177] Applying 5000 warmup steps to the learning rate
    I0407 21:20:11.417149 139785568282432 optimizer_factory.py:147] Scaling the learning rate based on the batch size multiplier. New base_lr: 0.016000
    I0407 21:20:11.417238 140114967496512 optimizer_factory.py:152] Using exponential learning rate with: initial_learning_rate: 0.016000, decay_steps: 2400, decay_rate: 0.970000
    I0407 21:20:11.417272 139785568282432 optimizer_factory.py:152] Using exponential learning rate with: initial_learning_rate: 0.016000, decay_steps: 2400, decay_rate: 0.970000
    I0407 21:20:11.417267 140463173154624 optimizer_factory.py:59] Building rmsprop optimizer with params {'name': 'rmsprop', 'decay': 0.9, 'epsilon': 0.001, 'momentum': 0.9, 'nesterov': None, 'moving_average_decay': 0.0, 'lookahead': False, 'beta_1': None, 'beta_2': None}
    I0407 21:20:11.417302 140114967496512 optimizer_factory.py:177] Applying 5000 warmup steps to the learning rate
    I0407 21:20:11.417313 140463173154624 optimizer_factory.py:74] Using RMSProp
    I0407 21:20:11.417335 139785568282432 optimizer_factory.py:177] Applying 5000 warmup steps to the learning rate
    I0407 21:20:11.417409 139785568282432 optimizer_factory.py:59] Building rmsprop optimizer with params {'name': 'rmsprop', 'decay': 0.9, 'epsilon': 0.001, 'momentum': 0.9, 'nesterov': None, 'moving_average_decay': 0.0, 'lookahead': False, 'beta_1': None, 'beta_2': None}
    I0407 21:20:11.417370 140114967496512 optimizer_factory.py:59] Building rmsprop optimizer with params {'name': 'rmsprop', 'decay': 0.9, 'epsilon': 0.001, 'momentum': 0.9, 'nesterov': None, 'moving_average_decay': 0.0, 'lookahead': False, 'beta_1': None, 'beta_2': None}
    I0407 21:20:11.417413 140114967496512 optimizer_factory.py:74] Using RMSProp
    I0407 21:20:11.417457 139785568282432 optimizer_factory.py:74] Using RMSProp
    I0407 21:20:11.417449 140034574755648 optimizer_factory.py:147] Scaling the learning rate based on the batch size multiplier. New base_lr: 0.016000
    I0407 21:20:11.417574 140034574755648 optimizer_factory.py:152] Using exponential learning rate with: initial_learning_rate: 0.016000, decay_steps: 2400, decay_rate: 0.970000
    I0407 21:20:11.417633 140034574755648 optimizer_factory.py:177] Applying 5000 warmup steps to the learning rate
    I0407 21:20:11.417696 140034574755648 optimizer_factory.py:59] Building rmsprop optimizer with params {'name': 'rmsprop', 'decay': 0.9, 'epsilon': 0.001, 'momentum': 0.9, 'nesterov': None, 'moving_average_decay': 0.0, 'lookahead': False, 'beta_1': None, 'beta_2': None}
    I0407 21:20:11.417742 140034574755648 optimizer_factory.py:74] Using RMSProp
    I0407 21:20:11.417922 139742514054976 optimizer_factory.py:147] Scaling the learning rate based on the batch size multiplier. New base_lr: 0.016000
    I0407 21:20:11.418047 139742514054976 optimizer_factory.py:152] Using exponential learning rate with: initial_learning_rate: 0.016000, decay_steps: 2400, decay_rate: 0.970000
    I0407 21:20:11.418115 139742514054976 optimizer_factory.py:177] Applying 5000 warmup steps to the learning rate
    I0407 21:20:11.418181 139742514054976 optimizer_factory.py:59] Building rmsprop optimizer with params {'name': 'rmsprop', 'decay': 0.9, 'epsilon': 0.001, 'momentum': 0.9, 'nesterov': None, 'moving_average_decay': 0.0, 'lookahead': False, 'beta_1': None, 'beta_2': None}
    I0407 21:20:11.418227 139742514054976 optimizer_factory.py:74] Using RMSProp
    I0407 21:20:11.511743 140463173154624 classifier_trainer.py:210] Load from checkpoint is enabled.
    I0407 21:20:11.511771 140136124327744 classifier_trainer.py:210] Load from checkpoint is enabled.
    I0407 21:20:11.511830 140201244120896 classifier_trainer.py:210] Load from checkpoint is enabled.
    I0407 21:20:11.511852 139785568282432 classifier_trainer.py:210] Load from checkpoint is enabled.
    I0407 21:20:11.511979 140463173154624 classifier_trainer.py:212] latest_checkpoint: None
    I0407 21:20:11.511931 140114967496512 classifier_trainer.py:210] Load from checkpoint is enabled.
    I0407 21:20:11.512035 140463173154624 classifier_trainer.py:214] No checkpoint detected.
    I0407 21:20:11.512013 140136124327744 classifier_trainer.py:212] latest_checkpoint: None
    I0407 21:20:11.511967 140189656872768 classifier_trainer.py:210] Load from checkpoint is enabled.
    I0407 21:20:11.512070 140136124327744 classifier_trainer.py:214] No checkpoint detected.
    I0407 21:20:11.512050 140201244120896 classifier_trainer.py:212] latest_checkpoint: None
    I0407 21:20:11.512109 140201244120896 classifier_trainer.py:214] No checkpoint detected.
    I0407 21:20:11.512080 139785568282432 classifier_trainer.py:212] latest_checkpoint: None
    I0407 21:20:11.512132 139785568282432 classifier_trainer.py:214] No checkpoint detected.
    I0407 21:20:11.512024 140034574755648 classifier_trainer.py:210] Load from checkpoint is enabled.
    I0407 21:20:11.512168 140114967496512 classifier_trainer.py:212] latest_checkpoint: None
    I0407 21:20:11.512199 140189656872768 classifier_trainer.py:212] latest_checkpoint: None
    I0407 21:20:11.512237 140114967496512 classifier_trainer.py:214] No checkpoint detected.
    I0407 21:20:11.512254 140034574755648 classifier_trainer.py:212] latest_checkpoint: None
    I0407 21:20:11.512263 140189656872768 classifier_trainer.py:214] No checkpoint detected.
    I0407 21:20:11.512320 140034574755648 classifier_trainer.py:214] No checkpoint detected.
    I0407 21:20:11.512342 139742514054976 classifier_trainer.py:210] Load from checkpoint is enabled.
    I0407 21:20:11.512574 139742514054976 classifier_trainer.py:212] latest_checkpoint: None
    I0407 21:20:11.512646 139742514054976 classifier_trainer.py:214] No checkpoint detected.
    I0407 21:20:11.512824 140463173154624 classifier_trainer.py:281] Saving experiment configuration to /home/ubuntu/log_hpu_8/params.yaml
    I0407 21:20:11.512844 140201244120896 classifier_trainer.py:281] Saving experiment configuration to /home/ubuntu/log_hpu_8/params.yaml
    I0407 21:20:11.512879 140136124327744 classifier_trainer.py:281] Saving experiment configuration to /home/ubuntu/log_hpu_8/params.yaml
    I0407 21:20:11.512916 139785568282432 classifier_trainer.py:281] Saving experiment configuration to /home/ubuntu/log_hpu_8/params.yaml
    I0407 21:20:11.513067 140114967496512 classifier_trainer.py:281] Saving experiment configuration to /home/ubuntu/log_hpu_8/params.yaml
    I0407 21:20:11.513097 140034574755648 classifier_trainer.py:281] Saving experiment configuration to /home/ubuntu/log_hpu_8/params.yaml
    I0407 21:20:11.513111 140189656872768 classifier_trainer.py:281] Saving experiment configuration to /home/ubuntu/log_hpu_8/params.yaml
    I0407 21:20:11.513427 139742514054976 classifier_trainer.py:281] Saving experiment configuration to /home/ubuntu/log_hpu_8/params.yaml
    INFO:tensorflow:Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:18.266501 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:18.273411 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:18.516657 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:18.519622 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:18.530623 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:18.541112 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:18.560963 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:18.637652 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:18.986170 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:18.993381 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:18.995272 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.000288 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.002371 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.008608 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.009221 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.015446 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.017377 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.023323 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.024164 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.030233 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.031977 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.038779 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.230454 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.237581 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.244430 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.247164 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.252501 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.254329 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.259245 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.261216 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.267061 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.269414 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.273861 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.276272 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.284127 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.290983 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.333514 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.338494 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.341590 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.346345 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.349385 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.353952 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.358115 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.358681 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.363032 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.365307 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.366400 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.370576 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.372233 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.375322 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.375967 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.379305 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.380400 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.383089 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.384177 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.386934 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.387247 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.392137 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.395114 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.401542 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.401992 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.409463 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.418614 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:19.426564 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:22.537983 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:22.542536 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:22.836477 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:22.927213 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:22.944903 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:22.953875 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.111991 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.325517 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.329089 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.333294 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.336935 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.342117 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.345850 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.349701 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.353518 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.358371 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.362257 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.365881 140136124327744 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.369845 140201244120896 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.377268 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 213 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.546396 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.553503 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.561465 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.568419 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.576227 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.583041 140114967496512 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.659709 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.666837 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.674949 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.681954 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.682129 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.689415 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.689947 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.697319 139742514054976 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.697648 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.702454 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.704830 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.710147 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.712905 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.718914 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.719922 140034574755648 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.726433 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.734972 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.742423 140463173154624 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.840401 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.847713 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.855868 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.862914 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.870940 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:23.877923 139785568282432 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:24.087826 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:24.094863 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:24.102927 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:24.109852 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:24.117700 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    INFO:tensorflow:Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:20:24.124553 140189656872768 cross_device_ops.py:1152] Collective all_reduce tensors: 1 all_reduces, num_devices = 1, group_size = 8, implementation = CommunicationImplementation.AUTO, num_packs = 1
    I0407 21:21:07.697500 140136124327744 keras_utils.py:145] TimeHistory: 53.15 seconds, 481.65 examples/second between steps 0 and 100
    I0407 21:21:07.697577 140201244120896 keras_utils.py:145] TimeHistory: 53.08 seconds, 482.25 examples/second between steps 0 and 100
    I0407 21:21:07.697674 140034574755648 keras_utils.py:145] TimeHistory: 52.86 seconds, 484.33 examples/second between steps 0 and 100
    I0407 21:21:07.697801 139742514054976 keras_utils.py:145] TimeHistory: 53.07 seconds, 482.35 examples/second between steps 0 and 100
    I0407 21:21:07.697997 140189656872768 keras_utils.py:145] TimeHistory: 52.83 seconds, 484.55 examples/second between steps 0 and 100
    I0407 21:21:07.698034 140114967496512 keras_utils.py:145] TimeHistory: 53.08 seconds, 482.27 examples/second between steps 0 and 100
    I0407 21:21:07.698210 139785568282432 keras_utils.py:145] TimeHistory: 53.09 seconds, 482.21 examples/second between steps 0 and 100
    I0407 21:21:07.698484 140463173154624 keras_utils.py:145] TimeHistory: 53.15 seconds, 481.61 examples/second between steps 0 and 100
    I0407 21:21:14.613924 140034574755648 keras_utils.py:145] TimeHistory: 6.91 seconds, 3703.28 examples/second between steps 100 and 200
    I0407 21:21:14.614000 140136124327744 keras_utils.py:145] TimeHistory: 6.91 seconds, 3703.05 examples/second between steps 100 and 200
    I0407 21:21:14.614020 139742514054976 keras_utils.py:145] TimeHistory: 6.91 seconds, 3703.29 examples/second between steps 100 and 200
    I0407 21:21:14.614002 140189656872768 keras_utils.py:145] TimeHistory: 6.91 seconds, 3703.37 examples/second between steps 100 and 200
    I0407 21:21:14.614032 140201244120896 keras_utils.py:145] TimeHistory: 6.91 seconds, 3703.14 examples/second between steps 100 and 200
    I0407 21:21:14.614092 140114967496512 keras_utils.py:145] TimeHistory: 6.91 seconds, 3703.23 examples/second between steps 100 and 200
    I0407 21:21:14.614236 140463173154624 keras_utils.py:145] TimeHistory: 6.91 seconds, 3703.37 examples/second between steps 100 and 200
    I0407 21:21:14.614216 139785568282432 keras_utils.py:145] TimeHistory: 6.91 seconds, 3703.29 examples/second between steps 100 and 200
    I0407 21:21:21.484448 140114967496512 keras_utils.py:145] TimeHistory: 6.87 seconds, 3727.39 examples/second between steps 200 and 300
    I0407 21:21:21.484506 140136124327744 keras_utils.py:145] TimeHistory: 6.87 seconds, 3727.31 examples/second between steps 200 and 300
    I0407 21:21:21.484404 140201244120896 keras_utils.py:145] TimeHistory: 6.87 seconds, 3727.46 examples/second between steps 200 and 300
    I0407 21:21:21.484558 139742514054976 keras_utils.py:145] TimeHistory: 6.87 seconds, 3727.22 examples/second between steps 200 and 300
    I0407 21:21:21.484634 139785568282432 keras_utils.py:145] TimeHistory: 6.87 seconds, 3727.49 examples/second between steps 200 and 300
    I0407 21:21:21.484652 140034574755648 keras_utils.py:145] TimeHistory: 6.87 seconds, 3727.21 examples/second between steps 200 and 300
    I0407 21:21:21.484851 140463173154624 keras_utils.py:145] TimeHistory: 6.87 seconds, 3727.29 examples/second between steps 200 and 300
    I0407 21:21:21.485041 140189656872768 keras_utils.py:145] TimeHistory: 6.87 seconds, 3726.96 examples/second between steps 200 and 300
    I0407 21:21:28.420698 139742514054976 keras_utils.py:145] TimeHistory: 6.93 seconds, 3691.99 examples/second between steps 300 and 400
    I0407 21:21:28.420896 140463173154624 keras_utils.py:145] TimeHistory: 6.93 seconds, 3692.27 examples/second between steps 300 and 400
    I0407 21:21:28.420929 140201244120896 keras_utils.py:145] TimeHistory: 6.93 seconds, 3691.93 examples/second between steps 300 and 400
    I0407 21:21:28.421085 140189656872768 keras_utils.py:145] TimeHistory: 6.93 seconds, 3692.09 examples/second between steps 300 and 400
    I0407 21:21:28.421093 140034574755648 keras_utils.py:145] TimeHistory: 6.93 seconds, 3691.95 examples/second between steps 300 and 400
    I0407 21:21:28.421246 139785568282432 keras_utils.py:145] TimeHistory: 6.93 seconds, 3691.87 examples/second between steps 300 and 400
    I0407 21:21:28.421290 140114967496512 keras_utils.py:145] TimeHistory: 6.93 seconds, 3691.64 examples/second between steps 300 and 400
    I0407 21:21:28.421599 140136124327744 keras_utils.py:145] TimeHistory: 6.93 seconds, 3691.55 examples/second between steps 300 and 400
    I0407 21:21:35.329704 139785568282432 keras_utils.py:145] TimeHistory: 6.91 seconds, 3706.83 examples/second between steps 400 and 500
    I0407 21:21:35.329709 140136124327744 keras_utils.py:145] TimeHistory: 6.90 seconds, 3707.51 examples/second between steps 400 and 500
    I0407 21:21:35.329788 140034574755648 keras_utils.py:145] TimeHistory: 6.91 seconds, 3707.11 examples/second between steps 400 and 500
    I0407 21:21:35.329867 139742514054976 keras_utils.py:145] TimeHistory: 6.91 seconds, 3706.66 examples/second between steps 400 and 500
    I0407 21:21:35.329879 140201244120896 keras_utils.py:145] TimeHistory: 6.91 seconds, 3706.65 examples/second between steps 400 and 500
    I0407 21:21:35.330195 140463173154624 keras_utils.py:145] TimeHistory: 6.91 seconds, 3706.32 examples/second between steps 400 and 500
    I0407 21:21:35.330370 140189656872768 keras_utils.py:145] TimeHistory: 6.91 seconds, 3706.85 examples/second between steps 400 and 500
    I0407 21:21:35.330735 140114967496512 keras_utils.py:145] TimeHistory: 6.91 seconds, 3706.47 examples/second between steps 400 and 500
    I0407 21:21:42.188033 139742514054976 keras_utils.py:145] TimeHistory: 6.86 seconds, 3734.02 examples/second between steps 500 and 600
    I0407 21:21:42.188151 140114967496512 keras_utils.py:145] TimeHistory: 6.85 seconds, 3734.60 examples/second between steps 500 and 600
    I0407 21:21:42.188249 140201244120896 keras_utils.py:145] TimeHistory: 6.86 seconds, 3733.89 examples/second between steps 500 and 600
    I0407 21:21:42.188445 140136124327744 keras_utils.py:145] TimeHistory: 6.86 seconds, 3733.64 examples/second between steps 500 and 600
    I0407 21:21:42.188469 140463173154624 keras_utils.py:145] TimeHistory: 6.86 seconds, 3733.85 examples/second between steps 500 and 600
    I0407 21:21:42.188690 139785568282432 keras_utils.py:145] TimeHistory: 6.86 seconds, 3733.67 examples/second between steps 500 and 600
    I0407 21:21:42.189284 140034574755648 keras_utils.py:145] TimeHistory: 6.86 seconds, 3733.27 examples/second between steps 500 and 600
    I0407 21:21:42.189564 140189656872768 keras_utils.py:145] TimeHistory: 6.86 seconds, 3733.72 examples/second between steps 500 and 600
    I0407 21:21:49.130795 139742514054976 keras_utils.py:145] TimeHistory: 6.94 seconds, 3688.42 examples/second between steps 600 and 700
    I0407 21:21:49.130985 140114967496512 keras_utils.py:145] TimeHistory: 6.94 seconds, 3688.49 examples/second between steps 600 and 700
    I0407 21:21:49.130992 140189656872768 keras_utils.py:145] TimeHistory: 6.94 seconds, 3689.18 examples/second between steps 600 and 700
    I0407 21:21:49.131092 140136124327744 keras_utils.py:145] TimeHistory: 6.94 seconds, 3688.49 examples/second between steps 600 and 700
    I0407 21:21:49.131217 139785568282432 keras_utils.py:145] TimeHistory: 6.94 seconds, 3688.52 examples/second between steps 600 and 700
    I0407 21:21:49.131289 140034574755648 keras_utils.py:145] TimeHistory: 6.94 seconds, 3688.79 examples/second between steps 600 and 700
    I0407 21:21:49.131284 140201244120896 keras_utils.py:145] TimeHistory: 6.94 seconds, 3688.21 examples/second between steps 600 and 700
    I0407 21:21:49.131366 140463173154624 keras_utils.py:145] TimeHistory: 6.94 seconds, 3688.42 examples/second between steps 600 and 700
    I0407 21:21:55.977806 139742514054976 keras_utils.py:145] TimeHistory: 6.84 seconds, 3740.15 examples/second between steps 700 and 800
    I0407 21:21:55.977958 140189656872768 keras_utils.py:145] TimeHistory: 6.84 seconds, 3740.16 examples/second between steps 700 and 800
    I0407 21:21:55.977973 140136124327744 keras_utils.py:145] TimeHistory: 6.84 seconds, 3740.10 examples/second between steps 700 and 800
    I0407 21:21:55.978061 139785568282432 keras_utils.py:145] TimeHistory: 6.84 seconds, 3740.22 examples/second between steps 700 and 800
    I0407 21:21:55.978130 140201244120896 keras_utils.py:145] TimeHistory: 6.84 seconds, 3740.13 examples/second between steps 700 and 800
    I0407 21:21:55.978198 140114967496512 keras_utils.py:145] TimeHistory: 6.85 seconds, 3739.92 examples/second between steps 700 and 800
    I0407 21:21:55.978354 140034574755648 keras_utils.py:145] TimeHistory: 6.84 seconds, 3740.07 examples/second between steps 700 and 800
    I0407 21:21:55.979142 140463173154624 keras_utils.py:145] TimeHistory: 6.85 seconds, 3739.68 examples/second between steps 700 and 800
    I0407 21:22:02.911350 140136124327744 keras_utils.py:145] TimeHistory: 6.93 seconds, 3693.52 examples/second between steps 800 and 900
    I0407 21:22:02.911496 140114967496512 keras_utils.py:145] TimeHistory: 6.93 seconds, 3693.48 examples/second between steps 800 and 900
    I0407 21:22:02.911551 140201244120896 keras_utils.py:145] TimeHistory: 6.93 seconds, 3693.73 examples/second between steps 800 and 900
    I0407 21:22:02.911565 140189656872768 keras_utils.py:145] TimeHistory: 6.93 seconds, 3693.34 examples/second between steps 800 and 900
    I0407 21:22:02.911564 140034574755648 keras_utils.py:145] TimeHistory: 6.93 seconds, 3693.49 examples/second between steps 800 and 900
    I0407 21:22:02.911469 139742514054976 keras_utils.py:145] TimeHistory: 6.93 seconds, 3693.60 examples/second between steps 800 and 900
    I0407 21:22:02.911684 140463173154624 keras_utils.py:145] TimeHistory: 6.93 seconds, 3694.25 examples/second between steps 800 and 900
    I0407 21:22:02.911754 139785568282432 keras_utils.py:145] TimeHistory: 6.93 seconds, 3693.39 examples/second between steps 800 and 900
    I0407 21:22:09.836450 139742514054976 keras_utils.py:145] TimeHistory: 6.92 seconds, 3698.20 examples/second between steps 900 and 1000
    I0407 21:22:09.836554 140463173154624 keras_utils.py:145] TimeHistory: 6.92 seconds, 3698.34 examples/second between steps 900 and 1000
    I0407 21:22:09.836533 140189656872768 keras_utils.py:145] TimeHistory: 6.92 seconds, 3698.29 examples/second between steps 900 and 1000
    I0407 21:22:09.836512 140114967496512 keras_utils.py:145] TimeHistory: 6.92 seconds, 3698.00 examples/second between steps 900 and 1000
    I0407 21:22:09.836662 139785568282432 keras_utils.py:145] TimeHistory: 6.92 seconds, 3698.17 examples/second between steps 900 and 1000
    I0407 21:22:09.836697 140136124327744 keras_utils.py:145] TimeHistory: 6.92 seconds, 3697.82 examples/second between steps 900 and 1000
    I0407 21:22:09.836765 140034574755648 keras_utils.py:145] TimeHistory: 6.92 seconds, 3698.22 examples/second between steps 900 and 1000
    
    Epoch 1: saving model to /home/ubuntu/log_hpu_8/workertemp_3/model.ckpt-0001
    
    Epoch 1: saving model to /home/ubuntu/log_hpu_8/model.ckpt-0001
    
    Epoch 1: saving model to /home/ubuntu/log_hpu_8/workertemp_6/model.ckpt-0001
    
    Epoch 1: saving model to /home/ubuntu/log_hpu_8/workertemp_7/model.ckpt-0001
    
    Epoch 1: saving model to /home/ubuntu/log_hpu_8/workertemp_4/model.ckpt-0001
    
    Epoch 1: saving model to /home/ubuntu/log_hpu_8/workertemp_1/model.ckpt-0001
    
    Epoch 1: saving model to /home/ubuntu/log_hpu_8/workertemp_5/model.ckpt-0001
    I0407 21:22:09.839954 140201244120896 keras_utils.py:145] TimeHistory: 6.93 seconds, 3696.16 examples/second between steps 900 and 1000
    
    Epoch 1: saving model to /home/ubuntu/log_hpu_8/workertemp_2/model.ckpt-0001
    1000/1000 - 116s - loss: 1.4975 - accuracy: 0.9747 - top_5_accuracy: 0.9762 - 116s/epoch - 116ms/step
    1000/1000 - 116s - loss: 1.4975 - accuracy: 0.9747 - top_5_accuracy: 0.9762 - 116s/epoch - 116ms/step
    1000/1000 - 116s - loss: 1.4975 - accuracy: 0.9747 - top_5_accuracy: 0.9762 - 116s/epoch - 116ms/step
    1000/1000 - 116s - loss: 1.4975 - accuracy: 0.9747 - top_5_accuracy: 0.9762 - 116s/epoch - 116ms/step
    1000/1000 - 116s - loss: 1.4975 - accuracy: 0.9747 - top_5_accuracy: 0.9762 - 116s/epoch - 116ms/step
    1000/1000 - 116s - loss: 1.4975 - accuracy: 0.9747 - top_5_accuracy: 0.9762 - 116s/epoch - 116ms/step
    1000/1000 - 116s - loss: 1.4975 - accuracy: 0.9747 - top_5_accuracy: 0.9762 - 116s/epoch - 116ms/step
    1000/1000 - 116s - loss: 1.4975 - accuracy: 0.9747 - top_5_accuracy: 0.9762 - 116s/epoch - 116ms/step
    I0407 21:22:11.052654 140463173154624 classifier_trainer.py:461] Run stats:
    {'loss': 1.497470736503601, 'training_accuracy_top_1': 0.9747382998466492, 'step_timestamp_log': ['BatchTimestamp<batch_index: 0, timestamp: 1649366414.5435808>', 'BatchTimestamp<batch_index: 100, timestamp: 1649366467.6982994>', 'BatchTimestamp<batch_index: 200, timestamp: 1649366474.6141982>', 'BatchTimestamp<batch_index: 300, timestamp: 1649366481.4846933>', 'BatchTimestamp<batch_index: 400, timestamp: 1649366488.4208562>', 'BatchTimestamp<batch_index: 500, timestamp: 1649366495.3301537>', 'BatchTimestamp<batch_index: 600, timestamp: 1649366502.1884263>', 'BatchTimestamp<batch_index: 700, timestamp: 1649366509.131313>', 'BatchTimestamp<batch_index: 800, timestamp: 1649366515.9790943>', 'BatchTimestamp<batch_index: 900, timestamp: 1649366522.9115124>', 'BatchTimestamp<batch_index: 1000, timestamp: 1649366529.8365154>'], 'train_finish_time': 1649366530.979233, 'avg_exp_per_second': 2198.6505107675516}
    I0407 21:22:11.055377 140034574755648 classifier_trainer.py:461] Run stats:
    {'loss': 1.497470736503601, 'training_accuracy_top_1': 0.9747382998466492, 'step_timestamp_log': ['BatchTimestamp<batch_index: 0, timestamp: 1649366414.8409789>', 'BatchTimestamp<batch_index: 100, timestamp: 1649366467.6975083>', 'BatchTimestamp<batch_index: 200, timestamp: 1649366474.6138842>', 'BatchTimestamp<batch_index: 300, timestamp: 1649366481.484506>', 'BatchTimestamp<batch_index: 400, timestamp: 1649366488.4210246>', 'BatchTimestamp<batch_index: 500, timestamp: 1649366495.3297474>', 'BatchTimestamp<batch_index: 600, timestamp: 1649366502.1892464>', 'BatchTimestamp<batch_index: 700, timestamp: 1649366509.1312516>', 'BatchTimestamp<batch_index: 800, timestamp: 1649366515.9783092>', 'BatchTimestamp<batch_index: 900, timestamp: 1649366522.9115179>', 'BatchTimestamp<batch_index: 1000, timestamp: 1649366529.836607>'], 'train_finish_time': 1649366530.9795616, 'avg_exp_per_second': 2204.2750397974496}
    I0407 21:22:11.056213 140201244120896 classifier_trainer.py:461] Run stats:
    {'loss': 1.497470736503601, 'training_accuracy_top_1': 0.9747382998466492, 'step_timestamp_log': ['BatchTimestamp<batch_index: 0, timestamp: 1649366414.6126447>', 'BatchTimestamp<batch_index: 100, timestamp: 1649366467.6974003>', 'BatchTimestamp<batch_index: 200, timestamp: 1649366474.6139913>', 'BatchTimestamp<batch_index: 300, timestamp: 1649366481.4842606>', 'BatchTimestamp<batch_index: 400, timestamp: 1649366488.4208622>', 'BatchTimestamp<batch_index: 500, timestamp: 1649366495.3298304>', 'BatchTimestamp<batch_index: 600, timestamp: 1649366502.1882102>', 'BatchTimestamp<batch_index: 700, timestamp: 1649366509.1312447>', 'BatchTimestamp<batch_index: 800, timestamp: 1649366515.9780846>', 'BatchTimestamp<batch_index: 900, timestamp: 1649366522.9115133>', 'BatchTimestamp<batch_index: 1000, timestamp: 1649366529.839788>'], 'train_finish_time': 1649366530.9814363, 'avg_exp_per_second': 2199.9139147410424}
    I0407 21:22:11.061334 140114967496512 classifier_trainer.py:461] Run stats:
    {'loss': 1.497470736503601, 'training_accuracy_top_1': 0.9747382998466492, 'step_timestamp_log': ['BatchTimestamp<batch_index: 0, timestamp: 1649366414.6152813>', 'BatchTimestamp<batch_index: 100, timestamp: 1649366467.697873>', 'BatchTimestamp<batch_index: 200, timestamp: 1649366474.6140425>', 'BatchTimestamp<batch_index: 300, timestamp: 1649366481.4844086>', 'BatchTimestamp<batch_index: 400, timestamp: 1649366488.4212291>', 'BatchTimestamp<batch_index: 500, timestamp: 1649366495.3305938>', 'BatchTimestamp<batch_index: 600, timestamp: 1649366502.1881087>', 'BatchTimestamp<batch_index: 700, timestamp: 1649366509.1309469>', 'BatchTimestamp<batch_index: 800, timestamp: 1649366515.9781585>', 'BatchTimestamp<batch_index: 900, timestamp: 1649366522.9114444>', 'BatchTimestamp<batch_index: 1000, timestamp: 1649366529.836465>'], 'train_finish_time': 1649366530.9856896, 'avg_exp_per_second': 2199.884352095361}
    I0407 21:22:11.061111 139742514054976 classifier_trainer.py:461] Run stats:
    {'loss': 1.497470736503601, 'training_accuracy_top_1': 0.9747382998466492, 'step_timestamp_log': ['BatchTimestamp<batch_index: 0, timestamp: 1649366414.6241438>', 'BatchTimestamp<batch_index: 100, timestamp: 1649366467.6976368>', 'BatchTimestamp<batch_index: 200, timestamp: 1649366474.6139808>', 'BatchTimestamp<batch_index: 300, timestamp: 1649366481.4845064>', 'BatchTimestamp<batch_index: 400, timestamp: 1649366488.4205527>', 'BatchTimestamp<batch_index: 500, timestamp: 1649366495.3298266>', 'BatchTimestamp<batch_index: 600, timestamp: 1649366502.187994>', 'BatchTimestamp<batch_index: 700, timestamp: 1649366509.130752>', 'BatchTimestamp<batch_index: 800, timestamp: 1649366515.97775>', 'BatchTimestamp<batch_index: 900, timestamp: 1649366522.911326>', 'BatchTimestamp<batch_index: 1000, timestamp: 1649366529.8363912>'], 'train_finish_time': 1649366530.9875371, 'avg_exp_per_second': 2200.0050956501373}
    I0407 21:22:11.069775 140189656872768 classifier_trainer.py:461] Run stats:
    {'loss': 1.497470736503601, 'training_accuracy_top_1': 0.9747382998466492, 'step_timestamp_log': ['BatchTimestamp<batch_index: 0, timestamp: 1649366414.865072>', 'BatchTimestamp<batch_index: 100, timestamp: 1649366467.6978402>', 'BatchTimestamp<batch_index: 200, timestamp: 1649366474.6139507>', 'BatchTimestamp<batch_index: 300, timestamp: 1649366481.4850001>', 'BatchTimestamp<batch_index: 400, timestamp: 1649366488.421016>', 'BatchTimestamp<batch_index: 500, timestamp: 1649366495.3302214>', 'BatchTimestamp<batch_index: 600, timestamp: 1649366502.1895208>', 'BatchTimestamp<batch_index: 700, timestamp: 1649366509.1309476>', 'BatchTimestamp<batch_index: 800, timestamp: 1649366515.9779165>', 'BatchTimestamp<batch_index: 900, timestamp: 1649366522.9115176>', 'BatchTimestamp<batch_index: 1000, timestamp: 1649366529.836486>'], 'train_finish_time': 1649366530.9903367, 'avg_exp_per_second': 2204.5252856335596}
    I0407 21:22:11.069629 139785568282432 classifier_trainer.py:461] Run stats:
    {'loss': 1.497470736503601, 'training_accuracy_top_1': 0.9747382998466492, 'step_timestamp_log': ['BatchTimestamp<batch_index: 0, timestamp: 1649366414.6088657>', 'BatchTimestamp<batch_index: 100, timestamp: 1649366467.6980593>', 'BatchTimestamp<batch_index: 200, timestamp: 1649366474.6140864>', 'BatchTimestamp<batch_index: 300, timestamp: 1649366481.4845717>', 'BatchTimestamp<batch_index: 400, timestamp: 1649366488.4212039>', 'BatchTimestamp<batch_index: 500, timestamp: 1649366495.329666>', 'BatchTimestamp<batch_index: 600, timestamp: 1649366502.1886487>', 'BatchTimestamp<batch_index: 700, timestamp: 1649366509.1311684>', 'BatchTimestamp<batch_index: 800, timestamp: 1649366515.9780223>', 'BatchTimestamp<batch_index: 900, timestamp: 1649366522.9115884>', 'BatchTimestamp<batch_index: 1000, timestamp: 1649366529.8366232>'], 'train_finish_time': 1649366530.9912195, 'avg_exp_per_second': 2199.6549542214775}
    I0407 21:22:11.070821 140136124327744 classifier_trainer.py:461] Run stats:
    {'loss': 1.497470736503601, 'training_accuracy_top_1': 0.9747382998466492, 'step_timestamp_log': ['BatchTimestamp<batch_index: 0, timestamp: 1649366414.5464873>', 'BatchTimestamp<batch_index: 100, timestamp: 1649366467.6973379>', 'BatchTimestamp<batch_index: 200, timestamp: 1649366474.613953>', 'BatchTimestamp<batch_index: 300, timestamp: 1649366481.4844654>', 'BatchTimestamp<batch_index: 400, timestamp: 1649366488.4214494>', 'BatchTimestamp<batch_index: 500, timestamp: 1649366495.3296707>', 'BatchTimestamp<batch_index: 600, timestamp: 1649366502.1884062>', 'BatchTimestamp<batch_index: 700, timestamp: 1649366509.1310532>', 'BatchTimestamp<batch_index: 800, timestamp: 1649366515.9779215>', 'BatchTimestamp<batch_index: 900, timestamp: 1649366522.9113102>', 'BatchTimestamp<batch_index: 1000, timestamp: 1649366529.83654>'], 'train_finish_time': 1649366530.990949, 'avg_exp_per_second': 2198.481605965419}
    2022-04-07 21:22:12.202472: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/habana_device.cpp:93] HabanaDevice (HPU) was not closed properly by TensorFlow. It can happen, when working with Keras or Eager Mode. Resulting log in dmesg: "user released device without removing its memory mappings" can be ignored.
    2022-04-07 21:22:12.205619: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/habana_device.cpp:93] HabanaDevice (HPU) was not closed properly by TensorFlow. It can happen, when working with Keras or Eager Mode. Resulting log in dmesg: "user released device without removing its memory mappings" can be ignored.
    2022-04-07 21:22:12.217010: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/habana_device.cpp:93] HabanaDevice (HPU) was not closed properly by TensorFlow. It can happen, when working with Keras or Eager Mode. Resulting log in dmesg: "user released device without removing its memory mappings" can be ignored.
    2022-04-07 21:22:12.229000: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/habana_device.cpp:93] HabanaDevice (HPU) was not closed properly by TensorFlow. It can happen, when working with Keras or Eager Mode. Resulting log in dmesg: "user released device without removing its memory mappings" can be ignored.
    2022-04-07 21:22:12.240637: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/habana_device.cpp:93] HabanaDevice (HPU) was not closed properly by TensorFlow. It can happen, when working with Keras or Eager Mode. Resulting log in dmesg: "user released device without removing its memory mappings" can be ignored.
    2022-04-07 21:22:12.269095: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/habana_device.cpp:93] HabanaDevice (HPU) was not closed properly by TensorFlow. It can happen, when working with Keras or Eager Mode. Resulting log in dmesg: "user released device without removing its memory mappings" can be ignored.
    2022-04-07 21:22:12.270621: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/habana_device.cpp:93] HabanaDevice (HPU) was not closed properly by TensorFlow. It can happen, when working with Keras or Eager Mode. Resulting log in dmesg: "user released device without removing its memory mappings" can be ignored.
    2022-04-07 21:22:12.614057: W /home/jenkins/workspace/cdsoftwarebuilder/create-tensorflow-module---bpt-d/tensorflow-training/habana_device/habana_device.cpp:93] HabanaDevice (HPU) was not closed properly by TensorFlow. It can happen, when working with Keras or Eager Mode. Resulting log in dmesg: "user released device without removing its memory mappings" can be ignored.


From the output above, you can see that with 8 Gaudi cards, the training throughput is significantly improved to around 3713 images/sec.


```python

```
