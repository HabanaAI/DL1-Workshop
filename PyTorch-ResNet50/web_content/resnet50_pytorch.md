# Set PythonPath and cd into appropriate directory


```python
%set_env PYTHONPATH=/home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision:/root/examples/models:/usr/lib/habanalabs/:/root
```

    env: PYTHONPATH=/home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision:/root/examples/models:/usr/lib/habanalabs/:/root



```python
%cd /home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision
```

    /home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision


# Import libraries for pytorch training


```python
# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.
from __future__ import print_function

#Import local copy of the model only for ResNext101_32x4d
#which is not part of standard torchvision package.
import datetime
import os
import time
import sys

import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
import random
import utils
from resnet50_notebook_utils import *

```

# Main training function

Insert the following code block in the appropriate places (after backward loss computation and after optimizer step).

```
if args.run_lazy_mode:
    import habana_frameworks.torch.core as htcore
    htcore.mark_step()
```


```python
def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ",device=device)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)
    step_count = 0
    last_print_time= time.time()

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)

        dl_ex_start_time=time.time()

        if args.channels_last:
            image = image.contiguous(memory_format=torch.channels_last)


        if args.run_lazy_mode:
            import habana_frameworks.torch.core as htcore
            htcore.mark_step()

        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad(set_to_none=True)

        # performance gain seen for these models using this mark_step.
        if args.run_lazy_mode:
            import habana_frameworks.torch.core as htcore
            htcore.mark_step()


        loss.backward()
        
        if args.run_lazy_mode:
            import habana_frameworks.torch.core as htcore
            htcore.mark_step()

        optimizer.step()

        ##
        if args.run_lazy_mode:
            import habana_frameworks.torch.core as htcore
            htcore.mark_step()

        if step_count % print_freq == 0:
            output_cpu = output.detach().to('cpu')
            acc1, acc5 = utils.accuracy(output_cpu, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size*print_freq)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size*print_freq)
            current_time = time.time()
            last_print_time = dl_ex_start_time if args.dl_time_exclude else last_print_time
            metric_logger.meters['img/s'].update(batch_size*print_freq / (current_time - last_print_time))
            last_print_time = time.time()

        step_count = step_count + 1
        if step_count >= args.num_train_steps:
            break
```

# Replicate command line args for single HPU resnet50 training


```python
os.environ["MAX_WAIT_ATTEMPTS"] = "50"
os.environ['HCL_CPU_AFFINITY'] = '1'
os.environ['PT_HPU_ENABLE_SYNC_OUTPUT_HOST'] = 'false'
parser = get_resnet50_argparser()
   

args = parser.parse_args(["--batch-size", "256", "--epochs", "20", "--workers", "12",
"--dl-time-exclude", "False", "--print-freq", "20", "--channels-last", "True", "--seed", "123",
"--run-lazy-mode", "--hmp",  "--hmp-bf16", "/home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision/ops_bf16_Resnet.txt",
"--hmp-fp32", "/home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision/ops_fp32_Resnet.txt",
"--deterministic"])
```

Main training loop for single node training. Use fake data to train


```python
from habana_frameworks.torch.utils.library_loader import load_habana_module
load_habana_module()

try:
    # Default 'fork' doesn't work with synapse. Use 'forkserver' or 'spawn'
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass

if args.run_lazy_mode:
    os.environ["PT_HPU_LAZY_MODE"] = "1"
if args.is_hmp:
    from habana_frameworks.torch.hpex import hmp
    hmp.convert(opt_level=args.hmp_opt_level, bf16_file_path=args.hmp_bf16,
                fp32_file_path=args.hmp_fp32, isVerbose=args.hmp_verbose)

torch.manual_seed(args.seed)

if args.deterministic:
    seed = args.seed
    random.seed(seed)

else:
    seed = None

device = torch.device('hpu')

torch.backends.cudnn.benchmark = True

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

dataset = datasets.FakeData(transform=transforms.Compose([transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,]))
dataset_test = datasets.FakeData(transform=transforms.Compose([transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,]))


train_sampler = torch.utils.data.RandomSampler(dataset)
test_sampler = torch.utils.data.SequentialSampler(dataset_test)	

if args.workers > 0:
    # patch torch cuda functions that are being unconditionally invoked
    # in the multiprocessing data loader
    torch.cuda.current_device = lambda: None
    torch.cuda.set_device = lambda x: None

data_loader_type = torch.utils.data.DataLoader

data_loader = data_loader_type(
    dataset, batch_size=args.batch_size, sampler=train_sampler,
    num_workers=args.workers, pin_memory=True)

data_loader_test = data_loader_type(
    dataset_test, batch_size=args.batch_size, sampler=test_sampler,
    num_workers=args.workers, pin_memory=True)

print("Creating model")
model = torchvision.models.__dict__['resnet50'](pretrained=False)
model.to(device)


criterion = nn.CrossEntropyLoss()

if args.run_lazy_mode:
    from habana_frameworks.torch.hpex.optimizers import FusedSGD
    sgd_optimizer = FusedSGD
else:
    sgd_optimizer = torch.optim.SGD
optimizer = sgd_optimizer(
    model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

permute_params(model, True, args.run_lazy_mode)
permute_momentum(optimizer, True, args.run_lazy_mode)

model_for_train = model

print("Start training")
start_time = time.time()
for epoch in range(args.start_epoch, args.epochs):
    train_one_epoch(model_for_train, criterion, optimizer, data_loader,
            device, epoch, print_freq=args.print_freq)


if args.run_lazy_mode:
    os.environ.pop("PT_HPU_LAZY_MODE")

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print('Training time {}'.format(total_time_str))
```

    Loading Habana modules from /home/ubuntu/.local/lib/python3.8/site-packages/habana_frameworks/torch/lib
    hmp:verbose_mode  False
    hmp:opt_level O1
    Creating model


    synapse_logger INFO. pid=272572 at /home/jenkins/workspace/cdsoftwarebuilder/create-pytorch---bpt-d/repos/pytorch-integration/pytorch_helpers/synapse_logger/synapse_logger.cpp:340 Done command: restart


    Start training


    /home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision/utils.py:183: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      pred_cpu = torch.tensor(pred, device='cpu')
    /home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision/utils.py:184: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      target_cpu = torch.tensor(target, device='cpu')


    Epoch: [0]  [0/4]  eta: 0:01:50  lr: 0.1  img/s: 185.4799704549534  loss: 7.2180 (7.2180)  acc1: 0.0000 (0.0000)  acc5: 0.0000 (0.0000)  time: 27.6041  data: 1.9372
    Epoch: [0] Total time: 0:00:50
    Epoch: [1]  [0/4]  eta: 0:00:08  lr: 0.1  img/s: 2378.7478973259235  loss: 23.4875 (23.4875)  acc1: 7.8125 (7.8125)  acc5: 45.7031 (45.7031)  time: 2.1524  data: 2.0303
    Epoch: [1] Total time: 0:00:02
    Epoch: [2]  [0/4]  eta: 0:00:08  lr: 0.1  img/s: 2409.5847875105924  loss: 12.9637 (12.9637)  acc1: 13.6719 (13.6719)  acc5: 45.3125 (45.3125)  time: 2.1249  data: 2.0100
    Epoch: [2] Total time: 0:00:02
    Epoch: [3]  [0/4]  eta: 0:00:08  lr: 0.1  img/s: 2336.4900375549432  loss: 12.9147 (12.9147)  acc1: 10.5469 (10.5469)  acc5: 48.8281 (48.8281)  time: 2.1913  data: 2.0846
    Epoch: [3] Total time: 0:00:02
    Epoch: [4]  [0/4]  eta: 0:00:08  lr: 0.1  img/s: 2389.7676082144535  loss: 35.2041 (35.2041)  acc1: 10.1562 (10.1562)  acc5: 48.8281 (48.8281)  time: 2.1425  data: 2.0342
    Epoch: [4] Total time: 0:00:02
    Epoch: [5]  [0/4]  eta: 0:00:08  lr: 0.1  img/s: 2425.4386779490987  loss: 8.4856 (8.4856)  acc1: 11.7188 (11.7188)  acc5: 49.2188 (49.2188)  time: 2.1110  data: 1.9933
    Epoch: [5] Total time: 0:00:02
    Epoch: [6]  [0/4]  eta: 0:00:08  lr: 0.1  img/s: 2359.2729890244354  loss: 4.4995 (4.4995)  acc1: 10.5469 (10.5469)  acc5: 49.2188 (49.2188)  time: 2.1702  data: 2.0659
    Epoch: [6] Total time: 0:00:02
    Epoch: [7]  [0/4]  eta: 0:00:08  lr: 0.1  img/s: 2429.487090925279  loss: 3.3681 (3.3681)  acc1: 15.2344 (15.2344)  acc5: 48.4375 (48.4375)  time: 2.1074  data: 1.9898
    Epoch: [7] Total time: 0:00:02
    Epoch: [8]  [0/4]  eta: 0:00:08  lr: 0.1  img/s: 2402.2007660030577  loss: 3.1179 (3.1179)  acc1: 14.4531 (14.4531)  acc5: 51.5625 (51.5625)  time: 2.1314  data: 2.0090
    Epoch: [8] Total time: 0:00:02
    Epoch: [9]  [0/4]  eta: 0:00:08  lr: 0.1  img/s: 2383.6994431258304  loss: 3.7898 (3.7898)  acc1: 8.5938 (8.5938)  acc5: 46.0938 (46.0938)  time: 2.1479  data: 2.0297
    Epoch: [9] Total time: 0:00:02
    Epoch: [10]  [0/4]  eta: 0:00:08  lr: 0.1  img/s: 2406.5181311650335  loss: 2.4575 (2.4575)  acc1: 11.7188 (11.7188)  acc5: 50.7812 (50.7812)  time: 2.1276  data: 2.0087
    Epoch: [10] Total time: 0:00:02
    Epoch: [11]  [0/4]  eta: 0:00:08  lr: 0.1  img/s: 2407.4116422753  loss: 2.5138 (2.5138)  acc1: 13.2812 (13.2812)  acc5: 53.1250 (53.1250)  time: 2.1268  data: 2.0189
    Epoch: [11] Total time: 0:00:02
    Epoch: [12]  [0/4]  eta: 0:00:08  lr: 0.1  img/s: 2378.7341958355055  loss: 2.3638 (2.3638)  acc1: 9.3750 (9.3750)  acc5: 54.2969 (54.2969)  time: 2.1524  data: 2.0348
    Epoch: [12] Total time: 0:00:02
    Epoch: [13]  [0/4]  eta: 0:00:08  lr: 0.1  img/s: 2421.260677446273  loss: 2.4239 (2.4239)  acc1: 11.7188 (11.7188)  acc5: 49.6094 (49.6094)  time: 2.1146  data: 2.0085
    Epoch: [13] Total time: 0:00:02
    Epoch: [14]  [0/4]  eta: 0:00:08  lr: 0.1  img/s: 2410.286866165488  loss: 2.4540 (2.4540)  acc1: 14.8438 (14.8438)  acc5: 50.0000 (50.0000)  time: 2.1242  data: 2.0086
    Epoch: [14] Total time: 0:00:02
    Epoch: [15]  [0/4]  eta: 0:00:08  lr: 0.1  img/s: 2426.772388487688  loss: 2.4589 (2.4589)  acc1: 8.2031 (8.2031)  acc5: 48.0469 (48.0469)  time: 2.1098  data: 2.0041
    Epoch: [15] Total time: 0:00:02
    Epoch: [16]  [0/4]  eta: 0:00:08  lr: 0.1  img/s: 2428.8377885928016  loss: 2.4314 (2.4314)  acc1: 9.3750 (9.3750)  acc5: 48.0469 (48.0469)  time: 2.1080  data: 1.9894
    Epoch: [16] Total time: 0:00:02
    Epoch: [17]  [0/4]  eta: 0:00:09  lr: 0.1  img/s: 2255.444087981789  loss: 2.3371 (2.3371)  acc1: 10.1562 (10.1562)  acc5: 51.1719 (51.1719)  time: 2.2701  data: 2.1479
    Epoch: [17] Total time: 0:00:02
    Epoch: [18]  [0/4]  eta: 0:00:08  lr: 0.1  img/s: 2377.39300459928  loss: 2.4550 (2.4550)  acc1: 9.7656 (9.7656)  acc5: 50.3906 (50.3906)  time: 2.1536  data: 2.0352
    Epoch: [18] Total time: 0:00:02
    Epoch: [19]  [0/4]  eta: 0:00:08  lr: 0.1  img/s: 2396.679608649094  loss: 2.4275 (2.4275)  acc1: 11.7188 (11.7188)  acc5: 49.6094 (49.6094)  time: 2.1363  data: 2.0191
    Epoch: [19] Total time: 0:00:02
    Training time 0:01:42


# Distributed Training

**Restart the kernel before running the next section of the notebook**

We will use the Model-References repo command line to demo multinode training. 

Multinode training differs in the following ways.

1. Initialization with hccl
```
import habana_frameworks.torch.core.hccl    
dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
```
2. Omit mark steps in lazy mode 
3. Use the torch distributed data sampler. ex:
```
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
```
4. Distributed data parallel pytorch model initalization. ex:
```
model = torch.nn.parallel.DistributedDataParallel(model, bucket_cap_mb=bucket_size_mb, broadcast_buffers=False,
                    gradient_as_bucket_view=is_grad_view)
```
__Note__: regarding step 4 you must use the DistributedDataParallel API, as the DataParallel API is unsupported by Habana


```python
%set_env PYTHONPATH=/home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision:/root/examples/models:/usr/lib/habanalabs/:/root
```

    env: PYTHONPATH=/home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision:/root/examples/models:/usr/lib/habanalabs/:/root



```python
%cd /home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision
```

    /home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision


Apply the following patch to use fake data and remove evaluation


```python
! git apply /home/ubuntu/work/DL1-Workshop/1.3.0/PyTorch-ResNet50/fake_data_no_eval.patch
```


```python
! python3 -u demo_resnet.py  --world-size 8 --batch-size 256 --model resnet50 --device hpu --print-freq 1 \
  --channels-last True --deterministic --data-path $HOME --mode lazy \
  --epochs 30 --data-type bf16  --custom-lr-values 0.275,0.45,0.625,0.8,0.08,0.008,0.0008 \
  --custom-lr-milestones 1,2,3,4,30,60,80 --dl-time-exclude=False --dl-worker-type=MP
```

    Namespace(batch_size=256, channels_last=True, custom_lr_milestones='1,2,3,4,30,60,80', custom_lr_values='0.275,0.45,0.625,0.8,0.08,0.008,0.0008', data_path='/home/ubuntu', data_type='bf16', deterministic=True, device='hpu', dist=False, dl_time_exclude=False, dl_worker_type='MP', epochs=30, mode='lazy', model='resnet50', num_eval_steps=None, num_train_steps=None, output_dir='.', print_freq=1, process_per_node=0, resume='', save_checkpoint=False, seed=123, workers=10, world_size=8)
    dist training = False, world_size = 8
    self.world_size = 8
    HLS (8): HCL_CONFIG_PATH = None
    TrainingRunner run(): command = mpirun -n 8 --bind-to core --map-by slot:PE=6 --rank-by core --report-bindings --allow-run-as-root /usr/bin/python3 /home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision/train.py --data-path=/home/ubuntu --model=resnet50 --device=hpu --batch-size=256 --epochs=30 --workers=10 --dl-worker-type=MP --dl-time-exclude=False --print-freq=1 --output-dir=. --channels-last=True --seed=123 --run-lazy-mode --hmp --hmp-bf16 /home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision/ops_bf16_Resnet.txt --hmp-fp32 /home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision/ops_fp32_Resnet.txt --custom-lr-values 0.275 0.45 0.625 0.8 0.08 0.008 0.0008 --custom-lr-milestones 1 2 3 4 30 60 80 --deterministic 
    [ip-172-31-23-63:312009] MCW rank 5 bound to socket 1[core 30[hwt 0-1]], socket 1[core 31[hwt 0-1]], socket 1[core 32[hwt 0-1]], socket 1[core 33[hwt 0-1]], socket 1[core 34[hwt 0-1]], socket 1[core 35[hwt 0-1]]: [../../../../../../../../../../../../../../../../../../../../../../../..][../../../../../../BB/BB/BB/BB/BB/BB/../../../../../../../../../../../..]
    [ip-172-31-23-63:312009] MCW rank 6 bound to socket 1[core 36[hwt 0-1]], socket 1[core 37[hwt 0-1]], socket 1[core 38[hwt 0-1]], socket 1[core 39[hwt 0-1]], socket 1[core 40[hwt 0-1]], socket 1[core 41[hwt 0-1]]: [../../../../../../../../../../../../../../../../../../../../../../../..][../../../../../../../../../../../../BB/BB/BB/BB/BB/BB/../../../../../..]
    [ip-172-31-23-63:312009] MCW rank 7 bound to socket 1[core 42[hwt 0-1]], socket 1[core 43[hwt 0-1]], socket 1[core 44[hwt 0-1]], socket 1[core 45[hwt 0-1]], socket 1[core 46[hwt 0-1]], socket 1[core 47[hwt 0-1]]: [../../../../../../../../../../../../../../../../../../../../../../../..][../../../../../../../../../../../../../../../../../../BB/BB/BB/BB/BB/BB]
    [ip-172-31-23-63:312009] MCW rank 0 bound to socket 0[core 0[hwt 0-1]], socket 0[core 1[hwt 0-1]], socket 0[core 2[hwt 0-1]], socket 0[core 3[hwt 0-1]], socket 0[core 4[hwt 0-1]], socket 0[core 5[hwt 0-1]]: [BB/BB/BB/BB/BB/BB/../../../../../../../../../../../../../../../../../..][../../../../../../../../../../../../../../../../../../../../../../../..]
    [ip-172-31-23-63:312009] MCW rank 1 bound to socket 0[core 6[hwt 0-1]], socket 0[core 7[hwt 0-1]], socket 0[core 8[hwt 0-1]], socket 0[core 9[hwt 0-1]], socket 0[core 10[hwt 0-1]], socket 0[core 11[hwt 0-1]]: [../../../../../../BB/BB/BB/BB/BB/BB/../../../../../../../../../../../..][../../../../../../../../../../../../../../../../../../../../../../../..]
    [ip-172-31-23-63:312009] MCW rank 2 bound to socket 0[core 12[hwt 0-1]], socket 0[core 13[hwt 0-1]], socket 0[core 14[hwt 0-1]], socket 0[core 15[hwt 0-1]], socket 0[core 16[hwt 0-1]], socket 0[core 17[hwt 0-1]]: [../../../../../../../../../../../../BB/BB/BB/BB/BB/BB/../../../../../..][../../../../../../../../../../../../../../../../../../../../../../../..]
    [ip-172-31-23-63:312009] MCW rank 3 bound to socket 0[core 18[hwt 0-1]], socket 0[core 19[hwt 0-1]], socket 0[core 20[hwt 0-1]], socket 0[core 21[hwt 0-1]], socket 0[core 22[hwt 0-1]], socket 0[core 23[hwt 0-1]]: [../../../../../../../../../../../../../../../../../../BB/BB/BB/BB/BB/BB][../../../../../../../../../../../../../../../../../../../../../../../..]
    [ip-172-31-23-63:312009] MCW rank 4 bound to socket 1[core 24[hwt 0-1]], socket 1[core 25[hwt 0-1]], socket 1[core 26[hwt 0-1]], socket 1[core 27[hwt 0-1]], socket 1[core 28[hwt 0-1]], socket 1[core 29[hwt 0-1]]: [../../../../../../../../../../../../../../../../../../../../../../../..][BB/BB/BB/BB/BB/BB/../../../../../../../../../../../../../../../../../..]
    hmp:verbose_mode  False
    hmp:opt_level O1
    hmp:verbose_mode  False
    hmp:opt_level O1
    hmp:verbose_mode  False
    hmp:opt_level O1
    hmp:verbose_mode  False
    hmp:opt_level O1
    hmp:verbose_mode  False
    hmp:opt_level O1
    hmp:verbose_mode  False
    hmp:opt_level O1
    hmp:verbose_mode  False
    hmp:opt_level O1
    hmp:verbose_mode  False
    hmp:opt_level O1
    | distributed init (rank 3): env://
    | distributed init (rank 0): env://
    | distributed init (rank 1): env://
    | distributed init (rank 2): env://
    | distributed init (rank 5): env://
    | distributed init (rank 4): env://
    | distributed init (rank 6): env://
    | distributed init (rank 7): env://
    synapse_logger INFO. pid=312014 at /home/jenkins/workspace/cdsoftwarebuilder/create-pytorch---bpt-d/repos/pytorch-integration/pytorch_helpers/synapse_logger/synapse_logger.cpp:340 Done command: restart
    synapse_logger INFO. pid=312016 at /home/jenkins/workspace/cdsoftwarebuilder/create-pytorch---bpt-d/repos/pytorch-integration/pytorch_helpers/synapse_logger/synapse_logger.cpp:340 Done command: restart
    synapse_logger INFO. pid=312020 at /home/jenkins/workspace/cdsoftwarebuilder/create-pytorch---bpt-d/repos/pytorch-integration/pytorch_helpers/synapse_logger/synapse_logger.cpp:340 synapse_logger INFO. pid=312019 at /home/jenkins/workspace/cdsoftwarebuilder/create-pytorch---bpt-d/repos/pytorch-integration/pytorch_helpers/synapse_logger/synapse_logger.cppsynapse_logger INFO. pid=312017 at /home/jenkins/workspace/cdsoftwarebuilder/create-pytorch---bpt-d/repos/pytorch-integration/pytorch_helpers/synapse_logger/synapse_logger.cpp:340 Done command: restart
    Done command: restart
    :340 Done command: restart
    synapse_logger INFO. pid=312018 at /home/jenkins/workspace/cdsoftwarebuilder/create-pytorch---bpt-d/repos/pytorch-integration/pytorch_helpers/synapse_logger/synapse_logger.cpp:340 Done command: restart
    synapse_logger INFO. pid=312013 at /home/jenkins/workspace/cdsoftwarebuilder/create-pytorch---bpt-d/repos/pytorch-integration/pytorch_helpers/synapse_logger/synapse_logger.cpp:340 Done command: restart
    synapse_logger INFO. pid=312015 at /home/jenkins/workspace/cdsoftwarebuilder/create-pytorch---bpt-d/repos/pytorch-integration/pytorch_helpers/synapse_logger/synapse_logger.cpp:340 Done command: restart
    Namespace(apex=False, apex_opt_level='O1', batch_size=256, cache_dataset=False, channels_last=True, custom_lr_milestones=[1, 2, 3, 4, 30, 60, 80], custom_lr_values=[0.275, 0.45, 0.625, 0.8, 0.08, 0.008, 0.0008], data_path='/home/ubuntu', deterministic=True, device='hpu', dist_backend='hccl', dist_url='env://', distributed=True, dl_time_exclude=False, dl_worker_type='MP', epochs=30, hls_type='HLS1', hmp_bf16='/home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision/ops_bf16_Resnet.txt', hmp_fp32='/home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision/ops_fp32_Resnet.txt', hmp_opt_level='O1', hmp_verbose=False, is_hmp=True, lr=0.1, lr_gamma=0.1, lr_step_size=30, model='resnet50', momentum=0.9, num_eval_steps=9223372036854775807, num_train_steps=9223372036854775807, output_dir='.', pretrained=False, print_freq=1, process_per_node=8, rank=0, resume='', run_lazy_mode=True, save_checkpoint=False, seed=123, start_epoch=0, sync_bn=False, test_only=False, weight_decay=0.0001, workers=10, world_size=8)
    Loading Habana modules from /home/ubuntu/.local/lib/python3.8/site-packages/habana_frameworks/torch/lib
    Loading data
    Loading training data
    Took 0.00014853477478027344
    Loading validation data
    Creating data loaders
    Creating model
    Converting model params to channels_last format on Habana
    Start training
    /home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision/utils.py:183: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      pred_cpu = torch.tensor(pred, device='cpu')
    /home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision/utils.py:184: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      target_cpu = torch.tensor(target, device='cpu')
    /home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision/utils.py:183: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      pred_cpu = torch.tensor(pred, device='cpu')
    /home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision/utils.py:184: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      target_cpu = torch.tensor(target, device='cpu')
    /home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision/utils.py:183: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      pred_cpu = torch.tensor(pred, device='cpu')
    /home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision/utils.py:184: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      target_cpu = torch.tensor(target, device='cpu')
    /home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision/utils.py:183: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      pred_cpu = torch.tensor(pred, device='cpu')
    /home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision/utils.py:184: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      target_cpu = torch.tensor(target, device='cpu')
    /home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision/utils.py:183: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      pred_cpu = torch.tensor(pred, device='cpu')
    /home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision/utils.py:184: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      target_cpu = torch.tensor(target, device='cpu')
    /home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision/utils.py:183: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      pred_cpu = torch.tensor(pred, device='cpu')
    /home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision/utils.py:184: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      target_cpu = torch.tensor(target, device='cpu')
    Epoch: [0]  [0/1]  eta: 0:00:25  lr: 0.1  img/s: 4.968440300198461  loss: 7.3282 (7.3282)  acc1: 0.0000 (0.0000)  acc5: 0.0000 (0.0000)  time: 25.1588  data: 2.6801
    Epoch: [0] Total time: 0:00:25
    /home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision/utils.py:183: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      pred_cpu = torch.tensor(pred, device='cpu')
    /home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision/utils.py:184: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      target_cpu = torch.tensor(target, device='cpu')
    /home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision/utils.py:183: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      pred_cpu = torch.tensor(pred, device='cpu')
    /home/ubuntu/work/Model-References/PyTorch/computer_vision/classification/torchvision/utils.py:184: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      target_cpu = torch.tensor(target, device='cpu')
    Epoch: [1]  [0/1]  eta: 0:00:24  lr: 0.275  img/s: 5.020899746100226  loss: 6.5814 (6.5814)  acc1: 10.4000 (10.4000)  acc5: 50.4000 (50.4000)  time: 24.8959  data: 2.8295
    Epoch: [1] Total time: 0:00:25
    Epoch: [2]  [0/1]  eta: 0:00:23  lr: 0.45  img/s: 5.250388528130194  loss: 41.2407 (41.2407)  acc1: 7.2000 (7.2000)  acc5: 50.4000 (50.4000)  time: 23.8078  data: 2.6896
    Epoch: [2] Total time: 0:00:24
    Epoch: [3]  [0/1]  eta: 0:00:02  lr: 0.625  img/s: 41.84581710237702  loss: 94.2396 (94.2396)  acc1: 13.6000 (13.6000)  acc5: 53.6000 (53.6000)  time: 2.9872  data: 2.7430
    Epoch: [3] Total time: 0:00:03
    Epoch: [4]  [0/1]  eta: 0:00:03  lr: 0.8  img/s: 41.306034852912326  loss: 56.9921 (56.9921)  acc1: 10.4000 (10.4000)  acc5: 50.4000 (50.4000)  time: 3.0262  data: 2.7317
    Epoch: [4] Total time: 0:00:03
    Epoch: [5]  [0/1]  eta: 0:00:02  lr: 0.8  img/s: 42.23005572916214  loss: 157.1179 (157.1179)  acc1: 4.8000 (4.8000)  acc5: 45.6000 (45.6000)  time: 2.9600  data: 2.7793
    Epoch: [5] Total time: 0:00:03
    Epoch: [6]  [0/1]  eta: 0:00:02  lr: 0.8  img/s: 41.86795902643707  loss: 34.7934 (34.7934)  acc1: 12.8000 (12.8000)  acc5: 46.4000 (46.4000)  time: 2.9856  data: 2.6695
    Epoch: [6] Total time: 0:00:03
    Epoch: [7]  [0/1]  eta: 0:00:03  lr: 0.8  img/s: 40.741010701695465  loss: 46.5065 (46.5065)  acc1: 8.0000 (8.0000)  acc5: 42.4000 (42.4000)  time: 3.0682  data: 2.7634
    Epoch: [7] Total time: 0:00:03
    Epoch: [8]  [0/1]  eta: 0:00:02  lr: 0.8  img/s: 42.07145216916882  loss: 41.1925 (41.1925)  acc1: 12.0000 (12.0000)  acc5: 49.6000 (49.6000)  time: 2.9711  data: 2.7575
    Epoch: [8] Total time: 0:00:03
    Epoch: [9]  [0/1]  eta: 0:00:03  lr: 0.8  img/s: 41.4880652809035  loss: 21.0469 (21.0469)  acc1: 2.4000 (2.4000)  acc5: 48.8000 (48.8000)  time: 3.0129  data: 2.8697
    Epoch: [9] Total time: 0:00:03
    Epoch: [10]  [0/1]  eta: 0:00:03  lr: 0.8  img/s: 41.61600809129601  loss: 20.8391 (20.8391)  acc1: 8.0000 (8.0000)  acc5: 39.2000 (39.2000)  time: 3.0037  data: 2.8567
    Epoch: [10] Total time: 0:00:03
    Epoch: [11]  [0/1]  eta: 0:00:02  lr: 0.8  img/s: 42.94716534029547  loss: 5.4058 (5.4058)  acc1: 9.6000 (9.6000)  acc5: 44.8000 (44.8000)  time: 2.9106  data: 2.7637
    Epoch: [11] Total time: 0:00:03
    Epoch: [12]  [0/1]  eta: 0:00:03  lr: 0.8  img/s: 41.562006177287614  loss: 5.1169 (5.1169)  acc1: 8.8000 (8.8000)  acc5: 43.2000 (43.2000)  time: 3.0076  data: 2.8644
    Epoch: [12] Total time: 0:00:03
    Epoch: [13]  [0/1]  eta: 0:00:02  lr: 0.8  img/s: 42.5130412529463  loss: 5.2326 (5.2326)  acc1: 8.8000 (8.8000)  acc5: 40.8000 (40.8000)  time: 2.9403  data: 2.7048
    Epoch: [13] Total time: 0:00:03
    Epoch: [14]  [0/1]  eta: 0:00:02  lr: 0.8  img/s: 42.45355933427603  loss: 4.0984 (4.0984)  acc1: 10.4000 (10.4000)  acc5: 50.4000 (50.4000)  time: 2.9444  data: 2.7506
    Epoch: [14] Total time: 0:00:03
    Epoch: [15]  [0/1]  eta: 0:00:03  lr: 0.8  img/s: 41.51929737255945  loss: 3.8364 (3.8364)  acc1: 8.8000 (8.8000)  acc5: 45.6000 (45.6000)  time: 3.0107  data: 2.7901
    Epoch: [15] Total time: 0:00:03
    Epoch: [16]  [0/1]  eta: 0:00:03  lr: 0.8  img/s: 41.62420194305238  loss: 4.3854 (4.3854)  acc1: 11.2000 (11.2000)  acc5: 51.2000 (51.2000)  time: 3.0031  data: 2.7007
    Epoch: [16] Total time: 0:00:03
    Epoch: [17]  [0/1]  eta: 0:00:03  lr: 0.8  img/s: 40.668464300989825  loss: 3.2956 (3.2956)  acc1: 16.8000 (16.8000)  acc5: 56.8000 (56.8000)  time: 3.0736  data: 2.8971
    Epoch: [17] Total time: 0:00:03
    Epoch: [18]  [0/1]  eta: 0:00:02  lr: 0.8  img/s: 42.20944236290695  loss: 2.9554 (2.9554)  acc1: 10.4000 (10.4000)  acc5: 45.6000 (45.6000)  time: 2.9614  data: 2.8133
    Epoch: [18] Total time: 0:00:03
    Epoch: [19]  [0/1]  eta: 0:00:03  lr: 0.8  img/s: 41.58997438941339  loss: 2.8381 (2.8381)  acc1: 8.0000 (8.0000)  acc5: 48.0000 (48.0000)  time: 3.0055  data: 2.6764
    Epoch: [19] Total time: 0:00:03
    Epoch: [20]  [0/1]  eta: 0:00:03  lr: 0.8  img/s: 41.60279239445777  loss: 2.6087 (2.6087)  acc1: 12.8000 (12.8000)  acc5: 48.8000 (48.8000)  time: 3.0046  data: 2.8617
    Epoch: [20] Total time: 0:00:03
    Epoch: [21]  [0/1]  eta: 0:00:02  lr: 0.8  img/s: 42.399485727673266  loss: 2.6218 (2.6218)  acc1: 10.4000 (10.4000)  acc5: 47.2000 (47.2000)  time: 2.9482  data: 2.8060
    Epoch: [21] Total time: 0:00:03
    Epoch: [22]  [0/1]  eta: 0:00:03  lr: 0.8  img/s: 41.20787178753197  loss: 2.6490 (2.6490)  acc1: 12.0000 (12.0000)  acc5: 44.8000 (44.8000)  time: 3.0334  data: 2.8084
    Epoch: [22] Total time: 0:00:03
    Epoch: [23]  [0/1]  eta: 0:00:03  lr: 0.8  img/s: 41.4376367471335  loss: 2.5754 (2.5754)  acc1: 5.6000 (5.6000)  acc5: 50.4000 (50.4000)  time: 3.0166  data: 2.6020
    Epoch: [23] Total time: 0:00:03
    Epoch: [24]  [0/1]  eta: 0:00:02  lr: 0.8  img/s: 42.232630838968035  loss: 2.5278 (2.5278)  acc1: 8.8000 (8.8000)  acc5: 58.4000 (58.4000)  time: 2.9598  data: 2.7428
    Epoch: [24] Total time: 0:00:03
    Epoch: [25]  [0/1]  eta: 0:00:03  lr: 0.8  img/s: 41.65710893682634  loss: 2.3649 (2.3649)  acc1: 13.6000 (13.6000)  acc5: 54.4000 (54.4000)  time: 3.0007  data: 2.8517
    Epoch: [25] Total time: 0:00:03
    Epoch: [26]  [0/1]  eta: 0:00:02  lr: 0.8  img/s: 41.85482008441887  loss: 2.5271 (2.5271)  acc1: 6.4000 (6.4000)  acc5: 45.6000 (45.6000)  time: 2.9865  data: 2.7654
    Epoch: [26] Total time: 0:00:03
    Epoch: [27]  [0/1]  eta: 0:00:03  lr: 0.8  img/s: 39.594482201748285  loss: 2.3276 (2.3276)  acc1: 16.0000 (16.0000)  acc5: 56.0000 (56.0000)  time: 3.1570  data: 3.0035
    Epoch: [27] Total time: 0:00:03
    Epoch: [28]  [0/1]  eta: 0:00:03  lr: 0.8  img/s: 40.98443056290586  loss: 2.4163 (2.4163)  acc1: 14.4000 (14.4000)  acc5: 54.4000 (54.4000)  time: 3.0499  data: 2.9039
    Epoch: [28] Total time: 0:00:03
    Epoch: [29]  [0/1]  eta: 0:00:02  lr: 0.8  img/s: 41.927704522793746  loss: 2.3607 (2.3607)  acc1: 11.2000 (11.2000)  acc5: 51.2000 (51.2000)  time: 2.9813  data: 2.7511
    Epoch: [29] Total time: 0:00:03
    Training time 0:02:46

