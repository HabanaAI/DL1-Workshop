def get_resnet50_argparser():
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    parser.add_argument('--dl-time-exclude', default='True', type=lambda x: x.lower() == 'true', help='Set to False to include data load time')
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('--device', default='hpu', help='device')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers (default: 10)')
    parser.add_argument('--process-per-node', default=8, type=int, metavar='N',
                        help='Number of process per node')
    parser.add_argument('--hls_type', default='HLS1', help='Node type')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')

    parser.add_argument('--channels-last', default='True', type=lambda x: x.lower() == 'true',
                        help='Whether input is in channels last format.'
                        'Any value other than True(case insensitive) disables channels-last')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--num-train-steps', type=int, default=sys.maxsize, metavar='T',
                        help='number of steps a.k.a iterations to run in training phase')
    parser.add_argument('--num-eval-steps', type=int, default=sys.maxsize, metavar='E',
                        help='number of steps a.k.a iterations to run in evaluation phase')
    parser.add_argument('--save-checkpoint', action="store_true",
                        help='Whether or not to save model/checkpont; True: to save, False to avoid saving')
    parser.add_argument('--run-lazy-mode', action='store_true',
                        help='run model in lazy execution mode')
    parser.add_argument('--deterministic', action="store_true",
                        help='Whether or not to make data loading deterministic;This does not make execution deterministic')
    parser.add_argument('--hmp', dest='is_hmp', action='store_true', help='enable hmp mode')
    parser.add_argument('--hmp-bf16', default='', help='path to bf16 ops list in hmp O1 mode')
    parser.add_argument('--hmp-fp32', default='', help='path to fp32 ops list in hmp O1 mode')
    parser.add_argument('--hmp-opt-level', default='O1', help='choose optimization level for hmp')
    parser.add_argument('--hmp-verbose', action='store_true', help='enable verbose mode for hmp')

    return parser

#permute the params from filters first (KCRS) to filters last(RSCK) or vice versa.
#and permute from RSCK to KCRS is used for checkpoint saving
def permute_params(model, to_filters_last, lazy_mode):
    import torch
    import habana_frameworks.torch.core as htcore
    if htcore.is_enabled_weight_permute_pass() is True:
        return
    with torch.no_grad():
        for name, param in model.named_parameters():
            if(param.ndim == 4):
                if to_filters_last:
                    param.data = param.data.permute((2, 3, 1, 0))
                else:
                    param.data = param.data.permute((3, 2, 0, 1))  # permute RSCK to KCRS
    if lazy_mode:
        import habana_frameworks.torch.core as htcore
        htcore.mark_step()


# permute the momentum from filters first (KCRS) to filters last(RSCK) or vice versa.
# and permute from RSCK to KCRS is used for checkpoint saving
# Used for Habana device only


def permute_momentum(optimizer, to_filters_last, lazy_mode):
    import habana_frameworks.torch.core as htcore
    if htcore.is_enabled_weight_permute_pass() is True:
        return
    # Permute the momentum buffer before using for checkpoint
    for group in optimizer.param_groups:
        for p in group['params']:
            param_state = optimizer.state[p]
            if 'momentum_buffer' in param_state:
                buf = param_state['momentum_buffer']
                if(buf.ndim == 4):
                    if to_filters_last:
                        buf = buf.permute((2,3,1,0))
                    else:
                        buf = buf.permute((3,2,0,1))
                    param_state['momentum_buffer'] = buf
    if lazy_mode:
        import habana_frameworks.torch.core as htcore
        htcore.mark_step()