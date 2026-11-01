"""Full-split inference, nuScenes evaluation and synchronized GPU timing."""
import argparse
from tools.runtime import add_common_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--warmup', type=int, default=10, help='batches omitted from latency statistics')
    args = parser.parse_args()
    if args.warmup < 0 or args.workers < 0 or (args.batch_size is not None and args.batch_size < 1):
        parser.error('invalid warmup, workers or batch size')

    import json
    import pickle
    import time
    import torch
    from tqdm import tqdm
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils
    from tools.runtime import (build_loader, disable_image_pretrain, load_config,
                               require_cuda, snapshot_config)

    require_cuda()
    common_utils.set_random_seed(args.seed)
    cfg, output = load_config(args)
    output = output / 'eval'
    output.mkdir(exist_ok=True)
    logger = common_utils.create_logger(output / 'test.log')
    disable_image_pretrain(cfg)
    snapshot_config(cfg, output / 'config.yaml')
    (output / 'arguments.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    dataset, loader = build_loader(cfg, args, logger, training=False)
    model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), dataset).cuda()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model._load_state_dict(checkpoint['model_state'], strict=True)
    model.eval()
    predictions, times, timed_samples = [], [], 0
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for index, batch in enumerate(tqdm(loader, desc='inference')):
            load_data_to_gpu(batch)
            torch.cuda.synchronize()
            start = time.perf_counter()
            boxes, _ = model(batch)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if index >= args.warmup:
                times.append(elapsed)
                timed_samples += batch['batch_size']
            predictions.extend(dataset.generate_prediction_dicts(batch, boxes, cfg.CLASS_NAMES))
    with (output / 'result.pkl').open('wb') as stream:
        pickle.dump(predictions, stream)
    timing = dict(
        milliseconds_per_batch=1000 * sum(times) / len(times) if times else None,
        milliseconds_per_sample=1000 * sum(times) / timed_samples if timed_samples else None,
        timed_batches=len(times), warmup_batches=args.warmup, batch_size=loader.batch_size,
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
        precision='fp32', scope='model forward including postprocessing; excludes data loading and H2D',
    )
    (output / 'timing.json').write_text(json.dumps(timing, indent=2), encoding='utf-8')
    logger.info('Timing: %s', timing)
    result, _ = dataset.evaluation(predictions, cfg.CLASS_NAMES,
                                   eval_metric='nuscenes', output_path=output)
    logger.info('%s', result)


if __name__ == '__main__':
    main()
