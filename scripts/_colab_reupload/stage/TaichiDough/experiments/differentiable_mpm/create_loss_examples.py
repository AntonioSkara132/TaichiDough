"""Create separate paper-loss examples without changing the source configuration."""
import argparse
import copy
import json
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parent


def example_definitions():
    sampling = {'target_sample_count': 512, 'predicted_sample_count': 1024,
                'sampling_seed': 0, 'max_assignment_pairs': 4_000_000,
                'max_assignment_bytes': 268_435_456}
    return [
        ('episode18_dpsi_pcd_cd_v1.json',
         {'version': 'dpsi-pcd-cd-v1', 'target_voxel_size_m': 0.005}, ()),
        ('episode18_dpsi_pcd_emd_sampled_v1.json',
         {'version': 'dpsi-pcd-emd-v1', 'target_voxel_size_m': 0.005, **sampling}, ()),
        ('episode18_empm_geometry_only_v1.json',
         {'version': 'empm-offline-v1', 'geometric_weight': 1.0, 'tracking_weight': 0.0}, ()),
        ('episode18_dpsi_prt_cd_TEMPLATE_NOT_RUNNABLE.json',
         {'version': 'dpsi-prt-cd-v1', 'target_source': 'external'}, ('loss_point_targets',)),
        ('episode18_dpsi_prt_emd_TEMPLATE_NOT_RUNNABLE.json',
         {'version': 'dpsi-prt-emd-v1', 'target_source': 'external', **sampling}, ('loss_point_targets',)),
        ('episode18_empm_tracks_TEMPLATE_NOT_RUNNABLE.json',
         {'version': 'empm-offline-v1', 'geometric_weight': 1.0, 'tracking_weight': 1.0,
          'empty_track_policy': 'error'}, ('loss_tracks',)),
        ('episode18_empm_masks_TEMPLATE_NOT_RUNNABLE.json',
         {'version': 'empm-mask-inspired-v1', 'geometric_weight': 1.0, 'mask_weight': 1.0,
          'mask_epsilon': 1e-8}, ('loss_masks',)),
    ]


def create_examples(base_path, output_dir):
    base_path = Path(base_path).resolve()
    output_dir = Path(output_dir).resolve()
    base = json.loads(base_path.read_text())
    definitions = example_definitions()
    existing = [output_dir / name for name, _, _ in definitions if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(f'Refusing to overwrite existing examples: {existing}')
    records = []
    for name, loss, targets in definitions:
        config = copy.deepcopy(base)
        config['name'] = name.removesuffix('.json').replace('_', '-')
        config['loss'] = loss
        for target in targets:
            for key, extension in ((target, 'npz'), (target + '_metadata', 'json')):
                config['paths'][key] = f'REPLACE_WITH_VERIFIED_{key.upper()}.{extension}'
                config['expected_sha256'][key] = 'REPLACE_WITH_SHA256_OF_THIS_FILE'
        records.append((output_dir / name, config))
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, config in records:
        with path.open('x') as stream:
            json.dump(config, stream, indent=2, allow_nan=False)
            stream.write('\n')
    return [path for path, _ in records]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-config', type=Path,
                        default=EXPERIMENT_ROOT / 'configs/episode18_table_aligned_registered_tools.json')
    parser.add_argument('--output-dir', type=Path, default=EXPERIMENT_ROOT / 'configs')
    args = parser.parse_args()
    for path in create_examples(args.base_config, args.output_dir):
        print(path)
    print('Material, density, contact and scoring windows were copied unchanged from the base.')
    print('Templates are intentionally not runnable without verified target files and hashes.')


if __name__ == '__main__':
    main()
