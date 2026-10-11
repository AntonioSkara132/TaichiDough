"""Write a von Mises copy of a stretch-clamp temporal dataset.

For every episode config: plasticity -> von-mises, von_mises_yield_stress_pa -> YIELD,
parameters.youngs_modulus -> E, plastic_min/plastic_max removed from fit_parameters and
parameter_bounds. The dataset's shared initial/fit/bounds are changed the same way.
Each config is written next to its source as <name>_von_mises_<tag>_v1.json, and the
dataset entry points at the new file.

Defaults reproduce the Episode 18 exclude-chunk20 von Mises fit (1000 Pa, E = 20 kPa).
For Episode 11 pass --data-dir, --source, --output and --name explicitly.

Usage (defaults = Episode 18):
    python scripts/make_von_mises_dataset.py DATA_DIR
    python scripts/make_von_mises_dataset.py DATA_DIR --source dataset_episode11_dynamics_temporal_v1.json \
        --output dataset_episode11_dynamics_temporal_von_mises_1000pa_v1.json \
        --name episode11-dynamics-icp-temporal-von-mises-1000pa-e20k-v1
"""
import argparse
import json
from pathlib import Path

PLASTIC = ("plastic_min", "plastic_max")


def to_von_mises(config, yield_pa, youngs_modulus):
    config["simulation"]["plasticity"] = "von-mises"
    config["simulation"]["von_mises_yield_stress_pa"] = yield_pa
    config["parameters"]["youngs_modulus"] = youngs_modulus
    config["fit_parameters"] = [n for n in config["fit_parameters"] if n not in PLASTIC]
    for name in PLASTIC:
        config["parameter_bounds"].pop(name, None)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", type=Path, help="folder holding the source dataset JSON and configs/")
    parser.add_argument("--source", default="dataset_episode18_dynamics_temporal_exclude_chunk20_v1.json")
    parser.add_argument("--output", default="dataset_episode18_dynamics_temporal_exclude_chunk20_von_mises_1000pa_v1.json")
    parser.add_argument("--name", default="episode18-dynamics-icp-temporal-von-mises-1000pa-e20k-exclude-chunk20-v1")
    parser.add_argument("--yield-pa", type=float, default=1000.0)
    parser.add_argument("--youngs-modulus", type=float, default=20000.0)
    parser.add_argument("--tag", default="von_mises_1000pa_v1", help="suffix for the new config files")
    args = parser.parse_args()

    data = args.data_dir
    source = json.loads((data / args.source).read_text())
    for episode in source["episodes"]:
        old = data / episode["config"]
        new = old.with_name(old.name.replace("_stretch_clamp_v1", f"_{args.tag}"))
        config = to_von_mises(json.loads(old.read_text()), args.yield_pa, args.youngs_modulus)
        new.write_text(json.dumps(config, indent=1))
        episode["config"] = str(new.relative_to(data))

    shared = source["shared_parameters"]
    shared["initial"]["youngs_modulus"] = args.youngs_modulus
    shared["fit"] = [n for n in shared["fit"] if n not in PLASTIC]
    for name in PLASTIC:
        shared["bounds"].pop(name, None)
    source["name"] = args.name
    out = data / args.output
    out.write_text(json.dumps(source, indent=1))
    print(out, len(source["episodes"]), shared["fit"])


if __name__ == "__main__":
    main()
