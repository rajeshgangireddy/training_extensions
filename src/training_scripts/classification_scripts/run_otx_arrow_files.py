from otx.tools.converter import ConfigConverter
import argparse
import os
import time
import json
WORK_DIR_ROOT = "/home/rgangire/workspace/data/OTX_WS"



if __name__ == "__main__":

    # argparse
    parser = argparse.ArgumentParser(description="Train and test a model on the OTX dataset using arrow and a "
                                                 "config file")
    parser.add_argument("--exp_name", type=str, help="Experiment name")
    parser.add_argument("--data_dir", type=str, help="Path to the data directory")
    # parse args
    args = parser.parse_args()

    data_dir = args.data_dir
    geti_config_path = os.path.join(data_dir, "config.json")
    # list of files with .arraow extension
    arrow_files = [f for f in os.listdir(data_dir) if f.endswith(".arrow")]
    if len(arrow_files) > 0:
        # the first file is enough
        arrow_file_path = os.path.join(data_dir, arrow_files[0])
    else:
        raise ValueError("No arrow files found in the data directory")

    time_stamp_str = time.strftime("%Y%m%d-%H%M%S")
    work_dir = os.path.join(WORK_DIR_ROOT, f"{args.exp_name}_{time_stamp_str}")
    results_dir = os.path.join(data_dir, f"{args.exp_name}_{time_stamp_str}")

    otx_config = ConfigConverter.convert(config_path=geti_config_path)

    otx_config["data"]["data_format"] = "arrow"
    otx_config["data"]["train_subset"]["subset_name"] = "TRAINING"
    otx_config["data"]["val_subset"]["subset_name"] = "VALIDATION"
    otx_config["data"]["test_subset"]["subset_name"] = "TESTING"
    tic = time.time()
    engine, train_kwargs = ConfigConverter.instantiate(
        config=otx_config,
        work_dir=work_dir,
        data_root=arrow_file_path,
    )
    engine.train(**train_kwargs)
    metrics = engine.test()
    time_taken = time.time() - tic

    metrics["time_taken"] = time_taken

    # dump metrics to a file
    metrics_file = os.path.join(results_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {metrics_file}")
    print(f"Metrics: {metrics}")




