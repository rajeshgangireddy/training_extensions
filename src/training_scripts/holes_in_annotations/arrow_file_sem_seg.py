
from otx.tools.converter import ConfigConverter



if __name__ == "__main__":
    work_dir = "customer/otx-workspace"
    arrow_dir = "/home/rgangire/workspace/datasets/SemanticSegmentation/semantic_seg_arrow"
    geti_config_path = arrow_dir + "/config.json"
    arrow_file_path = arrow_dir + "/datum-0-of-1.arrow"
    otx_config = ConfigConverter.convert(config_path=geti_config_path)

    otx_config["data"]["data_format"] = "arrow"
    otx_config["data"]["train_subset"]["subset_name"] = "TRAINING"
    otx_config["data"]["val_subset"]["subset_name"] = "VALIDATION"
    otx_config["data"]["test_subset"]["subset_name"] = "TESTING"

    # replace num_workers to be 0
    otx_config["data"]["train_subset"]["num_workers"] = 0
    otx_config["data"]["val_subset"]["num_workers"] = 0
    otx_config["data"]["test_subset"]["num_workers"] = 0

    engine, train_kwargs = ConfigConverter.instantiate(
         config=otx_config,
        work_dir=work_dir,
     data_root=arrow_file_path,
    )
    engine.train(**train_kwargs)
    engine.test()

