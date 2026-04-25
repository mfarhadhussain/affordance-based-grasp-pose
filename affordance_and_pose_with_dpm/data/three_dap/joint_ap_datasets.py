import torch 
from torch.utils.data import Dataset, DataLoader
import pickle as pkl 
import yaml 
from scipy.spatial.transform import Rotation as R
import os
import numpy as np 
import random 

seed = 1234 
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

class JointAPDatsets(Dataset): 
    """Dataset for Joint Affordance-Pose."""
    def __init__(self, data_file_path, mode="train", threshold: float = 0.03):
        """Initialize dataset."""
        super().__init__()
        with open(data_file_path, "rb") as f: 
            dataset = pkl.load(f) 

        # data_mode_path = os.path.join("/", *data_file_path.split("/")[:-1], f"{mode}.txt")
        # with open(data_mode_path, "r") as f: 
        #     shape_ids = [line.strip() for line in f]

        all_data = []

        for data_point in dataset:
            # if data_point["shape_id"] not in shape_ids:
            #     continue
            for affordance in data_point["affordance"]:
                for pose in data_point["pose"][affordance]:
                    coords = np.array(data_point["full_shape"]["coordinate"])  
                    transform_matrix = np.eye(4)
                    transform_matrix[:3, :3] = pose[:3, :3]
                    transform_matrix[:3, 3] = pose[:3, 3]
                    
                    if threshold is not None:
                        if np.min(np.linalg.norm(coords - (transform_matrix @ np.array([0., 0., 6.6e-02, 1.]))[:3], axis=1)) > threshold:
                            continue 

                    centroid = coords.mean(axis=0)             # (3,)
                    centered = coords - centroid  
                    dists = np.linalg.norm(centered, axis=1)    # (N,)
                    scale = np.percentile(dists, 95.0)    # scalar
                    if scale < 1e-6:
                        scale = 1e-6

                    coords_norm = centered / scale   

                    trans_centered = pose[:3,3] - centroid         # (3,)
                    trans_norm     = trans_centered / scale        # (3,)

                    quat = R.from_matrix(pose[:3,:3]).as_quat()    # (4,)

                    new_data_dict = {
                        "shape_id": data_point["shape_id"],
                        "semantic class": data_point["semantic class"],
                        "centroid": centroid,
                        "scale": scale,
                        "coordinate": centered, 
                        "affordance": affordance,
                        "affordance label": data_point["full_shape"]["label"][affordance],
                        "translation": trans_centered,
                        "rotation": quat
                    }
                    all_data.append(new_data_dict) 

        random.seed(42)
        random.shuffle(all_data)

        # 3) Compute cut-points
        n = len(all_data)
        n_train = int(0.92 * n)
        n_val   = int(0.04 * n)
        n_test  = n - n_train - n_val 
        
        if mode=="train": 
            self.data = all_data[:n_train]
        elif mode=="val":
            self.data = all_data[n_train:n_train+n_val]
        elif mode=="test":
            self.data  = all_data[n_train+n_val:] 
        
    

    def __len__(self): 
        """Return dataset size."""
        return len(self.data)
    
    def __getitem__(self, index): 
        """Return item at index."""
        data_dict = self.data[index]
        return (
            data_dict['shape_id'], 
            data_dict['semantic class'],
            torch.tensor(data_dict['centroid'], dtype=torch.float32).reshape(-1, 1),  
            torch.tensor(data_dict["scale"]),
            torch.tensor(data_dict['coordinate'], dtype=torch.float32),
            data_dict['affordance'], 
            torch.tensor(data_dict['affordance label'], dtype=torch.float32).reshape(-1, 1), 
            torch.cat([
                torch.tensor(data_dict['translation'], dtype=torch.float32).flatten(),
                torch.tensor(data_dict['rotation'], dtype=torch.float32).flatten()
            ]).reshape(-1, 1)
        )

def main(): 
    import argparse
    parser = argparse.ArgumentParser() 
    parser.add_argument("--config_file", type=str, help="Path of configuration file")
    args = parser.parse_args() 

    with open(args.config_file, "r") as f: 
        config = yaml.safe_load(f)

    data_file_path = config["dataset"]["data_file_path"] 
    batch_size = config["training"]["batch_size"]

    train_dataset = JointAPDatsets(data_file_path=data_file_path, mode="train")
    val_dataset = JointAPDatsets(data_file_path=data_file_path, mode="val")  
    test_dataset = JointAPDatsets(data_file_path=data_file_path, mode="test")  

    print(f"number of sample in:\ntraining: {len(train_dataset)}\nVal: {len(val_dataset)}\nTest: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for _, _, centroid, scale, pcd, text, a, p in train_loader: 
        print("Centroid:", centroid.shape)  
        print("Pose:", p.shape)
        print(centroid[0])
        print(f"Shape of pose: {p.shape}, shape of a: {centroid.shape}")
        print(p[0]) 
        p[:, 0:3, :] += centroid
        print(p[0]) 

        print(f"scale: {scale.shape}, PCD: {pcd.shape}")
        break

if __name__ == "__main__": 
    main()
