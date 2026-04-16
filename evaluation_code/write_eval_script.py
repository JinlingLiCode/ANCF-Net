import os
from glob import glob
import argparse
import time
from tqdm import tqdm

def write_eval_script(args):
    start_time = time.time()  # 记录开始时间
    pcds = glob(os.path.join(args.upsampled_pcd_path, '*.xyz'))
    if args.dataset == 'pu1k':
        mesh_dir = "../data/PU1K/test/original_meshes/"
        script_name = "eval_pu1k.sh"
    elif args.dataset == 'Sketchfab':
        mesh_dir = "/data1/lijinling/Sketchfab/meshes/test"
        script_name = "eval_Sketchfab.sh"
    else:
        mesh_dir = "../data/PU-GAN/test/"
        script_name = "eval_pugan.sh"
    with open(script_name, 'w') as f:
        for pcd_path in tqdm(pcds, desc="Processing point clouds"):  # 使用tqdm显示进度条
        # for pcd_path in pcds:
            pcd_name = pcd_path.split("/")[-1]
            prefix = os.path.splitext(pcd_name)[0]
            prefix = prefix.split('_')[-1]
            if prefix == 'distance':
                continue
            mesh_name = pcd_name.replace(".xyz", ".off")
            mesh_path = os.path.join(mesh_dir, mesh_name)
            f.write("./evaluation {} {}\n".format(mesh_path, pcd_path))

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time
    print(f"Script generation completed in {elapsed_time:.2f} seconds.")  # 打印时间

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation Arguments')
    parser.add_argument('--dataset', default='', type=str, help='datasetname, pu1k or pugan')
    parser.add_argument('--upsampled_pcd_path', default='', type=str, help='the upsampled point cloud path')
    args = parser.parse_args()

    assert args.upsampled_pcd_path != ''
    assert args.dataset in ['pu1k', 'pugan', 'Sketchfab']

    write_eval_script(args)
