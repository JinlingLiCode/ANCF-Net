import open3d as o3d


# def visualize_off_file(file_path):
#     # 读取OFF格式的点云文件
#     point_cloud = o3d.io.read_point_cloud(file_path)
#
#     # 可视化点云
#     o3d.visualization.draw_geometries([point_cloud])
#
# # 替换文件路径为你的OFF文件路径
# off_file_path = r"E:\A-linux-server\Grad-PU-main\data\PU1K\test\input_2048\gt_8192\11509_Panda_v4.xyz"
# visualize_off_file(off_file_path)
#



#
import open3d as o3d
mesh = o3d.io.read_triangle_mesh(r"E:\A-linux-server\Grad-PU-main\data\PU1K\test\original_meshes\02691156.37f2f187a1582704a29fef5d2b2f3d7.off") #读取.off文件

mesh.compute_vertex_normals()  #计算mesh的法向量
mesh.paint_uniform_color([1,0.7,0])  #上色，方便可视化
o3d.visualization.draw_geometries([mesh])
