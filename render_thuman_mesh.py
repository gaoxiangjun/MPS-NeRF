import numpy as np
from parser_config import *
import pyrender
import trimesh
import cv2
import math
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import glob


def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm


def make_rotate(rx, ry, rz):

    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3,3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3,3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3,3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz,Ry),Rx)
    return R


class Renderer(object):
    def __init__(self, focal_length=1000, height=512, width=512):
        self.height = height
        self.width = width
        self.renderer = pyrender.OffscreenRenderer(height, width)
        self.focal_length = focal_length

    def render(self, vertices, K, R, T, save_path=None, return_depth=False):
        # Need to flip x-axis
        rot = trimesh.transformations.rotation_matrix(np.radians(180),
                                                      [1, 0, 0])

        self.renderer.viewport_height = self.height
        self.renderer.viewport_width = self.width
        # Create a scene for each image and render all meshes
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                               ambient_light=(0.5, 0.5, 0.5))
        camera_pose = np.eye(4)
        camera = pyrender.camera.IntrinsicsCamera(fx=K[0, 0],
                                                  fy=K[1, 1],
                                                  cx=K[0, 2],
                                                  cy=K[1, 2])
        scene.add(camera, pose=camera_pose)
        # Create light source
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)

        mesh = trimesh.load(mesh_path)
        vertices = mesh.vertices

        vertices = vertices @ R.T + T
        mesh.vertices = vertices

        mesh.apply_transform(rot)
        normals = compute_normal(mesh.vertices, mesh.faces)
        colors = ((0.5 * normals + 0.5) * 255).astype(np.uint8)
        mesh.visual.vertex_colors[:, :3] = colors

        trans = [0, 0, 0]

        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene.add(mesh, 'mesh')

        # Use 3 directional lights
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, -1, 1]) + trans
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([0, 1, 1]) + trans
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([1, 1, 2]) + trans
        scene.add(light, pose=light_pose)

        # Alpha channel was not working previously need to check again
        # Until this is fixed use hack with depth image to get the opacity
        color, rend_depth = self.renderer.render(
                scene, flags=pyrender.RenderFlags.FLAT)
        color = color.astype(np.uint8)

        msk = (rend_depth != 0).astype(np.uint8)
        color[msk == 0] = 255
        msk[msk == 1] = 255
        color = color[..., [2, 1, 0]]
        color = np.concatenate([color, msk[..., None]], axis=2)
        
        y,x = 0,130 #- 10
        h,w = 512,256 #+ 80# THuman
        color = color[y:y+h, x:x+w]
        cv2.imwrite(save_path, color)
        


renderer = Renderer(height=512, width=512)

parser = config_parser()
global_args = parser.parse_args()
expname = global_args.expname
data_root_list = [
     "./data/THuman/nerf_data_/results_gyx_20181012_sty_1_M",
     "./data/THuman/nerf_data_/results_gyx_20181012_xsx_2_M",
     "./data/THuman/nerf_data_/results_gyx_20181013_hyd_1_M",
     "./data/THuman/nerf_data_/results_gyx_20181012_lw_2_F",
     "./data/THuman/nerf_data_/results_gyx_20181013_xyz_1_F",
]

for data_root in data_root_list:
    annots_path = os.path.join(data_root, 'annots.npy')
    annots = np.load(annots_path, allow_pickle=True).item()
    cameras = annots['cams']
    ims = annots['ims']
    Ks = np.array(cameras['K'])
    Rs = np.array(cameras['R'])
    Ts = np.array(cameras['T']).transpose(0, 2, 1)
    Ds = np.array(cameras['D'])

    mesh_paths = []
    mesh_path_pre = "./objs/THuman/" + expname
    obj_dir_path = os.path.join(mesh_path_pre, os.path.basename(data_root))
    all_files = os.listdir(obj_dir_path)
    obj_files = [os.path.join(obj_dir_path, x) for x in all_files if x.endswith("obj")]
    mesh_paths.extend(obj_files)
    
    print(data_root)
    for mesh_path in mesh_paths:
        print(os.path.basename(mesh_path))
        # view_num = 3
        # index_list = [x for x in range(view_num)]
        normal_map_path = "{}_view_{:03d}_normal.png".format(mesh_path[:-4], 0)
        renderer.render(mesh_path, Ks[4], Rs[4], Ts[4], normal_map_path)
        # renderer.render(mesh_path, Ks[0], Rs[0], Ts[0], normal_map_path)

        normal_map_path = "{}_view_{:03d}_normal.png".format(mesh_path[:-4], 1)
        renderer.render(mesh_path, Ks[12], Rs[12], Ts[12], normal_map_path)
        # renderer.render(mesh_path, Ks[6], Rs[6], Ts[6], normal_map_path)
        
        normal_map_path = "{}_view_{:03d}_normal.png".format(mesh_path[:-4], 2)
        renderer.render(mesh_path, Ks[20], Rs[20], Ts[20], normal_map_path)
        # renderer.render(mesh_path, Ks[12], Rs[12], Ts[12], normal_map_path)
