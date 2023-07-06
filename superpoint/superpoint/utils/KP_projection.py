import numpy as np

class Projection():
    def __init__(self,
                 camera_transforms: np.ndarray,
                 depth_data: np.ndarray,
                 H: int,
                 W: int,
                 fov: float) -> None:
        
        self.camera_transforms = camera_transforms
        self.depth_data = depth_data
        self.H = H
        self.W = W
        self.fov = np.deg2rad(fov)
        self.K = self.get_camera_intrinsic()
    
    def get_camera_intrinsic(self):
        c_x = self.W//2
        c_y = self.H//2
        F_L = c_y/np.tan(self.fov/2)
        cam_intrinsic_matrix = np.array([ [F_L,0,c_x] , 
                                          [0,F_L,c_y] , 
                                          [ 0, 0, 1 ] ],dtype=np.float32)
        
        return cam_intrinsic_matrix


    def flip_yz(self,transformation_matrix: np.ndarray)-> np.ndarray:
        reverse = np.diag([1, -1, -1, 1])
        transformation_matrix =  transformation_matrix @ reverse
        return transformation_matrix
    
    
    def inv(self,x: np.ndarray) -> np.ndarray:
        return np.linalg.inv(x)
    
    def project(self,
                src:np.ndarray,
                src_idx:int,
                dst_idx:int) -> np.ndarray:

        src_T = self.flip_yz(self.camera_transforms[:,:,src_idx])
        dst_T = self.flip_yz(self.camera_transforms[:,:,dst_idx])
        src_depth = self.depth_data[src[1],src[0],src_idx]

        P2_CAM = self.inv(self.K) @ src
        P2_CAM /= np.linalg.norm(P2_CAM)
        P2_CAM = P2_CAM * src_depth

        P_WORLD = src_T[:3,:3] @ P2_CAM + src_T[:3,3].reshape(3,1)

        P1_CAM = (self.inv(dst_T[:3,:3]) @ P_WORLD) - (self.inv(dst_T[:3,:3]) @ dst_T[:3,3].reshape(3,1)) 
        P1 = self.K @ P1_CAM
        P1 = P1 / P1[2]
       
        return P1
