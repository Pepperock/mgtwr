import numpy as np
from scipy.spatial.distance import cdist


class GWRKernel:

    def __init__(
            self,
            coords: np.ndarray,
            bw: float = None,
            fixed: bool = True,
            spherical = False,
            function: str = 'triangular',
            eps: float = 1.0000001):

        self.coords = coords
        self.function = function
        self.bw = bw
        self.fixed = fixed
        self.function = function
        self.eps = eps
        self.bandwidth = None
        self.kernel = None
        self.spherical = spherical

    def great_circle_cdist(coords_i, coords):

        dLat = np.radians(coords[:, 1] - coords_i[1])
        dLon = np.radians(coords[:, 0] - coords_i[0])
        lat1 = np.radians(coords[:, 1])
        lat2 = np.radians(coords_i[1])
        a = np.sin(
            dLat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        R = 6371.0

        return R * c
    
    def cal_distance(
            self,
            i: int):
        if self.spherical == True:
           spatial_distance = great_circle_cdist(self.coords[i], self.coords)
        else:
           spatial_distance = np.sqrt(np.sum((self.coords[i] - self.coords)**2, axis=1))
        return spatial_distance

    def cal_kernel(
            self,
            distance
    ):

        if self.fixed:
            self.bandwidth = float(self.bw)
        else:
            self.bandwidth = np.partition(
                distance,
                int(self.bw) - 1)[int(self.bw) - 1] * self.eps  # partial sort in O(n) Time

        self.kernel = self._kernel_funcs(distance / self.bandwidth)

        if self.function == "bisquare":  # Truncate for bisquare
            self.kernel[(distance >= self.bandwidth)] = 0
        return self.kernel

    def _kernel_funcs(self, zs):
        # functions follow Anselin and Rey (2010) table 5.4
        if self.function == 'triangular':
            return 1 - zs
        elif self.function == 'uniform':
            return np.ones(zs.shape) * 0.5
        elif self.function == 'quadratic':
            return (3. / 4) * (1 - zs ** 2)
        elif self.function == 'quartic':
            return (15. / 16) * (1 - zs ** 2) ** 2
        elif self.function == 'gaussian':
            return np.exp(-0.5 * zs ** 2)
        elif self.function == 'bisquare':
            return (1 - zs ** 2) ** 2
        elif self.function == 'exponential':
            return np.exp(-zs)
        else:
            print('Unsupported kernel function', self.function)


class GTWRKernel(GWRKernel):

    def __init__(
            self,
            coords: np.ndarray,
            t: np.ndarray,
            bw: float = None,
            tau: float = None,
            fixed: bool = True,
            spherical = False,
            function: str = 'triangular',
            eps: float = 1.0000001):

        super(GTWRKernel, self).__init__(coords, bw, fixed=fixed, spherical = spherical, function=function, eps=eps)

        self.t = t
        self.tau = tau
        self.coords_new = None
        self.spherical = spherical

    def great_circle_cdist(coords_i, coords):

        dLat = np.radians(coords[:, 1] - coords_i[1])
        dLon = np.radians(coords[:, 0] - coords_i[0])
        lat1 = np.radians(coords[:, 1])
        lat2 = np.radians(coords_i[1])
        a = np.sin(
            dLat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        R = 6371.0

        return R * c
    
    def cal_distance(
            self,
            i: int):
        if self.spherical == True:
           spatial_distance = great_circle_cdist(self.coords[i], self.coords)
        else:
           spatial_distance = np.sqrt(np.sum((self.coords[i] - self.coords)**2, axis=1))
        
        if self.tau == 0:
            return spatial_distance
        else:
            spatial_temporal_distance = np.sqrt(spatial_distance**2 + self.tau * (self.t - self.t[i])**2)
   
        return spatial_temporal_distance
