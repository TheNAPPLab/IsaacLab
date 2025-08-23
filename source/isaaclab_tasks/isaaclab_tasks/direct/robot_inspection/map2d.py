import time
import numpy as np
import math
import cv2

class Map2D():
    def __init__(self, x_lower, y_lower, x_upper, y_upper, resolution=0.05, dynamic_z = False,
                 occ_prob = 0.85, free_prob = 0.15, log_odds_clip = 10.0):
        
        self.x_lower = x_lower
        self.y_lower = y_lower
        self.x_upper = x_upper
        self.y_upper = y_upper
        self.resolution = resolution

        self.static_min_z = 0.05
        self.static_max_z = 2.0


        self.log_odds_occ = math.log(occ_prob / (1 - occ_prob))
        self.log_odds_free = math.log(free_prob / (1 - free_prob))
        self.log_odds_min = -abs(log_odds_clip)
        self.log_odds_max = abs(log_odds_clip)

        '''
          ----> x  x( -10.5 -> 9.5)
        |          y (-12.2 -> 18.0)
        |
        y
        '''


        self.prob_occ_thresh = occ_prob
        self.prob_free_thresh = free_prob

        self.x_width = int(np.ceil((x_upper - x_lower) / self.resolution))
        self.y_width = int(np.ceil((y_upper - y_lower) / self.resolution))
        self.log_odds_map = np.zeros((self.x_width, self.y_width), dtype=np.float64)
        self.visibility_grid = np.zeros((self.x_width, self.y_width), dtype=np.uint8)

        # Color themes
        self.COLOR_OCCUPIED = (0, 0, 0)     # Black
        self.COLOR_FREE = (255, 255, 255)    # White
        self.COLOR_UNKNOWN = (128, 128, 128) # Gray
        self.COLOR_ROBOT = (0, 0, 255)       # Red for robot position

    def _world_to_grid(self, x, y):
        """Converts world coordinates (meters) to grid indices."""
        gx = int((x - self.x_lower) / self.resolution)
        gy = int((y - self.y_lower) / self.resolution)
        return gx, gy
    
    def _is_in_bounds(self, gx, gy):
        """Checks if grid indices are within the map boundaries."""
        return 0 <= gx < self.x_width and 0 <= gy < self.y_width

    def _bresenham_line(self, x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        gx, gy = x0, y0
        while True:
            yield (gx, gy)
            if gx == x1 and gy == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                gx += sx
            if e2 <= dx:
                err += dx
                gy += sy
    
    def _process_point_cloud_rays(self, point_cloud_3D, robot_position):
        gx_robot, gy_robot = self._world_to_grid(robot_position[0], robot_position[1])
        endpoints = []
        for x, y, z in point_cloud_3D:
            if self.static_min_z <= z <= self.static_max_z:
                gx, gy = self._world_to_grid(x, y)
                if self._is_in_bounds(gx, gy):
                    endpoints.append((gx, gy))

        for gx_end, gy_end in set(endpoints):
            ray_cells = list(self._bresenham_line(gx_robot, gy_robot, gx_end, gy_end))
            yield ray_cells
    def update_occupancy_map(self, point_cloud_3D, robot_position):
        for ray_cells in self._process_point_cloud_rays(point_cloud_3D, robot_position):

            # Perform ray casting for each endpoint
            for gx, gy in ray_cells[:-1]:
                # Mark intermediate cells as free
                if self._is_in_bounds(gx, gy):
                    self.log_odds_map[gx, gy] += self.log_odds_free
                
            # Mark the endpoint cell as occupied
            ex, ey = ray_cells[-1]
            if self._is_in_bounds(ex, ey):
                self.log_odds_map[ex, ey] += self.log_odds_occ

        # Clamp values to prevent them from becoming too large or small
        np.clip(self.log_odds_map, self.log_odds_min, self.log_odds_max, out=self.log_odds_map)

    def update_visibility_map(self, point_cloud_3D, robot_position):
        for ray_cells in self._process_point_cloud_rays(point_cloud_3D, robot_position):
          # Perform ray casting for each endpoint
            for gx, gy in ray_cells[:-1]:
                # Get all cells along the ray from robot to endpoint
                    if self._is_in_bounds(gx, gy):
                        self.visibility_grid[gx, gy] = 1

  

    def get_map_probabilities(self):
        return 1.0 - 1.0 / (1.0 + np.exp(self.log_odds_map))

    def get_map_three_channel(self):
        """Return a 3-channel occupancy tensor (occupied/free/unknown)."""
        prob_map = self.get_map_probabilities()
        occ = (prob_map > self.prob_occ_thresh).astype(np.float32)
        free = (prob_map < self.prob_free_thresh).astype(np.float32)
        unknown = 1.0 - np.clip(occ + free, 0, 1)
        return np.stack([free, occ, unknown], axis=-1)
    
    def get_map_single_channel(self):
        """Returns the occupancy probability map as a single channel numpy array."""
        return self.get_map_probabilities().astype(np.float32)
    
    def get_visibility_map(self):
        return self.visibility_grid.astype(np.float32)

    def calculate_entropy(self):
        prob_map = self.get_map_probabilities()
        epsilon = 1e-12
        p = np.clip(prob_map, epsilon, 1 - epsilon)
        entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        return np.sum(entropy)
    
    def calculate_visible_area(self):
        return np.sum(self.visibility_grid)

    def reset_map(self):
        """Reset the occupancy map to all unknown."""
        self.log_odds_map.fill(0.0)
    
    def display_map(self, robot_position=None):
        prob_map = self.get_map_probabilities()
        prob_map_transposed = prob_map.T
        occ_img = np.full((self.y_width, self.x_width, 3), self.COLOR_UNKNOWN, dtype=np.uint8)


        occ_img[prob_map_transposed > self.prob_occ_thresh] = self.COLOR_OCCUPIED
        occ_img[prob_map_transposed < self.prob_free_thresh] = self.COLOR_FREE

        #visiblity image
        vis_grid_transposed = self.visibility_grid.T
        vis_img = np.full((self.y_width, self.x_width, 3), self.COLOR_UNKNOWN, dtype=np.uint8)
        vis_img[vis_grid_transposed == 1] = self.COLOR_FREE

        if robot_position is not None:
            gx, gy = self._world_to_grid(robot_position[0], robot_position[1])
            if self._is_in_bounds(gx, gy):
                # Draw a red circle for the robot
                cv2.circle(occ_img, (gx, gy), radius=4, color=self.COLOR_ROBOT, thickness=-1)
                cv2.circle(vis_img, (gx, gy), radius=4, color=self.COLOR_ROBOT, thickness=-1)
        combined_img = np.hstack((occ_img, vis_img))
        # Show the map in an OpenCV window
        cv2.imshow("Occupancy (Left) and Visibility (Right) Maps", cv2.flip(combined_img, 0))
        cv2.waitKey(1)


    # def update_occupancy_map(self, point_cloud_3D, robot_position):
        
    #     gx_robot, gy_robot = self._world_to_grid(robot_position[0], robot_position[1])

    #     endpoints = []

    #     for x,y,z in point_cloud_3D:
    #         if self.static_min_z <= z <= self.static_max_z:
    #             gx, gy = self._world_to_grid(x, y)
    #             if self._is_in_bounds(gx, gy):
    #                 endpoints.append((gx, gy))

    #     # Perform ray casting for each endpoint
    #     for gx_end, gy_end in endpoints:
    #         # Get all cells along the ray from robot to endpoint
    #         ray_cells = list(self._bresenham_line(gx_robot, gy_robot, gx_end, gy_end))

    #         # Mark intermediate cells as free
    #         for gx, gy in ray_cells[:-1]:
    #             if self._is_in_bounds(gx, gy):
    #                 self.log_odds_map[gx, gy] += self.log_odds_free
            
    #         # Mark the endpoint cell as occupied
    #         ex, ey = gx_end, gy_end
    #         self.log_odds_map[ex, ey] += self.log_odds_occ

    #     # Clamp values to prevent them from becoming too large or small
    #     np.clip(self.log_odds_map, self.log_odds_min, self.log_odds_max, out=self.log_odds_map)