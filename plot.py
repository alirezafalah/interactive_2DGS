import numpy as np
import pyvista as pv

class GaussianPrimitive:
    """A class to represent and generate a 2D Gaussian Primitive in 3D space."""
    
    def __init__(self, position, rotation_angles, scale):
        self.p = np.array(position, dtype=float)
        self.angles = np.array(rotation_angles, dtype=float) # Euler angles: Yaw, Pitch, Roll
        self.s = np.array(scale, dtype=float) # su, sv
        self.points = None
        self.generate_points()

    def update_params(self, position=None, rotation_angles=None, scale=None):
        """Update parameters and regenerate the point cloud."""
        if position is not None:
            self.p = np.array(position, dtype=float)
        if rotation_angles is not None:
            self.angles = np.array(rotation_angles, dtype=float)
        if scale is not None:
            self.s = np.array(scale, dtype=float)
        self.generate_points()

    def generate_points(self):
        """
        Generates the 3D point cloud for the primitive based on its parameters,
        implementing the math from the paper.
        """
        # 1. Define the local 2D grid (the "u, v" space)
        n_points = 50
        u_coords = np.linspace(-2.5, 2.5, n_points)
        v_coords = np.linspace(-2.5, 2.5, n_points)
        u, v = np.meshgrid(u_coords, v_coords)
        uv = np.vstack([u.ravel(), v.ravel()])

        # 2. Implement Equation (2): The Gaussian "Fuzziness"
        # G(u) = exp( -(u^2 + v^2) / 2 )
        # This will determine the color/opacity of each point.
        distance_sq = uv[0,:]**2 + uv[1,:]**2
        gaussian_values = np.exp(-distance_sq / 2.0)

        # 3. Define the orientation from Euler angles (t_u, t_v)
        # We create a rotation matrix from yaw, pitch, roll for intuitive control
        yaw, pitch, roll = self.angles
        rotation = pv.transformations.yaw_pitch_roll_to_matrix([yaw, pitch, roll])
        t_u = rotation[:3, 0] # First column of rotation matrix
        t_v = rotation[:3, 1] # Second column of rotation matrix

        # 4. Implement Equation (1): The 3D position of each point
        # P(u, v) = p + s_u*t_u*u + s_v*t_v*v
        points_3d = self.p[:, np.newaxis] + self.s[0] * t_u[:, np.newaxis] * uv[0,:] + self.s[1] * t_v[:, np.newaxis] * uv[1,:]
        
        # 5. Create a PyVista object to hold the points and their color
        self.points = pv.PolyData(points_3d.T)
        self.points['gaussian'] = gaussian_values # Assign the "fuzziness" as a color value

# --- Main Visualization Setup ---

# Create two initial primitives with different properties
plate1 = GaussianPrimitive(position=[0, 0, 0], rotation_angles=[0, 0, 0], scale=[0.5, 0.5])
plate2 = GaussianPrimitive(position=[1.5, 0, 0], rotation_angles=[0, 45, 0], scale=[0.7, 0.3])

# Create the plotter
plotter = pv.Plotter(window_size=[1600, 1000])

# Add the two plates to the scene
actor1 = plotter.add_mesh(plate1.points, cmap='viridis', point_size=5)
actor2 = plotter.add_mesh(plate2.points, cmap='plasma', point_size=5)

# --- Define Callback Functions for Sliders ---
# These functions will be called whenever a slider is moved

def update_plate1_position(value, index):
    new_pos = plate1.p
    new_pos[index] = value
    plate1.update_params(position=new_pos)
    actor1.overwrite(plate1.points) # Update the mesh in the plotter

def update_plate1_rotation(value, index):
    new_angles = plate1.angles
    new_angles[index] = value
    plate1.update_params(rotation_angles=new_angles)
    actor1.overwrite(plate1.points)

def update_plate1_scale(value, index):
    new_scale = plate1.s
    new_scale[index] = value
    plate1.update_params(scale=new_scale)
    actor1.overwrite(plate1.points)

def update_plate2_position(value, index):
    new_pos = plate2.p
    new_pos[index] = value
    plate2.update_params(position=new_pos)
    actor2.overwrite(plate2.points)

def update_plate2_rotation(value, index):
    new_angles = plate2.angles
    new_angles[index] = value
    plate2.update_params(rotation_angles=new_angles)
    actor2.overwrite(plate2.points)

def update_plate2_scale(value, index):
    new_scale = plate2.s
    new_scale[index] = value
    plate2.update_params(scale=new_scale)
    actor2.overwrite(plate2.points)

# --- Add all the sliders to the UI ---
# Plate 1
plotter.add_slider_widget(lambda v: update_plate1_position(v, 0), [-2, 2], value=0, title="Plate 1: Pos X")
plotter.add_slider_widget(lambda v: update_plate1_position(v, 1), [-2, 2], value=0, title="Plate 1: Pos Y", pointa=(.1, .87), pointb=(.45, .87))
plotter.add_slider_widget(lambda v: update_plate1_position(v, 2), [-2, 2], value=0, title="Plate 1: Pos Z", pointa=(.55, .87), pointb=(.9, .87))

plotter.add_slider_widget(lambda v: update_plate1_rotation(v, 0), [-180, 180], value=0, title="Plate 1: Yaw (Z)")
plotter.add_slider_widget(lambda v: update_plate1_rotation(v, 1), [-180, 180], value=0, title="Plate 1: Pitch (Y)", pointa=(.1, .67), pointb=(.45, .67))
plotter.add_slider_widget(lambda v: update_plate1_rotation(v, 2), [-180, 180], value=0, title="Plate 1: Roll (X)", pointa=(.55, .67), pointb=(.9, .67))

plotter.add_slider_widget(lambda v: update_plate1_scale(v, 0), [0.1, 2], value=0.5, title="Plate 1: Scale U")
plotter.add_slider_widget(lambda v: update_plate1_scale(v, 1), [0.1, 2], value=0.5, title="Plate 1: Scale V", pointa=(.1, .47), pointb=(.45, .47))

# Plate 2
plotter.add_slider_widget(lambda v: update_plate2_position(v, 0), [-2, 2], value=1.5, title="Plate 2: Pos X", pointa=(.1, .35), pointb=(.45, .35))
plotter.add_slider_widget(lambda v: update_plate2_position(v, 1), [-2, 2], value=0, title="Plate 2: Pos Y", pointa=(.55, .35), pointb=(.9, .35))
plotter.add_slider_widget(lambda v: update_plate2_position(v, 2), [-2, 2], value=0, title="Plate 2: Pos Z")

plotter.add_slider_widget(lambda v: update_plate2_rotation(v, 0), [-180, 180], value=0, title="Plate 2: Yaw (Z)", pointa=(.1, .15), pointb=(.45, .15))
plotter.add_slider_widget(lambda v: update_plate2_rotation(v, 1), [-180, 180], value=45, title="Plate 2: Pitch (Y)", pointa=(.55, .15), pointb=(.9, .15))

plotter.add_slider_widget(lambda v: update_plate2_scale(v, 0), [0.1, 2], value=0.7, title="Plate 2: Scale U")
plotter.add_slider_widget(lambda v: update_plate2_scale(v, 1), [0.1, 2], value=0.3, title="Plate 2: Scale V", pointa=(.1, .05), pointb=(.45, .05))

# Add a ground plane and axes for reference
plotter.add_axes()
plotter.add_ground_plane()

# Show the interactive window
plotter.show()