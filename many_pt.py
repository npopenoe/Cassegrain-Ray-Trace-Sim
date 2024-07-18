import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cassegrain_geo import CassegrainGeometry, Point, Hyperbolic, Parabolic

def calculate_normal(point, mirror):
    if isinstance(mirror, Parabolic):
        return np.array([2 * point.x, 2 * point.y, -2 * mirror.radius_curv])
    elif isinstance(mirror, Hyperbolic):
        R = mirror.radius_curv
        return np.array([point.x / np.sqrt(R**2 + 3 * (point.x**2 + point.y**2)), 
                         point.y / np.sqrt(R**2 + 3 * (point.x**2 + point.y**2)), 
                         -1])

def surface_primary(x, y, primary_radius_curv):
    return (x**2 + y**2) / (4 * primary_radius_curv)

# uses R = I - 2(I.N)N to calc reflected ray
def bounce_1(ray_direction, normal):
    normal = normal / np.linalg.norm(normal)
    return ray_direction - 2 * np.dot(ray_direction, normal) * normal

def trace_ray(telescope, point):
    # Calculate the direction of the ray to the primary mirror (vertical downward)
    ray_direction = np.array([0, 0, -1])
    
    # Find the intersection with the primary mirror
    z_primary = surface_primary(point.x, point.y, telescope.primary.radius_curv)  # The primary mirror surface
    t_primary = (z_primary - point.z) / ray_direction[2]
    target_primary = Point(
        point.x + t_primary * ray_direction[0],
        point.y + t_primary * ray_direction[1],
        z_primary
    )
    
    # Calculate the normal to the primary mirror at the target point
    normal_primary = calculate_normal(target_primary, telescope.primary)
    reflected_primary = bounce_1(ray_direction, normal_primary)
    
    # Calculate intersection with the secondary mirror
    z_secondary = telescope.secondary.z_position
    t_secondary = (z_secondary - target_primary.z) / reflected_primary[2]
    target_secondary = Point(
        target_primary.x + t_secondary * reflected_primary[0],
        target_primary.y + t_secondary * reflected_primary[1],
        z_secondary
    )
    
    # Calculate the normal to the secondary mirror at the intersection point
    normal_secondary = calculate_normal(target_secondary, telescope.secondary)
    reflected_secondary = bounce_1(reflected_primary, normal_secondary)

    return point, target_primary, target_secondary, reflected_secondary

def visualize_rays(telescope, rays):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for ray_path in rays:
        if ray_path is None:
            continue

        point, target_primary, target_secondary, reflected_secondary = ray_path

        # Plot the initial point
        ax.scatter(point.x, point.y, point.z, color='blue', marker='o', label='Atmospheric Particle')

        # Plot the primary reflection
        ax.plot([point.x, target_primary.x], [point.y, target_primary.y], [point.z, target_primary.z], color='orange', label='Photon Ray')
        ax.plot([target_primary.x, target_secondary.x], [target_primary.y, target_secondary.y], [target_primary.z, target_secondary.z], color='steelblue', label='Primary Reflection')

        # Plot the green line using the halved angle of reflection
        green_line_length = 18.0  # Adjust this length as needed
        green_endpoint = Point(target_secondary.x + green_line_length * reflected_secondary[0],
                               target_secondary.y + green_line_length * reflected_secondary[1],
                               target_secondary.z + green_line_length * reflected_secondary[2])
        ax.plot([target_secondary.x, green_endpoint.x], [target_secondary.y, green_endpoint.y], [target_secondary.z, green_endpoint.z], color='palevioletred', linewidth =0.7, label='Secondary Reflection')

    # Create a mask for the circular mirrors
    def circular_mask(X, Y, radius):
        return np.sqrt(X**2 + Y**2) <= radius

    # Plot the primary mirror with a circular mask
    theta = np.linspace(0, 2 * np.pi, 100)
    r_primary = 5.4745  # Actual radius of the primary mirror
    x_primary = np.linspace(-r_primary, r_primary, 100)
    y_primary = np.linspace(-r_primary, r_primary, 100)
    X_primary, Y_primary = np.meshgrid(x_primary, y_primary)
    mask_primary = circular_mask(X_primary, Y_primary, r_primary)
    Z_primary = np.zeros_like(X_primary)
    Z_primary[mask_primary] = (X_primary[mask_primary]**2 + Y_primary[mask_primary]**2) / (4 * telescope.primary.radius_curv)
    Z_primary[~mask_primary] = np.nan
    ax.plot_wireframe(X_primary, Y_primary, Z_primary, color='green', alpha=0.3)

    # Plot the secondary mirror with a circular mask
    r_secondary = 0.7090145  # Actual radius of the secondary mirror
    x_secondary = np.linspace(-r_secondary, r_secondary, 100)
    y_secondary = np.linspace(-r_secondary, r_secondary, 100)
    X_secondary, Y_secondary = np.meshgrid(x_secondary, y_secondary)
    mask_secondary = circular_mask(X_secondary, Y_secondary, r_secondary)
    Z_secondary = np.zeros_like(X_secondary)
    Z_secondary[mask_secondary] = telescope.secondary.surface(X_secondary[mask_secondary], Y_secondary[mask_secondary])
    Z_secondary[~mask_secondary] = np.nan
    ax.plot_wireframe(X_secondary, Y_secondary, Z_secondary, color='purple', alpha=0.3)

    # Plot focal points for reference
    #ax.scatter(0, 0, 17.5, color='green', marker='o', label='Primary Focal Point', alpha = 0.3)
    #ax.scatter(0, 0, -0.72, color='purple', marker='o', label='Back Focal Dist', alpha = 0.3)  

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=0, azim=-90)
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1
    plt.show()

if __name__ == "__main__":
    F = 149.583
    f1 = 17.489
    b = 2.5
    cassegrain_geo = CassegrainGeometry(F, b, f1)


    print("R1:", cassegrain_geo.primary_radius_curv)
    print("R2:", cassegrain_geo.secondary_radius_curv)
    print("K1:", cassegrain_geo.primary_K)
    print("K2:", cassegrain_geo.secondary_K)
    print("D:", cassegrain_geo.secondary_z_position)
    print("B:", cassegrain_geo.B)
    

    # Create a line of points across the diameter of the primary mirror spaced by 0.1
    x_values = np.arange(-5, 5.1, 0.55)
    test_points = [Point(x, 0, 20) for x in x_values]

    rays = [trace_ray(cassegrain_geo, point) for point in test_points]
    visualize_rays(cassegrain_geo, rays)

