#!/usr/bin/env python3
"""
Blender script to import PLY point clouds with vertex colors.

Usage in Blender:
    1. Open Blender
    2. Go to Scripting workspace
    3. Open this script
    4. Edit the PLY_PATH variable below
    5. Run the script (Alt+P or click "Run Script")

Or from command line:
    blender --python blender_import_pointcloud.py

This script:
    - Imports a PLY point cloud
    - Creates a mesh with vertex colors
    - Sets up a material that displays the vertex colors
    - Optionally converts points to small spheres/icospheres for better visibility
"""

import bpy
import struct
import os

# ============================================================================
# CONFIGURATION - Edit this path to your PLY file
# ============================================================================
PLY_PATH = "/srv/searidge_share/outputs/lyra/bright23_h200_lyra_converted.ply"

# Display mode: "points", "spheres", "cubes", or "surface"
# - "points": Just vertices (fastest, but colors won't show)
# - "spheres"/"cubes": Small shapes at each point (colors visible, sparse look)
# - "surface": Attempt to create a solid surface using ball pivoting (experimental)
DISPLAY_MODE = "spheres"

# Size of spheres/cubes 
# - For sparse point clouds, use larger values (0.02-0.05)
# - For dense point clouds, use smaller values (0.005-0.01)
POINT_SIZE = 0.02

# Maximum number of points to import (set to None for all)
# Useful for testing with large point clouds
MAX_POINTS = None  # e.g., 100000 for first 100k points


# ============================================================================
# FUNCTIONS
# ============================================================================

def read_ply_with_colors(filepath, max_points=None):
    """Read a PLY file and extract positions and colors.
    
    Supports two formats:
    1. Point cloud format: x, y, z (float), red, green, blue (uchar)
    2. 3DGS format: x, y, z, nx, ny, nz, f_dc_0, f_dc_1, f_dc_2, ... (all float)
    """
    positions = []
    colors = []
    
    with open(filepath, 'rb') as f:
        # Read header
        header_lines = []
        vertex_count = 0
        
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            
            if line == 'end_header':
                break
        
        print(f"PLY Header: {vertex_count} vertices")
        print(f"Header properties: {[l for l in header_lines if 'property' in l][:10]}...")
        
        # Determine format from header
        has_rgb_colors = any('red' in line or 'green' in line or 'blue' in line 
                            for line in header_lines)
        has_sh_colors = any('f_dc_0' in line for line in header_lines)
        has_normals = any('property float nx' in line for line in header_lines)
        
        if has_rgb_colors:
            print("Detected: Point cloud format (RGB colors)")
            # Format: x, y, z (float32), r, g, b (uint8)
            vertex_size = 12 + 3  # 3 floats + 3 bytes
            color_format = "rgb"
        elif has_sh_colors:
            print("Detected: 3DGS format (Spherical Harmonics)")
            # Format: x,y,z (3f), nx,ny,nz (3f), f_dc_0,f_dc_1,f_dc_2 (3f), opacity (f), scale (3f), rot (4f)
            # = 3 + 3 + 3 + 1 + 3 + 4 = 17 floats = 68 bytes
            vertex_size = 17 * 4  # 17 floats
            color_format = "sh"
        else:
            print("Warning: PLY file doesn't have recognized color properties")
            print("Will use default gray color")
            # Just positions
            vertex_size = 12
            color_format = "none"
        
        count = min(vertex_count, max_points) if max_points else vertex_count
        print(f"Reading {count:,} vertices (vertex size: {vertex_size} bytes)...")
        
        for i in range(count):
            data = f.read(vertex_size)
            if len(data) < vertex_size:
                print(f"Warning: Unexpected end of file at vertex {i}")
                break
            
            # Position is always first 3 floats
            x, y, z = struct.unpack('<3f', data[:12])
            positions.append((x, y, z))
            
            if color_format == "rgb":
                r, g, b = struct.unpack('<3B', data[12:15])
                colors.append((r / 255.0, g / 255.0, b / 255.0, 1.0))
            elif color_format == "sh":
                # Skip normals (3 floats = 12 bytes), read SH DC (3 floats)
                # Offset: 12 (pos) + 12 (normals) = 24
                f_dc_0, f_dc_1, f_dc_2 = struct.unpack('<3f', data[24:36])
                # Convert SH DC to RGB: color = SH_C0 * dc + 0.5
                SH_C0 = 0.28209479177387814
                r = max(0, min(1, f_dc_0 * SH_C0 + 0.5))
                g = max(0, min(1, f_dc_1 * SH_C0 + 0.5))
                b = max(0, min(1, f_dc_2 * SH_C0 + 0.5))
                colors.append((r, g, b, 1.0))
            else:
                colors.append((0.5, 0.5, 0.5, 1.0))  # Gray
            
            if (i + 1) % 100000 == 0:
                print(f"  Read {i + 1:,} / {count:,} vertices...")
    
    print(f"Loaded {len(positions):,} vertices with colors")
    if colors:
        sample = colors[0]
        print(f"Sample color: R={sample[0]:.3f}, G={sample[1]:.3f}, B={sample[2]:.3f}")
    return positions, colors


def create_point_cloud_mesh(name, positions, colors, as_cubes=True):
    """Create a mesh object from points with vertex colors.
    
    Args:
        name: Object name
        positions: List of (x, y, z) tuples
        colors: List of (r, g, b, a) tuples (0-1 range)
        as_cubes: If True, create tiny cubes at each point (slower but colors work)
                  If False, create just vertices (faster but colors won't show)
    """
    import bmesh
    
    if as_cubes and len(positions) > 0:
        # Create tiny cubes at each point - this ensures colors are visible
        # because Blender needs faces to display vertex colors
        
        bm = bmesh.new()
        
        # Get or create color layer
        color_layer = bm.loops.layers.color.new('Color')
        
        cube_size = POINT_SIZE
        
        # Limit for performance - cubes are expensive
        max_cubes = min(len(positions), 500000)
        if len(positions) > max_cubes:
            print(f"  Warning: Limiting to {max_cubes:,} points for performance")
            # Sample evenly
            step = len(positions) // max_cubes
            indices = range(0, len(positions), step)
        else:
            indices = range(len(positions))
        
        for idx, i in enumerate(indices):
            pos = positions[i]
            color = colors[i] if i < len(colors) else (0.5, 0.5, 0.5, 1.0)
            
            # Create a tiny cube
            half = cube_size / 2
            verts = [
                bm.verts.new((pos[0] - half, pos[1] - half, pos[2] - half)),
                bm.verts.new((pos[0] + half, pos[1] - half, pos[2] - half)),
                bm.verts.new((pos[0] + half, pos[1] + half, pos[2] - half)),
                bm.verts.new((pos[0] - half, pos[1] + half, pos[2] - half)),
                bm.verts.new((pos[0] - half, pos[1] - half, pos[2] + half)),
                bm.verts.new((pos[0] + half, pos[1] - half, pos[2] + half)),
                bm.verts.new((pos[0] + half, pos[1] + half, pos[2] + half)),
                bm.verts.new((pos[0] - half, pos[1] + half, pos[2] + half)),
            ]
            
            # Create faces (6 faces for a cube)
            face_indices = [
                (0, 1, 2, 3),  # bottom
                (4, 7, 6, 5),  # top
                (0, 4, 5, 1),  # front
                (2, 6, 7, 3),  # back
                (0, 3, 7, 4),  # left
                (1, 5, 6, 2),  # right
            ]
            
            for fi in face_indices:
                try:
                    face = bm.faces.new([verts[j] for j in fi])
                    # Set color for each loop (corner) of the face
                    for loop in face.loops:
                        loop[color_layer] = color
                except:
                    pass  # Skip if face already exists
            
            if (idx + 1) % 50000 == 0:
                print(f"  Created {idx + 1:,} / {len(list(indices)):,} cubes...")
        
        # Create mesh from bmesh
        mesh = bpy.data.meshes.new(name)
        bm.to_mesh(mesh)
        bm.free()
        
    else:
        # Simple vertex-only mesh (colors won't display properly)
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(positions, [], [])
        mesh.update()
        
        # Add color attribute anyway
        if hasattr(mesh, 'color_attributes'):
            if 'Color' in mesh.color_attributes:
                mesh.color_attributes.remove(mesh.color_attributes['Color'])
            
            color_attr = mesh.color_attributes.new(
                name='Color',
                type='FLOAT_COLOR',
                domain='POINT'
            )
            
            for i, color in enumerate(colors):
                color_attr.data[i].color = color
    
    # Create object
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    
    return obj


def create_point_cloud_with_geometry_nodes(obj, point_size=0.01):
    """Add geometry nodes modifier to display points as visible spheres WITH colors."""
    # Create geometry nodes modifier
    mod = obj.modifiers.new(name="PointCloud", type='NODES')
    
    # Create node group
    node_group = bpy.data.node_groups.new(name="PointCloudDisplay", type='GeometryNodeTree')
    mod.node_group = node_group
    
    # Create nodes
    nodes = node_group.nodes
    links = node_group.links
    
    # Input/Output - use interface for Blender 4.0+
    try:
        # Blender 4.0+ style
        node_group.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
        node_group.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    except:
        # Blender 3.x style
        node_group.inputs.new('NodeSocketGeometry', 'Geometry')
        node_group.outputs.new('NodeSocketGeometry', 'Geometry')
    
    input_node = nodes.new('NodeGroupInput')
    input_node.location = (-600, 0)
    
    output_node = nodes.new('NodeGroupOutput')
    output_node.location = (600, 0)
    
    # Mesh to Points (converts mesh vertices to point cloud)
    mesh_to_points = nodes.new('GeometryNodeMeshToPoints')
    mesh_to_points.location = (-400, 0)
    
    # Capture the color attribute BEFORE converting to points
    # Named Attribute node to get color
    named_attr = nodes.new('GeometryNodeInputNamedAttribute')
    named_attr.location = (-400, -200)
    named_attr.data_type = 'FLOAT_COLOR'
    named_attr.inputs['Name'].default_value = 'Color'
    
    # Instance on Points
    instance_on_points = nodes.new('GeometryNodeInstanceOnPoints')
    instance_on_points.location = (-100, 0)
    
    # Ico Sphere for point visualization
    ico_sphere = nodes.new('GeometryNodeMeshIcoSphere')
    ico_sphere.location = (-300, -100)
    ico_sphere.inputs['Radius'].default_value = point_size
    ico_sphere.inputs['Subdivisions'].default_value = 1
    
    # Realize Instances
    realize = nodes.new('GeometryNodeRealizeInstances')
    realize.location = (100, 0)
    
    # Store Named Attribute - transfer color to the realized geometry
    store_attr = nodes.new('GeometryNodeStoreNamedAttribute')
    store_attr.location = (300, 0)
    store_attr.data_type = 'FLOAT_COLOR'
    store_attr.domain = 'POINT'
    store_attr.inputs['Name'].default_value = 'Color'
    
    # Connect nodes
    links.new(input_node.outputs['Geometry'], mesh_to_points.inputs['Mesh'])
    links.new(mesh_to_points.outputs['Points'], instance_on_points.inputs['Points'])
    links.new(ico_sphere.outputs['Mesh'], instance_on_points.inputs['Instance'])
    links.new(instance_on_points.outputs['Instances'], realize.inputs['Geometry'])
    links.new(realize.outputs['Geometry'], store_attr.inputs['Geometry'])
    links.new(named_attr.outputs['Attribute'], store_attr.inputs['Value'])
    links.new(store_attr.outputs['Geometry'], output_node.inputs['Geometry'])


def create_vertex_color_material(obj):
    """Create a material that displays vertex colors."""
    mat = bpy.data.materials.new(name="VertexColorMaterial")
    mat.use_nodes = True
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear default nodes
    nodes.clear()
    
    # Create nodes
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)
    
    # Use Emission shader for point clouds - shows colors without lighting
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (0, 0)
    emission.inputs['Strength'].default_value = 1.0
    
    # Color Attribute node (Blender 3.2+)
    # Try the newer "Color Attribute" node first
    try:
        color_attr = nodes.new('ShaderNodeAttribute')
        color_attr.location = (-300, 0)
        color_attr.attribute_name = 'Color'
        color_output = color_attr.outputs['Color']
    except:
        # Fallback to older Vertex Color node
        color_attr = nodes.new('ShaderNodeVertexColor')
        color_attr.location = (-300, 0)
        color_attr.layer_name = 'Color'
        color_output = color_attr.outputs['Color']
    
    # Connect - use emission for better visibility of point clouds
    links.new(color_output, emission.inputs['Color'])
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    # Assign material to object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    
    return mat


def main():
    """Main function to import and display point cloud."""
    print("=" * 60)
    print("Blender Point Cloud Importer")
    print("=" * 60)
    
    if not os.path.exists(PLY_PATH):
        print(f"ERROR: File not found: {PLY_PATH}")
        print("Please check the PLY_PATH variable at the top of this script.")
        return
    
    file_size = os.path.getsize(PLY_PATH)
    print(f"Loading: {PLY_PATH}")
    print(f"File size: {file_size / 1024 / 1024:.1f} MB")
    print(f"Display mode: {DISPLAY_MODE}")
    print(f"Point size: {POINT_SIZE}")
    print(f"Max points: {MAX_POINTS if MAX_POINTS else 'All'}")
    print("")
    
    # Read PLY file
    print("Reading PLY file...")
    positions, colors = read_ply_with_colors(PLY_PATH, MAX_POINTS)
    
    if not positions:
        print("ERROR: No vertices loaded!")
        print("The PLY file may be empty or in an unsupported format.")
        return
    
    print(f"\nCreating mesh with {len(positions):,} vertices...")
    print(f"Display mode: {DISPLAY_MODE}")
    
    # Create mesh object
    # Use cubes for "spheres" or "cubes" mode, vertices only for "points"
    obj_name = os.path.splitext(os.path.basename(PLY_PATH))[0]
    use_cubes = DISPLAY_MODE in ("spheres", "cubes")
    
    if use_cubes:
        print("Creating colored cubes at each point (this may take a while for large point clouds)...")
    else:
        print("Creating vertices only (colors may not display - use 'spheres' mode for colors)")
    
    obj = create_point_cloud_mesh(obj_name, positions, colors, as_cubes=use_cubes)
    
    print(f"Created object: {obj.name}")
    print(f"Mesh vertices: {len(obj.data.vertices):,}")
    if obj.data.polygons:
        print(f"Mesh faces: {len(obj.data.polygons):,}")
    
    # Create material with vertex colors
    print("Creating material...")
    create_vertex_color_material(obj)
    
    # Select the object
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    # Set viewport shading to show vertex colors
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    # Use Material Preview mode
                    space.shading.type = 'MATERIAL'
                    # Also configure Solid mode to show vertex colors
                    space.shading.color_type = 'VERTEX'
                    break
    
    # Frame the object in view
    try:
        bpy.ops.view3d.view_selected()
    except:
        pass  # May fail if no 3D view is active
    
    print("")
    print("=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print(f"Created object: {obj_name}")
    print(f"Vertices: {len(positions):,}")
    print("Viewport set to Material Preview mode")
    print("")
    print("If you don't see the object:")
    print("  1. Press Numpad . to focus on the object")
    print("  2. Press Z and select 'Material Preview'")
    print("  3. The object may be very small - zoom in!")
    print("  4. Check the Outliner (top right) for the object")
    print("=" * 60)


# Run if executed in Blender
if __name__ == "__main__":
    main()

