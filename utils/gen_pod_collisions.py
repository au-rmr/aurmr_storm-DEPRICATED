"""
Created by Collin Dang at the Sensor Systems Lab at UW. Generates temporary collision boxes for pod in IssacGym.

Directions for use:

1. Tweak OFFSET_X, OFFSET_Y, and OFFSET_Z as needed to shift entire collection of collision boxes.
2. Run 'python gen_pod_colllisions.py'.
3. Go to output_pod.txt and copy-paste the text into collision_primitives_3d_collision.yml under the line where it says "pod collisions below this line".
4. Run ur16e_collision.py.

"""


from os import linesep

FILE_NAME = 'output_pod.txt'

OFFSET_X = 0
OFFSET_Y = 0
OFFSET_Z = 0

BIN_WIDTH = .2286
RECIPE = [8.75, 5.5, 7.5, 6, 4.5, 8.75, 5, 6, 5, 8.75, 5, 6, 11, 0] # represents heights of bins from bottom to top

def main():
    # write to output file (output.txt)
    f = open(FILE_NAME, 'w')
    cube_num = 100

    # generate vertical slices
    f.write('# generating vertical slices\n')
    dims = [0.001, 1, 2.5]
    x_pos = 1.2 + OFFSET_X
    y_pos = 1.4 + OFFSET_Y
    z_pos = -0.4 + OFFSET_Z
    for i in range(5):
        f.write(f"cube{cube_num}:\n")
        f.write(f"  dims: [{dims[0]}, {dims[1]}, {dims[2]}]\n")
        f.write(f"  pose: [{x_pos}, {y_pos}, {z_pos}, 0.5, 0.5, 0.5, -0.5]\n")

        cube_num += 1
        z_pos += BIN_WIDTH

    f.write("\n# generating horizontal slices\n")
    # generate horizontal slices
    dims = [1, 1, 0.001]
    x_pos = 1.2 + OFFSET_X
    y_pos = 0.34 + OFFSET_Y
    z_pos = 0.05 + OFFSET_Z
    
    INCHES_TO_M = 0.0254
    for i in RECIPE:
        f.write(f"cube{cube_num}:\n")
        f.write(f"  dims: [{dims[0]}, {dims[1]}, {dims[2]}]\n")
        f.write(f"  pose: [{x_pos}, {y_pos}, {z_pos}, 0.5, 0.5, 0.5, -0.5]\n")

        cube_num += 1
        y_pos += i * INCHES_TO_M

    f.write("\n")
    f.write('# generating back slice\n')
    # generate back slice
    dims = [1, 0.001, 2.5]
    x_pos = 0.845 + OFFSET_X
    y_pos = 1.35 + OFFSET_Y
    z_pos = 0.05 + OFFSET_Z
    f.write(f"cube{cube_num}:\n")
    f.write(f"  dims: [{dims[0]}, {dims[1]}, {dims[2]}]\n")
    f.write(f"  pose: [{x_pos}, {y_pos}, {z_pos}, 0.5, 0.5, 0.5, -0.5]\n")

    cube_num += 1

    f.write("\n")
    f.write('# generating left bar\n')
    # generate left bar
    dims = [0.05, 0.001, 2.5]
    x_pos = 0.695 + OFFSET_X
    y_pos = 1.35 + OFFSET_Y
    z_pos = -.39 + OFFSET_Z
    f.write(f"cube{cube_num}:\n")
    f.write(f"  dims: [{dims[0]}, {dims[1]}, {dims[2]}]\n")
    f.write(f"  pose: [{x_pos}, {y_pos}, {z_pos}, 0.5, 0.5, 0.5, -0.5]\n")

    cube_num += 1

    f.write("\n")
    f.write('# generating right bar\n')
    # generate right bar
    dims = [0.05, 0.001, 2.5]
    x_pos = 0.695 + OFFSET_X
    y_pos = 1.35 + OFFSET_Y
    z_pos = 0.506 + OFFSET_Z
    f.write(f"cube{cube_num}:\n")
    f.write(f"  dims: [{dims[0]}, {dims[1]}, {dims[2]}]\n")
    f.write(f"  pose: [{x_pos}, {y_pos}, {z_pos}, 0.5, 0.5, 0.5, -0.5]\n")

    cube_num += 1

    f.write("\n")
    f.write('# generating top bar\n')
    # generate top
    dims = [1, 0.001, 0.05]
    x_pos = 0.7 + OFFSET_X
    y_pos = 2.6 + OFFSET_Y
    z_pos = 0.05 + OFFSET_Z
    f.write(f"cube{cube_num}:\n")
    f.write(f"  dims: [{dims[0]}, {dims[1]}, {dims[2]}]\n")
    f.write(f"  pose: [{x_pos}, {y_pos}, {z_pos}, 0.5, 0.5, 0.5, -0.5]\n")

    cube_num += 1

    f.write("\n")
    f.write('# generating bottom bar\n')
    # generate bottom bar
    dims = [1, 0.001, 0.05]
    x_pos = 0.6817 + OFFSET_X
    y_pos = 0.27 + OFFSET_Y
    z_pos = 0.05 + OFFSET_Z
    f.write(f"cube{cube_num}:\n")
    f.write(f"  dims: [{dims[0]}, {dims[1]}, {dims[2]}]\n")
    f.write(f"  pose: [{x_pos}, {y_pos}, {z_pos}, 0.5, 0.5, 0.5, -0.5]\n")

    cube_num += 1

if __name__ == "__main__":
    main()