
from os import linesep

FILE_NAME = 'pod.urdf'
BASE_LINK = 'pod_base_link'

# finds index of nth occurrence of needle within haystack
def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

# given a property, finds the values between the two quotation marks ("") after the property. returns string
def extract_property(property, cur_line):
    after_prop = cur_line.index(property) + len(property)
    cur_line = cur_line[after_prop:]
    pos = cur_line.index("\"")
    endpos = cur_line.index("\"", pos + 1)
    return cur_line[pos + 1:endpos]


def main():
    with open(FILE_NAME) as f:
        urdf = f.readlines()
        f.close()
    
    # take out newline chars
    for i in range(0, len(urdf)):
        urdf[i] = urdf[i][:-1]
        # print(urdf[i])
    
    # construct links and collision boxes and transforms -------
    links = []
    i = 0
    transforms = {}
    transforms[BASE_LINK] = (0, 0, 0)

    # iterate through urdf
    while i < len(urdf):
        line = urdf[i]

        # if we see a transform, add it!
        if "parent" in line:
            parent = extract_property("parent", line)

            i += 1
            line = urdf[i]

            # get child (always after parent)
            child = extract_property("child", line)

            i += 1
            line = urdf[i]

            # get transform
            xyz = extract_property("xyz", line)
            transform_arr = xyz.split(" ")
            transform = (float(transform_arr[0]), float(transform_arr[1]), float(transform_arr[2]))

            transforms[child] = tuple(sum(x) for x in zip(transforms[parent], transform))
        
        # if we see a link, snip it out and add to array
        if "link" in line and "/link" not in line and "parent" not in line and "child" not in line:
            pos = line.index("name=") + 6
            endpos = line.rindex("\"")
            link = Link(line[pos:endpos])
            links.append(link)

            # add collision boxes under link
            while "/link" not in line:
                i += 1
                line = urdf[i]
                if "<collision>" in line:
                    collision = link.Collision() # create collision object
                    # go forward until we see origin
                    while "origin" not in line:
                        i += 1
                        line = urdf[i]
                    # we now have origin. extract rpy and xyz
                    rpy = extract_property("rpy", line)
                    rpy_split = rpy.split()
                    collision.set_rpy(float(rpy_split[0]), float(rpy_split[1]), float(rpy_split[2]))

                    xyz = extract_property("xyz", line)
                    xyz_split = xyz.split()
                    collision.set_xyz(float(xyz_split[0]), float(xyz_split[1]), float(xyz_split[2]))

                    # extra dimensions (box)
                    while "box" not in line:
                        i += 1
                        line = urdf[i]

                    size = extract_property("size", line)
                    xyz_split = size.split()
                    collision.set_dim(float(xyz_split[0]), float(xyz_split[1]), float(xyz_split[2]))

                    link.add_collision(collision)

        # increment i
        i += 1
    
    # print out results
    for link in links:
        print(str(link))
        for collision in link.children:
            print('\t' + str(collision))

    for key, val in transforms.items():
        print(key + " -> " + str(val))

    f = open('output.txt', 'w')
    cube_num = 1
    for link in links:
        link_name = link.name
        for collision in link.children:
            # print('DEBUG')
            # print(str(collision.x) + " | " + str(transforms[link_name][0]))
            x_pos = collision.x + transforms[link_name][0]
            y_pos = collision.y + transforms[link_name][1]
            z_pos = collision.z + transforms[link_name][2]

            f.write(f"cube{cube_num}:\n")
            f.write(f"  dims: [{collision.x_dim}, {collision.y_dim}, {collision.z_dim}]\n")
            f.write(f"  pose: [{x_pos}, {z_pos}, {y_pos}, -0.707, 0, 0, 0.707]\n")

            cube_num += 1


# specifies a link's name and children(collision objects)
class Link():

    # specifies dimensions, xyz, and rpy of each collision
    class Collision():
        def __init__(self):
            pass

        def __str__(self):
            return f'dim: {self.x_dim}, {self.y_dim}, {self.z_dim}. xyz: {self.x}, {self.y}, {self.z}. rpy: {self.r}, {self.p}, {self.y}. '

        def set_dim(self, x_dim, y_dim, z_dim):
            self.x_dim = x_dim
            self.y_dim = y_dim
            self.z_dim = z_dim
        
        def set_xyz(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

        def set_rpy(self, r, p, y):
            self.r = r
            self.p = p
            self.y = y

    def __init__(self, name):
        self.name = name
        self.children = []
    
    def __str__(self):
        return self.name
    
    def add_collision(self, collision):
        self.children.append(collision)

if __name__ == "__main__":
    main()