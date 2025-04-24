
import math


# Class for vectors
# class Vector:
#     def __init__(self, x, y, z): #values are initialized (x,y,z) dimensions
#         self.x = x
#         self.y = y
#         self.z = z

#     def __sub__(self, other):  # vector subtraction
#         return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

#     def __add__(self, other):  # vector addition -> returns new vector after this operation
#         return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

#     def dot(self, other): #scalar dot product of vector
#         return self.x * other.x + self.y * other.y + self.z * other.z

#     def length(self): # uses dot product to find the magnitude of vector
#         return np.sqrt(self.dot(self))

#     def normalize(self): #returns unit vector (divides each component by its magnitude: length here)
#         length = self.length()
#         return Vector(self.x / length, self.y / length, self.z / length)

""" Define vector using NumPy array. The operation carries our further mathematical operations and returns 
NumPy arrays that contains the x,y,z coords of the shape we are rendering"""

def vector(x,y,z):
    """Creates a vector as a tuple"""
    return (x,y,z)

def vector_subtraction(v1,v2):
    """returns 2 vector's subtraction"""
    return (v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])

def vector_addition(v1,v2):
    """returns 2 vector's addition"""
    return (v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2])

def dot_product(v1,v2):
    """Returns dot product of 2 vectors"""
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

def length_of_vector(v):
    """returns magnitude of a vector"""
    return math.sqrt(dot_product(v,v))

def normalize(v):
    """returns vector after converting it to its unit vector"""
    mag = length_of_vector(v)
    return (v[0]/mag, v[1]/mag, v[2]/mag)


class Ray: #class for Ray: uses origin (vector) and direction Vec
    def __init__(self, origin, direction):
        self.origin = origin #tuple (x,y,z)
        self.direction = normalize(direction) #to make direction vector a unit vector


class Sphere: #class for Sphere (the object we working on) takes center (vector) and radius
    def __init__(self, center, radius):
        self.center = center #tuple: (x,y,z)
        self.radius = radius

    def intersect(self, ray): #this checks if ray intersects, using quadratic formula (if we get discriminant >0 it means existing roots, means that light intersects sphere. else no)
        origin_center = vector_subtraction(ray.origin, self.center)
        a = dot_product(ray.direction, ray.direction)
        b = 2.0 * dot_product(origin_center, ray.direction)
        c = dot_product(origin_center, origin_center) - self.radius ** 2
        discriminant = b * b - 4 * a * c

        if discriminant < 0: #negative? no intersection
            return None  # No intersection

        sqrt_d = math.sqrt(discriminant)
        t1 = (-b - sqrt_d) / (2 * a)
        t2 = (-b + sqrt_d) / (2 * a)

        #find smallest valid t amongsts these two :
        if t1 > 0 and t2 > 0: #both are positive values,
            return min(t1, t2) #t = in float
        elif t1 > 0 and t2 < 0:
            return t1
        elif t2 > 0 and t1 < 0:
            return t2
        else:
            return None

class KdNode: #part of kdtree
    def __init__(self, left, right, objects):
        self.left = left
        self.right = right
        self.objects = objects #spheres

# Build a simple kd-tree (for demonstration purposes)
def build_kd_tree(objects, depth=0):
    if not objects:
        return None

    # Choose axis based on depth
    axis = depth % 3 # 0= x, 1= y, 2= z

    # Sort objects along the chosen axis
    def sort_key(obj):
        if axis == 0:
            return obj.center.x
        elif axis == 1:
            return obj.center.y
        else:
            return obj.center.z

    objects.sort(key=sort_key)

    # Find the median
    median = len(objects) // 2

    # Recursively build the tree
    return KdNode(
        left=build_kd_tree(objects[:median], depth + 1),
        right=build_kd_tree(objects[median + 1:], depth + 1),
        objects=[objects[median]]
    )

# Traverse the kd-tree to find intersections
def intersect_kd_tree(ray, tree): #find the smallest intersection
    if tree is None:
        return None

    # Check intersection with the current tree's objects
    hit = None
    for obj in tree.objects:
        t = obj.intersect(ray)
        if t is not None and (hit is None or t < hit):
            hit = t

    # Traverse the left and right subtrees
    hit_left = intersect_kd_tree(ray, tree.left)
    hit_right = intersect_kd_tree(ray, tree.right)

    # Find the closest hit
    if hit_left is not None and (hit is None or hit_left < hit):
        hit = hit_left
    if hit_right is not None and (hit is None or hit_right < hit):
        hit = hit_right

    return hit


# Test Case 1: Single Sphere, Ray Intersects
def test_case_1():
    # Define the sphere
    sphere_center = vector(0,0,0)
    sphere1 = Sphere(sphere_center, 0.5)  # Sphere at origin with radius 0.5

    # Define the ray
    ray_origin = vector(0,0,-1)
    ray_direction = vector(0,0,1)
    ray1 = Ray(ray_origin, ray_direction)  # Ray starts at (0, 0, -1) and points towards the sphere
    # Build the kd-tree
    kd_tree = build_kd_tree([sphere1])

    # # Check for intersection
    t = intersect_kd_tree(ray1, kd_tree)
    # render(400,400)

    # Assert that the intersection is valid
    assert t is not None, "Test Case 1 Failed: No intersection found"
    print(f"Test Case 1 Passed: Intersection at t = {t}")

# Run the test

# Test Case 2: Single Sphere, Ray Misses
def test_case_2():
    # Define the sphere
    sphere = Sphere(vector(0, 0, 0), 0.5)

     # Sphere at origin with radius 0.5

    # Define the ray
    ray = Ray(vector(2, 2, -1), vector(0, 0, 1))  # Ray starts at (2, 2, -1) and points away from the sphere

    # Build the kd-tree
    kd_tree = build_kd_tree([sphere])


    # Check for intersection
    t = intersect_kd_tree(ray, kd_tree)
    #render(400,400)

    # Assert that no intersection is found
    assert t is None, "Test Case 2 Failed: Unexpected intersection found"
    print("Test Case 2 Passed: No intersection (ray misses the sphere)")

# Run the test

# Test Case 3: Multiple Spheres, Ray Intersects Closest Sphere
def test_case_3():
    # Define two spheres
    sphere1 = Sphere(vector(0, 0, 1), 0.5)  # Sphere at (0, 0, 1) with radius 0.5
    sphere2 = Sphere(vector(0, 0, 3), 0.5)  # Sphere at (0, 0, 3) with radius 0.5

    # Define the ray
    ray = Ray(vector(0, 0, -1), vector(0, 0, 1))  # Ray starts at (0, 0, -1) and points towards both spheres

    # Build the kd-tree
    kd_tree = build_kd_tree([sphere1, sphere2])

    # Check for intersection
    t = intersect_kd_tree(ray, kd_tree)

    # Assert that the intersection is valid and corresponds to the closer sphere
    assert t is not None, "Test Case 3 Failed: No intersection found"
    assert t < 2.0, "Test Case 3 Failed: Intersection with the farther sphere"
    print(f"Test Case 3 Passed: Intersection at t = {t} (closer sphere)")

# Run the test

def test_case_4():
    # Define two spheres
    sphere1 = Sphere(vector(0, 0, 1), 0.5)  # Sphere at (0, 0, 1) with radius 0.5
    sphere2 = Sphere(vector(0, 0.6, 3), 0.5)  # Sphere at (0, 0.6, 3) with radius 0.5

    # Define the ray
    ray = Ray(vector(0, 1, -1), vector(0, 0, 1))  # Ray starts at (0, 1, -1) and points towards the spheres

    # Build the kd-tree
    kd_tree = build_kd_tree([sphere1, sphere2])

    # Check for intersection
    t = intersect_kd_tree(ray, kd_tree)

    # Assert that the intersection is valid and corresponds to the farther sphere
    assert t is not None, "Test Case 4 Failed: No intersection found"
    assert t > 2.0, "Test Case 4 Failed: Intersection with the closer sphere"
    print(f"Test Case 4 Passed: Intersection at t = {t} (farther sphere)")

# Test Case 5: Empty kd-tree
def test_case_5():
    # Define an empty list of spheres
    spheres = []

    # Define the ray
    ray = Ray(vector(0, 0, -1), vector(0, 0, 1))  # Ray starts at (0, 0, -1) and points forward

    # Build the kd-tree
    kd_tree = build_kd_tree(spheres)

    # Check for intersection
    t = intersect_kd_tree(ray, kd_tree)

    # Assert that no intersection is found
    assert t is None, "Test Case 5 Failed: Unexpected intersection found"
    print("Test Case 5 Passed: No intersection (empty kd-tree)")

# # Run the test

# test_case_1()
# test_case_2()
# test_case_3()
# test_case_4()
# test_case_5()

# --------------------------
# Test Case 1: Basic Intersection
# --------------------------
def test_basic_intersection():
    """Demonstrates KD-Tree returns valid intersection for a single sphere."""
    sphere = Sphere(vector(0, 0, 2), 1.0)
    ray = Ray(vector(0, 0, 0), vector(0, 0, 1))  # Shoots directly at sphere
    kd_tree = build_kd_tree([sphere])
    t = intersect_kd_tree(ray, kd_tree)
    assert t == 1.0, f"Expected t=1.0 (front surface), got {t}"
    print("[Demo] Test 1: Basic intersection (Validates core ray-sphere collision)")

# # --------------------------
# # Test Case 2: Empty Tree Handling
# # --------------------------
def test_empty_tree():
    """Validates the KD-Tree gracefully handles zero objects."""
    ray = Ray(vector(0, 0, 0), vector(1, 0, 0))
    kd_tree = build_kd_tree([])
    t = intersect_kd_tree(ray, kd_tree)
    assert t is None, "Expected no intersection with empty tree"
    print("[Demo] Test 2: Empty tree (Tests edge case and tree initialization)")

# # --------------------------
# # Test Case 3: Axis Splitting (X vs Y vs Z)
# # --------------------------
def test_axis_splitting():
    """Validates KD-Tree splits objects across X/Y/Z axes but ray misses all."""
    spheres = [
        Sphere(vector(-5, 0, 0), 0.5),   # Left subtree (X-axis split)
        Sphere(vector(0, 5, 0), 0.5),    # Right subtree (Y-axis split at depth=1)
        Sphere(vector(0, 0, 5), 0.5)     # Z-axis split at depth=2 (but ray misses)
    ]
    # Ray shoots diagonally through empty regions
    ray = Ray(vector(0, 0, 0), vector(1, 1, 1).normalize())
    kd_tree = build_kd_tree(spheres)
    t = intersect_kd_tree(ray, kd_tree)
    assert t is None, "KD-Tree splits correctly, but ray should miss all spheres"
    print("[Demo] Test 3: Axis splitting ✔️ (Tree splits spheres, ray misses)")

# # --------------------------
# # Test Case 4: Early Termination (Fixed)
# # --------------------------
# def test_early_termination():
#     """Validates KD-Tree skips irrelevant subtrees for efficiency."""
#     spheres = [
#         Sphere(vector(-5, 0, 0), 1.0),  # Left subtree (X-axis split)
#         Sphere(vector(5, 0, 2), 1.0)     # Right subtree
#     ]
#     ray = Ray(vector(5, 0, 0), vector(0, 0, 1))  # Only intersects right subtree
#     kd_tree = build_kd_tree(spheres)
#     t = intersect_kd_tree(ray, kd_tree)
#     assert t == 1.0, f"Expected t=1.0, got {t}"
#     print("[Demo] Test 4: Early termination ✔️ (Optimized traversal confirmed)")

# # --------------------------
# # Test Case 5: Closest Intersection
# # --------------------------
# def test_closest_intersection():
#     """Validates KD-Tree returns the closest hit, not just any hit."""
#     spheres = [
#         Sphere(vector(0, 0, 3), 1.0),  # Far sphere
#         Sphere(vector(0, 0, 1), 1.0)   # Near sphere
#     ]
#     ray = Ray(vector(0, 0, 0), vector(0, 0, 1))
#     kd_tree = build_kd_tree(spheres)
#     t = intersect_kd_tree(ray, kd_tree)
#     assert t == 0.0, "Closest sphere surface is at t=0.0 (ray origin inside sphere)"
#     print("[Demo] Test 5: Closest hit ✔️ (Critical for rendering accuracy)")

# # --------------------------
# # Run Tests
# # --------------------------

test_basic_intersection()
test_empty_tree()
test_axis_splitting()
# test_early_termination()
# test_closest_intersection()
