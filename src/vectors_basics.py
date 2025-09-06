import numpy as np

# create a vector 
x = 3
y = 4
v_1 = np.array([x, y])
v_2 = np.array([y, x])

# Magnitude (length)
v_1_magnitude = np.linalg.norm(v_1) # Euclidian
v_2_magnitude = np.linalg.norm(v_2) # Euclidian
np.linalg.norm(v_1, ord=1) # manhattan

# Unit vector (normalize)
v_1_unit = v_1 / v_1_magnitude # divide by the magnitude to get the unit vector
v_2_unit = v_2 / v_2_magnitude # divide by the magnitude to get the unit vector

# Dot product
v_1_2_dotproduct = np.dot(v_1, v_2) 

# Cosine similarity
v_1_2_dotproduct / (v_1_magnitude * v_2_magnitude) # divide the dotproduct by the product of the original magnitudes of the vectors
