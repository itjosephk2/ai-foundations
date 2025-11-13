import numpy as np

# --- Single vector: magnitude (Euclidean/L2 norm) ---
# --- This is euclidian norm in the case of 2d vector its pythagreon theory ---
v = np.array([3.0, 4.0])  # float dtype avoids integer division surprises
mag_v = np.linalg.norm(v) # used for getting the magnitude euclidian norm
print("v:", v)
print("‖v‖ (L2):", mag_v)  # expect 5.0

# --- Unit vector (normalization) ---
# --- For a unit vector you just divide by the magnitude to get one unit ---
u = v / mag_v # gets the vector unit 
print("v normalized (unit vector):", u)
print("‖u‖:", np.linalg.norm(u))   # this gets the magnitude of the vector unit of v 

# --- Batch of vectors: row-wise normalization ---
# --- 
X = np.array([
    [3.0, 4.0],
    [1.0, 2.0],
    [0.0, -5.0],
]) # creates a list of vectors
row_norms = np.linalg.norm(X, axis=1, keepdims=True)  # shape (n,1)
X_unit = X / row_norms
np.set_printoptions(precision=4, floatmode='maxprec_equal') # dont worry about this for now copy and pasted for formatting
print("\nX:", X)
print("Row norms:\n", row_norms.ravel())
print("Row-normalized X (each row has length 1):\n", X_unit)
print("Row norms after normalization:",
      np.linalg.norm(X_unit, axis=1))  # gets the euclidian distance (magnitude p2) of the unit vectors of the vectors un the list


# --- we can come back to this later ---
# --- Safe normalization helper (handles zero vectors) ---
def normalize(x: np.ndarray, axis=None, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize x along 'axis' with L2 norm. Adds eps to avoid divide-by-zero.
    - axis=None: normalize whole array as a single vector
    - axis=1: normalize each row (common for ML features)
    """
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)

z = np.array([0.0, 0.0])
print("\nSafe normalize zero-vector:", normalize(z))

print("\nSanity checks:")
assert abs(mag_v - 5.0) < 1e-9
assert abs(np.linalg.norm(u) - 1.0) < 1e-9
assert np.allclose(np.linalg.norm(X_unit, axis=1), 1.0, atol=1e-9)
print("✅ magnitude & normalization look good")
