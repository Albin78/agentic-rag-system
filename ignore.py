import sys
import numpy as np

# x = 1.28796
# x_64 = np.float64(x)
# z_u8 = x_64.astype("uint8", copy=False)
# z_u16 = x_64.astype("uint16", copy=False)
# y = np.float32(1.2)

# print(sys.getsizeof(x)) 
# print(y.nbytes)   
# print(f"X 64 bit: {x_64} and bytes: {x_64.nbytes}")
# print(f"Z u8: {z_u8} and bytes: {z_u8.nbytes}")       
# print(f"Z u16: {z_u16} and bytes: {z_u16.nbytes}")       

# a = np.random.rand(1,768).astype(np.float32)

# b = a.astype("float64", copy=False)

# c = np.float64(1.2)

# print(a is b)         
# print(a.dtype)
# print(b.dtype)
# print(c.nbytes)



# Example vectors
# x = np.array([1000.0, 0.0])
# z = np.array([10.0, 0.0])
# y = np.array([1000.0, 1.0])

# # ---- L2 distance ----
# def l2(a, b):
#     return np.linalg.norm(a - b)

# # ---- Cosine similarity ----
# def cosine(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# print("L2(x, z):", l2(x, z))
# print("Cosine(x, z):", cosine(x, z))

# print("L2(x, y):", l2(x, y))
# print("Cosine(x, y):", cosine(x, y))

# # ---- Normalized L2 vs Cosine equivalence ----
# x_norm = x / np.linalg.norm(x)
# y_norm = y / np.linalg.norm(y)

# print("L2(normalized x, normalized y):", l2(x_norm, y_norm))
# print("Cosine(x, y):", cosine(x, y))


# # assume x_norm and y_norm are normalized

# cos_val = np.dot(x_norm, y_norm)
# l2_val = np.linalg.norm(x_norm - y_norm)

# print("LHS (L2^2):", l2_val**2)
# print("RHS (2 - 2cos):", 2 - 2*cos_val)


x = np.random.randn(100, 768)
y = x[:, ::2]

# print("y result:", y)
print("Is contigous (x):", x.flags['C_CONTIGUOUS'])
print("Is contigous (y):", y.flags['C_CONTIGUOUS'])
