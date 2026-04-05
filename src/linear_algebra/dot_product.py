def dot_product(x, y):
    for i, vec in enumerate([x, y], start=1):
        if not isinstance(vec, (list, tuple)):
            raise TypeError(f"Argument #{i} must be a list or tuple")
    if len(x) != len(y):
        raise ValueError("Vectors must be the same length")
 
    dot_product = sum(xi * yi for xi, yi in zip(x, y))
    return dot_product

print(dot_product([4, 0], [5, 5]))      
print(dot_product([1, 1], [3, 2]))      
print(dot_product([1, 2, 3], [4, 5, 6]))
