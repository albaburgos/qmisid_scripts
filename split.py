import pickle

with open("/Users/albaburgosmondejar/Desktop/Dataset/file_4.pkl", "rb") as f:
    data = pickle.load(f)

sc = data["same_charge"]

print("type:", type(sc))
print("shape:", getattr(sc, "shape", None))
print("count >1:", (sc > 0).sum())
