import pickle
import os.path

if __name__ == "__main__":
    file_path = "/tmp/detectron-download-cache/ImageNetPretrained/MSRA/"
    weights_file = "R-101.pkl"
    new_weights_file = "R-101_py2.pkl"
    testpkl = pickle.loads(open(os.path.join(file_path, weights_file), "rb").read())
    pickle.dump(testpkl, open(os.path.join(file_path, new_weights_file),"wb"), protocol=2)