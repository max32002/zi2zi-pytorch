import os
import pickle
from io import BytesIO

from PIL import Image


def bytes_to_file(bytes_img):
    return BytesIO(bytes_img)


class PickledImageProvider(object):
    def __init__(self, obj_path):
        self.obj_path = obj_path
        self.offsets = self.build_index()

    def build_index(self):
        """Build an index of file offsets for each pickled object."""
        offsets = []
        if not os.path.exists(self.obj_path):
            raise FileNotFoundError(f"Data file not found: {self.obj_path}")
        
        print(f"Indexing {self.obj_path} ...")
        with open(self.obj_path, "rb") as of:
            while True:
                try:
                    pos = of.tell()
                    pickle.load(of) # Read and discard data to advance pointer
                    offsets.append(pos)
                    if len(offsets) % 10000 == 0:
                        print(f"Indexed {len(offsets)} examples")
                except EOFError:
                    break
                except Exception as e:
                    print(f"Error indexing object at pos {pos}: {e}")
                    pass
        print(f"Indexed total {len(offsets)} examples")
        return offsets

    def __getitem__(self, index):
        """Retrieve a single item from the file using the offset."""
        if index < 0 or index >= len(self.offsets):
            raise IndexError("Index out of range")
        
        offset = self.offsets[index]
        with open(self.obj_path, "rb") as of:
            of.seek(offset)
            return pickle.load(of)

    def __len__(self):
        return len(self.offsets)


def chk_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def unpickle_train_object(data_dir, sample_dir, label):
    obj_path = os.path.join(data_dir, 'train.obj')
    image_provider = PickledImageProvider(obj_path)
    
    # Iterate using index since image_provider is now a sequence
    for i in range(len(image_provider)):
        each_pickle = image_provider[i]
        img_bytes = each_pickle[1]
        image_file = bytes_to_file(img_bytes)
        img = Image.open(image_file)
        try:
            img.save(os.path.join(sample_dir, "%d_%05d.png" % (label, i)))
        finally:
            image_file.close()

if __name__ == "__main__":
    data_dir = "experiments/data"
    chk_mkdir(data_dir)

    sample_dir = "source/paired_images"
    chk_mkdir(sample_dir)

    label = 0
    unpickle_train_object(data_dir, sample_dir, label)
