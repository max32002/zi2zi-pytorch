import os
import pickle
from io import BytesIO

from PIL import Image


def bytes_to_file(bytes_img):
    return BytesIO(bytes_img)


class PickledImageProvider(object):
    def __init__(self, obj_path):
        self.obj_path = obj_path
        self.examples = self.load_pickled_examples()

    def load_pickled_examples(self):
        with open(self.obj_path, "rb") as of:
            examples = list()
            while True:
                try:
                    e = pickle.load(of)
                    examples.append(e)
                    if len(examples) % 100000 == 0:
                        print("processed %d examples" % len(examples))
                except EOFError:
                    break
                except Exception:
                    pass
            print("unpickled total %d examples" % len(examples))
            return examples

def chk_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def unpickle_train_object(data_dir, sample_dir, label):
    obj_path = os.path.join(data_dir, 'train.obj')
    image_provider = PickledImageProvider(obj_path)
    
    image_index = -1
    for each_pickle in image_provider.examples:
        image_index += 1
        img_bytes = each_pickle[1]
        image_file = bytes_to_file(img_bytes)
        img = Image.open(image_file)
        try:
            img.save(os.path.join(sample_dir, "%d_%05d.png" % (label, image_index)))
        finally:
            image_file.close()

if __name__ == "__main__":
    data_dir = "experiments/data"
    chk_mkdir(data_dir)

    sample_dir = "source/paired_images"
    chk_mkdir(sample_dir)

    label = 0
    unpickle_train_object(data_dir, sample_dir, label)
