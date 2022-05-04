from torchvision.io import read_image

class ImageLoader():

    @classmethod
    def load_image_to_torch(cls, image_path):
        return read_image(path=image_path)