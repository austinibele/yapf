import glob
import os
from yapf.yapflib.yapf_api import FormatFile


class Adder:
    def __init__(self, parent_dir=".", in_place=True):
        self.parent_dir = parent_dir
        self.in_place = in_place

    def run(self):
        files = self._find_files()
        for i, file in enumerate([files[0]]):
            reformatted_code, encoding, changed = FormatFile(filename=file, in_place=self.in_place)


    def _find_files(self):
        src_path = os.path.join(self.parent_dir, "src")
        return glob.glob(src_path + "/**/*.py")


if __name__ == "__main__":
    adder = Adder(parent_dir="tests/fixtures", in_place=False)
    adder.run()
