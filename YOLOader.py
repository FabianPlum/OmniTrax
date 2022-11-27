from pathlib import Path


class YOLOader(object):
    """
    YOLO handler class to manage loading trained YOLO models.
    Provides exception handling for incorrectly specified paths and functions to create placeholder files in case
    .data files are faulty or .names are missing.
    """

    def __init__(self, cfg, weights, data=None, names=None):
        """
        initialise YOLOader with paths to the respective files. Requires only cfg and weights files,
        as data and names can be generated as placeholders.
        :param cfg: path to YOLO config file
        :param weights: path to YOLO trained weights file
        :param data: path to data file
        :param names: path to class names file
        """
        self.cfg = cfg
        self.weights = weights
        if data == "":
            self.data = None
        else:
            self.data = data
        if names == "":
            self.names = None
        else:
            self.names = names

    def create_names(self):
        """
        creates a .names file in the location of the config file with enumerated classes as the names and sets the path
        to the output file as the new .names file.
        """
        with open(self.cfg) as f:
            lines = f.readlines()
            for line in lines:
                if line[:7] == "classes":
                    num_classes = int(line.split("=")[1].split("\n")[0])

        self.names = str(Path(self.cfg).parent.joinpath(str(Path(self.cfg).name)[:-4] + ".names"))

        with open(self.names, 'w', encoding="utf-8") as f:
            for class_name in range(num_classes):
                f.write(str(class_name) + "\n")

        print("Created new .names file for the loaded network.")

    def create_data(self):
        """
        creates a .data file in the location of the config file with the current location of the .names file and sets
        the path to the output file as the new .data file.
        """
        with open(self.cfg) as f:
            lines = f.readlines()
            for line in lines:
                if line[:7] == "classes":
                    num_classes = int(line.split("=")[1].split("\n")[0])

        self.data = str(Path(self.cfg).parent.joinpath(str(Path(self.cfg).name)[:-4] + ".data"))

        with open(self.data, 'w', encoding="utf-8") as f:
            f.write("classes = " + str(num_classes) + "\n")
            f.write("train = data/train.txt\n")
            f.write("test = data/test.txt\n")
            f.write("names = " + str(self.names).replace("\\", "/") + "\n")
            f.write("backup = backup/\n")

        print("Created new .data file for the loaded network.")

    def update_cfg(self, nw_width, nw_height, encoding="utf-8"):
        """
        Updates the cfg file and creates a copy of it in the same directory as the original. The function updates the
        network width and height according to the passed arguments (must be multiples of 32) and ensures the number of
        batches and subdivisions is set to 1. Afterwards, self.cfg is set to the updated file.

        :param nw_width: network width (must be a multiple of 32)
        :param nw_height: network height (must be a multiple of 32)
        """
        # round to the nearest multiple of 32
        nw_width_rounded = int(max(32 * round(nw_width / 32), 32))
        nw_height_rounded = int(max(32 * round(nw_height / 32), 32))

        correct_w, correct_h = False, False
        with open(self.cfg) as f:
            lines = f.readlines()
            for line in lines:
                if line.split("=")[0] == "width":
                    if int(line.split("=")[1].split("\n")[0]) == nw_width_rounded:
                        correct_w = True
                if line.split("=")[0] == "height":
                    if int(line.split("=")[1].split("\n")[0]) == nw_height_rounded:
                        correct_h = True

        if not correct_w and not correct_h:
            self.cfg = str(Path(self.cfg).parent.joinpath(str(Path(self.cfg).name)[:-4] +
                                                          "_width_" + str(nw_width_rounded) +
                                                          "_height_" + str(nw_height_rounded) + ".cfg"))
            with open(self.cfg, 'w', encoding="utf-8") as f:
                for line in lines:
                    if line.split("=")[0] == "batch":
                        f.write("batch=1\n")
                    elif line.split("=")[0] == "subdivisions":
                        f.write("subdivisions=1\n")
                    elif line.split("=")[0] == "width":
                        f.write("width=" + str(nw_width_rounded) + "\n")
                    elif line.split("=")[0] == "height":
                        f.write("height=" + str(nw_height_rounded) + "\n")
                    else:
                        f.write(line)

            print("Updated .cfg file with network width", nw_width_rounded, "and height", nw_height_rounded)


if __name__ == "__main__":
    test_yolo = YOLOader(cfg="C:\\Users\\Legos\\Desktop\\yolov4\\data\\yolov4-big_and_small_ants.cfg",
                         weights="C:\\Users\\Legos\\Desktop\\yolov4\\data\\yolov4-big_and_small_ants_21000.weights")

    test_yolo.update_cfg(nw_width=320, nw_height=320)
    test_yolo.create_names()
    test_yolo.create_data()
