import re

class imagen:
    def __init__(self,entrada):
        self.deteccion=entrada

    def tobytes(self):
        return self.deteccion

def load_network(config_file, data_file, weights, batch_size):
    f = open(config_file, "r")
    s = f.read()
    width=int(re.search("width=\d+",s)[0].replace("width=",""))
    height=int(re.search("height=\d+",s)[0].replace("height=",""))
    network=(width,height)
    f.close()
    return network, width, height

def network_width(network):
    return network[0]

def network_height(network):
    return network[1]

def make_image(width, height,otro):
    return imagen([])


def copy_image_from_bytes(darknetimagen,imagen):
    darknetimagen.deteccion=imagen


def detect_image(network, class_names, darknet_image, thresh):
    return darknet_image.deteccion
