import pickle
from os.path import join, splitext, isfile

mn=3
class imagen:
    def __init__(self,entrada):
        self.deteccion=entrada

    def tobytes(self):
        return self.deteccion



class capturado:
    def __init__(self, entrada):
        self.detecciones = pickle.load(open(splitext(entrada)[0] + '.pkl', 'rb'))


    def read(self):
        if len(self.detecciones):
            return True,imagen(self.detecciones.pop(0))
        else:
            return False,imagen([])

    def release(self):
        return False

    def isOpened(self):
        if len(self.detecciones):
            return True
        else:
            return False

class COLOR_BGR2RGB:
    def __init__(self):
      self.colores=1

class COLOR_RGB2BGR:
    def __init__(self):
      self.colores=1

class INTER_LINEAR:
    def __init__(self):
      self.colores=1




def VideoCapture(entrada):
    return capturado(entrada)

def cvtColor(frame, otro):
    return frame

def resize(frame, otro,interpolation):
    return frame

def imshow(c1,c2):
    return False

def waitKey(numero):
    return False

def destroyAllWindows():
    return False

def circle(image, x, radius, color,thickness):
    return False



