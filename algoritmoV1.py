from ctypes import *
import random
import os
import ntpath
import math
import glob
import multiprocessing as mp
import cv2 as cv2
import time
import darknet as darknet
import argparse
from threading import Thread, enumerate
from queue import Queue
import numpy as np
from os.path import join, splitext, isfile
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import linear_sum_assignment
from functools import partial
import pickle


def set_saved_video(input_video, output_video, size):
    import cv2
    # input_video=input_video.replace('.pkl','.avi')
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video




def guardarpkl(inputvideo):
    import pickle
    path_data = os.getcwd()
    carpeta = 'cooke_1211'
    config_file = join(path_data, glob.glob(join(path_data, carpeta + '/*.cfg'))[0])
    data_file = join(path_data, glob.glob(join(path_data, carpeta + '/*.data'))[0])
    weights = join(path_data, glob.glob(join(path_data, carpeta + '/*.weights'))[0])
    network, class_names, class_colors = darknet.load_network(config_file, data_file, weights, batch_size=1)
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)
    for s in inputvideo:
        input_source = join(path_data, s)
        if isfile(input_source):
            print("generando pkl para..."+input_source)
            cap = cv2.VideoCapture(input_source)
            detecciones = [] #[[width, height, config_file, data_file, weights]]
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = cv2.resize(frame_rgb, (416, 416),interpolation=cv2.INTER_LINEAR)
                darknet.copy_image_from_bytes(darknet_image, image.tobytes())
                detections = darknet.detect_image(network, class_names, darknet_image,thresh=0.1)
                detecciones.append(detections)
            pickle.dump(detecciones, open(splitext(input_source)[0] + '.pkl', 'wb'))
            print("pkl generado...")
        else:
            print("error: archivo no existe")




def contar(parametros,input_source):
    carpeta = 'cooke_1211'
    class chancho:
        def __init__(self, x, y, w, h, rut):
            self.x = x
            self.y = y
            self.w = w
            self.h = h
            self.rut = rut
            self.edad = 1
            self.desaparecio = 0
            self.bautizado = False
            self.recorrido = 0
    path_data = os.getcwd()
    # parametros = [0.12258329004931656, 0.13604936188941608, 0.7778348438634688, 0.4065490843381925, 0.40331703629945426,
    #               0.1640518327013193, 0.420739806554848, 0.5245631693900111, 0.21502020872607353, 0.482186332440264,
    #               0.4176353359845454, 0.6843880355248698, 0.5203910748394105, 0.9972326581009332]
    config_file = join(path_data, glob.glob(join(path_data, carpeta + '/*.cfg'))[0])
    data_file = join(path_data, glob.glob(join(path_data, carpeta + '/*.data'))[0])
    weights = join(path_data, glob.glob(join(path_data, carpeta + '/*.weights'))[0])
    network, class_names, class_colors = darknet.load_network(config_file, data_file, weights, batch_size=1)
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)
    thresh = parametros[0]
    maximadistancia = parametros[1] * width
    y2 = parametros[2]  # bautizadps
    y1 = parametros[3]  # bautizadps
    y22 = parametros[4]  # no bautizados
    y11 = parametros[5]  # no bautizadps
    maxdesap1 = parametros[6] * 100
    maxdesap2 = parametros[7] * 100
    edadbautizo = parametros[8] * 200
    recorridobautizo = parametros[9] * width
    recorridodesbautizo = parametros[10] * width
    recorridodedescuento = parametros[11] * width
    recorridoeliminacion = parametros[12] * width
    desaparecidassalida = parametros[13] * 100
    areamin = parametros[14]*width*height
    areamax = parametros[15]*width*height
    chanchos = []
    contador = 0
    cap = cv2.VideoCapture(input_source)
    #out=cv2.VideoWriter("prueba.avi", cv2.VideoWriter_fourcc(*'XVID'), 24.0, (int(cap.get(3)), int(cap.get(4))))
    #out = cv2.VideoWriter('output.mp4', -1, 20.0, (416, 416))
    out = cv2.VideoWriter("output2.avi",
                             cv2.VideoWriter_fourcc(*"MJPG"), 30, (416, 416))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(frame_rgb, (416, 416),
                           interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(darknet_image, image.tobytes())
        detections=darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
        matrix = []
        mostrar = []  # id a mostrar
        funcion1 = lambda xx: abs(xx * (y2 - y1) / 208 + (y1 - xx * (y2 - y1) / 208))  # funcion para bautizados
        funcion2 = lambda xx: abs(xx * (y22 - y11) / 208 + (y11 - xx * (y22 - y11) / 208))  # funcion para no bautizados
        if len(chanchos)>15:
            return 90000000000
        for x in chanchos:
            if (x.bautizado and x.desaparecio * funcion1(x.x) > maxdesap1) or (
                    not x.bautizado and x.desaparecio * funcion2(x.x) > maxdesap2):
                chanchos.remove(x)
            elif x.desaparecio > desaparecidassalida and x.recorrido > recorridoeliminacion:
                chanchos.remove(x)
        if detections:
            matrix2 = []
            for x in detections:
                if x[0] == 'salmon' and float(x[1]) / 100 > thresh and areamin<float(x[2][2]*x[2][3])<areamax:
                    matrix.append([x[2][0], x[2][1],x[2][2], x[2][3]])
            if len(chanchos):
                matrixtracker = []
                for x in chanchos:
                    matrixtracker.append([x.x, x.y, x.w, x.h])
                if len(matrix) > 0:
                    costos = euclidean_distances(matrixtracker, matrix)
                    costos[costos > maximadistancia] = 1000000000000
                    row_ind, col_ind = linear_sum_assignment(costos)
                    row_ind = list(row_ind)
                    col_ind = list(col_ind)
                else:
                    row_ind = []
                    col_ind = []
                mostrar = []
                inditoc = []
                for x in range(0, len(matrix)):  # para cada cerdo detectado actualmente
                    if x in col_ind:  # si se asignó una deteccion a un tracker
                        indi = row_ind[col_ind.index(x)]  # indice del chancho en tracker
                        if costos[indi][x] > maximadistancia:  # condicion para quitar asignamiento
                            chanchos.append(chancho(matrix[x][0], matrix[x][1], matrix[x][2], matrix[x][3], 0))
                            mostrar.append(len(chanchos) - 1)
                        else:
                            chanchos[indi].recorrido += matrix[x][0] - chanchos[indi].x
                            chanchos[indi].x = matrix[x][0]
                            chanchos[indi].y = matrix[x][1]
                            chanchos[indi].w = matrix[x][2]
                            chanchos[indi].h = matrix[x][3]
                            chanchos[indi].edad += 1
                            mostrar.append(indi)
                    else:
                        chanchos.append(chancho(matrix[x][0], matrix[x][1], matrix[x][2], matrix[x][3], 0))
                        mostrar.append(len(chanchos) - 1)
            else:
                mostrar = []
                km = 0
                for x in matrix:
                    chanchos.append(chancho(x[0], x[1], x[2], x[3], 0))
                    mostrar.append(len(chanchos) - 1)
                km += 1
            for x in range(len(chanchos)):  # para cada tracker
                if x in mostrar:  # si el tracker viejo se muestra
                    if chanchos[x].edad > edadbautizo and chanchos[x].recorrido > recorridobautizo and not chanchos[
                        x].bautizado:  # si el chancho llega a cierta edad
                        contador += 1
                        # q.put(1)
                        chanchos[x].rut = contador
                        chanchos[x].bautizado = True
                    chanchos[x].desaparecio = 0
                else:  # si el tracker viejo no se muestra
                    chanchos[x].desaparecio += 1
                if (chanchos[x].recorrido < recorridodesbautizo and chanchos[x].bautizado) or (
                        chanchos[x].recorrido < -recorridodedescuento and not chanchos[x].bautizado):
                    chanchos[x].rut = 0
                    chanchos[x].recorrido = 0
                    contador -= 1
                    # q.put(0)
                    chanchos[x].bautizado = False
            for x in range(0, len(matrix)):  # para cada deteccion actual mostrar algo
                if chanchos[mostrar[x]].rut == 0:
                    image = cv2.circle(image, (int(matrix[x][0]), int(matrix[x][1])), radius=8, color=(57, 255, 20),
                                       thickness=-1)

                else:
                    image = cv2.circle(image, (int(matrix[x][0]), int(matrix[x][1])), radius=8, color=(255, 0, 0),
                                       thickness=-1)
                    image = cv2.putText(image, str(chanchos[mostrar[x]].rut), (int(matrix[x][0]), int(matrix[x][1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                   (255, 0, 0), 2, cv2.LINE_AA)

                # image = cv2.circle(image, (int(matrix[x][0]), int(matrix[x][1])), radius=4, color=(0, 0, 255), thickness=-1)
                # image = cv2.putText(image, valor, (int(matrix[x][0]), int(matrix[x][1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                #                    (0, 0, 255), 2, cv2.LINE_AA)
        image = cv2.putText(image, str(contador), (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        out.write(image)
        #
        #print(contador)
        # with lock:
        #   outputFrame = image.copy()

    cap.release()
    cv2.destroyAllWindows()
    out.release()
    #print(contador)
    return contador

pool=mp.Pool(5)

def algoritmogenetico():
    import pickle
    mejor = []


    def evaluar(individuo,mostrar):
        extensions = '.pkl'
        error = 0
        porpescado = 0
        s = glob.glob(os.getcwd()+'/contados/*/*.pkl')
        cantidad=pool.map(partial(contar,individuo),s)
        #print(cantidad)
        error=sum(list(map(lambda x,y: math.sqrt((x-y)**2),cantidad,[int(x[-7:-4]) for x in s])))
        if mostrar:
            print(list(map(lambda x,y: str(x)+":"+str(y),cantidad,[int(x[-7:-4]) for x in s])))
            print(list(map(lambda x,y: abs(x-y),cantidad,[int(x[-7:-4]) for x in s])))
        return error

    cantidadparametros = 16
    individuos = 20
    path_data = os.getcwd()
    poblaciones = []
    poblacion = []
    carpetaentrenamiento = path_data + '/entrenamiento/'
    for kkk in range(0, 1):
        poblacion = []
        for k in range(0, individuos):
            poblacion.append([random.random() for i in range(0, cantidadparametros)])
        poblaciones.append(poblacion)

    while True:

        for kkk in range(0, 1):
            poblacion = poblaciones[kkk]
            if os.path.isfile('parametros.pkl'):
                f = open('parametros.pkl', 'rb')
                mejor = pickle.load(f)
                f.close()
                for ll in range(0, cantidadparametros - len(mejor)):
                    mejor.append(random.random())
                poblacion[-1] = mejor
            adaptacion = []
            for indiv in poblacion:
                if len(adaptacion) < len(poblacion)-1:
                    adaptacion.append(1 / (evaluar(indiv,False) + 0.00001))
                else:
                    adaptacion.append(1 / (evaluar(indiv, True) + 0.00001))




            dice = adaptacion.index(max(adaptacion))
            errorahora = (1 / adaptacion[dice]) - 0.00001
            print("error actual: ", errorahora)
            if os.path.isfile('errorg.pkl'):
                f = open('errorg.pkl', 'rb')
                error = pickle.load(f)
                f.close()
                print("error global: ", error)
                m = open('errorg.pkl', 'wb')
                pickle.dump(errorahora, m)
                m.close()
                m2 = open('parametros.pkl', 'wb')
                pickle.dump(poblacion[dice], m2)
                m2.close()
            else:
                m = open('errorg.pkl', 'wb')
                pickle.dump(errorahora, m)
                m.close()
                m2 = open('parametros.pkl', 'wb')
                pickle.dump(poblacion[dice], m2)
                m2.close()
            nuevapoblacion = poblacion.copy()
            for kl in range(0, individuos - 1):
                cruzarhijos = random.choices(range(0, individuos), adaptacion, k=2)
                for mm in range(0, cantidadparametros):
                    nuevapoblacion[kl][mm] = poblacion[random.choice(cruzarhijos)][mm]
                    if random.random() < 0.1:
                        nuevapoblacion[kl][mm] = random.random()
            nuevapoblacion[-1] = poblacion[dice]
            poblaciones[kkk] = nuevapoblacion.copy()
            # poblacion=nuevapoblacion.copy()



#algoritmogenetico()
#s = glob.glob(os.getcwd()+'/contados/*/*.mp4')
#guardarpkl(s)
parametros=[0.6707239020507427, 0.24920841953842188, 0.6690455158133256, 0.30924822675922004, 0.4537408544607523, 0.048285299766918754, 0.6111454676469459, 0.28447953847666807, 0.014618176015858797, 0.10956187335244405, 0.04344376803173755, 0.7884955310438393, 0.003626018564747868, 0.8326239606204782, 0.022820082089734628, 0.6382815178249167]
#[0.2745502279799066, 0.28808338185508775, 0.3556421604771698, 0.4517778528270111, 0.185497635697233, 0.08318465553892151, 0.6519535309400868, 0.5779365745141609, 0.028141320444893925, 0.21949240887122778, 0.06561427995372227, 0.5853292657401884, 0.37534627815121513, 0.6976157244141202, 0.01091919739780478, 0.623923135037621]

# algoritmogenetico()
s=glob.glob('/home/jesus/pruebas/nuevo/contados/*/*155.mp4')
#s'/home/jesus/pruebas/nuevo/contados/*/'
# parametros=pickle.load(open('parametros.pkl','rb'))
contar(parametros,s[0])
# procesos=[]
# for km in range(9,26):
#     procesos.append(mp.Process(target=algoritmogenetico,args=(km, )))
#     procesos[-1].start()
#
# for km in procesos:
#     km.join()

#input_source = join(os.getcwd(), "04_004.mp4")
#contar(input_source)

# import pickle
# s=glob.glob('/home/jesus/pruebas/nuevo/contados/*/Largo*.mp4')
# parametros=pickle.load(open('parametros.pkl','rb'))
# contar(parametros,s[0])
# guardarpkl(s)
#algoritmogenetico()
# pool=mp.Pool(4)
# parametros = [0.12258329004931656, 0.13604936188941608, 0.7778348438634688, 0.4065490843381925, 0.40331703629945426,
#                   0.1640518327013193, 0.420739806554848, 0.5245631693900111, 0.21502020872607353, 0.482186332440264,
#                   0.4176353359845454, 0.6843880355248698, 0.5203910748394105, 0.9972326581009332, 0.1, 0.9]
# s = glob.glob('/home/jesus/pruebas/nuevo/contados/*/*.pkl')
# cantidad=pool.map(partial(contar,parametros),s)
# print(cantidad)