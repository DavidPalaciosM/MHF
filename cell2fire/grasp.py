import numpy as np
import pandas as pd
import random as rd
import os
import math
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import seaborn as sns
import imageio
import csv


def bin_to_nod2(l,j,file): #funcion que pasa una lista de elementos a un archivo .csv que contiene a los cortafuegos
    datos=[np.insert(l,0,1)] #inserto el elemento 1 que corresponde al ano que necesita el archivo 
    if len(l)==0: #si no hay cortafuegos
        cols=['Year Number'] #creo solamente una columna correspondiente al ano
    else: #si hay cortafuegos
        colu=['Year Number',"Cell Numbers"] #creo 2 columnas
        col2=[""]*(len(l)-1) #creo el resto de columnas correspondientes a los otros nodos
        cols=colu+col2 #junto ambas columnas
    df = pd.DataFrame(datos,columns=cols) #creo el dataframe
    df.to_csv(file+"//harvested_member_"+str(j)+".csv",index=False) #guardo el dataframe


def save_harvest_csv(solution,file,i,step): #funcion que genera csvs de una solucion
    path_ori= "../data/"
    flat_solution = [item for sublist in solution for item in sublist] #junto la solucion en una sola lista (la sol. original contiene una lista de listas, en donde cada lista corresponde a un cluster)
    nod=[i for i in flat_solution] #copio los valores de flat_solution
    obj_path=path_ori+file+"//Harvest_iteration"+str(i)+"//Step"+str(step)
    if not os.path.isdir(obj_path):
        os.makedirs(obj_path)
    bin_to_nod2(flat_solution,step,obj_path)



def get_grid_state(file): #funcion que me entrega la cantidad de veces que cada celda se incendio en una serie de simulaciones #NUEVOOOOOOOOOOOO
    path_ori= "../results/"
    path=path_ori+file+'/Grids'
    l=os.listdir(path)
    if l==[]:
        return 0
    i=0
    for grid in l: #recorro cada archivo grid
        path2=path+"/"+grid+"/"+'ForestGrid00.csv' #me fijo solo en el 00 ya que le solicito a la simulacion que me genere solo el finalGrid
        with open(path2) as csv_file:
            exc=csv.reader(csv_file)
            exc=list(exc)
            n=len(exc)
            nodos=n*n
            if i ==0:
                cells=[0]*nodos
                i=1
            flat_solution = [item for sublist in exc for item in sublist]
            flat_solution=[int(i) for i in flat_solution]
            #print(flat_solution)
            cells=[x + y for x, y in zip(cells, flat_solution)] #sumo los grids
    return cells #devuelvo la suma de los grids

def get_saved_grid(file): #funcion que calcula el porcentaje de celdas salvadas
    path_ori= "../results/"
    path=path_ori+file+'/Grids'
    l=os.listdir(path) #obtengo todos los archivos grids de la carpeta (cada grid viene dado por cada simulacion)
    if l==[]:
        return 0
    means=[] #inicializo la lista que contendra los porcentajes de celdas salvadas de cada grid
    for grid in l: #recorro cada archivo grid
        path2=path+"/"+grid+"/"+'ForestGrid00.csv' #me fijo solo en el 00 ya que le solicito a la simulacion que me genere solo el finalGrid
        with open(path2) as csv_file:
            exc=csv.reader(csv_file)
            exc=list(exc)
            n=len(exc)
            nodos=n*n
            non_burned=0
            for i in exc:
                non_burned+=i.count("0")
            percentage_non_burned=non_burned/nodos
            means.append(percentage_non_burned)
    saved=np.mean(means) #obtengo el promedio de todos los procentajes
    return saved #retorno el promedio

def check_touch(nod,not_touch):
    for value in not_touch:
        if value in nod:
            return True
    return False

def fitness(solution,file,knapsack_threshold,nsims): #funcion que recibe una solucion y calcula el fitness de la misma
    path_ori= "../data/"
    path=path_ori+file+'/Data.csv'
    with open(path) as csv_file:
        exc=csv.reader(csv_file)
        exc=list(exc)
        n=len(exc)-1
    cells=math.sqrt(n) #tamano de celdas o filas del bosque
    cells=int(cells) # valor entero del valor anterior
    flat_solution = [item for sublist in solution for item in sublist] #junto la solucion en una sola lista (la sol. original contiene una lista de listas, en donde cada lista corresponde a un cluster)
    nod=[i for i in flat_solution] #copio los valores de flat_solution
    datos=[np.insert(nod,0,1)] #inserto un 1 a estos valores, que corresponde a YEAR NUMBER
    if len(nod)==0: #si no hay cortafuegos
        cols=['Year Number'] #la unica columna del excel es YEAR NUMBER
    else: # si hay cortafuegos
        colu=['Year Number',"Cell Numbers"] #genero las dos columnas del excel
        col2=[""]*(len(nod)-1) #genero los espacios para las otras columnas del excel segun el numero de cortafuegos
        cols=colu+col2 #creo las columnas del excel
    #print(nod)

    df = pd.DataFrame(datos,columns=cols) #genero el dataframe con los cortafuegos
    df.to_csv("harvested_grasp.csv",index=False) #genero y guardo el csv con los cortafuegos de la solucion
    cost = 0 #inicializo el fitness
    #costs=[] #lista de costos por si se genera para mas de un punto de ignicion
    #for i in range(3): #recorro cada punto de ignicion
    state_grid=[]
    #forbidden_list=[3232, 3233, 3234, 3235, 3236, 3237, 3238, 3332, 3333, 3334, 3335, 3336, 3337, 3338, 3432, 3433, 3434, 3435, 3436, 3437, 3438, 3532, 3533, 3534, 3535, 3536, 3537, 3538, 3632, 3633, 3634, 3635, 3636, 3637, 3638, 3732, 3733, 3734, 3735, 3736, 3737, 3738, 3832, 3833, 3834, 3835, 3836, 3837, 3838,3262, 3263, 3264, 3265, 3266, 3267, 3268, 3362, 3363, 3364, 3365, 3366, 3367, 3368, 3462, 3463, 3464, 3465, 3466, 3467, 3468, 3562, 3563, 3564, 3565, 3566, 3567, 3568, 3662, 3663, 3664, 3665, 3666, 3667, 3668, 3762, 3763, 3764, 3765, 3766, 3767, 3768, 3862, 3863, 3864, 3865, 3866, 3867, 3868,6232, 6233, 6234, 6235, 6236, 6237, 6238, 6332, 6333, 6334, 6335, 6336, 6337, 6338, 6432, 6433, 6434, 6435, 6436, 6437, 6438, 6532, 6533, 6534, 6535, 6536, 6537, 6538, 6632, 6633, 6634, 6635, 6636, 6637, 6638, 6732, 6733, 6734, 6735, 6736, 6737, 6738, 6832, 6833, 6834, 6835, 6836, 6837, 6838,6262, 6263, 6264, 6265, 6266, 6267, 6268, 6362, 6363, 6364, 6365, 6366, 6367, 6368, 6462, 6463, 6464, 6465, 6466, 6467, 6468, 6562, 6563, 6564, 6565, 6566, 6567, 6568, 6662, 6663, 6664, 6665, 6666, 6667, 6668, 6762, 6763, 6764, 6765, 6766, 6767, 6768, 6862, 6863, 6864, 6865, 6866, 6867, 6868]
    forbidden_list=[]
    #print(len(nod))
    if len(set(nod))<=knapsack_threshold: #and not check_touch(nod,forbidden_list): #INCLUYE UNA I: #si incluyo menos cortafuegos que el maximo y ninguno de estos cortafuegos es adyacente a la solucion (caso deterministico)
        #creo la linea de simulacion
        path_sim="python3 main.py --input-instance-folder ../data/"+file+"/ --output-folder ../results/"+file+"   --sim-years 1 --nsims "+str(nsims)+" --seed " + str(rd.randint(1,1000000))+ " --finalGrid --weather random --nweathers 7  --Fire-Period-Length 1.0 --ROS-CV 0.0 --HarvestedCells ../cell2fire/harvested_grasp.csv"
        os.system(path_sim) #simulo
        cost=get_saved_grid(file) #obtengo el promedio de celds asalvadas de esta configuracion de cortafuegos
        state_grid=get_grid_state(file)
            #costs.append(get_saved_grid(file)) #obtengo el promedio de celdas salvadas de esta configuracion de cortafuegos para el punto de ignicion dado
        #print(cost) #printeo el valor
        #else:
        #    costs.append(0)
    #if costs==[]: #si ningun caso da un configuracion de cortafuegos viables
    #    costs=0 #el fitness es 0
    #else: #si no
    #    costs=sum(costs)/len(costs) #calculo el promedio de celdas salvadas
    #    print(costs)
    return cost,state_grid #costs #retorno el valor del fitness de la solucion

def n_closest(x,n,d):
    if n[0]-d<0 and n[1]-d<0:
        return x[0:n[0]+d+1,0:n[1]+d+1]
    elif n[0]-d<0:
        return x[0:n[0]+d+1,n[1]-d:n[1]+d+1]
    elif n[1]-d<0:
        return x[n[0]-d:n[0]+d+1,0:n[1]+d+1]
    else:
        return x[n[0]-d:n[0]+d+1,n[1]-d:n[1]+d+1]


def construct_rcl(forest_size,state_grid,rcl_size):
    sqrt_size=int(math.sqrt(forest_size))
    forest=np.arange(1,forest_size+1,1)
    forest=forest.reshape(sqrt_size,sqrt_size)
    rcl=[]
    state_grid=dict(enumerate(state_grid))
    #forbidden_list=[3232, 3233, 3234, 3235, 3236, 3237, 3238, 3332, 3333, 3334, 3335, 3336, 3337, 3338, 3432, 3433, 3434, 3435, 3436, 3437, 3438, 3532, 3533, 3534, 3535, 3536, 3537, 3538, 3632, 3633, 3634, 3635, 3636, 3637, 3638, 3732, 3733, 3734, 3735, 3736, 3737, 3738, 3832, 3833, 3834, 3835, 3836, 3837, 3838,3262, 3263, 3264, 3265, 3266, 3267, 3268, 3362, 3363, 3364, 3365, 3366, 3367, 3368, 3462, 3463, 3464, 3465, 3466, 3467, 3468, 3562, 3563, 3564, 3565, 3566, 3567, 3568, 3662, 3663, 3664, 3665, 3666, 3667, 3668, 3762, 3763, 3764, 3765, 3766, 3767, 3768, 3862, 3863, 3864, 3865, 3866, 3867, 3868,6232, 6233, 6234, 6235, 6236, 6237, 6238, 6332, 6333, 6334, 6335, 6336, 6337, 6338, 6432, 6433, 6434, 6435, 6436, 6437, 6438, 6532, 6533, 6534, 6535, 6536, 6537, 6538, 6632, 6633, 6634, 6635, 6636, 6637, 6638, 6732, 6733, 6734, 6735, 6736, 6737, 6738, 6832, 6833, 6834, 6835, 6836, 6837, 6838,6262, 6263, 6264, 6265, 6266, 6267, 6268, 6362, 6363, 6364, 6365, 6366, 6367, 6368, 6462, 6463, 6464, 6465, 6466, 6467, 6468, 6562, 6563, 6564, 6565, 6566, 6567, 6568, 6662, 6663, 6664, 6665, 6666, 6667, 6668, 6762, 6763, 6764, 6765, 6766, 6767, 6768, 6862, 6863, 6864, 6865, 6866, 6867, 6868]
    #forbidden_list_python=[i-1 for i in forbidden_list]
    forbidden_list_python=[]
    forbidden_list=[]
    for k in forbidden_list_python:
        state_grid.pop(k,None)
    while len(rcl)<rcl_size:
        if state_grid=={}:
            rcl=rd.sample(list(range(1,forest_size)),rcl_size)
            break
        nodo = max(state_grid, key=state_grid.get)
        nodo=nodo+1
        rcl.append(nodo)
        idxs=np.where(forest == nodo)
        idx_x=idxs[0][0]
        idx_y=idxs[1][0]
        radius=3
        adjacents=n_closest(forest,(idx_x,idx_y),radius)
        #print(adjacents)
        flat_list = [item for sublist in adjacents.tolist() for item in sublist]
        flat_list=[i-1 for i in flat_list]
        for k in flat_list:
            if k in state_grid:
                del state_grid[k]

    return rcl



def create_cluster(node,forest_size):
    structures=[1,2,3,4]
    selected=rd.sample(structures,1)[0]
    if selected==1:
        cluster=down_arco(node,forest_size)
    elif selected==2:
        cluster=left_arco(node,forest_size)
    elif selected==3:
        cluster=right_arco(node,forest_size)
    else:
        cluster=up_arco(node,forest_size)
    return cluster


def left_arco(node,forest_size):
    sqrt_size=int(math.sqrt(forest_size))
    cluster=[node]
    parte_vertical_inferior=[node]
    parte_vertical_superior=[node]
    while len(parte_vertical_superior)<=6:
        arriba=parte_vertical_superior[0]-sqrt_size
        if arriba<1:
            break
        else:
            parte_vertical_superior.insert(0,arriba)
    while len(parte_vertical_inferior)<=5:
        abajo=parte_vertical_inferior[len(parte_vertical_inferior)-1] + sqrt_size
        if abajo>forest_size:
            break
        else:
            parte_vertical_inferior.append(abajo)

    parte_horizontal_inferior=[parte_vertical_inferior[len(parte_vertical_inferior)-1]]
    parte_horizontal_superior=[parte_vertical_superior[0]]

    if len(parte_vertical_superior)==7:
        while len(parte_horizontal_superior)<5:
            izquierda=parte_horizontal_superior[0]-1
            if  parte_horizontal_superior[0]%sqrt_size==1:
                break
            else:
                parte_horizontal_superior.insert(0,izquierda)
    if len(parte_vertical_inferior)==6:
        while len(parte_horizontal_inferior)<5:
            izquierda=parte_horizontal_inferior[0]-1
            if parte_horizontal_inferior[0]%sqrt_size==1:
                break
            else:
                parte_horizontal_inferior.insert(0,izquierda)
    cluster=parte_vertical_inferior+parte_vertical_superior+parte_horizontal_superior+parte_horizontal_inferior
    cluster=list(set(cluster))
    return cluster


def right_arco(node,forest_size):
    sqrt_size=int(math.sqrt(forest_size))
    cluster=[node]
    parte_vertical_inferior=[node]
    parte_vertical_superior=[node]
    while len(parte_vertical_superior)<=6:
        arriba=parte_vertical_superior[0]-sqrt_size
        if arriba<1:
            break
        else:
            parte_vertical_superior.insert(0,arriba)
    while len(parte_vertical_inferior)<=5:
        abajo=parte_vertical_inferior[len(parte_vertical_inferior)-1] + sqrt_size
        if abajo>forest_size:
            break
        else:
            parte_vertical_inferior.append(abajo)

    parte_horizontal_inferior=[parte_vertical_inferior[len(parte_vertical_inferior)-1]]
    parte_horizontal_superior=[parte_vertical_superior[0]]

    if len(parte_vertical_superior)==7:
        while len(parte_horizontal_superior)<5:
            derecha=parte_horizontal_superior[0]+1
            if  parte_horizontal_superior[0]%sqrt_size==0:
                break
            else:
                parte_horizontal_superior.insert(0,derecha)
    if len(parte_vertical_inferior)==6:
        while len(parte_horizontal_inferior)<5:
            derecha=parte_horizontal_inferior[0]+1
            if parte_horizontal_inferior[0]%sqrt_size==0:
                break
            else:
                parte_horizontal_inferior.insert(0,derecha)
    cluster=parte_vertical_inferior+parte_vertical_superior+parte_horizontal_superior+parte_horizontal_inferior
    cluster=list(set(cluster))
    return cluster


              
def up_arco(node,forest_size): #funcion que recibe un nodo y el tamano del bosque y crea un cluster a partir del nodo entregado
    sqrt_size=int(math.sqrt(forest_size))
    cluster=[node]
    parte_horizontal_derecha=[node] #largo 5
    parte_horizontal_izquierda=[node] #largo 6
    while len(parte_horizontal_derecha)<=5:
        derecha=parte_horizontal_derecha[len(parte_horizontal_derecha)-1]
        if derecha%sqrt_size==0:
            break
        else:
            parte_horizontal_derecha.append(derecha+1)
    while len(parte_horizontal_izquierda)<=6:
        izquierda=parte_horizontal_izquierda[0]
        if izquierda%sqrt_size==1:
            break
        else:
            parte_horizontal_izquierda.insert(0,izquierda-1)
    parte_horizontal_izquierda.pop()
    parte_horizontal=parte_horizontal_izquierda+parte_horizontal_derecha
    parte_vertical_izquierda=[parte_horizontal[0]]
    parte_vertical_derecha=[parte_horizontal[len(parte_horizontal)-1]]
    if len(parte_horizontal_derecha)==6:
        while len(parte_vertical_derecha)<5:
            arriba=parte_vertical_derecha[0]
            if arriba-sqrt_size<1:
                break
            else:
                parte_vertical_derecha.insert(0,arriba-sqrt_size)
    if len(parte_horizontal_izquierda)==6:
        while len(parte_vertical_izquierda)<5:
            arriba=parte_vertical_izquierda[0]
            if arriba-sqrt_size<1:
                break
            else:
                parte_vertical_izquierda.insert(0,arriba-sqrt_size)
    #print(parte_horizontal_derecha)
    #print(parte_vertical_izquierda)
    cluster=parte_vertical_izquierda+parte_horizontal+parte_vertical_derecha
    cluster=list(set(cluster))
    return cluster


    


def down_arco(node,forest_size): #funcion que recibe un nodo y el tamano del bosque y crea un cluster a partir del nodo entregado
    sqrt_size=int(math.sqrt(forest_size))
    cluster=[node]
    parte_horizontal_derecha=[node] #largo 5
    parte_horizontal_izquierda=[node] #largo 6
    while len(parte_horizontal_derecha)<=5:
        derecha=parte_horizontal_derecha[len(parte_horizontal_derecha)-1]
        if derecha%sqrt_size==0:
            break
        else:
            parte_horizontal_derecha.append(derecha+1)
    while len(parte_horizontal_izquierda)<=6:
        izquierda=parte_horizontal_izquierda[0]
        if izquierda%sqrt_size==1:
            break
        else:
            parte_horizontal_izquierda.insert(0,izquierda-1)
    parte_horizontal_izquierda.pop()
    parte_horizontal=parte_horizontal_izquierda+parte_horizontal_derecha
    parte_vertical_izquierda=[parte_horizontal[0]]
    parte_vertical_derecha=[parte_horizontal[len(parte_horizontal)-1]]
    if len(parte_horizontal_derecha)==6:
        while len(parte_vertical_derecha)<5:
            abajo=parte_vertical_derecha[len(parte_vertical_derecha)-1]
            if abajo+sqrt_size>forest_size:
                break
            else:
                parte_vertical_derecha.append(abajo+sqrt_size)
    if len(parte_horizontal_izquierda)==6:
        while len(parte_vertical_izquierda)<5:
            abajo=parte_vertical_izquierda[len(parte_vertical_izquierda)-1]
            if abajo+sqrt_size>forest_size:
                break
            else:
                parte_vertical_izquierda.append(abajo+sqrt_size)
    #print(parte_horizontal_derecha)
    #print(parte_vertical_izquierda)
    parte_vertical_derecha.pop(0)
    parte_vertical_izquierda.pop(0)
    cluster=parte_vertical_izquierda+parte_horizontal+parte_vertical_derecha
    return cluster #retorno el cluster



def probability_one_two(n):  #crea una lista como la sucesion 1/2 1/4 1/8 1/16... para usar como distribucion de probabilidad
    if n==1:
        return [1]
    value=[]
    for i in range(1,n+1):
        if i+1<=n:
            actual=1/(2**i)
            value.append(actual)
        else:
            value.append(actual)

    return value


def get_move(file,not_touch):
    path_ori= "../results/"
    path=path_ori+file+'/Grids'
    l=os.listdir(path)
    i=0
    for grid in l: #recorro cada archivo grid
        path2=path+"/"+grid+"/"+'ForestGrid00.csv' #me fijo solo en el 00 ya que le solicito a la simulacion que me genere solo el finalGrid
        with open(path2) as csv_file:
            exc=csv.reader(csv_file)
            exc=list(exc)
            n=len(exc)
            nodos=n*n
            if i ==0:
                cells=[0]*nodos
                i=1
            flat_solution = [item for sublist in exc for item in sublist]
            flat_solution=[int(h) for h in flat_solution]

            cells=[x + y for x, y in zip(cells, flat_solution)] #sumo los grids
    
    cells=dict(enumerate(cells)) #paso la suma de los grids a diccionario
    cells2=cells
    for p in not_touch:
        cells.pop(p-1,None)

    cells=sorted(cells, key=cells.get, reverse=True)[:int(nodos*0.01)] #extraigo el 10% que mas se quema y lo identifico por su llave
    cells=[t+1 for t in cells]

    sampleNumbers = np.random.choice(cells, 1, p=probability_one_two(len(cells))) #elijo un nodo de estos
    return sampleNumbers[0]



def move_solution(solution,forest_size,not_touch,file,it): #funcion que recibe una solucion y el tamano del bosque y retorna una solucion modificada al sacar un cluster y entregar otro
    copy_solution=solution[:] #creo una copia de la solucion
    random_item_from_list = rd.choice(copy_solution) #escojo un cluster de la solucion
    copy_solution.remove(random_item_from_list) #extraigo ese cluster
    tof=True #auxiliar
    while tof: #ciclo auxiliar para no seleccionar un nuevo nodo a partir de un nodo que ya se encuentre en la solucion
        p=rd.random() #creo un numero al azar para definir que tipo de vecino escogere
        if p>0.6 or it==1: #en el 40% de los casos o si es la primera iteracion, el movimiento es al azar
            idx=rd.randint(1,forest_size) #escojo un valor al azar del bosque
        else: #en el 60% de los casos el movimiento es hacia zonas que mas se queman
            try: #intento correr la funcion ya que si no hay grids (no se ha simulado aun) la funcion tirara error (OSError)
                idx=get_move(file,not_touch) #obtengo un nodo de los que mas se queman
            except OSError as e: #si hay error
                idx=rd.randint(1,forest_size) #obtengo un indice al azar
        new_cluster=create_cluster(idx,forest_size) #creo un nuevo cluster a partir de este valor
        flat_copy=[item for sublist in copy_solution for item in sublist]
        tof=any(value in flat_copy for value in new_cluster) #chequeo que ese valor no este en la solucion, si esta el ciclo while me hara volver a escojer otro valor
    #new_cluster=create_cluster(idx,forest_size) #creo un nuevo cluster a partir de este valor
    copy_solution.append(new_cluster) #incluyo este cluster a la solucion
    
    return copy_solution,idx #retorno la nueva solucion y el movimiento realizado



def simulate(file,num_iterations, max_runTime,perc_treatment):
    path_ori= "../data/"
    path=path_ori+file+'/Data.csv'
    with open(path) as csv_file:
        exc=csv.reader(csv_file)
        exc=list(exc)
        n=len(exc)-1
    knapsack_threshold=n*perc_treatment #10 #porcentaje de nodos disponibles para ser harvested
    #print(knapsack_threshold)
    rcl_size=int(knapsack_threshold/20*0.2)
    n_local_search=5
    bfit=0
    clusters_at_time=int(knapsack_threshold/20*0.2)
    bx=[]
    bfithistory=[]
    bxhistory=[]
    timeshistory=[]
    start=time.time()   # start the timer
    for i in range(num_iterations):
        solution=[]
        p=rd.random()
        if p>0.6: #40% de los casos
            init_sol=rd.sample(list(range(1,n)),clusters_at_time)
            for nd in init_sol:
                new_cluster=create_cluster(nd,n)
                solution.append(new_cluster)
        fit,grids=fitness(solution,file,knapsack_threshold,3)
        lfit=0
        lx=[]
        #not_touch=[3232, 3233, 3234, 3235, 3236, 3237, 3238, 3332, 3333, 3334, 3335, 3336, 3337, 3338, 3432, 3433, 3434, 3435, 3436, 3437, 3438, 3532, 3533, 3534, 3535, 3536, 3537, 3538, 3632, 3633, 3634, 3635, 3636, 3637, 3638, 3732, 3733, 3734, 3735, 3736, 3737, 3738, 3832, 3833, 3834, 3835, 3836, 3837, 3838,3262, 3263, 3264, 3265, 3266, 3267, 3268, 3362, 3363, 3364, 3365, 3366, 3367, 3368, 3462, 3463, 3464, 3465, 3466, 3467, 3468, 3562, 3563, 3564, 3565, 3566, 3567, 3568, 3662, 3663, 3664, 3665, 3666, 3667, 3668, 3762, 3763, 3764, 3765, 3766, 3767, 3768, 3862, 3863, 3864, 3865, 3866, 3867, 3868,6232, 6233, 6234, 6235, 6236, 6237, 6238, 6332, 6333, 6334, 6335, 6336, 6337, 6338, 6432, 6433, 6434, 6435, 6436, 6437, 6438, 6532, 6533, 6534, 6535, 6536, 6537, 6538, 6632, 6633, 6634, 6635, 6636, 6637, 6638, 6732, 6733, 6734, 6735, 6736, 6737, 6738, 6832, 6833, 6834, 6835, 6836, 6837, 6838,6262, 6263, 6264, 6265, 6266, 6267, 6268, 6362, 6363, 6364, 6365, 6366, 6367, 6368, 6462, 6463, 6464, 6465, 6466, 6467, 6468, 6562, 6563, 6564, 6565, 6566, 6567, 6568, 6662, 6663, 6664, 6665, 6666, 6667, 6668, 6762, 6763, 6764, 6765, 6766, 6767, 6768, 6862, 6863, 6864, 6865, 6866, 6867, 6868]
        not_touch=[]
        sumador=0
        flat_sol=[item for sublist in solution for item in sublist]
        while len(set(flat_sol))<knapsack_threshold-21:
        #while len(solution)<=int(knapsack_threshold)/17:
            rcl=construct_rcl(n,grids,rcl_size)
            count_i=0
            while count_i<clusters_at_time:
                save_harvest_csv(solution,file,i,count_i+sumador*clusters_at_time+sumador) if p<0.6 else print()
                newflat_sol=[item for sublist in solution for item in sublist]
                if len(set(newflat_sol))>knapsack_threshold-20:
                    break
                selected_node=rd.sample(rcl,1)[0]
                new_cluster=create_cluster(selected_node,n)
                tof=True
                for forbid in not_touch:
                    if forbid in new_cluster:
                        tof=False
                if tof:
                    solution.append(new_cluster)
                    count_i=count_i+1
            fit,grids=fitness(solution,file,knapsack_threshold,3)
            flat_sol=[item for sublist in solution for item in sublist]
            save_harvest_csv(solution,file,i,count_i+sumador*clusters_at_time+sumador) if p<0.6 else print()

            sumador=sumador+1

            #if fit>lfit:
            #    lfit=fit
            #    xfit=solution
        xfit=solution
        lfit=fit#1-fit-knapsack_threshold/n #=fit
        print("-------------------------")

        for j in range(n_local_search):
            neighbor,idx_not_touch=move_solution(solution,n,not_touch,file,i) #genero un vecino modificando un cluster (eliminandolo y usando otro)
            not_touch.append(idx_not_touch)
            fit,state_grid=fitness(neighbor,file,knapsack_threshold,5) #calculo el fitness de la solucion
            fit=fit#1-fit-knapsack_threshold/n #=fir
            if fit>lfit:#fit<lfit: #fit>lfit
                lfit=fit
                xfit=neighbor
            save_harvest_csv(solution,file,i,count_i+sumador*clusters_at_time+sumador+1+j) if p<0.6 else print()


        #lfit=fitness(xfit,file,knapsack_threshold,20)[0]
        print(lfit)
        if lfit>bfit:#lfit<bfit: #si el fitness es mejor globalmente #lfit>bfit
            bfit,bx=lfit,xfit #actualizo el optimo local
            bfithistory.append(bfit)
            bxhistory.append(bx)
            timeshistory.append(time.time()-start)
            print("best fit: "+str(bfit)) #printeo el valor de la solucion
        if time.time() >= start+max_runTime:
            break
        print("iteration: "+str(i))
    return bfit,bx,bfithistory,bxhistory,timeshistory,n,i

def simulate_grasp(archivo,ntest,maxTime,perc_treatment):

    Best_FIT,Best_Sol,bfithistory,bxhistory,timeshistory,n,iterations=simulate(archivo,1000,maxTime,perc_treatment)


    #Best_Sol=[i for i in Best_Sol[1]] #extraigo los valores de la mejor solucion
    print(Best_Sol) #printeo nuevamente la mejor solucion
    print(Best_FIT) #printeo nuevamente la mejor solucion

    #print(tot_time)#printeo el tiempo total de ejecucion
    #print(soluciones) #printero la lista de soluciones globales
    #print(desempenos) #printeo la lista de fitnesses globales
    #print(tiempos) #printeo los tiempos

    flat_solution = [item for sublist in Best_Sol for item in sublist] #paso la lista de clusters de la mejor solucion a una sola lista
    #genero archivo con los cortafuegos
    nod=[i for i in flat_solution] #creo una copia de la solucion
    datos=[np.insert(nod,0,1)] #inserto el ano
    if len(nod)==0: #si no hay cortafuegos
        cols=['Year Number'] #solo la columna de year number
    else: #si no
        colu=['Year Number',"Cell Numbers"] 
        col2=[""]*(len(nod)-1)
        cols=colu+col2
        #print(nod)

    df = pd.DataFrame(datos,columns=cols)
    if not os.path.exists("grasp_optimals"):
        os.mkdir("grasp_optimals")
    name="grasp_optimals/harvested_grasp_optimal_" + str(ntest)+".csv"
    #df.to_csv("harvested_ga_optimal.csv",index=False)
    df.to_csv(name,index=False)


     #paso las soluciones a valores binarios
    bin_sols=[]
    for value in bxhistory: #recorro cada solucion
        value = [item for sublist in value for item in sublist] #paso la lista de clusters a una sola lista
        value=[1 if i in value else 0 for i in range(n)] #las paso a valores binarios
        bin_sols.append(value) #anado esto a una lista que contiene las soluciones globales en valores binarios

    path_ori= "../results/"
    path_ori2=path_ori+"grasp"
    path=path_ori2+'/HarvestedPlots_'+str(ntest)
    if not os.path.exists(path_ori2):
        os.mkdir(path_ori2)
    if not os.path.exists(path):
        os.mkdir(path)
    i=1
    images = []
    with open('outfile.txt','w') as f:
        for sol in bin_sols: #
            n=int((len(sol))**(1/2))
            x = [sol[i:i + n] for i in range(0, len(sol), n)] 
            df = pd.DataFrame.from_records(x)
            df[df<1] = 0
            df[df>1] = 0
            to_lst=df.to_numpy().flatten()
            nods=np.where(to_lst==1)
            nods=nods[0]+1
            #f.write("individuo "+str(i)+" : "  + "celdas salvadas: "+ str(final_fit[i-1])+ " %"+ "\n")
            #f.write(np.array2string(nods))
            #f.write("\n")
            plt.clf() 
            sns.set()
            sns_plot=sns.heatmap(df,cmap="BuPu")
    #         #sns_plot.scatter(49, 30 , marker='*', s=150, color='red')
    #         #sns_plot.scatter(49, 50, marker='*', s=150, color='red')
    #         #sns_plot.scatter(49, 80, marker='*', s=150, color='red')
            name=path+"/output"+str(i)+".png"
            sns_plot.set_title('individuo '+str(i))
            sns_plot.figure.savefig(name)
            images.append(imageio.imread(name))
            i+=1
    imageio.mimsave(path+'/history.gif', images,duration = 0.5)

    #graficar historia del mejor
    sin_plot = plt.figure()
    plt.plot(timeshistory, bfithistory)
    plt.title("Evolution Global Optimum GRASP")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Celdas Salvadas [%]")
    plt.savefig(path+'/fitness_history.png')

    pathtxt=path_ori+"grasp"+'/resultados_'+str(ntest)+"txt"
    file1 = open(pathtxt,"w")


    L1=["optimo:\n"]
    L2=[str(Best_FIT) + "\n"]
    L3=["cortafuegos:\n"]
    L4=[str(flat_solution) + "\n"]
    L5=["numero de iteraciones:\n"]
    L6=[str(iterations) +"\n"]
    L7=["historial de fitness:\n"]
    L8=[str(bfithistory)+"\n"]
    L9=["historial de tiempo:\n"]
    L10=[str(timeshistory)+"\n"]

    L = L1+L2+L3+L4+L5+L6+L7+L8+L9+L10

    file1.writelines(L)
    file1.close() #to change file access modes


    tst="python3 main.py --input-instance-folder ../data/"+archivo+"/ --output-folder ../results/grasp_stats_"+str(ntest)+" --sim-years 1 --nsims 5000 --finalGrid --weather random --nweathers 7  --Fire-Period-Length 1.0 --output-messages --ROS-CV 0.0 --seed 123 --stats  --HarvestedCells ../cell2fire/grasp_optimals/harvested_grasp_optimal_" + str(ntest)+".csv"
    os.system(tst)
