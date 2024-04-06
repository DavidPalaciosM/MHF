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


def bin_to_nod(l): #funcion que pasa una lista de elementos a un archivo .csv que contiene a los cortafuegos
    datos=[np.insert(l,0,1)] #inserto el elemento 1 que corresponde al ano que necesita el archivo 
    if len(l)==0: #si no hay cortafuegos
        cols=['Year Number'] #creo solamente una columna correspondiente al ano
    else: #si hay cortafuegos
        colu=['Year Number',"Cell Numbers"] #creo 2 columnas
        col2=[""]*(len(l)-1) #creo el resto de columnas correspondientes a los otros nodos
        cols=colu+col2 #junto ambas columnas
    df = pd.DataFrame(datos,columns=cols) #creo el dataframe
    df.to_csv("harvested_ga.csv",index=False) #guardo el dataframe


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


def adjacent_nodes(node,forest_size):
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

#print(construct_solution(population,400,10,4))

def initial_generation(file,percent): #funcion que crea la primera generacion y entrega los parametros iniciales del algoritmo y el bosque
    path_ori= "../data/"
    path=path_ori+file+'/Data.csv'
    with open(path) as csv_file:
        exc=csv.reader(csv_file)
        exc=list(exc)
        n=len(exc)-1
    knapsack_threshold = n*percent #maximo numero de cortafuegos a ubicar
    solutions_per_pop = 4 #numero de individuos en cada generacion
    clusters=int(knapsack_threshold/20)#+1#numero de clusters a tener, cada uno sera en promedio de tamano 4, por eso escojemos knapsack_threshoold/4
    num_generations = 100 #100  # 500 #numero de generaciones
    individuals = [] #creo matriz que contendra los individuos de mi poblacion
    for i in range(solutions_per_pop): #recorro cada individuo
        individual={} #inicializo el individuo como un diccionario
        for j in range(1,clusters+1): #recorro cada cluster del individuo
            previous_locations=list(individual.values()) #miro los otros cortafuegos del individuo
            location=rd.sample([x for x in range(1,n+1) if x not in previous_locations],1)[0] #escojo un elemento al azar del bosque que no este ya en el individuo
            individual[j]=adjacent_nodes(location,n) #creo un cluster alrededor de la posicion escogida
        individuals.append(individual) #anado el individuo creado a la poblacion
    initial_population=individuals #guardo la variable
    print(initial_population)
    #initial_population[0]={1: [705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 805, 816, 905, 916, 1005, 1016, 1105, 1116], 2: [725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 825, 836, 925, 936, 1025, 1036, 1125, 1136], 3: [745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 845, 856, 945, 956, 1045, 1056, 1145, 1156], 4: [765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 865, 876, 965, 976, 1065, 1076, 1165, 1176], 5: [785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 885, 896, 985, 996, 1085, 1096, 1185, 1196], 6: [2701, 2702, 2703, 2704, 2705, 2706, 2806, 2906, 3006, 3106], 7: [2715, 2716, 2717, 2718, 2719, 2720, 2721, 2722, 2723, 2724, 2725, 2726, 2815, 2826, 2915, 2926, 3015, 3026, 3115, 3126], 8: [2735, 2736, 2737, 2738, 2739, 2740, 2741, 2742, 2743, 2744, 2745, 2746, 2835, 2846, 2935, 2946, 3035, 3046, 3135, 3146], 9: [2755, 2756, 2757, 2758, 2759, 2760, 2761, 2762, 2763, 2764, 2765, 2766, 2855, 2866, 2955, 2966, 3055, 3066, 3155, 3166], 10: [2775, 2776, 2777, 2778, 2779, 2780, 2781, 2782, 2783, 2784, 2785, 2786, 2875, 2886, 2975, 2986, 3075, 3086, 3175, 3186], 11: [2795, 2796, 2797, 2798, 2799, 2800, 2895, 2995, 3095, 3195], 12: [4705, 4706, 4707, 4708, 4709, 4710, 4711, 4712, 4713, 4714, 4715, 4716, 4805, 4816, 4905, 4916, 5005, 5016, 5105, 5116], 13: [4725, 4726, 4727, 4728, 4729, 4730, 4731, 4732, 4733, 4734, 4735, 4736, 4825, 4836, 4925, 4936, 5025, 5036, 5125, 5136], 14: [4745, 4746, 4747, 4748, 4749, 4750, 4751, 4752, 4753, 4754, 4755, 4756, 4845, 4856, 4945, 4956, 5045, 5056, 5145, 5156], 15: [4765, 4766, 4767, 4768, 4769, 4770, 4771, 4772, 4773, 4774, 4775, 4776, 4865, 4876, 4965, 4976, 5065, 5076, 5165, 5176], 16: [4785, 4786, 4787, 4788, 4789, 4790, 4791, 4792, 4793, 4794, 4795, 4796, 4885, 4896, 4985, 4996, 5085, 5096, 5185, 5196], 17: [6701, 6702, 6703, 6704, 6705, 6706, 6806, 6906, 7006, 7106], 18: [6715, 6716, 6717, 6718, 6719, 6720, 6721, 6722, 6723, 6724, 6725, 6726, 6815, 6826, 6915, 6926, 7015, 7026, 7115, 7126], 19: [6735, 6736, 6737, 6738, 6739, 6740, 6741, 6742, 6743, 6744, 6745, 6746, 6835, 6846, 6935, 6946, 7035, 7046, 7135, 7146], 20: [6755, 6756, 6757, 6758, 6759, 6760, 6761, 6762, 6763, 6764, 6765, 6766, 6855, 6866, 6955, 6966, 7055, 7066, 7155, 7166], 21: [6775, 6776, 6777, 6778, 6779, 6780, 6781, 6782, 6783, 6784, 6785, 6786, 6875, 6886, 6975, 6986, 7075, 7086, 7175, 7186], 22: [8705, 8706, 8707, 8708, 8709, 8710, 8711, 8712, 8713, 8714, 8715, 8716, 8805, 8816, 8905, 8916, 9005, 9016, 9105, 9116], 23: [8725, 8726, 8727, 8728, 8729, 8730, 8731, 8732, 8733, 8734, 8735, 8736, 8825, 8836, 8925, 8936, 9025, 9036, 9125, 9136], 24: [8745, 8746, 8747, 8748, 8749, 8750, 8751, 8752, 8753, 8754, 8755, 8756, 8845, 8856, 8945, 8956, 9045, 9056, 9145, 9156], 25: [8765, 8766, 8767, 8768, 8769, 8770, 8771, 8772, 8773, 8774, 8775, 8776, 8865, 8876, 8965, 8976, 9065, 9076, 9165, 9176], 26: [8785, 8786, 8787, 8788, 8789, 8790, 8791, 8792, 8793, 8794, 8795, 8796, 8885, 8896, 8985, 8996, 9085, 9096, 9185, 9196]}

    return initial_population,solutions_per_pop,num_generations, knapsack_threshold,n #retorno la poblacion inicial, el numero de individuos, el numero de generaciones, el numero maximo de cortafuegos a ubicar y el tamano del bosque


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

def check_touch(nod,not_touch):
    for value in not_touch:
        if value in nod:
            return True
    return False


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


def save_harvest_csv(members,file,i,step): #funcion que genera csvs de una poblacion
    path_ori= "../data/"
    path=path_ori+file+'/Data.csv'
    with open(path) as csv_file:
        exc=csv.reader(csv_file)
        exc=list(exc)
        n=len(exc)-1
    cells=math.sqrt(n) 
    cells=int(cells) 
    for j in range(len(members)): 
        elements=list(members[j].values()) 
        elements=sum(elements, [])
        obj_path=path_ori+file+"//Harvest_Generation"+str(i)+"//Step"+str(step)
        if not os.path.isdir(obj_path):
	        os.makedirs(obj_path)
        bin_to_nod2(elements,j,obj_path)


            

def cal_fitness(population, threshold,file,nsims): #funcion que recibe una generacion entera y calcula el fitness de cada individuo
    path_ori= "../data/"
    path=path_ori+file+'/Data.csv'
    with open(path) as csv_file:
        exc=csv.reader(csv_file)
        exc=list(exc)
        n=len(exc)-1
    cells=math.sqrt(n) #obtengo el tamano de una fila o columna del bosque
    cells=int(cells) #obtengo la parte entera del valor anterior
    fitness = np.empty(len(population)) #defino la matriz de fitnesses como una matriz de 0
    actual_state=[0]*n #defino la matriz que guardara los grids del estado final de la poblacion
    for i in range(len(population)): #recorremos cada individuo de la poblacion
        elements=list(population[i].values()) #obtenemos los cortafuegos de cada individuo (quitando las keys del diccionario)
        elements=sum(elements, []) #juntamos los cortafuegos en una sola lista
        S2 = len(elements) #calculamos el numero de cortafuegos de la solucion
        print(S2) #printeamos el numero de cortafuegos
        #print(population[i]) #printeamos al individuo
        #fits=[] #lista que guardara los fits (caso en que hay mas de 1 pto de ignicion dado x usuario)
        #for j in range(3): #recorro cada punto de ignicion
        #not_touch=[3232, 3233, 3234, 3235, 3236, 3237, 3238, 3332, 3333, 3334, 3335, 3336, 3337, 3338, 3432, 3433, 3434, 3435, 3436, 3437, 3438, 3532, 3533, 3534, 3535, 3536, 3537, 3538, 3632, 3633, 3634, 3635, 3636, 3637, 3638, 3732, 3733, 3734, 3735, 3736, 3737, 3738, 3832, 3833, 3834, 3835, 3836, 3837, 3838,3262, 3263, 3264, 3265, 3266, 3267, 3268, 3362, 3363, 3364, 3365, 3366, 3367, 3368, 3462, 3463, 3464, 3465, 3466, 3467, 3468, 3562, 3563, 3564, 3565, 3566, 3567, 3568, 3662, 3663, 3664, 3665, 3666, 3667, 3668, 3762, 3763, 3764, 3765, 3766, 3767, 3768, 3862, 3863, 3864, 3865, 3866, 3867, 3868,6232, 6233, 6234, 6235, 6236, 6237, 6238, 6332, 6333, 6334, 6335, 6336, 6337, 6338, 6432, 6433, 6434, 6435, 6436, 6437, 6438, 6532, 6533, 6534, 6535, 6536, 6537, 6538, 6632, 6633, 6634, 6635, 6636, 6637, 6638, 6732, 6733, 6734, 6735, 6736, 6737, 6738, 6832, 6833, 6834, 6835, 6836, 6837, 6838,6262, 6263, 6264, 6265, 6266, 6267, 6268, 6362, 6363, 6364, 6365, 6366, 6367, 6368, 6462, 6463, 6464, 6465, 6466, 6467, 6468, 6562, 6563, 6564, 6565, 6566, 6567, 6568, 6662, 6663, 6664, 6665, 6666, 6667, 6668, 6762, 6763, 6764, 6765, 6766, 6767, 6768, 6862, 6863, 6864, 6865, 6866, 6867, 6868]
        not_touch=[]
        if S2<=threshold and S2>0 and not check_touch(elements,not_touch): #and S2>=threshold-5: #si los cortafuegos utilizados son menor al meximo numero de cortafuegos y no hay un cortafuego adjacente al pto de ignicion
            bin_to_nod(elements)      #paso la solucion actual a un archivo harvested como cortafuego
            path_sim="python3 main.py --input-instance-folder ../data/"+file+"/ --output-folder ../results/"+file+"  --sim-years 1 --nsims "+str(nsims)+" --seed " + str(rd.randint(1,1000000))+ " --finalGrid --weather random --nweathers 7 --Fire-Period-Length 1.0 --ROS-CV 0.0 --HarvestedCells ../cell2fire/harvested_ga.csv"
            #creo la linea de simulacion
            os.system(path_sim)
            #simulo
            S1=get_saved_grid(file) #calculo el porcentaje de celdas salvadas
            state_grid=get_grid_state(file) #NUEVOOOOOOOOOOOO #obtengo la cantidad de veces que se quemo cada celda
            actual_state=cells=[x + y for x, y in zip(actual_state,state_grid)] #sumo la cantidad acumulada de veces que se quemo cada celda a lo largo de la generacion
            print(S1) #printeo el porcentaje de celdas salvadas
                #fits.append(S1)
            fitness[i]=S1  #guardo el fitness del individuo
            #else :                #sino
                #fitness[i] = get_saved_orig(file)   #se salvan 0 (solucion infactible)
                #fits.append(0)
        else: #si hay mas cortafuegos que los permitidos o la solucion incluye un punto de ignicion
            fitness[i]=0 #el fitness es 0
        #if len(fits)==0:
        #    fitness[i]=0
        #else:
            #fitness[i]=sum(fits)/len(fits)

    return fitness.astype(float),actual_state #STATE_GRIDNUEVOOOOOOOO #retorno el fitness de la generacion y la cantidad de veces que se quemo cada celda del bosque

def selection(fitness, num_parents, population): #funcion que recibe el fitness de la generacion, el numero de padres y la generacion y selecciona a los mejores individuos como padres
    fitness = np.array(fitness) #transformo la matriz de fitnesses a un array
    parents = [] #creo un array vacio que contendra a los padres
    idxs= (-fitness).argsort()[:num_parents] #escojo a los n mejores padres, con n el numero de padres a seleccionar
    for idx in range(len(population)): #recorro cada elemento de la poblacion
        if idx in idxs: #si el indice es un padre
            parents.append(population[idx]) #anado el individuo correspondiente a la lista de padres
    return parents #retorno los padres 

def crossover(parents, num_offsprings,population): #funcion que recibe a los padres, el numero de hijos y la poblacion para retornar a los hijos como combinaciones de padres
    offsprings = [] #creo una matriz vacia que contendra a los hijos
    i=0 #seteo un indice auxiliar en 0
    while (i < num_offsprings): #mientras no he terminado de agregar hijos permitidos
        parents_indexs=list(range(0,len(parents))) #creo una lista con numeros desde el 0 al numero de padres
        parent1_index=rd.choice(parents_indexs) #agarro un indice de un padre al azar:padre 1
        parents_indexs.remove(parent1_index) #borro ese indice de la lista creada
        parent2_index=rd.choice(parents_indexs) #agarro otro indice: padre 2
        num_cluster=len(parents[parent1_index].keys()) #obtengo el numero de clusters de los individuos
        offs={} #creo un diccionario con los hijos
        for j in range(1,num_cluster+1): #recorro cada cluster
            p=rd.random() #lanzo una moneda
            if p>0.5: #si es cara
                of=parents[parent1_index][j] #escojo el cluster j del padre 1 para el hijo i
                if len(offsprings)>0:
                    if any(value in sum(list(offsprings[i-1].values()),[]) for value in of):
                        of=parents[parent2_index][j]
            else: #si es sello
                of=parents[parent2_index][j] #escojo el cluster j del padre 2 para el hijo i
                if len(offsprings)>0:
                    if any(value in sum(list(offsprings[i-1].values()),[]) for value in of):
                        of=parents[parent1_index][j]
            offs[j]=of #anado el cluster seleccionado a la lista de clusters del hijo i
        offsprings.append(offs) #anado el hijo creado a la lista de hijos
        i+=1 #sumo 1 al indice auxiliar
    return offsprings #retorno los hijos creados


# def crossover2(parents, num_offsprings,population,threshold,file): #funcion que recibe a los padres, el numero de hijos y la poblacion para retornar a los hijos como combinaciones de padres
# 	offsprings = [] #creo una matriz vacia que contendra a los hijos
# 	suboffsprings=[]
# 	fits=[]
# 	i=0 #seteo un indice auxiliar en 0
# 	while (i < num_offsprings): #mientras no he terminado de agregar hijos permitidos
# 		for h in range(10):
# 			parents_indexs=list(range(0,len(parents))) #creo una lista con numeros desde el 0 al numero de padres
# 			parent1_index=rd.choice(parents_indexs) #agarro un indice de un padre al azar:padre 1
# 			parents_indexs.remove(parent1_index) #borro ese indice de la lista creada
# 			parent2_index=rd.choice(parents_indexs) #agarro otro indice: padre 2
# 			num_cluster=len(parents[parent1_index].keys()) #obtengo el numero de clusters de los individuos
# 			offs={} #creo un diccionario con los hijos
# 			for j in range(1,num_cluster+1): #recorro cada cluster
# 				p=rd.random() #lanzo una moneda
# 				if p>0.5: #si es cara
# 					of=parents[parent1_index][j] #escojo el cluster j del padre 1 para el hijo i
# 					if len(offsprings)>0:
# 						if any(value in sum(list(offsprings[i-1].values()),[]) for value in of):
# 							of=parents[parent2_index][j]
# 				else: #si es sello
# 					of=parents[parent2_index][j] #escojo el cluster j del padre 2 para el hijo i
# 					if len(offsprings)>0:
# 						if any(value in sum(list(offsprings[i-1].values()),[]) for value in of):
# 							of=parents[parent1_index][j]
# 				offs[j]=of #anado el cluster seleccionado a la lista de clusters del hijo i
# 			suboffsprings.append(offs) #anado el hijo creado a la lista de hijos
# 		fits,grid=cal_fitness(suboffsprings,threshold,file,3)
# 		max_fit=np.where(fits == np.max(fits))
# 		offsprings.append(suboffsprings[max_fit[0][0]])
# 		i+=1 #sumo 1 al indice auxiliar
# 	return offsprings #retorno los hijos creados

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

def get_move(not_touch,actual_grid,forest_size): #funcion que toma el grid final de una generacion, el tamano del bosque y una lista de nodos prohibidos para entregar uno de los nodos de los que mas se quemaron
    actual_grid=dict(enumerate(actual_grid)) #paso la suma de los grids a diccionario
    for i in not_touch: #recorro cada elemento de la lista prohibida
        actual_grid.pop(i-1,None) #elemino del bosque el nodo  de la lista prohibida
    actual_grid=sorted(actual_grid, key=actual_grid.get, reverse=True)[:int(forest_size*0.01)] #extraigo el 10% que mas se quema y lo identifico por su llave
    actual_grid=[i+1 for i in actual_grid] #sumo 1 a la llave ya que el diccionario lo enumera desde el 0
    sampleNumbers = np.random.choice(actual_grid, 1, p=probability_one_two(len(actual_grid))) #elijo un nodo de estos al azar segun una distribucion de probabilidad
    return sampleNumbers[0] #retorno el elemento

def mutation(offsprings,state_grid,forest_size): #funcion que toma a los hijos, el estado final del bosque a lo largo de la generacion y el tamano del bosque y retorna a los hijos mutados 
    mutation_rate = 0.6 #defino la tasa de mutacion
    #not_touch=[3232, 3233, 3234, 3235, 3236, 3237, 3238, 3332, 3333, 3334, 3335, 3336, 3337, 3338, 3432, 3433, 3434, 3435, 3436, 3437, 3438, 3532, 3533, 3534, 3535, 3536, 3537, 3538, 3632, 3633, 3634, 3635, 3636, 3637, 3638, 3732, 3733, 3734, 3735, 3736, 3737, 3738, 3832, 3833, 3834, 3835, 3836, 3837, 3838,3262, 3263, 3264, 3265, 3266, 3267, 3268, 3362, 3363, 3364, 3365, 3366, 3367, 3368, 3462, 3463, 3464, 3465, 3466, 3467, 3468, 3562, 3563, 3564, 3565, 3566, 3567, 3568, 3662, 3663, 3664, 3665, 3666, 3667, 3668, 3762, 3763, 3764, 3765, 3766, 3767, 3768, 3862, 3863, 3864, 3865, 3866, 3867, 3868,6232, 6233, 6234, 6235, 6236, 6237, 6238, 6332, 6333, 6334, 6335, 6336, 6337, 6338, 6432, 6433, 6434, 6435, 6436, 6437, 6438, 6532, 6533, 6534, 6535, 6536, 6537, 6538, 6632, 6633, 6634, 6635, 6636, 6637, 6638, 6732, 6733, 6734, 6735, 6736, 6737, 6738, 6832, 6833, 6834, 6835, 6836, 6837, 6838,6262, 6263, 6264, 6265, 6266, 6267, 6268, 6362, 6363, 6364, 6365, 6366, 6367, 6368, 6462, 6463, 6464, 6465, 6466, 6467, 6468, 6562, 6563, 6564, 6565, 6566, 6567, 6568, 6662, 6663, 6664, 6665, 6666, 6667, 6668, 6762, 6763, 6764, 6765, 6766, 6767, 6768, 6862, 6863, 6864, 6865, 6866, 6867, 6868] #inicializo la lista con nodos prohibidos para construir en torno a ellos
    not_touch=[]
    for i in range(len(offsprings)): #recorro cada hijo
        random_value = rd.random() #creo un numero entre 0 y 1 al azar
        if random_value <= mutation_rate:  #si el numero entre 0 y 1 es menor que la tasa de mutacion mutare al hijo, sino se mantiene igual
            tof=True #auxiliar
            while tof: #ciclo auxiliar para no seleccionar un nuevo nodo a partir de un nodo que ya se encuentre en la solucion
                p=rd.random() #creo un numero al azar para definir que tipo de vecino escogere
                if p>0.6: #en el 40% de los casos el movimiento es al azar
                    idx=rd.randint(1,forest_size) #escojo un valor al azar del bosque
                else: #en el 60% de los casos el movimiento es hacia zonas que mas se queman
                    try: #intento correr la funcion ya que si no hay grids (no se ha simulado nunca) la funcion tirara error (OSError)
                        idx=get_move(not_touch,state_grid,forest_size) #obtengo un nodo de los que mas se queman
                    except OSError as e: #si hay error
                        idx=rd.randint(1,forest_size) #obtengo un indice al azar
                new_solution=adjacent_nodes(idx,forest_size) #creo un nuevo cluster alrededor del nodo seleccionado
                tof=any(value in sum(list(offsprings[i].values()),[]) for value in new_solution) #chequeo si el indice ya esta en algun cortafuego del individuo, si esta, el ciclo while me obligara a buscar otro indice
                #print(offsprings[i].values())
            not_touch.append(idx) #anado el movimiento que acabo de hacer como un nodo prohibido a visitar nuevamente en el resto de los hijos de esta generacion
            #new_solution=adjacent_nodes(idx,forest_size) #creo un nuevo cluster alrededor del nodo seleccionado
            mut=list(offsprings[i].keys()) #obtengo las llaves que hacen referencia a cada cluster del individuo
            mut=rd.sample(mut,1)[0] #escojo al azar algun cluster del individuo
            offsprings[i][mut]=new_solution #reemplazo el cluster seleccionado por el nuevo cluster creado
    return offsprings #retorno a los hijos mutados y a los no mutados


def solution_to_bin(solution,forest_size): #funcion que traspasa una solucion a una solucion de un array binario
    elements=list(solution.values()) #obtengo los cortafuegos del individuo (quito las llaves del diccionario)
    elements=sum(elements, []) #junto los cortafuegos en una sola lista
    sol=[1 if i+1 in elements else 0 for i in range(forest_size)] #transformo a binario los indices, chequeo si el indice+1 (ya que el indice del range va entre 0 y el tamano del bosque-1) esta en la lista de soluciones (que van del 1 al tamano del bosque)
    return sol #retorno la solucion modificada

def optimize( population, solutions_per_pop,forest_size, num_generations, threshold,file,maxTime): #funcion que ejecuta el algoritmo genetico
    parameters, fitness_history,times,fitness_history_mean,individuals = [], [], [], [],[] #inicializo los parametros optimos, el historial del mejor fitness de cada generacion, los tiempos, el historial de los promedios del fitness, y los mejores individuos de cada generacion
    num_parents = int(solutions_per_pop/2) #calculo el numero de padres como el la cantidad de individuos de una poblacion/2
    num_offsprings = solutions_per_pop - num_parents #calculo el numero de hijos como el numero de individuos de una generacion menos el numero de padres de una generacion
    initial_time=time.time() #obtengo el tiempo inicial
    for i in range(num_generations): #recorro cada generacion
        fitness,state_grid = cal_fitness(population, threshold,file,10) #calculo el fitness de la poblacion y obtengo el estado de las celdas a lo largo de la generacion
        fitness_history.append(max(fitness)) #anado el mejor fitness al historial de mejores fitnesses
        npfit=np.array(fitness) #traspaso el fitness a un array para calcular su promedio a lo largo de la generacion
        fitness_history_mean.append(npfit[np.nonzero(npfit)].mean()) #calculo el promedio del fitness de la generacion y lo anado a la lista del historial de promedio de fitnesses
        max_fitness_individual = np.where(fitness == np.max(fitness)) #obtengo el indice del mejor individuo de la poblacion
        best_individual=population[max_fitness_individual[0][0]] #obtengo al mejor individuo de la poblacion
        individuals.append(solution_to_bin(best_individual,forest_size)) #traspaso a binario al mejor individuo de la poblacion y lo anado a la lista con mejores individuos
        actual=time.time()-initial_time #actualizo el tiempo
        times.append(actual) #anado el tiempo a la lista de tiempos
        parents= selection(fitness, num_parents, population) #obtengo a los padres
        #save_harvest_csv(parents,file,i,0)
        offsprings = crossover(parents, num_offsprings,population) #obtengo a los hijos 
        #save_harvest_csv(offsprings,file,i,1)
        #offsprings = crossover2(parents, num_offsprings,population,threshold,file) #obtengo a los hijos 
        mutants = mutation(offsprings,state_grid,forest_size) #obtengo a los mutantes
        #save_harvest_csv(offsprings,file,i,2)
        population[0:num_parents] = parents #modifico la poblacion como un porcentaje correspondiente a los padres
        population[num_parents:] = mutants #modifico la poblacion como un porcentaje correspondiente a los hijos
        #save_harvest_csv(population,file,i,3)
        if actual>maxTime: #si me demoro mas de cierto tiempo y aun no paso de cierta generacion
            break #me detengo
        #print("------------------------------")
        #print(parents)
        #print("------------------------------")
        #print(offsprings)
        print("Fin de la generacion" + str(fitness)) #printeo el fin de una generacion
    fitness_last_gen,state_grid = cal_fitness(population, threshold,file,40) #obtengo el fitness de la poblacion final 
    max_fitness = np.where(fitness_last_gen == np.max(fitness_last_gen)) #obtengo el indice del mejor individuo de la poblacion final
    parameters.append(population[max_fitness[0][0]]) #guardo al mejor individuo de la poblacion final
    bin_population=[] #creo un array vacio que guarda a los individuos de la ultima generacion como binarios
    for i in population: #recorro cada individuo de la generacion final
        bin_sol=solution_to_bin(i,forest_size) #paso a binario el individuo
        bin_population.append(bin_sol) #anado al individuo a la lista


    return parameters, fitness_last_gen, bin_population,fitness_history,times,fitness_history_mean,individuals

def simulate_ga(archivo,ntest,treatment_perc,maxTime):

    #20x20 64.19 s esalvan
#archivo="Exp100x100_hetero_C1"#"Exp100x100_hetero_C1" #archivo de bosque a simular
    initial_population,solutions_per_pop,num_generations, knapsack_threshold,n=initial_generation(archivo,treatment_perc)
    parameters, fit_last_gen,bin_population,fitness_history,times,fitness_history_mean,individuals = optimize(initial_population, solutions_per_pop,n, num_generations,knapsack_threshold,archivo,maxTime)
    print("------------------------------")
    print(parameters)
    print("------------------------------")
    print(fit_last_gen)
    print("------------------------------")
#print(last_gen)
#print("------------------------------")
    print(fitness_history)
    print("------------------------------")
    print(times)
    print("------------------------------")
    print(fitness_history_mean)
#print("------------------------------")
#print(individuals)
    parameters=sum(list(parameters[0].values()),[])
#flat_solution = [item for sublist in parameters for item in sublist] #paso la lista de clusters de la mejor solucion a una sola lista
#genero archivo con los cortafuegos
    nod=[i for i in parameters] #creo una copia de la solucion
    datos=[np.insert(nod,0,1)] #inserto el ano
    if len(nod)==0: #si no hay cortafuegos
        cols=['Year Number'] #solo la columna de year number
    else: #si no
        colu=['Year Number',"Cell Numbers"] 
        col2=[""]*(len(nod)-1)
        cols=colu+col2
    #print(nod)

    df = pd.DataFrame(datos,columns=cols)
    if not os.path.exists("ga_optimals"):
        os.mkdir("ga_optimals")
    name="ga_optimals/harvested_ga_optimal_" + str(ntest)+".csv"
    #df.to_csv("harvested_ga_optimal.csv",index=False)
    df.to_csv(name,index=False)


    path_ori= "../results/"
    path_ori2=path_ori+"ga"
    path=path_ori2+'/HarvestedPlots_'+str(ntest)
    if not os.path.exists(path_ori2):
        os.mkdir(path_ori2)
    if not os.path.exists(path):
        os.mkdir(path)
    i=1
    images = []
    with open('outfile.txt','w') as f:
        for sol in bin_population:
            n=int((len(sol))**(1/2))
            x = [sol[i:i + n] for i in range(0, len(sol), n)] 
            df = pd.DataFrame.from_records(x)
            df[df<1] = 0
            df[df>1] = 0
            to_lst=df.to_numpy().flatten()
            nods=np.where(to_lst==1)
            nods=nods[0]+1

            plt.clf() 
            sns.set()
            sns_plot=sns.heatmap(df,cmap="BuPu")
        #sns_plot.scatter(49, 30 , marker='*', s=150, color='red')
        #sns_plot.scatter(49, 50, marker='*', s=150, color='red')
        #sns_plot.scatter(49, 80, marker='*', s=150, color='red')
        #sns_plot.scatter(0, 17, marker='*', s=150, color='red')
        #sns_plot.scatter(19, 9, marker='*', s=150, color='red')
        #sns_plot.scatter(19, 29, marker='*', s=150, color='red') 
            name=path+"/output"+str(i)+".png"
            sns_plot.set_title('individuo '+str(i))
            sns_plot.figure.savefig(name)
            images.append(imageio.imread(name))
            i+=1
    imageio.mimsave(path+'/history.gif', images,duration = 0.5)

#graficar historia del mejor
    sin_plot = plt.figure()
    plt.plot(times, fitness_history,label="Mejor individuo por generacion")
    plt.plot(times, fitness_history_mean,label="Promedio individuos por generacion")
    plt.title("Comportamiento algoritmo genetico segun tiempo")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Celdas Salvadas [%]")
    plt.legend()
    plt.savefig(path+'/fitness_history.png')


    i=1
    images = []
    with open('outfile.txt','w') as f:
        for sol in individuals:
            n=int((len(sol))**(1/2))
            x = [sol[i:i + n] for i in range(0, len(sol), n)] 
            df = pd.DataFrame.from_records(x)
            df[df<1] = 0
            df[df>1] = 0
            to_lst=df.to_numpy().flatten()
            nods=np.where(to_lst==1)
            nods=nods[0]+1

            plt.clf() 
            sns.set()
            sns_plot=sns.heatmap(df,cmap="BuPu")
            #sns_plot.scatter(49, 30 , marker='*', s=150, color='red')
            #sns_plot.scatter(49, 50, marker='*', s=150, color='red')
            #sns_plot.scatter(49, 80, marker='*', s=150, color='red')
            #sns_plot.scatter(0, 17, marker='*', s=150, color='red')
            #sns_plot.scatter(19, 9, marker='*', s=150, color='red')
            #sns_plot.scatter(19, 29, marker='*', s=150, color='red')
            name=path+"/best"+str(i)+".png"
            sns_plot.set_title('generacion '+str(i))
            sns_plot.figure.savefig(name)
            images.append(imageio.imread(name))
            i+=1
    imageio.mimsave(path+'/history_best.gif', images,duration = 0.5)

    pathtxt=path_ori+"ga"+'/resultados_'+str(ntest)+"txt"
    file1 = open(pathtxt,"w")

    L1=["optimo:\n"]
    L2=[str(max(fit_last_gen)) + "\n"]
    L3=["cortafuegos:\n"]
    L4=[str(parameters) + "\n"]
    L5=["numero de iteraciones:\n"]
    L6=[str(len(times)) +"\n"]
    L7=["historial de fitness promedio:\n"]
    L8=[str(fitness_history_mean)+"\n"]
    L9=["historial de tiempo:\n"]
    L10=[str(times)+"\n"]
    L11=["historial de fitness:\n"]
    L12=[str(fitness_history)+"\n"]

    L = L1+L2+L3+L4+L5+L6+L7+L8+L9+L10+L11+L12

    file1.writelines(L)
    file1.close() #to change file access modes
    tst="python3 main.py --input-instance-folder ../data/"+archivo+"/ --output-folder ../results/ga_stats_"+str(ntest)+" --sim-years 1 --nsims 5000 --finalGrid --weather random --nweathers 7 --Fire-Period-Length 1.0 --output-messages --ROS-CV 0.0 --seed 123 --stats  --HarvestedCells ../cell2fire/ga_optimals/harvested_ga_optimal_" + str(ntest)+".csv"
    os.system(tst)






