import os
import numpy as np
import sys
import csv
import glob
import time
import matplotlib.pyplot as plt
import psi4
import random
import math
from scipy.signal import argrelextrema
#author:ss0832
#reference about LUP method:J. Chem. Phys. 94, 751–760 (1991) https://doi.org/10.1063/1.460343

color_list = ["b","g","r","c","m","y","k"] #use for matplotlib

basic_set_and_function = 'b3lyp/6-31G*'

N_THREAD = 8
SET_MEMORY = '2GB'
NEB_NUM = 10
partition = 10
#please input psi4 inputfile.

np.set_printoptions(precision=13, floatmode="fixed", suppress=True)

spring_constant_k = 0.01

hartree2kcalmol = 627.509

#parameter_for_FIRE_method
FIRE_dt = 0.1
FIRE_N_accelerate = 5
FIRE_f_inc = 1.15
FIRE_f_accelerate = 0.99
FIRE_f_decelerate = 0.5
FIRE_a_start = 0.1
FIRE_dt_max = 10.0
FINISH_REQUIREMENT = 1e-6
APPLY_CI_NEB = False

try:
    partition = int(sys.argv[1])
    NEB_NUM = int(sys.argv[2])
except:
    pass
try:
    start_folder = sys.argv[3]
    #start_file_list = sys.argv[3:]   
    NEB_FOLDER_DIRECTORY = start_folder+"_LUP_"+basic_set_and_function.replace("/","_")+"_"+str(time.time())+"/"
    os.mkdir(NEB_FOLDER_DIRECTORY)
except Exception as e:
    print(e)
    print("usage: python LUP.py [partition(number of nodes)] [LUP_NUM(for optimize)] [initial structure folder name] ")
    time.sleep(3)
    sys.exit(1)

def force2velocity(gradient_list, element_list):
    velocity_list = []
    elements = ["H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn"]
    element_mass = [1.008, 4.0026, 6.94, 9.0122, 10.81, 12.011, 14.007, 15.999, 18.998, 20.180, 22.990, 24.305, 26.982, 28.085, 30.974, 32.06, 35.45, 39.948, 39.098, 40.078, 44.956, 47.867, 50.942, 51.996, 54.938,55.845,58.933,58.693,63.546,65.38,69.723,72.630,74.922,78.971,79.904,83.798,85.468,87.62 ,88.906, 91.224,92.906,95.95,98,101.07,102.91,106.42,107.87,112.41,114.82,118.71,121.76,127.60,126.90,131.29,132.91,137.33,138.91,140.12,140.91,144.24,145,150.36,151.96,157.25,158.93,162.50,164.93,167.26,168.93,173.05,174.97,178.49,180.95,183.84,186.21,190.23,192.22,195.08,196.97,200.59,204.38,207.2,208.98,209,210,222]
   
    for gradient in gradient_list:     
        for i in range(len(element_list)):
            for num, element in enumerate(elements): 
                if element_list[i] == element:
                    gradient[i] = gradient[i] / 1.0#element_mass[num]
        
        velocity_list.append(gradient)
    return np.array(velocity_list, dtype="float64")

def make_geometry_list(start_folder,partition_function):
    start_file_list = glob.glob(start_folder+"/*_[0-9].xyz") + glob.glob(start_folder+"/*_[0-9][0-9].xyz") + glob.glob(start_folder+"/*_[0-9][0-9][0-9].xyz") + glob.glob(start_folder+"/*_[0-9][0-9][0-9][0-9].xyz")
    loaded_geometry_list = []
    geometry_list = []
    for start_file in start_file_list:
        with open(start_file,"r") as f:
            reader = csv.reader(f, delimiter=' ')
            pre_start_data = [row for row in reader]
            start_data = []
            for i in pre_start_data:
                start_data.append([row.strip() for row in i if row != "" and row != "\t"])    
            loaded_geometry_list.append(start_data)
    
    
    electric_charge_and_multiplicity = start_data[0]
    element_list = []
    loaded_geometry_num_list = []

    for i in range(1, len(start_data)):
        element_list.append(start_data[i][0])
       
    for geom_list in loaded_geometry_list:
        num_list = []
        for i in range(1, len(start_data)):
            num_list.append(geom_list[i][1:4])
        loaded_geometry_num_list.append(num_list)
    geometry_list.append(loaded_geometry_list[0])
    for k in range(len(loaded_geometry_list)-1):    
        delta_num_geom = (np.array(loaded_geometry_num_list[k+1], dtype = "float64") - np.array(loaded_geometry_num_list[k], dtype = "float64")) / (partition_function+1)
        frame_geom = np.array(loaded_geometry_num_list[k], dtype = "float64")
        for i in range(0, partition_function+1):
            for j in range(1, len(start_data)):
                frame_geom = np.array(loaded_geometry_num_list[k], dtype = "float64") + delta_num_geom*i
                frame_file = [start_data[0]]+[[]*n for n in range(len(start_data)-1)]
                for x in range(0, len(start_data)-1):
                    frame_file[x+1].append(element_list[x])
                    frame_file[x+1].extend(frame_geom[x].tolist())
            geometry_list.append(frame_file)
    
    geometry_list.append(loaded_geometry_list[-1])
    
    
    print("\n geometry datas are loaded. \n")
    
    return geometry_list, element_list, electric_charge_and_multiplicity

def make_geometry_list_2(new_geometory, element_list, electric_charge_and_multiplicity):
    new_geometory = new_geometory.tolist()
   
    geometry_list = []
    for geometries in new_geometory:
        new_data = [electric_charge_and_multiplicity]
        for num, geometry in enumerate(geometries):
            geometory = list(map(str, geometry))
            geometory = [element_list[num]] + geometory
            new_data.append(geometory)
        
        geometry_list.append(new_data)
    return geometry_list

def make_psi4_input_file(geometry_list, optimize_num):
    file_directory = NEB_FOLDER_DIRECTORY+"samples_"+str(optimize_num)+"_"+str(start_folder)+"_"+str(time.time())
    try:
        os.mkdir(file_directory)
    except:
        pass
    for y, geometry in enumerate(geometry_list):
        with open(file_directory+"/"+start_folder+"_"+str(y)+".xyz","a") as w:
            for rows in geometry:
                for row in rows:
                    w.write(str(row))
                    w.write(" ")
                w.write("\n")
    return file_directory
    
def sinple_plot(num_list, energy_list, file_directory, optimize_num):
    fig, ax = plt.subplots()
    ax.plot(num_list,energy_list*np.array(hartree2kcalmol, dtype = "float64"), color_list[random.randint(0,len(color_list)-1)]+"--o" )

    ax.set_title(str(file_directory))
    ax.set_xlabel('step')
    ax.set_ylabel('Gibbs Energy [kcal/mol]')
    fig.tight_layout()
    fig.savefig(NEB_FOLDER_DIRECTORY+"Energy_plot_sinple_"+str(optimize_num)+"_"+str(time.time())+".png", format="png", dpi=200)
    plt.close()
   

    return
    
def psi4_calclation(file_directory, optimize_num,electric_charge_and_multiplicity,element_list ,fixing_geom_structure_num,pre_total_velocity):
    psi4.core.clean()
    gradient_list = []
    energy_list = []
    geometry_num_list = []
    num_list = []
    delete_pre_total_velocity = []
    try:
        os.mkdir(file_directory)
    except:
        pass
    file_list = glob.glob(file_directory+"/*_[0-9].xyz") + glob.glob(file_directory+"/*_[0-9][0-9].xyz") + glob.glob(file_directory+"/*_[0-9][0-9][0-9].xyz") + glob.glob(file_directory+"/*_[0-9][0-9][0-9][0-9].xyz")
    for num, input_file in enumerate(file_list):
        try:
            print("\n",input_file,"\n")
        
            logfile = file_directory+"/"+start_folder+'_'+str(num)+'.log'
            #psi4.set_options({'pcm': True})
            #psi4.pcm_helper(pcm)
            psi4.set_output_file(logfile)
            psi4.set_num_threads(nthread=N_THREAD)
            psi4.set_memory(SET_MEMORY)
            with open(input_file,"r") as f:
                input_data = f.read()
                input_data = psi4.geometry(input_data)
                input_data_for_display = np.array(input_data.geometry(), dtype = "float64")
            if np.nanmean(np.nanmean(input_data_for_display)) > 1e+5:
                raise Exception("geometry is abnormal.")
                #print('geometry:\n'+str(input_data_for_display))            
        
            e = []      
            g = np.array(psi4.gradient(basic_set_and_function, molecule=input_data), dtype = "float64")
            with open(input_file[:-4]+".log","r") as f:
                word_list = f.readlines()
                for word in word_list:
                    if "    Total Energy =             " in word:
                        word = word.replace("    Total Energy =             ","")
                        e.append(float(word))
            #print("gradient:\n"+str(g))
            print('energy:'+str(e[0])+" a.u.")
            gradient_list.append(g)
            energy_list.append(e[0])
            num_list.append(num)
            geometry_num_list.append(input_data_for_display)
        except Exception as error:
            print(error)
            print("This molecule could not be optimized.")
            if optimize_num != 0:
                delete_pre_total_velocity.append(num)
            
            
        psi4.core.clean()
    print("data sampling completed...")
    try:
        sinple_plot(num_list, energy_list, file_directory, optimize_num)
        print("energy graph plotted.")
    except Exception as e:
        print(e)
        print("Can't plot energy graph.")

    if type(optimize_num) == int:
        if optimize_num % 200 == 0 and optimize_num != 0:
            fixing_geom_structure_num, gradient_list = freq_analysis(np.array(geometry_num_list, dtype="float64")*psi4.constants.bohr2angstroms,basic_set_and_function,electric_charge_and_multiplicity,element_list,gradient_list,fixing_geom_structure_num)
        else:
            pass
    else:
        pass
    
    if optimize_num != 0:
        pre_total_velocity = pre_total_velocity.tolist()
        for i in sorted(delete_pre_total_velocity, reverse=True):
            pre_total_velocity.pop(i)
        pre_total_velocity = np.array(pre_total_velocity, dtype="float64")

    return np.array(energy_list, dtype = "float64"), np.array(gradient_list, dtype = "float64"), np.array(geometry_num_list, dtype = "float64"), fixing_geom_structure_num, pre_total_velocity



def freq_analysis(geometry_num_list,basic_set_and_function,electric_charge_and_multiplicity,element_list,gradient_list ,fixing_geom_structure_num):
    fixing_geom_structure_num = []
    for num, geometry in enumerate(geometry_num_list):
        print("image ",num)
        print("initializing frequency analysis...(for eigenvalue)")
        frame_file = [electric_charge_and_multiplicity]+[[]*n for n in range(len(geometry))]
            
        for x in range(1, len(geometry)+1):
            frame_file[x].append(element_list[x-1])
            frame_file[x].extend(geometry[x-1])
           
        input_data = str()
        for data in frame_file:#to input correctly to psi4   
            input_data += " ".join(list(map(str, data))) + "\n"
        input_data = psi4.geometry(input_data)
        psi4.set_num_threads(nthread=N_THREAD)
        psi4.set_memory(SET_MEMORY)
        
        e, wfn = psi4.frequency(basic_set_and_function, molecule=input_data, return_wfn=True)
        print("energy: ", e)
        freq_list = np.array(wfn.frequencies()).tolist()
        hessian = np.array(wfn.hessian())
        print("Frequencies:\n",freq_list)
        print("Hessian:\n",hessian)
        
        with open(NEB_FOLDER_DIRECTORY+"hess_list.txt","a") as w:
            w.write("image "+str(num)+"\n")
            w.write("Frequencies:\n"+str(freq_list)+"\n")
            w.write("Hessian:\n"+str(hessian)+"\n")
        
        
        if (freq_list[0] < 0 or freq_list[0] > 0) and freq_list[1] > 0:
            fixing_geom_structure_num.append(num)
    
        psi4.core.clean()
    
    print(fixing_geom_structure_num)
    return fixing_geom_structure_num, gradient_list


def xyz_file_make(file_directory):
    print("\ngeometry integration processing...\n")
    file_list = glob.glob(file_directory+"/*_[0-9].xyz") + glob.glob(file_directory+"/*_[0-9][0-9].xyz") + glob.glob(file_directory+"/*_[0-9][0-9][0-9].xyz") + glob.glob(file_directory+"/*_[0-9][0-9][0-9][0-9].xyz")
    
    for m, file in enumerate(file_list):
       
        with open(file,"r") as f:
            sample = f.readlines()
            with open(file_directory+"/"+start_folder+"_integration.xyz","a") as w:
                atom_num = len(sample)-1
                w.write(str(atom_num)+"\n")
                w.write("Frame "+str(m)+"\n")
            del sample[0]
            for i in sample:
                with open(file_directory+"/"+start_folder+"_integration.xyz","a") as w2:
                    w2.write(i)
    print("\ngeometry integration complete...\n")
    return


def extremum_list_index(energy_list):
    local_max_energy_list_index = argrelextrema(energy_list, np.greater)
    inverse_energy_list = (-1)*energy_list
    local_min_energy_list_index = argrelextrema(inverse_energy_list, np.greater)
    local_max_energy_list_index = local_max_energy_list_index[0].tolist()
    local_min_energy_list_index = local_min_energy_list_index[0].tolist()
    local_max_energy_list_index.append(0)
    local_min_energy_list_index.append(0)
    local_max_energy_list_index.append(0)
    local_min_energy_list_index.append(0)
    local_max_energy_list_index.append(0)
    local_min_energy_list_index.append(0)
  
    return local_max_energy_list_index, local_min_energy_list_index


def LUP_calc(geometry_num_list,energy_list, gradient_list, element_list):
    local_max_energy_list_index, local_min_energy_list_index = extremum_list_index(energy_list)

    total_force_list = [((-1)*np.array(gradient_list[0], dtype = "float64")).tolist()]
    for i in range(1,len(energy_list)-1):
      
        tau_plus, tau_minus, tau = [], [], []
        
        delta_max_energy = np.array(max([(energy_list[i+1]-energy_list[i]),(energy_list[i-1]-energy_list[i])]), dtype = "float64")
        delta_min_energy = np.array(min([(energy_list[i+1]-energy_list[i]),(energy_list[i-1]-energy_list[i])]), dtype = "float64")
        
        if (energy_list[i-1] < energy_list[i]) and (energy_list[i] < energy_list[i+1]):
            for t in range(len(geometry_num_list[i])):
                tau_vector = geometry_num_list[i+1][t]-geometry_num_list[i][t]
                tau_norm = np.linalg.norm(geometry_num_list[i+1][t]-geometry_num_list[i][t], ord=2)
                tau.append(np.divide(tau_vector, tau_norm, out=np.zeros_like(geometry_num_list[i][t]) ,where=np.linalg.norm(geometry_num_list[i+1][t]-geometry_num_list[i][t], ord=2)!=0).tolist())       
       
             
        elif (energy_list[i-1] > energy_list[i]) and (energy_list[i] > energy_list[i+1]):
            for t in range(len(geometry_num_list[i])):      
                tau_vector = geometry_num_list[i][t]-geometry_num_list[i-1][t]
                tau_norm = np.linalg.norm(geometry_num_list[i][t]-geometry_num_list[i-1][t], ord=2)
                tau.append(np.divide(tau_vector, tau_norm, out=np.zeros_like(geometry_num_list[i][t]), where=np.linalg.norm(geometry_num_list[i][t]-geometry_num_list[i-1][t], ord=2)!=0).tolist())       
      
        
        
        else: #((energy_list[i-1] >= energy_list[i]) and (energy_list[i] <= energy_list[i+1])) or ((energy_list[i-1] <= energy_list[i]) and (energy_list[i] >= energy_list[i+1])):
            for t in range(len(geometry_num_list[i])):         
                tau_minus_vector = geometry_num_list[i][t]-geometry_num_list[i-1][t]
                tau_minus_norm = np.linalg.norm(geometry_num_list[i][t]-geometry_num_list[i-1][t], ord=2)
                tau_minus.append(np.divide(tau_minus_vector, tau_minus_norm
                                 ,out=np.zeros_like(geometry_num_list[i][t]),
                                 where=np.linalg.norm(geometry_num_list[i][t]-geometry_num_list[i-1][t], ord=2)!=0).tolist())       

            for t in range(len(geometry_num_list[i])):
                tau_plus_vector = geometry_num_list[i+1][t]-geometry_num_list[i][t]
                tau_plus_norm = np.linalg.norm(geometry_num_list[i+1][t]-geometry_num_list[i][t], ord=2)
                tau_plus.append(np.divide(tau_plus_vector, tau_plus_norm, out=np.zeros_like(geometry_num_list[i][t]), where=np.linalg.norm(geometry_num_list[i+1][t]-geometry_num_list[i][t], ord=2)!=0).tolist())

            if energy_list[i-1] > energy_list[i+1]:
                for t in range(len(geometry_num_list[i])):
                    tau_vector = (tau_plus[t]*delta_min_energy+tau_minus[t]*delta_max_energy)
                    tau_norm = np.linalg.norm(tau_plus[t]*delta_min_energy+tau_minus[t]*delta_max_energy, ord=2)
                    tau.append(np.divide(tau_vector,tau_norm, out=np.zeros_like(tau_plus[0]) ,where=np.linalg.norm(tau_plus[t]*delta_min_energy+tau_minus[t]*delta_max_energy!=0)).tolist())
            else:
                for t in range(len(geometry_num_list[i])):
                    tau_vector = (tau_plus[t]*delta_max_energy+tau_minus[t]*delta_min_energy)
                    tau_norm = np.linalg.norm(tau_plus[t]*delta_min_energy+tau_minus[t]*delta_max_energy, ord=2)
                    tau.append(np.divide(tau_vector, tau_norm,out=np.zeros_like(tau_minus[0]) ,where=np.linalg.norm(tau_plus[t]*delta_min_energy+tau_minus[t]*delta_max_energy, ord=2)!=0 ).tolist())

        tau_plus, tau_minus, tau = np.array(tau_plus, dtype = "float64"), np.array(tau_minus, dtype = "float64"), np.array(tau, dtype = "float64")    

        
        force_perpendicularity = []
        if energy_list[i] == energy_list[local_max_energy_list_index[0]] and APPLY_CI_NEB == True: #CI-NEB
            for f in range(len(geometry_num_list[i])):
                force_perpendicularity.append(np.array((-1)*(gradient_list[i][f]-2.0*(np.dot(gradient_list[i][f], tau[f]))*tau[f]), dtype = "float64"))
                
            total_force = np.array(force_perpendicularity, dtype="float64")
            del local_max_energy_list_index[0]
         
        elif energy_list[i] == energy_list[local_min_energy_list_index[0]]: #for discovering intermidiate
            for f in range(len(geometry_num_list[i])):
                force_perpendicularity.append(np.array(((-1)*(gradient_list[i][f])), dtype = "float64"))
               
            total_force = np.array(force_perpendicularity, dtype="float64")
            del local_min_energy_list_index[0]
        else:    
            for f in range(len(geometry_num_list[i])):
                grad = 0.0
                
                for gg in range(len(gradient_list[i])):
                    grad += np.linalg.norm(gradient_list[i][gg], ord=2)
                
                grad = grad/len(gradient_list[i])
                
                spring_constant_k = 0.75*grad
              
                

                force_perpendicularity.append(np.array(gradient_list[i][f]-(np.dot(gradient_list[i][f], tau[f]))*tau[f], dtype = "float64"))
              
            
                
            
                
        
            force_perpendicularity = np.array(force_perpendicularity, dtype = "float64")
            total_force = np.array((-1)*force_perpendicularity , dtype = "float64")
        
        if np.nanmean(np.nanmean(total_force)) > 10:
            total_force = total_force / np.nanmean(np.nanmean(total_force))
        
        total_force_list.append(total_force.tolist())

            
    
    total_force_list.append(((-1)*np.array(gradient_list[-1], dtype = "float64")).tolist())
    
    return np.array(total_force_list, dtype = "float64")

 


def saddle_calc(geometry_list,local_max_energy_list_index, local_min_energy_list_index):
  
    local_max_energy_list_index = list(set(local_max_energy_list_index))
    local_min_energy_list_index = list(set(local_min_energy_list_index))
    for index in local_max_energy_list_index:
        try:
            print('appear TS index: ',index)
            os.mkdir(NEB_FOLDER_DIRECTORY+"saddle_calc_"+str(index))
            logfile = NEB_FOLDER_DIRECTORY+"saddle_calc_"+str(index)+"/saddle_calc_"+str(index)+"_"+str(time.time())+".log"
            psi4.set_output_file(logfile)
            psi4.set_num_threads(nthread=N_THREAD)
            psi4.set_memory(SET_MEMORY)
            psi4.set_options({'geom_maxiter': 300,"opt_type": "ts", "full_hess_every": 2})

            input_data = ""
            for data in geometry_list[index]:
                input_data += " ".join(list(map(str, data))) + "\n"
            input_data = psi4.geometry(input_data)
            
            e, wfn = psi4.optimize(basic_set_and_function, molecule=input_data, return_wfn=True)
            e, wfn = psi4.frequencies(basic_set_and_function, molecule=input_data,ref_gradient=wfn.gradient(),return_wfn=True)
            freqs = np.array(wfn.frequencies())
            print(f'TS optimized structure:\n{input_data.save_string_xyz()}')
            print(f'frequency:\n{freqs}')
            
            if freqs[0] < 0 and freqs[1] >= 0:
                print("discover TS!")
                with open(logfile+"_TS.xyz","w") as f:
                    f.write(input_data.save_string_xyz())

                    
        except Exception as e:
            print(e)
        psi4.core.clean()
    
    return 



def FIRE_calc(geometry_num_list, total_force_list, pre_total_velocity, optimize_num, total_velocity, dt, n_reset, a,  ):
    velocity_neb = []
    finish_frag = False
    for num, each_velocity in enumerate(total_velocity):
        part_velocity_neb = []
        for i in range(len(total_force_list[0])):
             
            part_velocity_neb.append((1.0-a)*total_velocity[num][i]+a*np.sqrt(np.dot(total_velocity[num][i],total_velocity[num][i])/np.dot(total_force_list[num][i],total_force_list[num][i]))*total_force_list[num][i])
        velocity_neb.append(part_velocity_neb)
        #print(part_velocity_neb)
    
    velocity_neb = np.array(velocity_neb)
    #print(velocity_neb)
    np_dot_param = 0
    if optimize_num != 0:
        for num_1, total_force in enumerate(total_force_list):
            for num_2, total_force_num in enumerate(total_force):
                np_dot_param += (np.dot(pre_total_velocity[num_1][num_2] ,total_force_num.T))
        print(np_dot_param)
    else:
        pass
    if optimize_num > 0 and np_dot_param > 0:
        if n_reset > FIRE_N_accelerate:
            dt = min(dt*FIRE_f_inc, FIRE_dt_max)
            a = a*FIRE_N_accelerate
        n_reset += 1
    else:
        velocity_neb = velocity_neb*0
        a = FIRE_a_start
        dt = dt*FIRE_f_decelerate
        n_reset = 0
    total_velocity = velocity_neb + dt*(total_force_list)
    if optimize_num != 0:
        total_delta = dt*(total_velocity+pre_total_velocity)#like conjugate gradient descent
    else:
        total_delta = dt*(total_velocity)
   
    total_delta_average = np.nanmean(total_delta)
    print("total_delta_average:",str(total_delta_average))
  
    new_geometory = (geometry_num_list + total_delta)*psi4.constants.bohr2angstroms
  
    if abs(total_delta_average) < FINISH_REQUIREMENT:
        finish_frag = True
        
    return new_geometory, dt, n_reset, a, finish_frag

def main():
    global APPLY_CI_NEB
    geometry_list, element_list, electric_charge_and_multiplicity = make_geometry_list(start_folder,partition)
    file_directory = make_psi4_input_file(geometry_list,0)
    pre_total_velocity = [[[]]]
    fixing_geom_structure_num = []
    #prepare for FIRE method
    dt = 0.5
    n_reset = 0
    a = FIRE_a_start
    
    for optimize_num in range(NEB_NUM):
        print("\n\n\nLUP:   "+str(optimize_num+1)+" time(s) \n\n\n")
        xyz_file_make(file_directory)    

        energy_list, gradient_list, geometry_num_list, fixing_geom_structure_num, pre_total_velocity = psi4_calclation(file_directory, optimize_num,electric_charge_and_multiplicity,element_list,fixing_geom_structure_num,pre_total_velocity )
        
        
        
        total_force = LUP_calc(geometry_num_list,energy_list, gradient_list, element_list)
       
        total_velocity = force2velocity(total_force, element_list)
        for i in fixing_geom_structure_num:
            total_velocity[i] *= 1.0e-9
        new_geometory, dt, n_reset, a, finish_frag = FIRE_calc(geometry_num_list, total_force, pre_total_velocity, optimize_num, total_velocity, dt, n_reset, a,  )
        print(str(dt),str(n_reset),str(a))
        geometry_list = make_geometry_list_2(new_geometory, element_list, electric_charge_and_multiplicity)
        file_directory = make_psi4_input_file(geometry_list, optimize_num+1)
        if finish_frag:
            break
        pre_total_velocity = total_velocity
  
    print("\n\n\nLUP: final\n\n\n")
    xyz_file_make(file_directory) 
    energy_list, gradient_list, geometry_num_list,fixing_geom_structure_num, pre_total_velocity = psi4_calclation(file_directory, "final",electric_charge_and_multiplicity,element_list,fixing_geom_structure_num, pre_total_velocity )
    geometry_list = make_geometry_list_2(new_geometory, element_list, electric_charge_and_multiplicity)
    energy_list = energy_list.tolist()
  
    local_max_energy_list_index, local_min_energy_list_index = extremum_list_index(np.array(energy_list,dtype="float64"))
    saddle_calc(geometry_list,local_max_energy_list_index, local_min_energy_list_index)
    
    print("Complete...")
    return


main()