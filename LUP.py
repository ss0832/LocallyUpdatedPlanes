import os
import numpy as np
import sys
import csv
import glob
import time
import matplotlib.pyplot as plt
import random
import math
import argparse
from scipy.signal import argrelextrema

try:
    import psi4
except:
    print("You can't use psi4.")

try:
    from tblite.interface import Calculator
except:
    print("You can't use extended tight binding method.")

#reference about LUP method:J. Chem. Phys. 94, 751–760 (1991) https://doi.org/10.1063/1.460343

color_list = ["g"] #use for matplotlib





def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", help='input folder')
    parser.add_argument("-bs", "--basisset", default='6-31G(d)', help='basisset (ex. 6-31G*)')
    parser.add_argument("-func", "--functional", default='b3lyp', help='functional(ex. b3lyp)')

    parser.add_argument("-ns", "--NSTEP",  type=int, default='10', help='iter. number')
    parser.add_argument("-o", "--OM", action='store_true', help='J. Chem. Phys. 155, 074103 (2021)  doi:https://doi.org/10.1063/5.0059593 This improved NEB method is inspired by the Onsager-Machlup (OM) action. (OM + DNEB)')
    parser.add_argument("-p", "--partition",  type=int, default='0', help='number of nodes')
    parser.add_argument("-core", "--N_THREAD",  type=int, default='8', help='threads')
    parser.add_argument("-mem", "--SET_MEMORY",  type=str, default='1GB', help='use mem(ex. 1GB)')
    parser.add_argument("-cineb", "--apply_CI_NEB",  type=int, default='99999', help='apply CI_NEB method')
    parser.add_argument("-sd", "--steepest_descent",  type=int, default='99999', help='apply steepest_descent method')
    parser.add_argument("-xtb", "--usextb",  type=str, default="None", help='use extended tight bonding method to calculate. default is not using extended tight binding method (ex.) GFN1-xTB, GFN2-xTB ')
    args = parser.parse_args()
    return args

def element_number(elem):
    num = {"H": 1, "He": 2,
        "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10, 
        "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
        "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
        "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42,"Tc": 43,"Ru": 44,"Rh": 45,"Pd": 46,"Ag": 47,"Cd": 48,"In": 49,"Sn": 50,"Sb": 51,"Te": 52,"I": 53,"Xe": 54,
        "Cs": 55 ,"Ba": 56, "La": 57,"Ce":58,"Pr": 59,"Nd": 60,"Pm": 61,"Sm": 62,"Eu": 63,"Gd": 64,"Tb": 65,"Dy": 66,"Ho": 67,"Er": 68,"Tm": 69,"Yb": 70,"Lu": 71,"Hf": 72,"Ta": 73,"W": 74,"Re": 75,"Os": 76,"Ir": 77,"Pt": 78,"Au": 79,"Hg": 80,"Tl": 81,"Pb":82,"Bi":83,"Po":84,"At":85,"Rn":86}
        
    return num[elem]
    
def atomic_mass(elem):
    if elem is int:
        elem_num = elem
    else:    
        elem_num = element_number(elem)
    mass = {1: 1.00782503223, 2: 4.00260325413,
    3: 7.0160034366, 4: 9.012183065, 5: 11.00930536, 6: 12.0, 7: 14.00307400443, 8: 15.99491461957, 9: 18.99840316273, 10: 19.9924401762,
    11: 22.989769282, 12: 23.985041697, 13: 26.98153853, 14: 27.97692653465, 15: 30.97376199842, 16: 31.9720711744, 17: 34.968852682, 18: 39.9623831237,
    19: 38.9637064864, 20: 39.962590863, 21: 44.95590828, 22: 47.94794198, 23: 50.94395704, 24: 51.94050623, 25: 54.93804391, 26: 55.93493633, 27: 58.93319429, 28: 57.93534241, 29: 62.92959772, 30: 63.92914201, 31: 68.9255735, 32: 73.921177761, 33: 74.92159457, 34: 79.9165218, 35: 78.9183376, 36: 83.9114977282,
    37: 84.9117897379, 38: 87.9056125, 39: 88.9058403, 40: 89.9046977, 41: 92.906373, 42: 97.90540482, 43: 96.9063667, 44: 101.9043441, 45: 102.905498, 46: 105.9034804, 47: 106.9050916, 48: 113.90336509, 49: 114.903878776, 50: 119.90220163, 51: 120.903812, 52: 129.906222748, 53: 126.9044719, 54: 131.9041550856,
    55: 132.905451961, 56: 137.905247, 57: 138.9063563, 58: 139.9054431, 59: 140.9076576, 60: 141.907729, 61: 144.9127559, 62: 151.9197397, 63: 152.921238, 64: 157.9241123, 65: 158.9253547, 66: 163.9291819, 67: 164.9303288,
    68: 165.9302995, 69: 168.9342179, 70: 173.9388664, 71: 174.9407752, 72: 179.946557, 73: 180.9479958, 74: 183.95093092, 75: 186.9557501, 76: 191.961477, 77: 192.9629216, 78: 194.9647917, 79: 196.96656879, 80: 201.9706434, 81: 204.9744278,
    82: 207.9766525, 83: 208.9803991, 84: 208.9824308, 85: 209.9871479, 86: 222.0175782}# https://www.nist.gov/pml/atomic-weights-and-isotopic-compositions-relative-atomic-masses
    return mass[elem_num]


class LUP:
    def __init__(self, args):
    
        self.basic_set_and_function = args.functional+"/"+args.basisset

        self.N_THREAD = args.N_THREAD
        self.SET_MEMORY = args.SET_MEMORY
        self.NEB_NUM = args.NSTEP
        self.partition = args.partition
        #please input psi4 inputfile.

        self.spring_constant_k = 0.01
        self.bohr2angstroms = 0.52917721067
        self.hartree2kcalmol = 627.509
        #parameter_for_FIRE_method
        self.FIRE_dt = 0.1
        self.FIRE_N_accelerate = 5
        self.FIRE_f_inc = 1.10
        self.FIRE_f_accelerate = 0.99
        self.FIRE_f_decelerate = 0.5
        self.FIRE_a_start = 0.1
        self.FIRE_dt_max = 3.0
        self.APPLY_CI_NEB = args.apply_CI_NEB
        self.start_folder = args.INPUT
        self.om = args.OM
        self.usextb = args.usextb
        self.sd = args.steepest_descent
        
        if args.usextb == "None":
            self.NEB_FOLDER_DIRECTORY = args.INPUT+"_LUP_"+self.basic_set_and_function.replace("/","_")+"_"+str(time.time())+"/"
        else:
            self.NEB_FOLDER_DIRECTORY = args.INPUT+"_LUP_"+self.usextb+"_"+str(time.time())+"/"
        self.args = args
        os.mkdir(self.NEB_FOLDER_DIRECTORY)
        return

    def force2velocity(self, gradient_list, element_list):
        velocity_list = gradient_list
        return np.array(velocity_list, dtype="float64")

    def make_geometry_list(self, start_folder, partition_function):
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
        #print(loaded_geometry_list)
        
        electric_charge_and_multiplicity = start_data[0]
        element_list = []
        loaded_geometry_num_list = []

        for i in range(1, len(start_data)):
            element_list.append(start_data[i][0])
            #start_num_list.append(start_data[i][1:4])
            #end_num_list.append(end_data[i][1:4])
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
        
        #print(geometry_list)
        print("\n geometry datas are loaded. \n")
        
        return geometry_list, element_list, electric_charge_and_multiplicity

    def make_geometry_list_2(self, new_geometory, element_list, electric_charge_and_multiplicity):
        new_geometory = new_geometory.tolist()
        #print(new_geometory)
        geometry_list = []
        for geometries in new_geometory:
            new_data = [electric_charge_and_multiplicity]
            for num, geometry in enumerate(geometries):
                geometory = list(map(str, geometry))
                geometory = [element_list[num]] + geometory
                new_data.append(geometory)
            
            geometry_list.append(new_data)
        return geometry_list

    def make_psi4_input_file(self, geometry_list, optimize_num):
        file_directory = self.NEB_FOLDER_DIRECTORY+"samples_"+str(optimize_num)+"_"+str(self.start_folder)+"_"+str(time.time())
        try:
            os.mkdir(file_directory)
        except:
            pass
        for y, geometry in enumerate(geometry_list):
            with open(file_directory+"/"+self.start_folder+"_"+str(y)+".xyz","w") as w:
                for rows in geometry:
                    for row in rows:
                        w.write(str(row))
                        w.write(" ")
                    w.write("\n")
        return file_directory
        
    def sinple_plot(self, num_list, energy_list, file_directory, optimize_num, axis_name_1="NODE #", axis_name_2="Electronic Energy [kcal/mol]", name="energy"):
        fig, ax = plt.subplots()
        ax.plot(num_list,energy_list, color_list[random.randint(0,len(color_list)-1)]+"--o" )

        ax.set_title(str(optimize_num))
        ax.set_xlabel(axis_name_1)
        ax.set_ylabel(axis_name_2)
        fig.tight_layout()
        fig.savefig(self.NEB_FOLDER_DIRECTORY+"Plot_"+name+"_"+str(optimize_num)+"_"+str(time.time())+".png", format="png", dpi=200)
        plt.close()
        #del fig, ax

        return
        
    def psi4_calculation(self, file_directory, optimize_num, pre_total_velocity):
        psi4.core.clean()
        gradient_list = []
        gradient_norm_list = []
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
            
                logfile = file_directory+"/"+self.start_folder+'_'+str(num)+'.log'
                #psi4.set_options({'pcm': True})
                #psi4.pcm_helper(pcm)
                psi4.set_output_file(logfile)
                psi4.set_num_threads(nthread=self.N_THREAD)
                psi4.set_memory(self.SET_MEMORY)
                with open(input_file,"r") as f:
                    input_data = f.read()
                    input_data = psi4.geometry(input_data)
                    input_data_for_display = np.array(input_data.geometry(), dtype = "float64")
                if np.nanmean(np.nanmean(input_data_for_display)) > 1e+5:
                    raise Exception("geometry is abnormal.")
                    #print('geometry:\n'+str(input_data_for_display))            
            
               
                g, wfn = psi4.gradient(self.basic_set_and_function, molecule=input_data, return_wfn=True)
                g = np.array(g, dtype = "float64")
                e = float(wfn.energy())
  
                #print("gradient:\n"+str(g))
                print('energy:'+str(e)+" a.u.")

                gradient_list.append(g)
                gradient_norm_list.append(np.sqrt(np.linalg.norm(g)**2/(len(g)*3)))#RMS
                energy_list.append(e)
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
            self.sinple_plot(num_list, np.array(energy_list, dtype="float64")*self.hartree2kcalmol, file_directory, optimize_num)
            print("energy graph plotted.")
        except Exception as e:
            print(e)
            print("Can't plot energy graph.")

        try:
            self.sinple_plot(num_list, gradient_norm_list, file_directory, optimize_num, axis_name_1="NODE #", axis_name_2="RMS Gradient [a.u.]", name="gradient")
          
            print("gradient graph plotted.")
        except Exception as e:
            print(e)
            print("Can't plot gradient graph.")
        
        if optimize_num != 0:
            pre_total_velocity = pre_total_velocity.tolist()
            for i in sorted(delete_pre_total_velocity, reverse=True):
                pre_total_velocity.pop(i)
            pre_total_velocity = np.array(pre_total_velocity, dtype="float64")

        return np.array(energy_list, dtype = "float64"), np.array(gradient_list, dtype = "float64"), np.array(geometry_num_list, dtype = "float64"), pre_total_velocity


    
    def tblite_calculation(self, file_directory, optimize_num, pre_total_velocity, element_number_list):
        #execute extended tight binding method calclation.
        gradient_list = []
        energy_list = []
        geometry_num_list = []
        gradient_norm_list = []
        delete_pre_total_velocity = []
        num_list = []
        finish_frag = False
        try:
            os.mkdir(file_directory)
        except:
            pass
        file_list = glob.glob(file_directory+"/*_[0-9].xyz") + glob.glob(file_directory+"/*_[0-9][0-9].xyz") + glob.glob(file_directory+"/*_[0-9][0-9][0-9].xyz") + glob.glob(file_directory+"/*_[0-9][0-9][0-9][0-9].xyz")
        for num, input_file in enumerate(file_list):
            try:
                print("\n",input_file,"\n")

                with open(input_file,"r") as f:
                    input_data = f.readlines()
                    
                positions = []
                for word in input_data[1:]:
                    positions.append(word.split()[1:4])
                        
                positions = np.array(positions, dtype="float64") / self.bohr2angstroms
                calc = Calculator(self.usextb, element_number_list, positions)
                calc.set("max-iter", 500)
                calc.set("verbosity", 1)
                res = calc.singlepoint()
                e = float(res.get("energy"))  #hartree
                g = res.get("gradient") #hartree/Bohr
                        
                print("\n")
                energy_list.append(e)
                gradient_list.append(g)
                gradient_norm_list.append(np.sqrt(np.linalg.norm(g)**2/(len(g)*3)))#RMS
                geometry_num_list.append(positions)
                num_list.append(num)
            except Exception as error:
                print(error)
                print("This molecule could not be optimized.")
                if optimize_num != 0:
                    delete_pre_total_velocity.append(num)
            
        try:
            self.sinple_plot(num_list, np.array(energy_list, dtype="float64")*self.hartree2kcalmol, file_directory, optimize_num)
            print("energy graph plotted.")
        except Exception as e:
            print(e)
            print("Can't plot energy graph.")

        try:
            self.sinple_plot(num_list, gradient_norm_list, file_directory, optimize_num, axis_name_1="NODE #", axis_name_2="RMS Gradient [a.u.]", name="gradient")
          
            print("gradient graph plotted.")
        except Exception as e:
            print(e)
            print("Can't plot gradient graph.")

        if optimize_num != 0:
            pre_total_velocity = pre_total_velocity.tolist()
            for i in sorted(delete_pre_total_velocity, reverse=True):
                pre_total_velocity.pop(i)
            pre_total_velocity = np.array(pre_total_velocity, dtype="float64")

        return np.array(energy_list, dtype = "float64"), np.array(gradient_list, dtype = "float64"), np.array(geometry_num_list, dtype = "float64"), pre_total_velocity

    



    def xyz_file_make(self, file_directory):
        print("\ngeometry integration processing...\n")
        file_list = glob.glob(file_directory+"/*_[0-9].xyz") + glob.glob(file_directory+"/*_[0-9][0-9].xyz") + glob.glob(file_directory+"/*_[0-9][0-9][0-9].xyz") + glob.glob(file_directory+"/*_[0-9][0-9][0-9][0-9].xyz")
        #print(file_list,"\n")
        for m, file in enumerate(file_list):
            #print(file,m)
            with open(file,"r") as f:
                sample = f.readlines()
                with open(file_directory+"/"+self.start_folder+"_integration.xyz","a") as w:
                    atom_num = len(sample)-1
                    w.write(str(atom_num)+"\n")
                    w.write("Frame "+str(m)+"\n")
                del sample[0]
                for i in sample:
                    with open(file_directory+"/"+self.start_folder+"_integration.xyz","a") as w2:
                        w2.write(i)
        print("\ngeometry integration complete...\n")
        return


    def extremum_list_index(self, energy_list):
        local_max_energy_list_index = argrelextrema(energy_list, np.greater)
        inverse_energy_list = (-1)*energy_list
        local_min_energy_list_index = argrelextrema(inverse_energy_list, np.greater)

        local_max_energy_list_index = local_max_energy_list_index[0].tolist()
        local_min_energy_list_index = local_min_energy_list_index[0].tolist()
        local_max_energy_list_index.append(0)
        local_min_energy_list_index.append(0)
        local_max_energy_list_index.append(0)
        local_min_energy_list_index.append(0)
        return local_max_energy_list_index, local_min_energy_list_index

    def OM_calc(self, geometry_num_list, energy_list, gradient_list, optimize_num, element_list):#J. Chem. Phys. 155, 074103 (2021)  doi:https://doi.org/10.1063/5.0059593
        #This improved NEB method is inspired by the Onsager-Machlup (OM) action.
        local_max_energy_list_index, local_min_energy_list_index = self.extremum_list_index(energy_list)
        delta_t = 1
        damping_const = 1
        OM_spring_const = 0.001
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

            L_minus = []
            L_neutral = []
            for t in range(len(geometry_num_list[i])):
                atom_mass = atomic_mass(element_list[t])
                L_minus.append(-(delta_t/(atom_mass * damping_const)) * gradient_list[i-1][t])
                L_neutral.append(-(delta_t/(atom_mass * damping_const)) * gradient_list[i][t])
            L_minus = np.array(L_minus, dtype="float64")
            L_neutral = np.array(L_neutral, dtype="float64")
        
            OM_action_force = OM_spring_const * (geometry_num_list[i+1] + geometry_num_list[i-1] - 2 * geometry_num_list[i] + L_minus - L_neutral)
            
            cos_phi = np.sum((geometry_num_list[i+1]-geometry_num_list[i]) * (geometry_num_list[i]-geometry_num_list[i-1])) /(np.linalg.norm(geometry_num_list[i+1]-geometry_num_list[i]) * np.linalg.norm(geometry_num_list[i]-geometry_num_list[i-1]))
            phi = np.arccos(cos_phi)
            
            if 0 <= phi and phi <= np.pi/2:
                f_phi = 0.5 * (1 + np.cos(np.pi * cos_phi))
            else:
                f_phi = 1.0
            OM_action_force_perpendicularity, OM_action_force_parallelism = [], []
            
            force_perpendicularity, force_parallelism, force_parallelism_perpendicularity , swiching_double_neb_force = [], [], [], []
            
            if energy_list[i] == energy_list[local_max_energy_list_index[0]] and self.APPLY_CI_NEB < optimize_num: #CI-NEB
                for f in range(len(geometry_num_list[i])):
                    force_perpendicularity.append(np.array((-1)*(gradient_list[i][f]-2.0*(np.dot(gradient_list[i][f], tau[f]))*tau[f]), dtype = "float64"))
                    #print(str(force_perpendicularity))
                total_force = np.array(force_perpendicularity, dtype="float64")
                del local_max_energy_list_index[0]

            else:    
                for f in range(len(geometry_num_list[i])):
                    grad = 0.0
                    
                    for gg in range(len(gradient_list[i])):
                        grad += np.linalg.norm(gradient_list[i][gg], ord=2)
                    
                    grad = grad/len(gradient_list[i])
                    
                    spring_constant_k = 0.001
                    #print("spring_constant:",spring_constant_k)
                    
                    force_parallelism.append(np.array((spring_constant_k*(np.linalg.norm(geometry_num_list[i+1][f]-geometry_num_list[i][f], ord=2))+(-1.0)*spring_constant_k*(np.linalg.norm(geometry_num_list[i][f]-geometry_num_list[i-1][f], ord=2)))*tau[f], dtype = "float64"))
                    OM_action_force_parallelism.append(np.array(OM_action_force[f] * np.dot(tau[f], tau[f]), dtype = "float64"))
                
                    force_perpendicularity.append(np.array(gradient_list[i][f]-(np.dot(gradient_list[i][f], tau[f]))*tau[f], dtype = "float64"))
                    OM_action_force_perpendicularity.append(f_phi * np.array(OM_action_force[f]-(np.dot(OM_action_force[f], tau[f]))*tau[f], dtype = "float64"))
                    #doubly nudged elastic band method :https://doi.org/10.1063/1.1636455
                
                    force_parallelism_perpendicularity.append(np.array(np.array((spring_constant_k*(np.linalg.norm(geometry_num_list[i+1][f]-geometry_num_list[i][f], ord=2))+(-1.0)*spring_constant_k*(np.linalg.norm(geometry_num_list[i][f]-geometry_num_list[i-1][f], ord=2))), dtype = "float64") - (np.dot(np.array((spring_constant_k*(np.linalg.norm(geometry_num_list[i+1][f]-geometry_num_list[i][f], ord=2))+(-1.0)*spring_constant_k*(np.linalg.norm(geometry_num_list[i][f]-geometry_num_list[i-1][f], ord=2))), dtype = "float64"), tau[f]))*tau[f], dtype = "float64"))
                
                    swiching_double_neb_force.append((2.0/np.pi)*np.arctan(np.divide(np.linalg.norm(force_parallelism_perpendicularity[f], ord=2)**2, np.linalg.norm(force_parallelism_perpendicularity[f], ord=2)**2 ,out=np.zeros_like(force_parallelism_perpendicularity[f]) ,where=np.linalg.norm(force_parallelism_perpendicularity[f], ord=2)!=0))*(force_parallelism_perpendicularity[f] - np.dot(force_parallelism_perpendicularity[f],force_perpendicularity[f])*force_perpendicularity[f]))
            
                force_perpendicularity, force_parallelism, force_parallelism_perpendicularity, swiching_double_neb_force = np.array(force_perpendicularity, dtype = "float64"), np.array(force_parallelism, dtype = "float64"),np.array(force_parallelism_perpendicularity, dtype = "float64"), np.array(swiching_double_neb_force, dtype = "float64")
                
                OM_action_force_perpendicularity, OM_action_force_parallelism = np.array(OM_action_force_perpendicularity, dtype = "float64"), np.array(OM_action_force_parallelism, dtype = "float64")
                
                
                
                
                total_force = np.array((-1)*force_perpendicularity - force_parallelism - swiching_double_neb_force + OM_action_force_parallelism + OM_action_force_perpendicularity, dtype = "float64")
            
            if np.nanmean(np.nanmean(total_force)) > 10:
                total_force = total_force / np.nanmean(np.nanmean(total_force))
            
            total_force_list.append(total_force.tolist())

                
        
        total_force_list.append(((-1)*np.array(gradient_list[-1], dtype = "float64")).tolist())
        
        return np.array(total_force_list, dtype = "float64")



    def LUP_calc(self, geometry_num_list, energy_list, gradient_list, optimize_num):
        local_max_energy_list_index, local_min_energy_list_index = self.extremum_list_index(energy_list)

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
            #print("tau_minus:\n",tau_minus)
            #print("tau_plus:\n",tau_plus)
            #print("tau:\n",str(tau))
            force_perpendicularity, force_parallelism, force_parallelism_perpendicularity , swiching_double_neb_force = [], [], [], []
            
            if energy_list[i] == energy_list[local_max_energy_list_index[0]] and self.APPLY_CI_NEB < optimize_num: #CI-NEB
                for f in range(len(geometry_num_list[i])):
                    force_perpendicularity.append(np.array((-1)*(gradient_list[i][f]-2.0*(np.dot(gradient_list[i][f], tau[f]))*tau[f]), dtype = "float64"))
                    #print(str(force_perpendicularity))
                total_force = np.array(force_perpendicularity, dtype="float64")
                del local_max_energy_list_index[0]
                #print(str(total_force))
            #elif energy_list[i] == energy_list[local_min_energy_list_index[0]]: #for discovering intermidiate
            #    for f in range(len(geometry_num_list[i])):
            #        force_perpendicularity.append(np.array(((-1)*(gradient_list[i][f])), dtype = "float64"))
            #        #print(str(force_perpendicularity))
            #    total_force = np.array(force_perpendicularity, dtype="float64")
            #    del local_min_energy_list_index[0]
            else:    
                for f in range(len(geometry_num_list[i])):
                    grad = 0.0
                    
                    for gg in range(len(gradient_list[i])):
                        grad += np.linalg.norm(gradient_list[i][gg], ord=2)
                    
                    grad = grad/len(gradient_list[i])
                    
                    spring_constant_k = 0.001
                    #print("spring_constant:",spring_constant_k)
                    
                    force_parallelism.append(np.array((spring_constant_k*(np.linalg.norm(geometry_num_list[i+1][f]-geometry_num_list[i][f], ord=2))+(-1.0)*spring_constant_k*(np.linalg.norm(geometry_num_list[i][f]-geometry_num_list[i-1][f], ord=2)))*tau[f], dtype = "float64"))  
                
                    force_perpendicularity.append(np.array(gradient_list[i][f]-(np.dot(gradient_list[i][f], tau[f]))*tau[f], dtype = "float64"))
                    #doubly nudged elastic band method :https://doi.org/10.1063/1.1636455
                
                    force_parallelism_perpendicularity.append(np.array(np.array((spring_constant_k*(np.linalg.norm(geometry_num_list[i+1][f]-geometry_num_list[i][f], ord=2))+(-1.0)*spring_constant_k*(np.linalg.norm(geometry_num_list[i][f]-geometry_num_list[i-1][f], ord=2))), dtype = "float64") - (np.dot(np.array((spring_constant_k*(np.linalg.norm(geometry_num_list[i+1][f]-geometry_num_list[i][f], ord=2))+(-1.0)*spring_constant_k*(np.linalg.norm(geometry_num_list[i][f]-geometry_num_list[i-1][f], ord=2))), dtype = "float64"), tau[f]))*tau[f], dtype = "float64"))
                
                    swiching_double_neb_force.append((2.0/np.pi)*np.arctan(np.divide(np.linalg.norm(force_parallelism_perpendicularity[f], ord=2)**2, np.linalg.norm(force_parallelism_perpendicularity[f], ord=2)**2 ,out=np.zeros_like(force_parallelism_perpendicularity[f]) ,where=np.linalg.norm(force_parallelism_perpendicularity[f], ord=2)!=0))*(force_parallelism_perpendicularity[f] - np.dot(force_parallelism_perpendicularity[f],force_perpendicularity[f])*force_perpendicularity[f]))
            
                force_perpendicularity, force_parallelism, force_parallelism_perpendicularity, swiching_double_neb_force = np.array(force_perpendicularity, dtype = "float64"), np.array(force_parallelism, dtype = "float64"),np.array(force_parallelism_perpendicularity, dtype = "float64"), np.array(swiching_double_neb_force, dtype = "float64")
                total_force = np.array((-1)*force_perpendicularity - force_parallelism - swiching_double_neb_force, dtype = "float64")
            
            if np.nanmean(np.nanmean(total_force)) > 10:
                total_force = total_force / np.nanmean(np.nanmean(total_force))
            
            total_force_list.append(total_force.tolist())

                
        
        total_force_list.append(((-1)*np.array(gradient_list[-1], dtype = "float64")).tolist())
        
        return np.array(total_force_list, dtype = "float64")

     


    def FIRE_calc(self, geometry_num_list, total_force_list, pre_total_velocity, optimize_num, total_velocity, dt, n_reset, a, cos_list):
        velocity_neb = []
        
        for num, each_velocity in enumerate(total_velocity):
            part_velocity_neb = []
            for i in range(len(total_force_list[0])):
                 
                part_velocity_neb.append((1.0-a)*total_velocity[num][i]+a*np.sqrt(np.dot(total_velocity[num][i],total_velocity[num][i])/np.dot(total_force_list[num][i],total_force_list[num][i]))*total_force_list[num][i])
            velocity_neb.append(part_velocity_neb)
           
        
        velocity_neb = np.array(velocity_neb)
        
        np_dot_param = 0
        if optimize_num != 0:
            for num_1, total_force in enumerate(total_force_list):
                for num_2, total_force_num in enumerate(total_force):
                    np_dot_param += (np.dot(pre_total_velocity[num_1][num_2] ,total_force_num.T))
            print(np_dot_param)
        else:
            pass
        if optimize_num > 0 and np_dot_param > 0:
            if n_reset > self.FIRE_N_accelerate:
                dt = min(dt*self.FIRE_f_inc, self.FIRE_dt_max)
                a = a*self.FIRE_N_accelerate
            n_reset += 1
        else:
            velocity_neb = velocity_neb*0
            a = self.FIRE_a_start
            dt = dt*self.FIRE_f_decelerate
            n_reset = 0
        total_velocity = velocity_neb + dt*(total_force_list)
        if optimize_num != 0:
            total_delta = dt*(total_velocity+pre_total_velocity)
        else:
            total_delta = dt*(total_velocity)
       
        total_delta_average = np.nanmean(total_delta)
        print("total_delta_average:",str(total_delta_average))
        
        #---------------------
        move_vector = [total_delta[0]]
        trust_radii_1_list = []
        trust_radii_2_list = []
        
        for i in range(1, len(total_delta)-1):
            total_delta[i] *= (abs(cos_list[i]) ** 0.1 + 0.1)
            trust_radii_1 = np.linalg.norm(geometry_num_list[i] - geometry_num_list[i-1]) / 2.0
            trust_radii_2 = np.linalg.norm(geometry_num_list[i] - geometry_num_list[i+1]) / 2.0
            
            trust_radii_1_list.append(str(trust_radii_1*2))
            trust_radii_2_list.append(str(trust_radii_2*2))
            
            
            if np.linalg.norm(total_delta[i]) > trust_radii_1:
                move_vector.append(total_delta[i]*trust_radii_1/np.linalg.norm(total_delta[i]))
            elif np.linalg.norm(total_delta[i]) > trust_radii_2:
                move_vector.append(total_delta[i]*trust_radii_2/np.linalg.norm(total_delta[i]))
            else:
                move_vector.append(total_delta[i])
            
        with open(self.NEB_FOLDER_DIRECTORY+"Procrustes_distance_1.csv", "a") as f:
            f.write(",".join(trust_radii_1_list)+"\n")
        
        with open(self.NEB_FOLDER_DIRECTORY+"Procrustes_distance_2.csv", "a") as f:
            f.write(",".join(trust_radii_2_list)+"\n")
        
        move_vector.append(total_delta[-1])
        #--------------------
        
        
        new_geometory = (geometry_num_list + move_vector)*self.bohr2angstroms
         
        return new_geometory, dt, n_reset, a

    def SD_calc(self, geometry_num_list, total_force_list):
        total_delta = []
        delta = 0.5
        for i in range(len(total_force_list)):
            total_delta.append(delta*total_force_list[i])

        #---------------------
        move_vector = [total_delta[0]]
        
        for i in range(1, len(total_delta)-1):
            trust_radii_1 = np.linalg.norm(geometry_num_list[i] - geometry_num_list[i-1]) / 2.0
            trust_radii_2 = np.linalg.norm(geometry_num_list[i] - geometry_num_list[i+1]) / 2.0
            if np.linalg.norm(total_delta[i]) > trust_radii_1:
                move_vector.append(total_delta[i]*trust_radii_1/np.linalg.norm(total_delta[i]))
            elif np.linalg.norm(total_delta[i]) > trust_radii_2:
                move_vector.append(total_delta[i]*trust_radii_2/np.linalg.norm(total_delta[i]))
            else:
                move_vector.append(total_delta[i])
        move_vector.append(total_delta[-1])
        #--------------------
        new_geometory = (geometry_num_list + move_vector)*self.bohr2angstroms

        return new_geometory


    def main(self):
        
        geometry_list, element_list, electric_charge_and_multiplicity = self.make_geometry_list(self.start_folder, self.partition)
        file_directory = self.make_psi4_input_file(geometry_list,0)
        pre_total_velocity = [[[]]]
       
        #prepare for FIRE method 
        dt = 0.5
        n_reset = 0
        a = self.FIRE_a_start

        if args.usextb == "None":
            pass
        else:
            element_number_list = []
            for elem in element_list:
                element_number_list.append(element_number(elem))
            element_number_list = np.array(element_number_list, dtype="int")
        exit_flag = False
        for optimize_num in range(self.NEB_NUM):
            
            exit_file_detect = glob.glob(self.NEB_FOLDER_DIRECTORY+"*.txt")
            for file in exit_file_detect:
                if "end.txt" in file:
                    exit_flag = True
                    break
            if exit_flag:
                if psi4:
                    psi4.core.clean()
                
                break
            print("\n\n\nLUP:   "+str(optimize_num)+" ITR. \n\n\n")
            self.xyz_file_make(file_directory)
            #------------------
            if args.usextb == "None":
                energy_list, gradient_list, geometry_num_list, pre_total_velocity = self.psi4_calculation(file_directory,optimize_num, pre_total_velocity)
            else:
                energy_list, gradient_list, geometry_num_list, pre_total_velocity = self.tblite_calculation(file_directory, optimize_num,pre_total_velocity, element_number_list)
            #------------------
            if self.om:
                total_force = self.OM_calc(geometry_num_list, energy_list, gradient_list, optimize_num, element_list)
            else:
                total_force = self.LUP_calc(geometry_num_list, energy_list, gradient_list, optimize_num)
            #------------------
            cos_list = []
            for i in range(len(total_force)):
                cos = np.sum(total_force[i]*gradient_list[i])/(np.linalg.norm(total_force[i])*np.linalg.norm(gradient_list[i]))
                cos_list.append(cos)
            
            self.sinple_plot([x for x in range(len(total_force))], cos_list, file_directory, optimize_num, axis_name_1="NODE #", axis_name_2="cosθ", name="orthogonality")
            
            #------------------
            if optimize_num < self.sd:
                total_velocity = self.force2velocity(total_force, element_list)
                new_geometory, dt, n_reset, a = self.FIRE_calc(geometry_num_list, total_force, pre_total_velocity, optimize_num, total_velocity, dt, n_reset, a, cos_list)
                
            else:
                new_geometory = self.SD_calc(geometry_num_list, total_force)
            #------------------
            geometry_list = self.make_geometry_list_2(new_geometory, element_list, electric_charge_and_multiplicity)
            file_directory = self.make_psi4_input_file(geometry_list, optimize_num+1)
 
            pre_total_velocity = total_velocity
            #------------------
            with open(self.NEB_FOLDER_DIRECTORY+"energy_plot.csv", "a") as f:
                f.write(",".join(list(map(str,energy_list.tolist())))+"\n")
            
            #------------------
    
        
        print("\n\n\nLUP: final\n\n\n")
        self.xyz_file_make(file_directory) 
        if args.usextb == "None":
            energy_list, gradient_list, geometry_num_list, pre_total_velocity = self.psi4_calculation(file_directory, optimize_num, pre_total_velocity)
        else:
            energy_list, gradient_list, geometry_num_list, pre_total_velocity = self.tblite_calculation(file_directory, optimize_num,pre_total_velocity, element_number_list)
            
        geometry_list = self.make_geometry_list_2(new_geometory, element_list, electric_charge_and_multiplicity)
        energy_list = energy_list.tolist()
        
        print("Complete...")
        return

if __name__ == "__main__":
    args = parser()
    l = LUP(args)
    l.main()
