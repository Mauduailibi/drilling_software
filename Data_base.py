
import numpy as np

class DataSet:

    def __init__(self,
                P0 : tuple[float,float], # X and Y coordenates from the initial point
                P3 : tuple[float,float], # X and Y coordenates objective
                ro_fluid: float, #Ro of the fluid
                ro_command: float, #Ro of the command pipe
                ro_drillpipe: float, #Ro of the drill pipe
                ro_heavypipe: float, #Ro of the heavy pipe
                diameters_command: tuple[float,float], # Informations about the command pipe [De,Di]
                diameters_drillpipe: tuple[float,float], # Informations about the drill pipe [De,Di]
                diameters_heavypipe: tuple[float,float], # Informations about the heavy pipe [De,Di]
                lp : float, # len of heavypipe
                µ: float, #Coefficient of friction
                z: float, # Rock reaction 
                max: float,
                radius: tuple[float,float],
                ) -> None:
        area_command = ( np.pi/4)*(diameters_command[0]**2  - diameters_command[1]**2)
        area_drill =(np.pi/4)*(diameters_drillpipe[0]**2  - diameters_drillpipe[1]**2)
        area_heavy =(np.pi/4)*(diameters_heavypipe[0]**2  - diameters_heavypipe[1]**2)
        self.P0 = P0 
        self.P3 = P3
        self.g = 9.81
        self.ro_fluid = ro_fluid #kg / m^3 
        self.ro_command = ro_command #kg / m^3 
        self.ro_drillpipe = ro_drillpipe #kg / m^3 
        self.ro_heavypipe = ro_heavypipe #kg / m^3 
        self.µ = µ
        self.z = z
        self.area_command = area_command
        self.area_drill = area_drill
        self.area_heavy = area_heavy
        self.lambd_command = ro_command * area_command
        self.lambd_drill = ro_drillpipe * area_drill
        self.lambd_heavy = ro_heavypipe * area_heavy
        self.d_ext_drill = diameters_drillpipe[0]
        self.d_int_drill = diameters_drillpipe[1]
        self.d_ext_command = diameters_command[0]
        self.d_int_command = diameters_command[1]
        self.d_ext_heavy = diameters_heavypipe[0]
        self.d_int_heavy = diameters_heavypipe[1]
        self.lp = lp
        self.max = max # m
        self.min_radius = radius[0]
        self.max_radius = radius[1]


        pass




class lithology:

    def sandstone(self,
            rop: int,
            ):
        self.rop = rop
        return None
    
    def limestone(self,
        rop: int,
        ):
        self.rop  = rop
        return None

    def dolomite(self,
        rop: int,
        ):
        self.rop = rop
        return None
    
    def evaporite(self,
        rop: int,
        ):
        self.rop = rop
        return None


class mesh:

    def __init__(self,
            sandstone : list, #  [[0,100],[400,500]]
            limestone : list, #  [100,200]
            dolomite : list, # [200,300]
            evaporite : list, # [300,400] 

            ): 
        

         

        while True:

            mesh_vector = []

            positions = []


            mesh_lithology = []

            i = 0
            for element in sandstone:
                soma = np.sum(element)
                dx = element[1] - element[0]
                print(soma)
                sandstone.pop(i)
                i+=1
            
            i = 0
            for element in limestone:
                soma = np.sum(element)
                dx = element[1] - element[0]
                print(soma)
                mesh_vector.append(dx)
                positions.append('Limestone')
                limestone.pop(i)
                i+= 1
        
            i= 0
            for element in dolomite:
                soma = np.sum(element)
                print(soma)
                dolomite.pop(i)
                i+= 1
            
            i = 0
            for element in evaporite:
                soma = np.sum(element)
                print(soma)
                evaporite.pop(i)
                i+= 1

            if len(sandstone) == 0 and len(limestone) == 0 and len(dolomite) == 0 and len(evaporite) == 0:
                break

        
        
        # mesh_dict = dict()
        
        # i_sandstone = 0
        # i_limestone = 0
        # i_dolomite = 0
        # i_evaporite = 0

        # for element in sandstone:

        #     length = element[1] - element[0]

        #     mesh_dict[f'sandstone {i_sandstone}'] = length
        #     i_sandstone += 1

        # for element in limestone:
        #     length = element[1] - element[0]
        #     mesh_dict[f'limestone {i_limestone}'] = length
        #     i_limestone += 1

        # for element in dolomite:
        #     length = element[1] - element[0]
        #     mesh_dict[f'dolomite {i_dolomite}'] = length
        #     i_dolomite += 1
        
        # for element in evaporite:
        #     length = element[1] - element[0]
        #     mesh_dict[f'evaporite {i_evaporite}'] = length
        #     i_evaporite += 1
        
                      
        # print(mesh_dict)
                

        return None
        