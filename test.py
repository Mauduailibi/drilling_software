from Minimal import *

from Data_base import mesh


Data = DataSet(P0 = (0, 0), # X , Y
                P3 = (1000, 3000),  # X , Y
                ro_fluid = 1737.5 , #kg / m^3 
                ro_command = 8000, #kg / m^3 
                ro_drillpipe = 8000, #kg / m^3 
                ro_heavypipe = 8000, #kg / m^3 
                diameters_command = (0.2032, 0.1143), # m 8 e 2.813
                diameters_drillpipe = (0.127,  0.1086104), # m 5 e 4.276
                diameters_heavypipe = (0.1524,  0.1143), # m 6 e 4
                µ = 0.23, # 
                z = (5000*8) * 4.44822 , # Newton
                lp = 36, # m 
                max = 2300,
                radius= (100,600)
                )

Mesh = mesh(
    sandstone =[[0,100],[400,500]],
    dolomite =[[100,200]],
    evaporite =[[200,300]],
    limestone =[[300,400]],
    )

l1,r = minimal_tension(Data)
drilling_informations_table(Data)
drilling_draw(Data)
tension_graphic(Data)

#l1_1,r_1 = minimal_torque(Data)
#tension_in_radius(Data,l1_1)
#tension_in_section1(Data,r_1)
#tension_in_radius(Data,l1_1)
#tension_in_section1(Data,r_1)

