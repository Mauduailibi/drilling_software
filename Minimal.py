import Auxiliaries as ax
import matplotlib.pyplot as plt
import numpy as np
from Data_base import DataSet
import pandas as pd


def minimal_tension(Data) -> list:
    t1_data = []
    l1_data = []
    R_data = []

    l1 = 100
    condition =  False 
    while condition == False:

        t1_data_R = []
        R_valor = []

        for R in np.arange(Data.min_radius,Data.max_radius+50,50):
            if ax.theta(Data,l1,R)*(180/np.pi) > 52:
                pass
            else: 
                up_t1,*_ = ax.up_tension(Data,l1,R)
                t1_data_R.append(up_t1)
                R_valor.append(R)
        if len(t1_data_R) != 0 :
            key_r = np.where(t1_data_R == min(t1_data_R))[0][0]
            t1_data.append(t1_data_R[key_r])
            l1_data.append(l1)
            R_data.append(R_valor[key_r])

        if round(l1,1) == Data.max:
            condition = True

        l1 += 10

    key = np.where(t1_data == min(t1_data))[0][0]
    l1_value = l1_data[key]
    R_value = R_data[key]

    return [l1_value,R_value]

def minimal_torque(Data) -> list:
    torque_data = []
    l1_data = []
    R_data = []

    l1 = 100

    condition =  False 

    while condition == False:

        torque_data_R = []
        R_valor = []

        for R in np.arange(Data.min_radius,Data.max_radius+50,50):
            if ax.theta(Data,l1,R)*(180/np.pi) > 52:
                pass
            else: 
                t1,t2,t3,torque = ax.down_tension(Data,l1,R)
                torque_data_R.append(torque)
                R_valor.append(R)

        if len(torque_data_R) != 0 :
            key_r = np.where(torque_data_R == min(torque_data_R))[0][0]
            torque_data.append(torque_data_R[key_r])
            l1_data.append(l1)
            R_data.append(R_valor[key_r])

        if round(l1,1) == Data.max:
            condition = True

        l1 += 10

    key = np.where(torque_data == min(torque_data))[0][0]
    l1_value = l1_data[key]
    R_value = R_data[key]

    return [l1_value,R_value]

def drilling_informations(Data)->list:

    l1,R = minimal_tension(Data)
    up_tensions = ax.up_tension(Data,l1,R)
    down_tensions = ax.down_tension(Data,l1,R)
    angle = ax.theta(Data,l1,R)*180/np.pi
    neutral_line = ax.Nl(Data,l1,R)
    lenghts = ax.lenght(Data,l1,R)
    lenght_command = ax.buckling(Data,l1,R)

    tension_results = [up_tensions,down_tensions,angle,neutral_line,lenghts,lenght_command,l1,R]

    l1,R = minimal_torque(Data)
    up_tensions = ax.up_tension(Data,l1,R)
    down_tensions = ax.down_tension(Data,l1,R)
    angle = ax.theta(Data,l1,R)*180/np.pi
    neutral_line = ax.Nl(Data,l1,R)
    lenghts = ax.lenght(Data,l1,R)
    lenght_command = ax.buckling(Data,l1,R)

    torque_results = [up_tensions,down_tensions,angle,neutral_line,lenghts,lenght_command,l1,R]

    return [tension_results,torque_results]

def drilling_informations_table(data):
    results = drilling_informations(data)
    for i in range(len(results)):
        up_tensions,down_tensions,angle,neutral_line,lenghts,lenght_command,l1,R = results[i]
        l1,l2,l3 = lenghts
        t1_up ,t2_up,t3_up = up_tensions
        t1_down,t2_down,t3_down,torque = down_tensions
        data = [l1,l2,l3,R,t1_up,t2_up,t3_up,t1_down,t2_down,t3_down,torque,angle,neutral_line,lenght_command ]
        data = np.round(data,2)
        index_labels = [
            "L1 (m):", "L2 (m):", "L3 (m):","radius (m):",
            "Up tension L1 (N/m²):", "Up tension L2 (N/m²):", "Up tension L3 (N/m²):",
            "Down tension L1 (N/m²):", "Down tension L2 (N/m²):", "Down tension L3 (N/m²):",
            "Torque (Nm):", "Angle (°):", "Neutral line (m):", "Length command (m):"]
        
        if i == 0:
            print(f"\n--- Result table for minimal tension ---")
        else:
            print(f"\n--- Result table for minimal torque ---")
        a = pd.DataFrame(data,columns=[''],index=index_labels)
        print(a)
        print('')


    pass

def drilling_draw(Data) -> None :

    l1, R = minimal_tension(Data)
    x1, y1 = ax.points_coordinates(Data, l1, R)
    l2, R2 = minimal_torque(Data)
    x2, y2 = ax.points_coordinates(Data, l2, R2)


    custom_style = {
        'font.size': 16,  # Tamanho adequado para leitura de gráficos
        'axes.labelsize': 18,  # Tamanho dos rótulos dos eixos
        'axes.titlesize': 14,  # Tamanho do título do gráfico
        'axes.linewidth': 1.5,  # Espessura das bordas dos gráficos
        'xtick.labelsize': 18,  # Tamanho do texto dos ticks no eixo x
        'ytick.labelsize': 18,  # Tamanho do texto dos ticks no eixo y
        'lines.linewidth': 2,  # Espessura das linhas dos gráficos
        'lines.markersize': 6,  # Tamanho dos marcadores
        'legend.fontsize': 18,  # Tamanho da legenda
        'legend.frameon': False,  # Remove a moldura ao redor da legenda
        'legend.loc': 'best',  # Melhor posição automática para a legenda
        'figure.figsize': (4, 3),  # Tamanho padrão da figura (polegadas)
        'savefig.dpi': 1000,  # Alta resolução para exportação (publicação)
        'savefig.bbox': 'tight',  # Salva a imagem sem cortar parte do gráfico
        }

    plt.rcParams.update(custom_style)

    plt.axes().set_aspect('equal')
    plt.plot(x1, y1, color="blue")
    plt.gca().invert_yaxis()
    plt.title("Type 1 well trajectory for the minimal tension")
    plt.xlabel("Distance ($m$)")
    plt.ylabel("Depth ($m$)")
    plt.grid(alpha=1,linewidth=2)
    plt.margins(x=0.1, y=0.1)
    plt.show()

    plt.axes().set_aspect('equal')
    plt.plot(x2, y2, color="red", linestyle="--")
    plt.gca().invert_yaxis()
    plt.title("Type 1 well trajectory for the minimal Torque")
    plt.xlabel("Distance ($m$)")
    plt.ylabel("Depth ($m$)")
    plt.grid(alpha=1,linewidth=2)
    plt.margins(x=0.1, y=0.1)
    plt.show()


    pass

def tension_in_radius(Data,l1) -> list:

    torque_data = [] 
    t1_data_R = []
    R_valor = []

    for R in np.arange(Data.min_radius,Data.max_radius+50,50):
        if ax.theta(Data,l1,R)*(180/np.pi) > 52:
            pass
        else: 
            up_t1,*_ = ax.up_tension(Data,l1,R)
            t1,t2,t3,torque = ax.down_tension(Data,l1,R)
            t1_data_R.append(up_t1)
            torque_data.append(torque)
            R_valor.append(R)


    custom_style = {
        'font.size': 16,  # Tamanho adequado para leitura de gráficos
        'axes.labelsize': 18,  # Tamanho dos rótulos dos eixos
        'axes.titlesize': 14,  # Tamanho do título do gráfico
        'axes.linewidth': 1.5,  # Espessura das bordas dos gráficos
        'xtick.labelsize': 18,  # Tamanho do texto dos ticks no eixo x
        'ytick.labelsize': 18,  # Tamanho do texto dos ticks no eixo y
        'lines.linewidth': 2,  # Espessura das linhas dos gráficos
        'lines.markersize': 6,  # Tamanho dos marcadores
        'legend.fontsize': 18,  # Tamanho da legenda
        'legend.frameon': False,  # Remove a moldura ao redor da legenda
        'legend.loc': 'best',  # Melhor posição automática para a legenda
        'figure.figsize': (4, 3),  # Tamanho padrão da figura (polegadas)
        'savefig.dpi': 1000,  # Alta resolução para exportação (publicação)
        'savefig.bbox': 'tight',  # Salva a imagem sem cortar parte do gráfico
        }

    plt.rcParams.update(custom_style)

    plt.plot(R_valor,t1_data_R, label=f'Tension in section of ${l1}m$', color="blue")
    plt.title("Tension behavior across the radius")
    plt.xlabel("Radius ($m$)")
    plt.ylabel("Tension ($N/m²$)")
    plt.legend()
    plt.grid(alpha=1,linewidth=2)
    plt.margins(x=0.1, y=0.1)
    plt.show()


    plt.plot(R_valor,torque_data, label=f'Torque in section of ${l1}m$', color="red", linestyle="--")
    plt.title("Torque behavior across the radius")
    plt.xlabel("Radius ($m$)")
    plt.ylabel("Torque ($N*m$)")
    plt.legend()
    plt.grid(alpha=1,linewidth=2)
    plt.margins(x=0.1, y=0.1)
    plt.show()
    
    pass

def tension_in_section1(Data,R) -> list:
    t1_data = []
    l1_data = []
    torque_data = []
    l1 = 100
    condition =  False 
    while condition == False:
        if ax.theta(Data,l1,R)*(180/np.pi) > 52:
            pass
        else:
            up_t1,*_ = ax.up_tension(Data,l1,R)
            t1,t2,t3,torque = ax.down_tension(Data,l1,R)
            t1_data.append(up_t1)
            l1_data.append(l1)
            torque_data.append(torque)

        if round(l1,1) == Data.max:
            condition = True
        else:
            l1 += 10

    custom_style = {
        'font.size': 16,  # Tamanho adequado para leitura de gráficos
        'axes.labelsize': 18,  # Tamanho dos rótulos dos eixos
        'axes.titlesize': 14,  # Tamanho do título do gráfico
        'axes.linewidth': 1.5,  # Espessura das bordas dos gráficos
        'xtick.labelsize': 18,  # Tamanho do texto dos ticks no eixo x
        'ytick.labelsize': 18,  # Tamanho do texto dos ticks no eixo y
        'lines.linewidth': 2,  # Espessura das linhas dos gráficos
        'lines.markersize': 6,  # Tamanho dos marcadores
        'legend.fontsize': 18,  # Tamanho da legenda
        'legend.frameon': False,  # Remove a moldura ao redor da legenda
        'legend.loc': 'best',  # Melhor posição automática para a legenda
        'figure.figsize': (4, 3),  # Tamanho padrão da figura (polegadas)
        'savefig.dpi': 1000,  # Alta resolução para exportação (publicação)
        'savefig.bbox': 'tight',  # Salva a imagem sem cortar parte do gráfico
        }


    plt.rcParams.update(custom_style)

    plt.plot(l1_data,t1_data, label=f'Tension with radius of ${R}m$', color="blue")
    plt.title("Tension behavior over the length of section one ")
    plt.xlabel("Length ($m$)")
    plt.ylabel("Tension ($N/m²$)")
    plt.legend()
    plt.grid(alpha=1,linewidth=2)
    plt.margins(x=0.1, y=0.1)
    plt.show()

    
    plt.plot(l1_data,torque_data, label=f'Torque with radius of ${R}m$', color="red", linestyle="--")
    plt.title("Torque behavior over the length of section one")
    plt.xlabel("Length ($m$)")
    plt.ylabel("Torque ($N*m$)")
    plt.legend()
    plt.grid(alpha=1,linewidth=2)
    plt.margins(x=0.1, y=0.1)
    plt.show()


    pass


def tension_graphic(Data) -> list:
    t1_data = []
    l1_data = []
    R_data = []
    torque_data = []

    l1 = 100

    condition =  False 

    while condition == False:

        t1_data_R = []
        R_valor = []
        torque_data_R = []

        for R in np.arange(Data.min_radius,Data.max_radius+50,50):
            if ax.theta(Data,l1,R)*(180/np.pi) > 52:
                pass
            else: 
                up_t1,*_ = ax.up_tension(Data,l1,R)
                t1,t2,t3,torque = ax.down_tension(Data,l1,R)
                t1_data_R.append(up_t1)
                torque_data_R.append(torque)
                R_valor.append(R)

        if len(t1_data_R) != 0 :
            key_r = np.where(t1_data_R == min(t1_data_R))[0][0]
            key_r1 = np.where(torque_data_R == min(torque_data_R))[0][0]
            t1_data.append(t1_data_R[key_r])
            l1_data.append(l1)
            R_data.append(R_valor[key_r])
            torque_data.append(torque_data_R[key_r1])

        if round(l1,1) == Data.max:
            condition = True

        l1 += 10

        custom_style = {
            'font.size': 16,  # Tamanho adequado para leitura de gráficos
            'axes.labelsize': 18,  # Tamanho dos rótulos dos eixos
            'axes.titlesize': 14,  # Tamanho do título do gráfico
            'axes.linewidth': 1.5,  # Espessura das bordas dos gráficos
            'xtick.labelsize': 18,  # Tamanho do texto dos ticks no eixo x
            'ytick.labelsize': 18,  # Tamanho do texto dos ticks no eixo y
            'lines.linewidth': 2,  # Espessura das linhas dos gráficos
            'lines.markersize': 6,  # Tamanho dos marcadores
            'legend.fontsize': 18,  # Tamanho da legenda
            'legend.frameon': False,  # Remove a moldura ao redor da legenda
            'legend.loc': 'best',  # Melhor posição automática para a legenda
            'figure.figsize': (4, 3),  # Tamanho padrão da figura (polegadas)
            'savefig.dpi': 1000,  # Alta resolução para exportação (publicação)
            'savefig.bbox': 'tight',  # Salva a imagem sem cortar parte do gráfico
            }


    plt.rcParams.update(custom_style)

    plt.figure(figsize=(12, 6)) 
    plt.plot(l1_data,t1_data, label="Tension", color="blue",)
    plt.title("Tension behavior depending on the best configurations for section 1", fontsize=10)
    plt.xlabel("Length ($m$)")
    plt.ylabel("Tension ($N/m²$)")
    plt.legend()
    plt.margins(x=0.1, y=0.1)
    plt.grid(alpha=1,linewidth=2)
    plt.show()


    plt.plot(l1_data,torque_data, label="Torque", color="red", linestyle="--")
    plt.title("Torque behavior depending on the best configurations for section 1", fontsize=10)
    plt.xlabel("Length ($m$)")
    plt.ylabel("Torque ($N*m$)")
    plt.legend()
    plt.margins(x=0.1, y=0.1)
    plt.grid(alpha=1,linewidth=2)
    plt.show()

    pass