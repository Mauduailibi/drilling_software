import numpy as np
import sympy as sp


def signal_change(s)->tuple: #return arrays with the signals of the value, -1 for negative and 1 for positive
    a = None
    codition = None
    cs = None
    for i in range(len(s)):
        if i == 0 :
            if s[i] > 0 :
                codition = True
            else: 
                codition = False
        else:
            if s[i] > 0 :
                a = True
                if codition != a:
                    value = 'Yes'
                    cs = i
                    break
                else:
                    value = 'No'
            else: 
                a = False
                if codition != a:
                    value = 'Yes'
                    cs = i
                    break
                else:
                    value = 'No'
    return (value, cs)

def theta( Data, l1, R ) ->  float: # Estimated final angle in the curve
    l1 = l1
    x = sp.Symbol('x')
    x3 , y3 = Data.P3
    l3 = np.sqrt( ((x3- R)**2) + (y3-l1)**2 - (R**2) )
    a1 = (2*np.arctan((R - np.sqrt(R**2 - l1**2 + 2*l1*y3 + l3**2 - y3**2))/(-l1 + l3 + y3)))
    a2 = (2*np.arctan((R + np.sqrt(R**2 - l1**2 + 2*l1*y3 + l3**2 - y3**2))/(-l1 + l3 + y3)))
    if R == x3:
         a = a1
    else:
        f1 = ( -R*sp.cos(x) + l3*sp.sin(x) )
        f2 = ( l3*sp.cos(x) + R*sp.sin(x) )
        if (np.sign((f1.evalf(subs={x: a1})))) == (np.sign((x3-R))) and (np.sign((f2.evalf(subs={x: a1})))) == (np.sign(((y3-l1)))):
            a = a1
        else:
            a = a2
    return float(a)

def lenght( Data, l1,R )  -> list: # Returns l1,l2 and l3 lengths from the each part

    l3 = np.sqrt( ((Data.P3[0]-R)**2) + (Data.P3[1]-l1)**2 - (R**2) )

    angle  = theta(Data, l1,R)

    l2 = (2*np.pi*R) * (((angle)*(180/np.pi))/360)

    return l1,l2,l3

def curve_points( Data,l1,R ) -> list:  # Estimated points in the curve part
    angle = np.linspace(np.pi, np.pi + theta(Data, l1, R),1000)
    l1, *_ = lenght(Data,l1,R)
    y = l1 -  R * np.sin(angle)
    x = R * (np.cos(angle)) + R
    return x,y

def points_coordinates( Data,l1,R) -> list : # Estimated points in well
    x,y = [Data.P0[0], 0],[Data.P0[1],l1]

    points = curve_points(Data,l1,R)

    for i in range(len(points[0])):
            x.append(points[0][i])
            y.append(points[1][i])
    x.append(Data.P3[0]),y.append(Data.P3[1])
    
    return x,y 

def buckling( Data, l1, R ) -> float:

    PSB = Data.z
    
    alpha =  1 - ((Data.ro_fluid) / (Data.ro_command))
    ws = Data.lambd_command

    lc = round((PSB*1.2) / (ws*alpha*np.cos(theta(Data, l1, R))))
    
    codition =  True

    while codition == True:

        if (lc % 9) >= 5 and (lc % 9) != 0:
            lc += 1
        elif (lc % 9) < 5 and (lc % 9) != 0:
            lc -= 1
        else:
            codition = False

    return lc
    
def Nl(Data, l1, R) -> float:
    lc = buckling(Data, l1, R)
    angle = theta(Data, l1, R)
    Y_F = Data.P3[1]
    ro_fluid = Data.ro_fluid
    ro_command = Data.ro_command
    ro_heavypipe = Data.ro_heavypipe
    ro_drillpipe = Data.ro_drillpipe
    g = Data.g
    lambd_command = Data.lambd_command
    lambd_heavy = Data.lambd_heavy
    lambd_drill = Data.lambd_drill
    d_ext_command = Data.d_ext_command
    d_ext_heavy = Data.d_ext_heavy
    d_ext_drill = Data.d_ext_drill
    d_int_command = Data.d_int_command
    d_int_heavy = Data.d_int_heavy
    d_int_drill = Data.d_int_drill
    z = Data.z
    µ = Data.µ
    lp = Data.lp

    ff = (ro_fluid * g * Y_F * (np.pi * (d_ext_command**2 - d_int_command**2) / 4))

    neutral_line = (z + ff) / (lambd_command * g * (np.cos(angle) - µ * (1 - (ro_fluid / ro_command)) * np.sin(angle)))

    if neutral_line < lc:  # Neutral line is in command
        return neutral_line

    if neutral_line > lc:  # Neutral line is in heavy pipe
        Y_ic = Y_F - (lc * np.cos(angle))
        fic = ( ro_fluid * g * Y_ic * np.pi * ( (d_ext_command**2 - d_ext_heavy**2) - (d_int_command**2 - d_int_heavy**2) ) )    / 4

        A = (ff + z - fic) / (lambd_heavy * g * (np.cos(angle) - µ * (1 - (ro_fluid / ro_heavypipe)) * np.sin(angle)))
        B = lc * ((lambd_command - lambd_heavy) * np.cos(angle) - (µ * ((lambd_command * (1 - (ro_fluid / ro_command))) - (lambd_heavy * (1 - (ro_fluid / ro_heavypipe))))) * np.sin(angle))
        C = lambd_heavy * (np.cos(angle) - µ * (1 - (ro_fluid / ro_heavypipe)) * np.sin(angle))

        neutral_line = A - (B / C)

        if neutral_line > (lc + lp):  # Neutral line is in drill
            Y_ip = Y_F - ((lc + lp) * np.cos(angle))
            fip = (ro_fluid * g * Y_ip * np.pi *( (d_ext_heavy**2 - d_ext_drill**2) - (d_int_heavy**2 - d_int_drill**2 )    ) )   / 4

            A = (ff + z - fic - fip) / (lambd_drill * g * (np.cos(angle) - µ * (1 - (ro_fluid / ro_drillpipe)) * np.sin(angle)))
            B = lc * ((lambd_command - lambd_drill) * np.cos(angle) - (µ * ((lambd_command * (1 - (ro_fluid / ro_command))) - (lambd_drill * (1 - (ro_fluid / ro_drillpipe))))) * np.sin(angle))
            C = lambd_drill * (np.cos(angle) - µ * (1 - (ro_fluid / ro_drillpipe)) * np.sin(angle))
            D = lp * ((lambd_heavy - lambd_drill) * np.cos(angle) - (µ * ((lambd_heavy * (1 - (ro_fluid / ro_heavypipe))) - (lambd_drill * (1 - (ro_fluid / ro_drillpipe))))) * np.sin(angle))

            neutral_line = A - (B / C) - (D / C)

    return neutral_line


def up_tension(Data, l1, R) -> list:  # Estimated tensions in upping
    lc = buckling(Data, l1, R)
    l1, l2, l3 = lenght(Data, l1, R)
    angle = theta(Data, l1, R)
    g = Data.g
    ro_fluid = Data.ro_fluid
    ro_drillpipe = Data.ro_drillpipe
    lambd_drill = Data.lambd_drill
    lambd_command = Data.lambd_command
    lambd_heavy = Data.lambd_heavy
    d_ext_command = Data.d_ext_command
    d_ext_heavy = Data.d_ext_heavy
    d_ext_drill = Data.d_ext_drill
    d_int_command = Data.d_int_command
    d_int_heavy = Data.d_int_heavy
    d_int_drill = Data.d_int_drill
    µ = Data.µ
    lp = Data.lp
    P3 = Data.P3
    area_drill = Data.area_drill
    area_command = Data.area_command
    area_heavy = Data.area_heavy

    weight_3 = ((lambd_command * lc) + (lambd_heavy * lp) + (lambd_drill * (l3 - lc - lp))) * g
    buoyancy_3 = (((area_drill * (l3 - lp - lc) * ro_fluid) + (area_command * lc * ro_fluid) + (area_heavy * lp * ro_fluid)) * g)

    Y_F = P3[1]
    Y_ic = Y_F - (lc * np.cos(angle))
    Y_ip = Y_F - ((lc + lp) * np.cos(angle))
    fic = ( ro_fluid * g * Y_ic * np.pi * ( (d_ext_command**2 - d_ext_heavy**2) - (d_int_command**2 - d_int_heavy**2) ) )    / 4
    fip = (ro_fluid * g * Y_ip * np.pi *( (d_ext_heavy**2 - d_ext_drill**2) - (d_int_heavy**2 - d_int_drill**2 )    ) )   / 4
    ff = (ro_fluid * g * Y_F * (np.pi * (d_ext_command**2 - d_int_command**2) / 4))

    tension_3 = weight_3 * np.cos(angle) + µ * (weight_3 - buoyancy_3) * np.sin(angle) + fic + fip - ff

    angle_variation = np.arange(0.001, angle, 0.001)[::-1]

    DNS = []

    for i in range(len(angle_variation)):
        DN = tension_3 * angle_variation[i] - (1 - (ro_fluid / ro_drillpipe)) * g * lambd_drill * R * np.sin(angle_variation[i]) * angle_variation[i]
        DNS.append(DN)

    signals = np.sign(DNS)

    coditions = signal_change(signals)

    condition_signal_change = coditions[0]
    condition_where_change = coditions[1]

    if condition_signal_change == "No":
        if DNS[0] > 0:  # DN POSTIVO SEM TROCA DE SINAL
            A = lambd_drill * g * R
            C1 = (A / (1 + µ**2)) * (((µ**2) * (1 - (ro_fluid / ro_drillpipe))) - 1)
            C2 = -((A * µ) / (1 + µ**2)) * (2 - (ro_fluid / ro_drillpipe))

            K = (tension_3 - C1 * np.sin(angle) - C2 * np.cos(angle)) / (np.exp(-µ * angle))

            tension_2 = C1 * np.sin(0) + C2 * np.cos(0) + K * (np.exp(-µ * 0))

        else:  # DN NEGATIVO SEM TROCA DE SINAL
            A = lambd_drill * g * R
            B1 = (A / (1 + µ**2)) * (((µ**2) * (1 - (ro_fluid / ro_drillpipe))) - 1)
            B2 = ((A * µ) / (1 + µ**2)) * (2 - (ro_fluid / ro_drillpipe))
            K = (tension_3 - B1 * np.sin(angle) - B2 * np.cos(angle)) / (np.exp(µ * angle))

            tension_2 = B1 * np.sin(0) + B2 * np.cos(0) + K * (np.exp(µ * 0))

    if condition_signal_change == "Yes":
        A = lambd_drill * g * R
        B1 = (A / (1 + µ**2)) * (((µ**2) * (1 - (ro_fluid / ro_drillpipe))) - 1)
        B2 = ((A * µ) / (1 + µ**2)) * (2 - (ro_fluid / ro_drillpipe))
        K = (tension_3 - B1 * np.sin(angle) - B2 * np.cos(angle)) / (np.exp(µ * angle))

        tension_change = B1 * np.sin(angle_variation[condition_where_change]) + B2 * np.cos(angle_variation[condition_where_change]) + K * (np.exp(µ * angle_variation[condition_where_change]))

        C1 = (A / (1 + µ**2)) * (((µ**2) * (1 - (ro_fluid / ro_drillpipe))) - 1)
        C2 = -((A * µ) / (1 + µ**2)) * (2 - (ro_fluid / ro_drillpipe))

        K = (tension_change - C1 * np.sin(angle) - C2 * np.cos(angle)) / (np.exp(-µ * angle))

        tension_2 = C1 * np.sin(0) + C2 * np.cos(0) + K * (np.exp(-µ * 0))

    tension_1 = tension_2 + (lambd_drill * l1) * g

    return tension_1, tension_2, tension_3


def down_tension(Data, l1, R) -> list:  # Estimated tensions in descent
    lc = buckling(Data, l1, R)
    l1, l2, l3 = lenght(Data, l1, R)
    angle = theta(Data, l1, R)
    ld = l3 - Data.lp - lc
    g = Data.g
    ro_fluid = Data.ro_fluid
    ro_drillpipe = Data.ro_drillpipe
    lambd_drill = Data.lambd_drill
    lambd_command = Data.lambd_command
    lambd_heavy = Data.lambd_heavy
    d_ext_command = Data.d_ext_command
    d_ext_heavy = Data.d_ext_heavy
    d_ext_drill = Data.d_ext_drill
    d_int_command = Data.d_int_command
    µ = Data.µ
    lp = Data.lp
    P3 = Data.P3
    area_drill = Data.area_drill
    area_command = Data.area_command
    area_heavy = Data.area_heavy
    d_int_heavy = Data.d_int_heavy
    d_int_drill = Data.d_int_drill
    z = Data.z

    weight_3 = ((lambd_command * lc) + (lambd_heavy * lp) + (lambd_drill * (l3 - lc - lp))) * g
    buoyancy_3 = (((area_drill * (l3 - lp - lc) * ro_fluid) + (area_command * lc * ro_fluid) + (area_heavy * lp * ro_fluid)) * g)

    Y_F = P3[1]
    Y_ic = Y_F - (lc * np.cos(angle))
    Y_ip = Y_F - ((lc + lp) * np.cos(angle))
    ff = (ro_fluid * g * Y_F * (np.pi * (d_ext_command**2 - d_int_command**2) / 4))
    fic = ( ro_fluid * g * Y_ic * np.pi * ( (d_ext_command**2 - d_ext_heavy**2) - (d_int_command**2 - d_int_heavy**2) ) )    / 4
    fip = (ro_fluid * g * Y_ip * np.pi *( (d_ext_heavy**2 - d_ext_drill**2) - (d_int_heavy**2 - d_int_drill**2 )    ) )   / 4

    tension_3 = weight_3 * (np.cos(angle) - µ * np.sin(angle)) + (µ * buoyancy_3 * np.sin(angle)) + fic + fip - z - ff

    angle_variation = np.arange(0.01, angle, 0.01)[::-1]

    DNS = []

    for i in range(len(angle_variation)):
        DN = tension_3 * angle_variation[i] - (1 - (ro_fluid / ro_drillpipe)) * g * lambd_drill * R * np.sin(angle_variation[i]) * angle_variation[i]
        DNS.append(DN)

    signals = np.sign(DNS)

    coditions = signal_change(signals)

    condition_signal_change = coditions[0]
    condition_where_change = coditions[1]

    if condition_signal_change == "No":
        if DNS[0] > 0:  # DN POSTIVO SEM TROCA DE SINAL
            A = lambd_drill * g * R
            B1 = (A / (1 + µ**2)) * (((µ**2) * (1 - (ro_fluid / ro_drillpipe))) - 1)
            B2 = ((A * µ) / (1 + µ**2)) * (2 - (ro_fluid / ro_drillpipe))
            K = (tension_3 - B1 * np.sin(angle) - B2 * np.cos(angle)) / (np.exp(µ * angle))

            tension_2 = B1 * np.sin(0) + B2 * np.cos(0) + K * (np.exp(µ * 0))

            B3 = B1 - ((1 - (ro_fluid / ro_drillpipe)) * lambd_drill * g * R)

            medial_N_force = B3 * (1 - np.cos(angle)) + B2 * np.sin(angle) + (K / µ) * ((np.exp(µ * angle)) - 1)

            frictional_torque_2 = (µ * medial_N_force * d_ext_drill) / 2

            fat3_command = µ * (1 - (ro_fluid / Data.ro_command)) * lambd_command * lc * g * np.sin(angle)
            fat3_heavy = µ * (1 - (ro_fluid / Data.ro_heavypipe)) * lambd_heavy * lp * g * np.sin(angle)
            fat3_drill = µ * (1 - (ro_fluid / ro_drillpipe)) * lambd_drill * ld * g * np.sin(angle)

            torque3_command = (d_ext_command / 2) * fat3_command
            torque3_heavy = (d_ext_heavy / 2) * fat3_heavy
            torque3_drill = (d_ext_drill / 2) * fat3_drill

            frictional_torque_3 = torque3_command + torque3_heavy + torque3_drill

            torque = frictional_torque_2 + frictional_torque_3

        else:  # DN NEGATIVO SEM TROCA DE SINAL
            A = lambd_drill * g * R
            C1 = (A / (1 + µ**2)) * (((µ**2) * (1 - (ro_fluid / ro_drillpipe))) - 1)
            C2 = -((A * µ) / (1 + µ**2)) * (2 - (ro_fluid / ro_drillpipe))
            K = (tension_3 - C1 * np.sin(angle) - C2 * np.cos(angle)) / (np.exp(-µ * angle))

            tension_2 = C1 * np.sin(0) + C2 * np.cos(0) + K * (np.exp(-µ * 0))

            C3 = C1 - ((1 - (ro_fluid / ro_drillpipe)) * lambd_drill * g * R)

            medial_N_force = C3 * (1 - np.cos(angle)) + C2 * np.sin(angle) - (K / µ) * ((np.exp(-µ * angle)) - 1)

            frictional_torque_2 = -((µ * medial_N_force * d_ext_drill) / 2)

            fat3_command = µ * (1 - (ro_fluid / Data.ro_command)) * lambd_command * lc * g * np.sin(angle)
            fat3_heavy = µ * (1 - (ro_fluid / Data.ro_heavypipe)) * lambd_heavy * lp * g * np.sin(angle)
            fat3_drill = µ * (1 - (ro_fluid / ro_drillpipe)) * lambd_drill * ld * g * np.sin(angle)

            torque3_command = (d_ext_command / 2) * fat3_command
            torque3_heavy = (d_ext_heavy / 2) * fat3_heavy
            torque3_drill = (d_ext_drill / 2) * fat3_drill

            frictional_torque_3 = torque3_command + torque3_heavy + torque3_drill

            torque = frictional_torque_2 + frictional_torque_3

    elif condition_signal_change == "Yes":  # negative - positive
        A = lambd_drill * g * R
        C1 = (A / (1 + µ**2)) * (((µ**2) * (1 - (ro_fluid / ro_drillpipe))) - 1)
        C2 = -((A * µ) / (1 + µ**2)) * (2 - (ro_fluid / ro_drillpipe))
        K = (tension_3 - C1 * np.sin(angle) - C2 * np.cos(angle)) / (np.exp(-µ * angle))

        C3 = C1 - ((1 - (ro_fluid / ro_drillpipe)) * lambd_drill * g * R)

        medial_N_force = -C3 * (np.cos(angle_variation[condition_where_change]) - np.cos(angle)) + C2 * (np.sin(angle_variation[condition_where_change]) - np.sin(angle)) - (K / µ) * (np.exp(-µ * angle_variation[condition_where_change]) - np.exp(-µ * angle))

        frictional_torque_2_change = -((µ * medial_N_force * d_ext_drill) / 2)

        tension_change = C1 * np.sin(angle_variation[condition_where_change]) + C2 * np.cos(angle_variation[condition_where_change]) + K * (np.exp(-µ * (angle_variation[condition_where_change])))

        B1 = (A / (1 + µ**2)) * (((µ**2) * (1 - (ro_fluid / ro_drillpipe))) - 1)
        B2 = ((A * µ) / (1 + µ**2)) * (2 - (ro_fluid / ro_drillpipe))
        K = (tension_change - B1 * np.sin(angle_variation[condition_where_change]) - B2 * np.cos(angle_variation[condition_where_change])) / (np.exp(µ * angle_variation[condition_where_change]))

        B3 = B1 - ((1 - (ro_fluid / ro_drillpipe)) * lambd_drill * g * R)

        medial_N_force = B3 * (1 - np.cos(angle_variation[condition_where_change])) + B2 * np.sin(angle_variation[condition_where_change]) + (K / µ) * ((np.exp(µ * (angle_variation[condition_where_change]))) - 1)

        frictional_torque_2 = ((µ * medial_N_force * d_ext_drill) / 2) + frictional_torque_2_change

        fat3_command = µ * (1 - (ro_fluid / Data.ro_command)) * lambd_command * lc * g * np.sin(angle)
        fat3_heavy = µ * (1 - (ro_fluid / Data.ro_heavypipe)) * lambd_heavy * lp * g * np.sin(angle)
        fat3_drill = µ * (1 - (ro_fluid / ro_drillpipe)) * lambd_drill * ld * g * np.sin(angle)

        torque3_command = (d_ext_command / 2) * fat3_command
        torque3_heavy = (d_ext_heavy / 2) * fat3_heavy
        torque3_drill = (d_ext_drill / 2) * fat3_drill

        frictional_torque_3 = torque3_command + torque3_heavy + torque3_drill

        torque = frictional_torque_2 + frictional_torque_3

        tension_2 = B1 * np.sin(0) + B2 * np.cos(0) + K * (np.exp(µ * 0))

    tension_1 = tension_2 + (lambd_drill * l1) * g

    return tension_1, tension_2, tension_3, torque
