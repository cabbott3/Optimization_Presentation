# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 20:48:19 2018

@author: cabbott3
"""
#import pip
#pip.main(['install','gekko'])
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gekko import GEKKO

#%% Data Import 
excel_file = 'Facebook_data2.xlsx'
data = pd.read_excel(excel_file)
v_data = data['v'] #m/s
rho_data = data['rho'] #kg/m3
t_air_data = data['t_air'] #K
re_data = data['re']
time_data = data['time'] #seconds
motor_data = data['p_n'] #W 
h_data = data['h'] #height (m)
charge_data = data['e_batt'] #MJ
mu_data = data['mu']
motor_efficiency_data = data['eta_motor']
motor_voltage_data = data['motor_voltage'] #V
motor_current_data = data['motor_current'] #A


#%% Compiling Data

#Selecting Data from large data set
n = len(data)-1  #Number of data points to calculate.
shift = 1  #Shift data that is selected by this many (max data point = 8635)
Optimal_temp = 298 #K (Assumed optimal battery temperature)
space = 6 #only include every (space)th data point
data_pieces = int(n/space)-2

#Storage for loop below
v = np.zeros(data_pieces) #velocity storage
rho = np.zeros(data_pieces) #air density storage
t_air = np.zeros(data_pieces) #ambient temperature storage
re2 = np.zeros(data_pieces) #reynolds number storage
time = np.zeros(data_pieces) #time storage 
motor = np.zeros(data_pieces) #motor power storage
motor[0] = motor_data[0+shift] #motor power initial value
T_SP = np.ones(data_pieces)*Optimal_temp #Creating an array of optimal temperatures (K)
mu = np.zeros(data_pieces) #viscosity storage
motor_efficiency = np.zeros(data_pieces)
motor_voltage = np.zeros(data_pieces)
motor_current = np.zeros(data_pieces)


#Loop to select data from large data set
for d in range(data_pieces):
    v[d] = float(v_data[d*space+shift])
    rho[d] = float(rho_data[d*space+shift])
    t_air[d] = float(t_air_data[d*space+shift])
    re2[d] = float(re_data[d*space+shift])
    time[d] = float(time_data[d*space+shift]-time_data[0+shift]) #makes the first set of data start at t=0
    motor[d] = float(motor_data[d*space+shift])
    mu[d] = float(mu_data[d*space+shift])
    motor_efficiency[d] = float(motor_efficiency_data[d*space+shift])
    motor_voltage[d] = float(motor_voltage_data[d*space+shift])
    motor_current[d] = float(motor_current_data[d*space+shift])
    


#%% Gekko model

m = GEKKO(server='http://byu.apmonitor.com')
m.time = time 

#%% Constants, Variables, Intermediates, Equations, Settings

#Constants
Optimal_Temp = m.Const(value = Optimal_temp,name='Optimal_Temp')
MW = m.Const(value=0.0289644,name='MW') #kg/mol
Cp_air = m.Const(value=975.4,name='Cp_air') #J/kgK (found using HYSYS calculations)
kf_air = m.Const(value=.01855,name='kf_air') #W/mK (found using HYSYS calculations)
mass_batt = m.Const(value=142/4,name='mass_batt') #kg
Cp_batt = m.Const(value=900,name='Cp_batt') #J/kgK
kf_batt = m.Const(value=6,name='kf_batt') #W/mK
radius_batt = m.Const(value=.5/2,name='radius_batt') #m (Estimation based on photographs)
length_total = m.Const(value=2,name='length_total') #m (Estimation based on photographs)
length_motor = m.Const(value=.5,name='length_motor') #m (Guess)
V_open_circuit = m.Const(value=44,name='V_open_circuit') #Volts
mu2 = m.Param(value=mu,name='mu2')

#Intermediates
length_batt = m.Intermediate(length_total - length_motor,name='length_batt') #m
Pr = m.Intermediate(Cp_air*mu2/kf_air,name='Pr')


################ Variables only discretized against time ###################### 


#Constants and values for convective heat transfer
motor_power = m.Param(value=motor,name='motor_power') #W (Not used)
t_air2 = m.Param(value=t_air,name='t_air2') #K
T_SP2 = m.Param(value=T_SP,name='T_SP2') #K
re = m.Param(value=re2,name='re')
eff_motor = m.Param(value=motor_efficiency,name='eff_motor')
I_load = m.Param(value=motor_current,name='I_load') #Amps
V_load = m.Param(value=motor_voltage,name='V_load') #Volts

R_batt = m.Const(value=.02,name='R_batt') #ohms (typical internal resistance for AA bateries)


######################### Spatial Dependent Variables #########################


############ CHANGE ME ###############

discretize = 10 #Number of discretizations spatially
thick_insulation = m.Const(value=0,name='thick_insulation') #m

######################################


thick_motor_insulation = m.Const(value=.01,name='thick_motor_insulation') #m
thick_carbon = m.Const(value=.0004*2,name='thick_carbon')        #m
num_slices = m.Const(value=discretize,name='num_slices') #Inputting the number of slices into Gekko model
kf_insulation = m.Const(value=.03,name='kf_insulation') #W/mK
kf_carbon = m.Const(value=100.5,name='kf_carbon') #W/mK
Cp_motor = m.Const(value=900,name='Cp_motor') #J/kgK
Cp_insulation = m.Const(value=1331,name='Cp_insulation') #J/kgK Calculated at 298K
Cp_carbon = m.Const(value=1130,name='Cp_carbon') #J/kgK
dens_insulation = m.Const(value=.205,name='dens_insulation') #kg/m3
dens_carbon = m.Const(value=1.800,name='dens_carbon') #kg/m3

length = m.Intermediate(length_batt/num_slices,name='length') #Length of slice
Nu = m.Intermediate( .0296*re**(4/5)*Pr**(1/3) ,name='Nu') #Nusselt number correlation
h = m.Intermediate( Nu*kf_air/length_batt ,name='h') #convective heat transfer coefficient
A1_inner = m.Intermediate( np.pi*2*radius_batt*length_motor+np.pi*radius_batt**2 ,name='A1_inner') #Surface area of motor casing touching air
A2_inner = m.Intermediate( np.pi*radius_batt**2 ,name='A2_inner') #Surface area of slice
A3_inner = m.Intermediate( np.pi*2*radius_batt*length ,name='A3_inner') #Surface area of slice touching air
A4_inner = m.Intermediate( np.pi*2*radius_batt*length+np.pi*radius_batt**2 ,name='A4_inner') #Surface area of last slice touching air
A1_mid = m.Intermediate( np.pi*2*(radius_batt+thick_insulation)*length_motor,name='A1_mid') #Surface area of motor touching insulation above
A2_mid = m.Intermediate( np.pi*(radius_batt+thick_insulation)**2-np.pi*radius_batt**2,name='A2_mid') #surface area of insulation touching insulation 
A3_mid = m.Intermediate( np.pi*2*(radius_batt+thick_insulation)*length,name='A3_mid') # surface area of insulation touching carbon
A4_mid = m.Intermediate( np.pi*2*(radius_batt+thick_insulation)*length+np.pi*(radius_batt+thick_insulation)**2,name='A4_mid') 
A1_out = m.Intermediate( np.pi*2*(radius_batt+thick_insulation+thick_carbon)*length_motor+(np.pi*(radius_batt+thick_insulation+thick_carbon)**2-np.pi*(radius_batt+thick_insulation)**2),name='A1_out')
A2_out = m.Intermediate( np.pi*(radius_batt+thick_insulation+thick_carbon)**2-np.pi*(radius_batt+thick_insulation)**2,name='A2_out') #Surface area of carbon touching carbon
A3_out = m.Intermediate( np.pi*2*(radius_batt+thick_insulation+thick_carbon)*length,name='A3_out') #Surface area of carbon touching air for each slice
A4_out = m.Intermediate( np.pi*2*(radius_batt+thick_insulation+thick_carbon)*length+np.pi*(radius_batt+thick_insulation+thick_carbon)**2+np.pi*(radius_batt+thick_insulation)**2,name='A4_out') #Surface area of carbon touching air for end piece
D1 = m.Intermediate( 2*radius_batt) #Diameter of the battery
D2 = m.Intermediate( 2*radius_batt+2*thick_insulation) #diameter of the insulation
D3 = m.Intermediate( 2*radius_batt+2*thick_insulation+2*thick_carbon) #diameter of the whole housing
#D4 = m.Intermediate( D2-2*thick_motor_insulation) #Diameter of the insulation around the motor (only used in U_motor calculations)
U_slice = m.Intermediate( 1/(m.log(D2/D1)/(2*np.pi*kf_insulation*length_batt/num_slices)+m.log(D3/D2)/(2*np.pi*kf_carbon*length_batt/num_slices)+1/(h*A3_out)))
#U_motor = m.Intermediate( 1/(m.log(D3/D2)/(2*np.pi*kf_carbon*length_motor)+1/(h*A1_out)+m.log(D2/D4)/(2*np.pi*kf_insulation*length_motor)))
U_end = m.Intermediate( 1/(m.log(D2/D1)/(2*np.pi*kf_insulation*length_batt/num_slices)+m.log(D3/D2)/(2*np.pi*kf_carbon*length_batt/num_slices)+1/(h*A4_out)))

Q_heater = m.Const(value=0,name='Q_heater') #Watts


############################# Equation Section ################################


T = [m.SV(value=298,name='T_'+str(i)) for i in range(discretize+1)] #Create a temperature variable for (discretize) number of sections +1 for the motor
    
for j in range(discretize+1):
    if j == 0:
        # Motor (Section 0)
        m.Equation( (mass_batt/4*Cp_motor)*T[j].dt() ==  motor_power*(1-eff_motor) #Heat generation (From Motor) 
                                                                 -kf_batt*A2_inner*(T[j]-T[j+1])  #Conduction to next slice
                                                                 -h*A1_inner*(T[j]-t_air2)) #Convection from motor 
#                                                                 -U_motor*(T[j]-t_air2))       #Convection from slice through carbon                                
    elif j > 0 and j < discretize-1:
        # Battery slice (Section 1 --> Section n-1)
        m.Equation( (mass_batt*Cp_batt)/num_slices*T[j].dt() == kf_batt*A2_inner*(T[j-1]-T[j]) #Conduction from previous slice
                                                                + I_load**2*R_batt/num_slices #Heat generation (Joule heating)
                                                                -kf_batt*A2_inner*(T[j]-T[j+1]) #Conduction to next slice
                                                                -U_slice*(T[j]-t_air2) #Conduction through both insulation and carbon with convection on outside
                                                                +Q_heater/(num_slices)) #Heat from external coil 
    else:
        # Battery (Last section) 
        m.Equation( (mass_batt*Cp_batt)/num_slices*T[j].dt() == kf_batt*A2_inner*(T[j-1]-T[j]) #Conduction from previous slice
                                                                + I_load**2*R_batt/num_slices #Heat generation (Joule heating)
                                                                -U_end*(T[j]-t_air2) #Conduction through both insulation and carbon with convection on outside
                                                                +Q_heater/(num_slices)) #Heat from external coil
                                                                

############################ Global Options ###################################


m.options.imode = 4
m.options.nodes = 3
m.options.solver = 3
#m.options.EV_Type = 1

m.solve() #Solve simulation

#Load results
sol = m.load_results()
solData = pd.DataFrame.from_dict(sol)

#%% Plot results
plt.figure(figsize=(7,7))
plt.title('Battery Temperature Profile')
plt.ylabel('Temperature (C)')
plt.xlabel('Time (hr)')
for i in range(discretize+1):
    plt.plot(solData.time/3600, solData['t_'+str(i)]-273.15,label='T_'+str(i))
if discretize < 20:
    plt.legend(["T_"+str(q) for q in range(discretize+1)])

