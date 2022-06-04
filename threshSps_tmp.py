import numpy as np

list_x=np.array([])#np.array([[]])
list_y=np.array([[70,110],[200,250]])


from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt

list_poly=[]
list_poly_parking=[]
list_gates=[]

#coords =[(0,0),(0,262),(500,250),(500,200),(25,200),(25,106),(550,110),(550,66),(25,56),(25,0),(0,0)]
#coords=[(0,0),(0,80), (169,107),(200,0)]
coords =[(0,0),(0,241),(580,241),(580,295),(640,295),(640,197),(60,197),(60,97),(640,97),(640,59),(60,54), (60,0), (0,0)]
poly = Polygon(coords)

#Nhung poly co the do xe trong bai
coords_parking_1=  Polygon([(61,0),(61,50),(640,58),(640,0)])
coords_parking_2=Polygon([(58,104),(65,191),(640,195), (640,104)])
coords_parking_3=Polygon([(0,304), (35,304),(35,250),(574,250),(580,349),(0,357)])



#polygon cua gate (moi gate co 2 polygon)
gate_1_poly_1 = Polygon([(0,0), (0,16), (59,16), (59,0)])
gate_1_poly_2= Polygon([(0,17), (0,30), (59,30), (59,17)])

list_gates.append([gate_1_poly_1, gate_1_poly_2])

#print(list_gates)



#list cua nhung poly do dc xe
list_poly_parking.append(coords_parking_1)
list_poly_parking.append(coords_parking_2)
list_poly_parking.append(coords_parking_3)

#list cua nhung poly duong ma xe di duoc
list_poly.append(poly)

