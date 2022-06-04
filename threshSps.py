import numpy as np

list_x=np.array([])#np.array([[]])
list_y=np.array([[70,110],[200,250]])


from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt

gate_file_path='./coords_tool/gate_area.txt'
travel_file_path ='./coords_tool/travel_area.txt'
parking_file_path='./coords_tool/parking_area.txt'

list_poly=[]
list_poly_parking=[]
list_gates=[]

list1=[]
list2=[]
list3=[]

#------------------------------------------------------------------------------------------------
f1=open(travel_file_path, "r")
line=f1.readlines()[0][1:-1]+', '
arr=line.split('], ')[:-1]

for elm in arr:
	one_polygon=[]
	one_polygon1=[]#---------------
	elm=elm[1:]+','
	arr_tmp=elm.split('),')[:-1]
	for x in arr_tmp:
		x=x.strip()
		x=x[1:]
		arr_num=x.split(', ')
		#print(arr_num)
		num1=int(arr_num[0])
		num2=int(arr_num[1])
		one_polygon.append((num1, num2))
		one_polygon1.append([num1, num2])#---------------------
	#print("one:" ,one_polygon)
	#break
	list_poly.append(Polygon(one_polygon))
	list1.append(one_polygon1)#------------------
	#list1_check.append(one_polygon1)

f1.close()

#-------------------------------------
f2=open(parking_file_path, "r")
line=f2.readlines()[0][1:-1]+', '
arr=line.split('], ')[:-1]

for elm in arr:
	one_polygon=[]
	one_polygon1=[]#--------------------
	elm=elm[1:]+','
	arr_tmp=elm.split('),')[:-1]
	for x in arr_tmp:
		x=x.strip()
		x=x[1:]
		arr_num=x.split(', ')
		#print(arr_num)
		num1=int(arr_num[0])
		num2=int(arr_num[1])
		one_polygon.append((num1, num2))
		one_polygon1.append([num1, num2])#-------------------
	#print("one:" ,one_polygon)
	#break
	list_poly_parking.append(Polygon(one_polygon))
	list2.append(one_polygon1)#------------------
	#list2_check.append(one_polygon1)

f2.close()

#------------------------------------

f3=open(gate_file_path, "r")

line=f3.readlines()[0][1:-1]+', '
# print("++++++++++++++++++", line)
arr=line.split(']], ')[:-1]
#print("1:",arr)

for elm in arr:
	one_gate=[]
	one1=[]
	elm=elm[1:]+'],'
	arr_tmp=elm.split('],')[:-1]
	#print("2:", arr_tmp)
	for xx in arr_tmp:	#[(46, 234), (85, 236), (83, 365), (44, 362)
		
		xx=xx.strip()
		#print("XX: ", xx)
		one_polygon=[]
		one_polygon1=[]#-------------------
		
		xx=xx[1:]+', '
		arr_tmp_tmp=xx.split('), ')[:-1]	#['(46, 234', '(85, 236', '(83, 365', '(44, 362']
		for x in arr_tmp_tmp:
			# print("3:", arr_tmp_tmp)
			# exit()
			#print("^^^^", x)
			x=x.strip()
			x=x[1:]
			arr_num=x.split(', ')
			#print(arr_num)
			num1=int(arr_num[0])
			num2=int(arr_num[1])
			one_polygon.append((num1, num2))
			one_polygon1.append([num1, num2])#----------------
		one_gate.append(Polygon(one_polygon))
		one1.append(one_polygon1)#----------------
	list_gates.append(one_gate)
	list3.append(one1)#------------------
	#list3_check.append(one1)
#print("**************", list_gates)

f3.close()




from hyperParams import *
if xy_inverse==True:
	for rd in list1:
		for vertex in rd:
			tg=vertex[0]
			vertex[0]=vertex[1]
			vertex[1]=tg
	for rd in list2:
		for vertex in rd:
			tg=vertex[0]
			vertex[0]=vertex[1]
			vertex[1]=tg

print("--------------------------------------------------------------------------------------------------")
print("travel: ", list1)
print("parking: ", list2)
print("gates: ", list3)
print("--------------------------------------------------------------------------------------------------")