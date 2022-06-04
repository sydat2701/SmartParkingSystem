import cv2
import numpy as np
import math
# from shapely.geometry import Point, Polygon
from FindPath_Multiroad.FillRoadCplusplus import convert_point_to_cell, convert_cell_to_point, fill_road
import cv2
import numpy as np
import math
from shapely.geometry import Point, Polygon

# # Convert point in image (HxW) to cell in grid (NxM)
# def convert_point_to_cell(h_coordinate, w_coordinate, H, W, N=80, M=80) :
#     delta_h = H/N
#     delta_w = W/M
#     # print(f'delta_h = {delta_h}, delta_w = {delta_w}')
#     x = math.ceil(h_coordinate / delta_h) - 1
#     y = math.ceil(w_coordinate / delta_w) - 1

#     return (x, y)
# def convert_cell_to_point(x, y, H, W, N=80, M=80):
#     x += 1
#     y += 1
#     delta_h = H/N
#     delta_w = W/M
#     h_coordinate = delta_h*(x + x - 1)/2
#     w_coordinate = delta_w*(y + y - 1)/2

#     return (h_coordinate, w_coordinate)
def test_convert(img, N, M, export_img = True):
    H, W, _ = img.shape
    h_coordinate, w_coordinate = (H//2, W//2)
    img = cv2.circle(img, (int(w_coordinate), int(h_coordinate)), radius=5, color=(0,0,255), thickness=-1)

    H, W, _ = img.shape
    x, y = convert_point_to_cell(h_coordinate, w_coordinate, H, W, N, M)
    print(f'x = {x}, y = {y}')
    h_conv, w_conv = convert_cell_to_point(x, y, H, W, N, M)
    print(f'h_conv = {h_conv}, w_conv = {w_conv}')
    img = cv2.circle(img, (int(w_conv), int(h_conv)), radius=5, color=(0,255,0), thickness=-1)
    if export_img:
        cv2.imwrite('test.jpg', img)


# Fill the ROAD
UNBLOCKED = 0 # const value, not change
BLOCKED = -1 # const value, not change
# def fill_road(road, H, W, N, M, grid=None):
#     if grid is None:
#         grid = np.full([N, M], BLOCKED)
#     for sub_road in road:
#         polygon = Polygon(sub_road)
#         for x in range(N):
#             for y in range(M):
#                 h_coordinate, w_coordinate = convert_cell_to_point(x, y, H, W, N, M)
#                 point = Point(h_coordinate, w_coordinate)
#                 if point.within(polygon):
#                     grid[x][y] = UNBLOCKED
#     return grid


# Fill all slots
TARGET = 1
def convert_slotsPoint_slotsCell(slotsPoint, H, W, N, M, add_x=0, add_y=0):
    slotsCell = []
    for slotPoint in slotsPoint:
        up_left_cell = convert_point_to_cell(slotPoint[0], slotPoint[1], H, W, N, M)
        up_left_cell = [max(0, up_left_cell[0] - add_x), max(up_left_cell[1] - add_y, 0)]

        down_right_cell = convert_point_to_cell(slotPoint[2], slotPoint[3], H, W, N, M)
        down_right_cell = [min(N-1, down_right_cell[0] + add_x), min(M-1, down_right_cell[1] + add_y)]
        slotsCell.append([up_left_cell[0], up_left_cell[1], down_right_cell[0], down_right_cell[1]])
    return slotsCell
def fill_slots(slotsPoint, H, W, N, M, grid=None, add_x=0, add_y=0):
    if grid is None:
        grid = np.full([N, M], BLOCKED)
    slotsCell = convert_slotsPoint_slotsCell(slotsPoint, H, W, N, M, add_x, add_y)
    for slot in slotsCell:
        for x in range(slot[0], slot[2] + 1):
            for y in range(slot[1], slot[3] + 1):
                grid[x][y] = TARGET
    return grid


# Fill all cars
SOURCE_LOWERBOUND = 2 
# CARS = [[212, 272], [214, 159], [74, 391], [149, 48]]
def convert_carsPoint_carsCell(carsPoint, H, W, N, M):
    carsCell = []
    for car in carsPoint:
        x, y = convert_point_to_cell(car[0], car[1], H, W, N, M)
        carsCell.append([x, y])
    return carsCell
def fill_cars(cars, H, W, N, M, grid=None):
    if grid is None:
        grid = np.full([N, M], BLOCKED)
    carsCell = convert_carsPoint_carsCell(cars, H, W, N, M)
    for i in range(len(carsCell)):
        car = carsCell[i]
        grid[car[0]][car[1]] = SOURCE_LOWERBOUND + i
    return grid

# Fill cars, slots, road
def fill_grid(CARS, ROAD, SLOTS, H, W, N, M):
    grid = np.full([N, M], int(BLOCKED))
    grid = fill_road(ROAD, H, W, N, M, grid)
    grid = fill_slots(SLOTS, H, W, N, M, grid=grid)
    grid = fill_cars(CARS, H, W, N, M, grid=grid)
    return grid

def draw(img, grid, H, W, N, M, slot=False, car=True, road=False):
    for x in range(N):
        for y in range(M):
            if grid[x][y] == TARGET and slot == True:
                h_conv, w_conv = convert_cell_to_point(x, y, H, W, N, M)
                img = cv2.circle(img, (int(w_conv), int(h_conv)), radius=math.ceil(min(H/N, W/M)), color=(0,0,255), thickness=-1)
            if grid[x][y] == UNBLOCKED and road == True:
                h_conv, w_conv = convert_cell_to_point(x, y, H, W, N, M)
                img = cv2.circle(img, (int(w_conv), int(h_conv)), radius=math.ceil(min(H/N, W/M)), color=(0,255,0), thickness=-1)
            if grid[x][y] >= SOURCE_LOWERBOUND and car == True:
                h_conv, w_conv = convert_cell_to_point(x, y, H, W, N, M)
                img = cv2.circle(img, (int(w_conv), int(h_conv)), radius=math.ceil(min(H/N, W/M)), color=(255,0,0), thickness=-1)
    return img


