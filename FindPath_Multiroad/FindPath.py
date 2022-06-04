from FindPath_Multiroad.FillGrid import *
from FindPath_Multiroad import FillGrid
from FindPath_Multiroad.FindPathCplusplus import BFS 
import cv2 
import math
import time

def findPath(CARS, ROAD, SLOTS, H, W, N, M):
    # start_time = time.time() 
    grid = FillGrid.fill_grid(CARS, ROAD, SLOTS, H, W, N, M)
    # end_time = time.time()
    # print(f'Fill Grid time ={end_time - start_time} second')
    slotsCell = FillGrid.convert_slotsPoint_slotsCell(SLOTS, H, W, N, M)
    # start_time = time.time() 
    findPath = BFS(N, M, grid, slotsCell)
    # end_time = time.time()
    # print(f'Find Path time ={end_time - start_time} second')
    pathsCell = findPath.shortestPath()
    pathsPoint = []
    for path in pathsCell:
        pathPoint = []
        for point in path:
            x = point//M;
            y = point%M;
            pathPoint.append(FillGrid.convert_cell_to_point(x, y, H, W, N, M))
        pathsPoint.append(pathPoint)
    return pathsPoint        
# Result = pathsPoint

colors_arr=[(255,0,0),(0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (255,255,255), (0,0,0)]

def draw_path(paths, CARS, ROAD, SLOTS, H, W, N, M, img):
    grid = FillGrid.fill_grid(CARS, ROAD, SLOTS, H, W, N, M)
    img = FillGrid.draw(img, grid, H, W, N, M)
    i = 0
    #rd_num=np.random.randint(8)
    for path in paths:
        for point in path:
            h, w = point
            img = cv2.circle(img, (int(w), int(h)), radius=2, color=(0,255,0), thickness=-1)
    #cv2.imwrite('test.jpg', img)
    return img


if __name__ == '__main__':
    CARS = [[74, 391], [212, 272], [214, 159], [149, 48]]
    ROAD = [[[2, 27], [2, 65], [47, 67], [52, 630], [101, 630], [92, 68]], [[191, 68], [190, 630], [342, 630], [342, 609], [248, 591], [248, 40], [298, 38], [300, 2]]]
    SLOTS = [[144, 234, 190, 263], [147, 365, 190, 396]]
    H = 360
    W = 640
    N = 80
    M = 80
    start_time = time.time() 
    paths = findPath(CARS, ROAD, SLOTS, H, W, N, M)
    end_time = time.time()
    print("paths: ", paths)
    print(f'Total time ={end_time - start_time} second')

    #draw(paths, CARS, ROAD, SLOTS, H, W, N, M)
