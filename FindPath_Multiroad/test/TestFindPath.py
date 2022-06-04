import sys
sys.path.append('..')
import FindPath
import cv2
import math
import time

if __name__ == '__main__':
    img = cv2.imread('istockphoto-456878070-640_adpp_is_314.jpg')
    CARS = [[212, 272], [149, 48], [214, 159], [74, 391]]
    ROAD = [[[2, 27], [2, 65], [47, 67], [52, 630], [101, 630], [92, 68], [191, 68], [190, 630], [342, 630], [342, 609], [248, 591], [248, 40], [298, 38], [300, 2]]]
    SLOTS = [[144, 234, 190, 263], [147, 365, 190, 396]]
    H = 360
    W = 640
    N = 80
    M = 80
    start_time = time.time() 
    paths = FindPath.findPath(CARS, ROAD, SLOTS, H, W, N, M)
    end_time = time.time()
    print(f'Total time ={end_time - start_time} second')

    FindPath.draw_path(paths, CARS, ROAD, SLOTS, H, W, N, M, img)
    cv2.imwrite('test.jpg', img)