import sys
sys.path.append('..')
import FillGrid
import cv2
import time


if __name__ == '__main__':
    img = cv2.imread('istockphoto-456878070-640_adpp_is_314.jpg')
    CARS = [[212, 272], [214, 159], [74, 391], [149, 48]]
    ROAD = [[[2, 27], [2, 65], [47, 67], [52, 630], [101, 630], [92, 68], [191, 68], [190, 630], [342, 630], [342, 609], [248, 591], [248, 40], [298, 38], [300, 2]]]
    SLOTS = [[144, 234, 190, 263], [147, 365, 190, 396]]
    H = 360
    W = 640
    N = 80
    M = 80

    start = time.time()
    FillGrid.test_convert(img, 40, 60, export_img = False)

    grid = FillGrid.fill_grid(CARS, ROAD, SLOTS, H, W, N, M)

    # export to image: test.jpg
    img = FillGrid.draw(img, grid, H, W, N, M, slot=True, car=True, road=True)
    end = time.time()
    print(f'time = {end-start}')
    cv2.imwrite('test.jpg', img)

    # # export to text: 1.inp
    # slotsCell = FillGrid.convert_slotsPoint_slotsCell(SLOTS, H, W, N, M)
    # carsCell = FillGrid.convert_carsPoint_carsCell(CARS, H, W, N, M)
    # with open("1.inp", 'w') as outfile:
    #     outfile.write(f'{N} {M}\n')
    #     for row in grid:
    #         line = ' '.join([str(v) for v in row])
    #         line = line + '\n'
    #         outfile.write(line)

    #     outfile.write(f'{len(slotsCell)}\n')
    #     for slot in slotsCell:
    #         outfile.write(f'{slot[0]} {slot[1]} {slot[2]} {slot[3]}\n')
      