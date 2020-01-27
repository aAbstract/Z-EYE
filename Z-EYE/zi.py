import cv2
import re
from zi_lib import grid
from zi_lib import pca


# Defenitions
cam = cv2.VideoCapture(2)  # Camera Handler Object
on = 40  # Outter Frame Diagonal Pixels Optimized: Values [40,50] | Accuracy [10,20,30]
bth = 30  # Black Value Threshold in Gray Scale
rc = .1  # Variance Ratio
avg_mine = 100  # Average Detectable Mine Area
color1 = (0, 0, 255)  # Color Red
color2 = (255, 255, 255)  # Color White

# Option Switch
print('1 -> Training Mode')
print('2 -> Testing Mode')
print('3 -> Action Mode')
print('4 -> Update Data Set')
ans = input('Mode: ')
if ans == 1:
    c = 0
    # Training Loop (First Time)
    while (True):
        r, f = cam.read()
        f = cv2.resize(f, dsize=(480, 480), interpolation=cv2.INTER_CUBIC)
        gc = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        current_grid = grid.get_grid(gc, on, 2)
        is_middle = grid.middle_object(current_grid, bth, rc)
        x = current_grid[len(current_grid) / 2][len(current_grid[0]) / 2][0]
        y = current_grid[len(current_grid) / 2][len(current_grid[0]) / 2][0]
        if (is_middle):
            f = cv2.circle(f, (x, y), 2, color1, -1)
            f = cv2.circle(f, (x, y), avg_mine / 2, color1, 2)
        else:
            f = cv2.circle(f, (x, y), 2, color2, -1)
            f = cv2.circle(f, (x, y), avg_mine / 2, color2, 2)
        cv2.imshow('Robot Camera', f)
        # Read Key Storkes
        k = cv2.waitKey(1) & 0xFF
        if (k == ord('s')):
            cor_image = gc[x - (avg_mine) / 2: x + ((avg_mine) / 2 + 1), y - (avg_mine) / 2: y + ((avg_mine) / 2 + 1)]
            img_dir = './db/img' + str(c) + '.jpg'
            cv2.imwrite(img_dir, cor_image)
            print('[IMAGE-WRITTEN]: ' + img_dir)
            c += 1
        if (k == ord('r')):
            print('PCA Analysis Started ...')
            pca.prep_dataset()
            print('PCA Analysis Done')
            break
        elif (k == ord('q')):
            break
elif ans == 2:
    # Prepare Environment
    print('Loading PCA Parameters ...')
    eigen_vects, mean_vect = pca.get_pcas()
    data_set = pca.get_dataset()
    print('Done Loading PCA Parameters')
    while (True):
        r, f = cam.read()
        f = cv2.resize(f, dsize=(480, 480), interpolation=cv2.INTER_CUBIC)
        gc = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        current_grid = grid.get_grid(gc, on, 2)
        is_middle = grid.middle_object(current_grid, bth, rc)
        x = current_grid[len(current_grid) / 2][len(current_grid[0]) / 2][0]
        y = current_grid[len(current_grid) / 2][len(current_grid[0]) / 2][0]
        if (is_middle):
            f = cv2.circle(f, (x, y), 2, color1, -1)
            f = cv2.circle(f, (x, y), avg_mine / 2, color1, 2)
        else:
            f = cv2.circle(f, (x, y), 2, color2, -1)
            f = cv2.circle(f, (x, y), avg_mine / 2, color2, 2)
        cv2.imshow('Robot Camera', f)
        # Read Key Storkes
        k = cv2.waitKey(1) & 0xFF
        if (k == ord('s')):
            cor_image = gc[x - (avg_mine) / 2: x + ((avg_mine) / 2 + 1), y - (avg_mine) / 2: y + ((avg_mine) / 2 + 1)]
            detected = pca.match_image(cor_image, eigen_vects, mean_vect, data_set)
            bool_detected = (len(re.findall('./db/mine[0-9]+.jpg', detected)) != 0)
            if (bool_detected):
                print('[Mine Detected]')
            else:
                print('[No Mine Detected]')
            feedback = input('Your Feedback (1 -> Right | 0 -> Wrong): ')
            if (feedback != 1):
                mines_no = pca.xlsx_read_cell(0, 0, './params.xlsx')
                img_no = pca.xlsx_read_cell(0, 1, './params.xlsx')
                new_img_name = ''
                if (not (bool_detected)):
                    new_img_name = './db/mine' + str(mines_no) + '.jpg'
                    cv2.imwrite(new_img_name, cor_image)
                    mines_no += 1
                else:
                    new_img_name = './db/img' + str(img_no) + '.jpg'
                    cv2.imwrite(new_img_name, cor_image)
                    img_no += 1
                pca.xlsx_write_row('./params.xlsx', [mines_no, img_no])
                print('[%s]: Data Sample Saved' % (new_img_name))
                print('Updating Data Set ...')
                pca.prep_dataset()
                print('Done Updating Data Set')
        if (k == ord('q')):
            break
elif ans == 3:
    # Prepare Environment
    print('Loading PCA Parameters ...')
    eigen_vects, mean_vect = pca.get_pcas()
    data_set = pca.get_dataset()
    print('Done Loading PCA Parameters')
    # Action Loop
    while (True):
        r, f = cam.read()
        f = cv2.resize(f, dsize=(480, 480), interpolation=cv2.INTER_CUBIC)
        gc = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        current_grid = grid.get_grid(gc, on, 2)
        detected_spots = []
        is_done = False
        for gx in range(len(current_grid)):
            for gy in range(len(current_grid[0])):
                x = current_grid[gx][gy][0]
                y = current_grid[gx][gy][1]
                if ((x > avg_mine / 2) & (x < 480 - 51) & (y > avg_mine / 2) & (y < 480 - 51)):
                    s_temp = grid.grid_corp(gc, current_grid, gx, gy, avg_mine)
                    detected = pca.match_image(s_temp, eigen_vects, mean_vect, data_set)
                    if (len(re.findall('./db/mine[0-9]+.jpg', detected)) != 0):
                        detected_spots.append(current_grid[gx][gy])
                        is_done = True
                        break
            if is_done:
                break
        for x in detected_spots:
            f = cv2.circle(f, (x[1], x[0]), avg_mine / 2, color1, 2)
        cv2.imshow('Robot Camera', f)
        # Read Key Storkes
        k = cv2.waitKey(1) & 0xFF
        if (k == ord('q')):
            break
elif ans == 4:
    print('Updating Data Set & PCA Parameters ...')
    pca.prep_dataset()
    print('Done Updating Data Set & PCA Parameters')
else:
    print('ERROR!')

# Clear Resources
cam.release()
cv2.destroyAllWindows()
