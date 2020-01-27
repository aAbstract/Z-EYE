import numpy as np


# Get Information of Some Points in The Frame
def get_grid(data, on, mode):
    out = []
    gx = on
    gy = on
    if (mode == 1):
        while (gx < data.shape[1]):
            col_grid = [(gx, gy, data[gx][gy])]
            while (gy < data.shape[0]):
                col_grid.append((gx, gy, data[gx][gy]))
                gy += on
            out.append(col_grid)
            gy = on
            gx += on
    elif (mode == 2):
        while (gx < data.shape[1]):
            col_grid = [(gx, gy, data[gx][gy])]
            while (gy < data.shape[0]):
                col_grid.append((gx, gy, (np.sum(data[gx - 1:gx + 2, gy - 1:gy + 2]) / 9)))
                gy += on
            out.append(col_grid)
            gy = on
            gx += on
    else:
        return -1
    return out


# Computes Approximated Center Point of Grid Points Reigon
def get_rcp(grox, groy, ls, mode):
    out = [0, 0]
    dl = (ls[1] - ls[0]) / 2
    if (ls[0] == ls[1]):
        out = [grox, groy]
    else:
        if (mode == 1):
            if (ls[0] > ls[1]):
                out[0] = grox - dl
                out[1] = groy - dl
            elif (ls[0] < ls[1]):
                out[0] = grox + dl
                out[1] = groy + dl
        elif (mode == 2):
            if (ls[0] > ls[1]):
                out[0] = grox - dl
                out[1] = groy + dl
            elif (ls[0] < ls[1]):
                out[0] = grox + dl
                out[1] = groy - dl
    return out


# Remove Sorrounding Pixles of a Mine
def point_bound_check(grid, grox, groy, bth, rc):
    d1 = False
    d2 = False
    d3 = False
    d4 = False
    ls0 = [0, 0]
    ls1 = [0, 0]
    cp = [0, 0]
    # D1->D1(d1) Check
    x1 = grox
    x2 = groy
    while (x1 >= 0 & x2 >= 0):
        x1 -= 1
        x2 -= 1
        if (grid[x1][x2][2] > (bth + rc * bth)):
            d1 = True
            break
        ls0[0] += 1
    # D2->D1(d2) Check
    if d1 != True:
        return False, [0, 0], [0, 0]
    else:
        x1 = grox
        x2 = groy
        while (x1 < len(grid) & x2 < len(grid[0])):
            x1 += 1
            x2 += 1
            if (grid[x1][x2][2] > (bth + rc * bth)):
                d2 = True
                break
            ls0[1] += 1
    temp = get_rcp(grox, groy, ls0, 1)
    # D1->D2(d3) Check
    x1 = temp[0]
    x2 = temp[1]
    while (x1 >= 0 & x2 < len(grid[0])):
        x1 -= 1
        x2 += 1
        if (grid[x1][x2][2] > (bth + rc * bth)):
            d3 = True
            break
        ls1[0] += 1
    # D2->D2(d4) Check
    if d3 != True:
        return False, [0, 0], [0, 0]
    else:
        x1 = temp[0]
        x2 = temp[1]
        while (x1 < len(grid) & x2 >= 0):
            x1 += 1
            x2 -= 1
            if (grid[x1][x2][2] > (bth + rc * bth)):
                d4 = True
                break
            ls1[1] += 1
    cp = get_rcp(temp[0], temp[1], ls1, 2)
    return (d1 & d2 & d3 & d4), cp, ((ls1[0] + ls1[1]) / 2)


def if_in(mat, element):
    try:
        mat.index(element)
        return True
    except:
        return False


# Get Grid Points That May Have Mines
def check_grid(grid_info, bth, rc):
    out = []
    for x in range(len(grid_info)):
        for y in range(len(grid_info[0])):
            if (grid_info[x][y][2] <= (bth + bth * rc)):
                is_bounded, cp, rr = point_bound_check(grid_info, x, y, bth, rc)
                if (is_bounded & (not (if_in(out, cp)))):
                    out.append((cp, rr))
    return out


# Corp Relative To Grid Coordinates
def grid_corp(data, grid, gx, gy, a):
    x = grid[gx][gy][0]
    y = grid[gx][gy][1]
    return data[x - a / 2: x + (a / 2 + 1), y - a / 2: y + (a / 2 + 1)]


# Detect If There's Object In The Middle of The Camera
def middle_object(grid, bth, rc):
    gx = len(grid) / 2
    gy = len(grid[0]) / 2
    if (grid[gx][gy][2] < bth + bth * rc):
        return True
    else:
        return False