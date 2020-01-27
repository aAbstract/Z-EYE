import os
import numpy as np
import cv2
from numpy import linalg as la
import xlsxwriter
import xlrd


def xlsx_write(data, file_name):
    workbook = xlsxwriter.Workbook(file_name)
    worksheet = workbook.add_worksheet()
    data = list(data)
    for rank, data_col in enumerate(data):
        worksheet.write_column(0, rank, data_col)
    workbook.close()


def xlsx_read_cell(x, y, file_name):
    wb = xlrd.open_workbook(file_name)
    s = wb.sheet_by_index(0)
    return s.cell_value(x, y)


def xlsx_write_row(file_name, data):
    workbook = xlsxwriter.Workbook(file_name)
    worksheet = workbook.add_worksheet()
    worksheet.write_row(0, 0, data)
    workbook.close()


def get_pcas():
    wb = xlrd.open_workbook('./train_resault.xlsx')
    s = wb.sheet_by_index(0)
    eigen_vects = []
    mean_vect = []
    for x in range(s.ncols):
        if (x == s.ncols - 1):
            mean_vect = np.array(s.col_values(x))
        else:
            eigen_vects.append(s.col_values(x))
    eigen_vects = np.array(eigen_vects)
    return eigen_vects, mean_vect


def get_dataset():
    wb = xlrd.open_workbook('./data_set.xlsx')
    s = wb.sheet_by_index(0)
    out = []
    for x in range(s.ncols):
        out.append(s.col_values(x))
    return np.array(out)


def match_image(data, eigen_vects, mean_vect, data_set):
    data = cv2.resize(data, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)
    vect = np.reshape(data, (10000))
    if_vect = vect - mean_vect
    weights = np.matmul(eigen_vects, np.reshape(if_vect, (if_vect.shape[0], 1)))
    weights = np.reshape(weights, (weights.shape[0]))
    errs = [la.norm(x - weights) for x in data_set]
    match_index = errs.index(np.amin(errs))
    img_dirs = []
    for r, d, f in os.walk('./db'):
        for file in f:
            if ((file != '.') & (file != '..')):
                img_dirs.append(os.path.join(r, file))
    return img_dirs[match_index]


def prep_dataset():
    img_dirs = []
    for r, d, f in os.walk('./db'):
        for file in f:
            if ((file != '.') & (file != '..')):
                img_dirs.append(os.path.join(r, file))
    img_vects = []
    for x in img_dirs:
        img = cv2.imread(x)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)
        temp_vect = np.reshape(img, (10000))
        img_vects.append(temp_vect)
    img_vects = np.array(img_vects)
    mean_vect = img_vects.mean(0)
    if_vects = [(x - mean_vect) for x in img_vects]
    if_vects = np.array(if_vects)
    co_mat = np.matmul(if_vects, if_vects.transpose())
    eigen_values, eigen_vectors = la.eig(co_mat)
    eigen_vectors = eigen_vectors[-10:]
    ev2 = np.matmul(if_vects.transpose(), eigen_vectors.transpose())
    ev2 = list(ev2.transpose())
    ev2.append(mean_vect)
    ev2 = np.array(ev2)
    xlsx_write(ev2, 'train_resault.xlsx')
    proj_vects = []
    for x in range(len(if_vects)):
        temp = np.matmul(ev2[:-1], np.reshape(if_vects[x], (if_vects[x].shape[0], 1)))
        proj_vects.append(np.reshape(temp, (temp.shape[0])))
    xlsx_write(np.array(proj_vects), 'data_set.xlsx')

