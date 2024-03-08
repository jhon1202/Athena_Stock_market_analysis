import sys
from PyQt5 import QtGui, QtCore, Qt, uic, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import pyqtSlot
from PyQt5.Qt import *

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QIcon
from PyQt5 import uic

from PyQt5 import QtChart
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt, QPointF
from PyQt5 import QtChart as qc

from difflib import SequenceMatcher
from time import gmtime, strftime
import datetime
import time
import numpy as np
import os
import math
import pandas as pd
import pathlib

import operator
import heapq
import statistics
import matplotlib as pyplot
import pyqtgraph as pg
from sklearn.model_selection import train_test_split
import heapq
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statistics
import xgboost

class Gui(QMainWindow):
    # flist = None

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        uic.loadUi('guidesign.ui', self)
        self.update()
        self.setWindowTitle("ATHENA")


        ########### tab-1
        self.sel_boolean_1 = False
        self.scale_1 = 1.0  ##scale
        self.initstart_1(0)




    ############# tab-1
    def initstart_1(self, i):
        if i == 0:
            if self.sel_boolean_1:
                self.sel_boolean_1 = False
            self.label_PATH_open_1.setText("")

        else:
            self.sel_boolean_1 = True
            self.label_PATH_open_1.setText(self.filename)


    @pyqtSlot()
    def on_pushButton_open_1_clicked(self):
        print("click pushButton_open_1")
        self.in_filename_1, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select files", "",
                                                                      "Excel Files (*.csv *.xlsx);;All Files (*)")
        self.filename = self.in_filename_1.split('/')[-1]
        if self.in_filename_1:
            self.initstart_1(1)
        else:
            self.initstart_1(0)
            print("cancel open_pushButton")
        self.sel_boolean_1 = True

    def AX_AY_arr_1(self, Z, F1, F2, XY):
        Data_close = pd.read_csv(self.in_filename_1, usecols=[5], header=None).values
        Data_low = pd.read_csv(self.in_filename_1, usecols=[4], header=None).values
        Data_high = pd.read_csv(self.in_filename_1, usecols=[3], header=None).values
        Data_open = pd.read_csv(self.in_filename_1, usecols=[2], header=None).values
        df_date = pd.read_csv(self.in_filename_1, usecols=[0], header=None).values
        df_time = pd.read_csv(self.in_filename_1, usecols=[1], header=None).values
        Rate1_step = int(len(Data_close) / Z)
        Rate1_arr = []
        for i in range(Rate1_step):
            Rate1 = Data_close[i * Z + Z-1][0] * 100 / Data_close[i * Z][0] - 100
            Rate1_arr.append(Rate1)

        Rate2_arr = []
        for j in range(len(Rate1_arr) - 1):
            Rate2 = Rate1_arr[j + 1] * 100 / Rate1_arr[j] - 100
            Rate2_arr.append(Rate2)
        No_arr = []
        index_arr = []

        for k in range(len(Rate2_arr)):
            var_max_arr = []
            var_min_arr = []
            if Rate2_arr[k] >= F1:
                for i in range(0, Z):
                    var_max_arr.append(Data_close[(k + 1) * Z + i][0])
                max_index, max_value = max(enumerate(var_max_arr), key=operator.itemgetter(1))
                No_arr.append(max_value)
                index_arr.append((k + 1) * Z + max_index)
            elif Rate2_arr[k] <= F2:
                for i in range(0, Z):
                    var_min_arr.append(Data_close[(k + 1) * Z + i][0])
                min_index, min_value = min(enumerate(var_min_arr), key=operator.itemgetter(1))
                No_arr.append(min_value)
                index_arr.append((k + 1) * Z + min_index)

        Axis_X_arr = []
        Axis_Y_arr = []

        for m in range(len(index_arr)-1):
            Axis_X = index_arr[m + 1] - index_arr[m]
            Axis_Y = No_arr[m]
            Axis_X_arr.append(Axis_X)
            Axis_Y_arr.append(math.floor(Axis_Y * 100) / 100)
        AX_AY_corresponding_date_times = []
        for n in range(len(Axis_X_arr)):
            AX_corresponding_date = df_date[index_arr[n]:index_arr[n + 1]]
            AX_corresponding_time = df_time[index_arr[n]:index_arr[n + 1]]
            AX_corresponding_date_time_arr = []
            for p in range(len(AX_corresponding_date)):
                AX_corresponding_date_time = AX_corresponding_date[p] + '-' + AX_corresponding_time[p]
                AX_corresponding_date_time_arr.append(AX_corresponding_date_time)
            AX_AY_corresponding_date_times.append([Axis_X_arr[n], Axis_Y_arr[n], index_arr[n], AX_corresponding_date_time_arr])

        self.Baseline_AX_arr = []
        self.Baseline_AY_arr = []
        self.Baseline_AX_AY_DATE_arr = []
        for i in range(0, XY):
            self.Baseline_AX_arr.append(Axis_X_arr[len(Axis_X_arr)-XY+i])
            self.Baseline_AY_arr.append(Axis_Y_arr[len(Axis_Y_arr)-XY+i])
            self.Baseline_AX_AY_DATE_arr.append(AX_AY_corresponding_date_times[len(Axis_X_arr)-XY+i])
        self.Baseline_AX_AY_DATE_arr = self.Baseline_AX_AY_DATE_arr
        self.candlestick_arr = []
        #self.comparison_arr = []
        for k in range(len(self.Baseline_AX_AY_DATE_arr)):
            #self.comparison_arr.append(Data_close[self.Baseline_AX_AY_DATE_arr[k][2]][0])
            for h in range(self.Baseline_AX_AY_DATE_arr[k][0]):
                if h == 0:
                    per_candlestick_value = (df_date[self.Baseline_AX_AY_DATE_arr[k][2]+h][0] +'-'+ df_time[self.Baseline_AX_AY_DATE_arr[k][2]+h][0], Data_open[self.Baseline_AX_AY_DATE_arr[k][2]+h][0], Data_high[self.Baseline_AX_AY_DATE_arr[k][2]+h][0], Data_low[self.Baseline_AX_AY_DATE_arr[k][2]+h][0], Data_close[self.Baseline_AX_AY_DATE_arr[k][2]+h][0], (Data_open[self.Baseline_AX_AY_DATE_arr[k][2]+h][0]+ Data_high[self.Baseline_AX_AY_DATE_arr[k][2]+h][0]+ Data_low[self.Baseline_AX_AY_DATE_arr[k][2]+h][0]+ Data_close[self.Baseline_AX_AY_DATE_arr[k][2]+h][0])/4, Data_close[self.Baseline_AX_AY_DATE_arr[k][2]][0])
                    self.candlestick_arr.append(per_candlestick_value)
                elif h != 0:
                    per_candlestick_value = (df_date[self.Baseline_AX_AY_DATE_arr[k][2] + h][0] + '-' +
                                             df_time[self.Baseline_AX_AY_DATE_arr[k][2] + h][0],
                                             Data_open[self.Baseline_AX_AY_DATE_arr[k][2] + h][0],
                                             Data_high[self.Baseline_AX_AY_DATE_arr[k][2] + h][0],
                                             Data_low[self.Baseline_AX_AY_DATE_arr[k][2] + h][0],
                                             Data_close[self.Baseline_AX_AY_DATE_arr[k][2] + h][0], (
                                                         Data_open[self.Baseline_AX_AY_DATE_arr[k][2] + h][0] +
                                                         Data_high[self.Baseline_AX_AY_DATE_arr[k][2] + h][0] +
                                                         Data_low[self.Baseline_AX_AY_DATE_arr[k][2] + h][0] +
                                                         Data_close[self.Baseline_AX_AY_DATE_arr[k][2] + h][0]) / 4,
                                             0)
                    self.candlestick_arr.append(per_candlestick_value)
        self.candlestic_arr = tuple(self.candlestick_arr)
        #self.comparison_arr = tuple(self.comparison_arr)

    @pyqtSlot()
    def on_pushButton_RUN_1_clicked(self):
        print("click pushButton_RUN_1")
        Z = int(self.textEdit_z.toPlainText())
        F1 = int(self.textEdit_f1.toPlainText())
        F2 = int(self.textEdit_f2.toPlainText())
        XY = int(self.textEdit_xy.toPlainText())
        self.AX_AY_arr_1(Z, F1, F2, XY)

        candlestic_arr = self.candlestic_arr
        #comparison_arr = self.comparison_arr
        print(candlestic_arr)

        series = QtChart.QCandlestickSeries()
        series.setDecreasingColor(QtCore.Qt.red)
        series.setIncreasingColor(QtCore.Qt.green)
        tm = []  # stores str type data
        ma5 = QtChart.QLineSeries()
        com5 = QtChart.QLineSeries()
        num = 0
        # in a loop,  series and ma5 append corresponding data
        for date, o, h, l, c, m, com in candlestic_arr:
            num = num + 1
            series.append(QtChart.QCandlestickSet(o, h, l, c))
            ma5.append(QtCore.QPointF(num, m))
            if com != 0:
                com5.append(QtCore.QPointF(num, com))
            tm.append(date)


        chart = QtChart.QChart()
        chart.addSeries(series)  # candle
        chart.addSeries(ma5)
        chart.addSeries(com5)

        chart.setAnimationOptions(QtChart.QChart.SeriesAnimations)
        chart.createDefaultAxes()
        chart.legend().hide()

        chart.axisX(series).setCategories(tm)

        chartview = QtChart.QChartView(chart)
        self.chart_container.setContentsMargins(0, 0, 0, 0)
        lay = QtWidgets.QHBoxLayout(self.chart_container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(chartview)
        lay.deleteLater()

    def initstart_2(self, i):
        if i == 0:
            if self.sel_boolean_2:
                self.sel_boolean_2 = False
            self.label_PATH_open_2.setText("")

        else:
            self.sel_boolean_2 = True
            self.label_PATH_open_2.setText(self.filename2)

    @pyqtSlot()
    def on_pushButton_open_2_clicked(self):
        print("click pushButton_open_2")
        self.in_filename_2, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select files", "",
                                                                      "Excel Files (*.csv *.xlsx);;All Files (*)")
        self.filename2 = self.in_filename_2.split('/')[-1]
        if self.in_filename_2:
            self.initstart_2(1)
        else:
            self.initstart_2(0)
            print("cancel open_pushButton")
        self.sel_boolean_2 = True



    def AX_AY_arr_2(self, Z, F1, F2, num_XY, filepath):
        Data_close = pd.read_csv(filepath, usecols=[5], header=None).values
        Data_low = pd.read_csv(filepath, usecols=[4], header=None).values
        Data_high = pd.read_csv(filepath, usecols=[3], header=None).values
        Data_open = pd.read_csv(filepath, usecols=[2], header=None).values
        df_date = pd.read_csv(filepath, usecols=[0], header=None).values
        df_time = pd.read_csv(filepath, usecols=[1], header=None).values
        Data_asset = pd.read_csv(filepath, usecols=[7], header=None).values
        Rate1_step = int(len(Data_close) / Z)
        Rate1_arr = []
        for i in range(Rate1_step):
            Rate1 = Data_close[i * Z + Z-1][0] * 100 / Data_close[i * Z][0] - 100
            Rate1_arr.append(Rate1)

        Rate2_arr = []
        for j in range(len(Rate1_arr) - 1):
            Rate2 = Rate1_arr[j + 1] * 100 / Rate1_arr[j] - 100
            Rate2_arr.append(Rate2)
        No_arr = []
        index_arr = []

        for k in range(len(Rate2_arr)):
            var_max_arr = []
            var_min_arr = []
            if Rate2_arr[k] >= F1:
                for i in range(0, Z):
                    var_max_arr.append(Data_close[(k + 1) * Z + i][0])
                max_index, max_value = max(enumerate(var_max_arr), key=operator.itemgetter(1))
                No_arr.append(max_value)
                index_arr.append((k + 1) * Z + max_index)
            elif Rate2_arr[k] <= F2:
                for i in range(0, Z):
                    var_min_arr.append(Data_close[(k + 1) * Z + i][0])
                min_index, min_value = min(enumerate(var_min_arr), key=operator.itemgetter(1))
                No_arr.append(min_value)
                index_arr.append((k + 1) * Z + min_index)

        Axis_X_arr = []
        Axis_Y_arr = []

        for m in range(len(index_arr)-1):
            Axis_X = index_arr[m + 1] - index_arr[m]
            Axis_Y = No_arr[m]
            Axis_X_arr.append(Axis_X)
            Axis_Y_arr.append(math.floor(Axis_Y * 100) / 100)


        adj_R2_X_arr = []
        adj_R2_Y_arr = []
        result_final_arr = []
        for n in range(len(Axis_Y_arr) - num_XY + 1):
            Gen_AX_arr = []
            Gen_AY_arr = []
            for k in range(0, num_XY):
                Gen_AX_arr.append(Axis_X_arr[n + k])
                Gen_AY_arr.append(Axis_Y_arr[n + k])
            mean1_X = statistics.mean(self.Baseline_AX_arr)
            mean2_X = statistics.mean(Gen_AX_arr)
            min1_ind_X, min1_X = min(enumerate(self.Baseline_AX_arr), key=operator.itemgetter(1))
            minimum1_X = min1_X-1
            min2_ind_X, min2_X = min(enumerate(Gen_AX_arr), key=operator.itemgetter(1))
            minimum2_X = min2_X - 1
            Base_Ax_1 = [i * mean2_X-minimum1_X for i in self.Baseline_AX_arr]
            Base_AX_2 = [i * mean1_X-minimum2_X for i in Gen_AX_arr]
            comparison_arr_X = []
            for i in range(len(Base_Ax_1)-1):
                comparison = (Base_Ax_1[i]/Base_AX_2[i])*100
                if comparison <= 100:
                    comparison = comparison
                elif comparison>100:
                    comparison = 100 - (comparison-100)
                comparison_arr_X.append(comparison)
            Result_X = statistics.mean(comparison_arr_X)

            mean1_Y = statistics.mean(self.Baseline_AY_arr)
            mean2_Y = statistics.mean(Gen_AY_arr)
            min1_ind_Y, min1_Y = min(enumerate(self.Baseline_AY_arr), key=operator.itemgetter(1))
            minimum1_Y = min1_Y - 1
            min2_ind_Y, min2_Y = min(enumerate(Gen_AY_arr), key=operator.itemgetter(1))
            minimum2_Y = min2_Y - 1
            Base_AY_1 = [i * mean2_Y-minimum1_Y for i in self.Baseline_AY_arr]
            Base_AY_2 = [i * mean1_Y-minimum2_Y for i in Gen_AY_arr]
            comparison_arr_Y = []
            for i in range(len(Base_AY_1)-1):
                comparison_Y = (Base_AY_1[i] / Base_AY_2[i]) * 100
                if comparison_Y <= 100:
                    comparison_Y = comparison_Y
                elif comparison_Y > 100:
                    comparison_Y = 100 - (comparison_Y - 100)
                comparison_arr_Y.append(comparison_Y)
            Result_Y = statistics.mean(comparison_arr_Y)
            Result_final = statistics.mean([Result_X, Result_Y])
            result_final_arr.append(Result_final)


        X = heapq.nlargest(7, result_final_arr)
        X_indexes = heapq.nlargest(7, range(len(result_final_arr)), key=result_final_arr.__getitem__)

        Y = heapq.nlargest(7, result_final_arr)
        Y_indexes = heapq.nlargest(7, range(len(result_final_arr)), key=result_final_arr.__getitem__)

        XY_arr = []
        for l in range(len(result_final_arr)):
            XY_val = statistics.mean([result_final_arr[l], result_final_arr[l]])
            XY_arr.append(XY_val)
        XY = heapq.nlargest(7, XY_arr)
        XY_indexes = heapq.nlargest(7, range(len(XY_arr)), key=XY_arr.__getitem__)

        X_corresponding_date_times = []

        for h in range(0, 7):
            index_inf = math.floor((index_arr[X_indexes[h]] + 1) / Z) * Z
            index_sup = (math.floor((index_arr[X_indexes[h]] + 1) / Z) + num_XY+1) * Z
            X_corresponding_date = df_date[index_inf:index_sup]
            X_corresponding_time = df_time[index_inf:index_sup]
            X_corresponding_date_time_arr = []
            for p in range(len(X_corresponding_date)):
                X_corresponding_date_time = X_corresponding_date[p] + '-' + X_corresponding_time[p]
                X_corresponding_date_time_arr.append(X_corresponding_date_time)
            X_corresponding_date_times.append([X[h], X_corresponding_date_time_arr])
        print('-----------X values and Times --------------')
        print(X_corresponding_date_times)

        Y_corresponding_date_times = []

        for h in range(0, 7):
            index_inf = math.floor((index_arr[Y_indexes[h]] + 1) / Z) * Z
            index_sup = (math.floor((index_arr[Y_indexes[h]] + 1) / Z) + num_XY+1) * Z
            Y_corresponding_date = df_date[index_inf:index_sup]
            Y_corresponding_time = df_time[index_inf:index_sup]
            Y_corresponding_date_time_arr = []
            for p in range(len(Y_corresponding_date)):
                Y_corresponding_date_time = Y_corresponding_date[p] + '-' + Y_corresponding_time[p]
                Y_corresponding_date_time_arr.append(Y_corresponding_date_time)
            Y_corresponding_date_times.append([Y[h], Y_corresponding_date_time_arr])

        print('-----------Y values and Times --------------')
        print(Y_corresponding_date_times)

        XY_corresponding_date_times = []

        for h in range(0, 7):
            index_inf = math.floor((index_arr[XY_indexes[h]] + 1) / Z) * Z
            index_sup = (math.floor((index_arr[XY_indexes[h]] + 1) / Z) + num_XY+1) * Z
            step_val = index_sup - index_inf
            print(index_sup)
            print(len(Data_close))
            if len(Data_close) > index_sup +30:
                new_index_sup = index_sup + 30
            elif len(Data_close) <= index_sup +30:
                new_index_sup = len(Data_close)-1
            print(new_index_sup)
            XY_corresponding_date = df_date[index_inf-30:new_index_sup-1]
            XY_corresponding_time = df_time[index_inf-30:new_index_sup-1]
            XY_close = Data_close[index_inf-30:new_index_sup-1]
            XY_open = Data_open[index_inf-30:new_index_sup-1]
            XY_low = Data_low[index_inf-30:new_index_sup-1]
            XY_high = Data_high[index_inf-30:new_index_sup-1]

            XY_asset = Data_asset[index_inf-30:new_index_sup-1]
            XY_corresponding_date_time_arr = []
            high_arr = []
            low_arr = []
            for p in range(len(XY_corresponding_date)):
                XY_corresponding_date_time = XY_corresponding_date[p][0] + '-' + XY_corresponding_time[p][0]
                XY_close_val = XY_close[p][0]
                XY_open_val = XY_open[p][0]
                XY_low_val = XY_low[p][0]
                XY_high_val = XY_high[p][0]
                XY_asset_val = XY_asset[p][0]
                high_arr.append(XY_high_val)
                low_arr.append(XY_low_val)

                XY_mean_val = (XY_open_val + XY_low_val +XY_high_val +XY_close_val)/4
                XY_values = (XY_corresponding_date_time, XY_open_val, XY_high_val, XY_low_val, XY_close_val, XY_mean_val, XY_asset_val)
                XY_corresponding_date_time_arr.append(XY_values)
            high_index, high_val = max(enumerate(high_arr), key=operator.itemgetter(1))
            low_index, low_val = min(enumerate(low_arr), key=operator.itemgetter(1))
            XY_corresponding_date_times.append([XY[h], XY_corresponding_date_time_arr, step_val, high_val, low_val])
        self.XY_corresponding_date_times = XY_corresponding_date_times

        print('-----------XY values and Times --------------')
        print(XY_corresponding_date_times)
        #print(self.candlestic_arr2)


    @pyqtSlot()
    def on_pushButton_RUN_2_clicked(self):
        print("click pushButton_RUN_2")
        Z = int(self.textEdit_z.toPlainText())
        F1 = int(self.textEdit_f1.toPlainText())
        F2 = int(self.textEdit_f2.toPlainText())
        XY2 = int(self.textEdit_xy2.toPlainText())
        self.AX_AY_arr_2(Z, F1, F2, XY2, self.in_filename_2)

        candlestic_arr2 = self.candlestic_arr

        series = QtChart.QCandlestickSeries()
        series.setDecreasingColor(QtCore.Qt.red)
        series.setIncreasingColor(QtCore.Qt.green)
        tm = []  # stores str type data
        ma5 = QtChart.QLineSeries()
        com5 = QtChart.QLineSeries()
        num = 0
        # in a loop,  series and ma5 append corresponding data
        for date, o, h, l, c, m, com in candlestic_arr2:
            num = num + 1
            series.append(QtChart.QCandlestickSet(o, h, l, c))
            ma5.append(QtCore.QPointF(num, m))
            if com != 0:
                com5.append(QtCore.QPointF(num, com))
            tm.append(date)

        chart = QtChart.QChart()
        chart.addSeries(series)  # candle
        chart.addSeries(ma5)
        chart.addSeries(com5)

        chart.setAnimationOptions(QtChart.QChart.SeriesAnimations)
        chart.createDefaultAxes()
        chart.legend().hide()

        chart.axisX(series).setCategories(tm)

        chartview = QtChart.QChartView(chart)
        self.chart_container.setContentsMargins(0, 0, 0, 0)
        lay = QtWidgets.QHBoxLayout(self.chart_container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(chartview)
        lay.deleteLater()

    @pyqtSlot()
    def on_pushButton_sel_tab1_clicked(self):

        XY_corresponding_date_times = tuple(self.XY_corresponding_date_times[0][1])
        print(XY_corresponding_date_times)
        self.label_PATH_open_4.setText(str(self.XY_corresponding_date_times[0][0]))
        high_val = self.XY_corresponding_date_times[0][3]
        low_val = self.XY_corresponding_date_times[0][4]
        step_val = self.XY_corresponding_date_times[0][2]
        series = QtChart.QCandlestickSeries()
        series.setDecreasingColor(QtCore.Qt.red)
        series.setIncreasingColor(QtCore.Qt.green)
        tm = []  # stores str type data
        ma5 = QtChart.QLineSeries()


        num = 0
        # in a loop,  series and ma5 append corresponding data
        for date, o, h, l, c, m, asset in XY_corresponding_date_times:
            num = num + 1
            if num == 30:
                series.append(QtChart.QCandlestickSet(low_val, high_val, low_val, low_val))
                ma5.append(QtCore.QPointF(num, m))
            elif num == 30+step_val:
                series.append(QtChart.QCandlestickSet(low_val, high_val, low_val, low_val))
                ma5.append(QtCore.QPointF(num, m))
            series.append(QtChart.QCandlestickSet(o, h, l, c))
            ma5.append(QtCore.QPointF(num, m))
            tm.append(date)
            asset = asset
        self.label_PATH_open_3.setText(str(asset))
        chart = QtChart.QChart()
        chart.addSeries(series)  # candle
        chart.addSeries(ma5)


        chart.setAnimationOptions(QtChart.QChart.SeriesAnimations)
        chart.createDefaultAxes()
        chart.legend().hide()
        chart.axisX(series).setCategories(tm)

        chartview = QtChart.QChartView(chart)

        self.chart_container2.setContentsMargins(0, 0, 0, 0)
        lay = QtWidgets.QHBoxLayout(self.chart_container2)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(chartview)
        lay.deleteLater()

    @pyqtSlot()
    def on_pushButton_sel_tab2_clicked(self):

        XY_corresponding_date_times = tuple(self.XY_corresponding_date_times[1][1])
        print(XY_corresponding_date_times)
        self.label_PATH_open_4.setText(str(self.XY_corresponding_date_times[1][0]))
        high_val = self.XY_corresponding_date_times[1][3]
        low_val = self.XY_corresponding_date_times[1][4]
        step_val = self.XY_corresponding_date_times[1][2]
        series = QtChart.QCandlestickSeries()
        series.setDecreasingColor(QtCore.Qt.red)
        series.setIncreasingColor(QtCore.Qt.green)
        tm = []  # stores str type data
        ma5 = QtChart.QLineSeries()


        num = 0
        # in a loop,  series and ma5 append corresponding data
        for date, o, h, l, c, m, asset in XY_corresponding_date_times:
            num = num + 1
            if num == 30:
                series.append(QtChart.QCandlestickSet(low_val, high_val, low_val, low_val))
                ma5.append(QtCore.QPointF(num, m))
            elif num == 30+step_val:
                series.append(QtChart.QCandlestickSet(low_val, high_val, low_val, low_val))
                ma5.append(QtCore.QPointF(num, m))
            series.append(QtChart.QCandlestickSet(o, h, l, c))
            ma5.append(QtCore.QPointF(num, m))
            tm.append(date)
            asset = asset
        self.label_PATH_open_3.setText(str(asset))
        chart = QtChart.QChart()
        chart.addSeries(series)  # candle
        chart.addSeries(ma5)


        chart.setAnimationOptions(QtChart.QChart.SeriesAnimations)
        chart.createDefaultAxes()
        chart.legend().hide()
        chart.axisX(series).setCategories(tm)

        chartview = QtChart.QChartView(chart)

        self.chart_container2.setContentsMargins(0, 0, 0, 0)
        lay = QtWidgets.QHBoxLayout(self.chart_container2)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(chartview)
        lay.deleteLater()

    @pyqtSlot()
    def on_pushButton_sel_tab3_clicked(self):

        XY_corresponding_date_times = tuple(self.XY_corresponding_date_times[2][1])
        print(XY_corresponding_date_times)
        self.label_PATH_open_4.setText(str(self.XY_corresponding_date_times[2][0]))
        high_val = self.XY_corresponding_date_times[2][3]
        low_val = self.XY_corresponding_date_times[2][4]
        step_val = self.XY_corresponding_date_times[2][2]
        series = QtChart.QCandlestickSeries()
        series.setDecreasingColor(QtCore.Qt.red)
        series.setIncreasingColor(QtCore.Qt.green)
        tm = []  # stores str type data
        ma5 = QtChart.QLineSeries()


        num = 0
        # in a loop,  series and ma5 append corresponding data
        for date, o, h, l, c, m, asset in XY_corresponding_date_times:
            num = num + 1
            if num == 30:
                series.append(QtChart.QCandlestickSet(low_val, high_val, low_val, low_val))
                ma5.append(QtCore.QPointF(num, m))
            elif num == 30+step_val:
                series.append(QtChart.QCandlestickSet(low_val, high_val, low_val, low_val))
                ma5.append(QtCore.QPointF(num, m))
            series.append(QtChart.QCandlestickSet(o, h, l, c))
            ma5.append(QtCore.QPointF(num, m))
            tm.append(date)
            asset = asset
        self.label_PATH_open_3.setText(str(asset))
        chart = QtChart.QChart()
        chart.addSeries(series)  # candle
        chart.addSeries(ma5)


        chart.setAnimationOptions(QtChart.QChart.SeriesAnimations)
        chart.createDefaultAxes()
        chart.legend().hide()
        chart.axisX(series).setCategories(tm)

        chartview = QtChart.QChartView(chart)

        self.chart_container2.setContentsMargins(0, 0, 0, 0)
        lay = QtWidgets.QHBoxLayout(self.chart_container2)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(chartview)
        lay.deleteLater()

    @pyqtSlot()
    def on_pushButton_sel_tab4_clicked(self):

        XY_corresponding_date_times = tuple(self.XY_corresponding_date_times[3][1])
        print(XY_corresponding_date_times)
        self.label_PATH_open_4.setText(str(self.XY_corresponding_date_times[3][0]))
        high_val = self.XY_corresponding_date_times[3][3]
        low_val = self.XY_corresponding_date_times[3][4]
        step_val = self.XY_corresponding_date_times[3][2]
        series = QtChart.QCandlestickSeries()
        series.setDecreasingColor(QtCore.Qt.red)
        series.setIncreasingColor(QtCore.Qt.green)
        tm = []  # stores str type data
        ma5 = QtChart.QLineSeries()


        num = 0
        # in a loop,  series and ma5 append corresponding data
        for date, o, h, l, c, m, asset in XY_corresponding_date_times:
            num = num + 1
            if num == 30:
                series.append(QtChart.QCandlestickSet(low_val, high_val, low_val, low_val))
                ma5.append(QtCore.QPointF(num, m))
            elif num == 30+step_val:
                series.append(QtChart.QCandlestickSet(low_val, high_val, low_val, low_val))
                ma5.append(QtCore.QPointF(num, m))
            series.append(QtChart.QCandlestickSet(o, h, l, c))
            ma5.append(QtCore.QPointF(num, m))
            tm.append(date)
            asset = asset
        self.label_PATH_open_3.setText(str(asset))
        chart = QtChart.QChart()
        chart.addSeries(series)  # candle
        chart.addSeries(ma5)


        chart.setAnimationOptions(QtChart.QChart.SeriesAnimations)
        chart.createDefaultAxes()
        chart.legend().hide()
        chart.axisX(series).setCategories(tm)

        chartview = QtChart.QChartView(chart)

        self.chart_container2.setContentsMargins(0, 0, 0, 0)
        lay = QtWidgets.QHBoxLayout(self.chart_container2)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(chartview)
        lay.deleteLater()

    @pyqtSlot()
    def on_pushButton_sel_tab5_clicked(self):

        XY_corresponding_date_times = tuple(self.XY_corresponding_date_times[4][1])
        print(XY_corresponding_date_times)
        self.label_PATH_open_4.setText(str(self.XY_corresponding_date_times[4][0]))
        high_val = self.XY_corresponding_date_times[5][3]
        low_val = self.XY_corresponding_date_times[5][4]
        step_val = self.XY_corresponding_date_times[5][2]
        series = QtChart.QCandlestickSeries()
        series.setDecreasingColor(QtCore.Qt.red)
        series.setIncreasingColor(QtCore.Qt.green)
        tm = []  # stores str type data
        ma5 = QtChart.QLineSeries()


        num = 0
        # in a loop,  series and ma5 append corresponding data
        for date, o, h, l, c, m, asset in XY_corresponding_date_times:
            num = num + 1
            if num == 30:
                series.append(QtChart.QCandlestickSet(low_val, high_val, low_val, low_val))
                ma5.append(QtCore.QPointF(num, m))
            elif num == 30+step_val:
                series.append(QtChart.QCandlestickSet(low_val, high_val, low_val, low_val))
                ma5.append(QtCore.QPointF(num, m))
            series.append(QtChart.QCandlestickSet(o, h, l, c))
            ma5.append(QtCore.QPointF(num, m))
            tm.append(date)
            asset = asset
        self.label_PATH_open_3.setText(str(asset))
        chart = QtChart.QChart()
        chart.addSeries(series)  # candle
        chart.addSeries(ma5)


        chart.setAnimationOptions(QtChart.QChart.SeriesAnimations)
        chart.createDefaultAxes()
        chart.legend().hide()
        chart.axisX(series).setCategories(tm)

        chartview = QtChart.QChartView(chart)

        self.chart_container2.setContentsMargins(0, 0, 0, 0)
        lay = QtWidgets.QHBoxLayout(self.chart_container2)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(chartview)
        lay.deleteLater()

    @pyqtSlot()
    def on_pushButton_sel_tab6_clicked(self):

        XY_corresponding_date_times = tuple(self.XY_corresponding_date_times[5][1])
        print(XY_corresponding_date_times)
        self.label_PATH_open_4.setText(str(self.XY_corresponding_date_times[5][0]))
        high_val = self.XY_corresponding_date_times[5][3]
        low_val = self.XY_corresponding_date_times[5][4]
        step_val = self.XY_corresponding_date_times[5][2]
        series = QtChart.QCandlestickSeries()
        series.setDecreasingColor(QtCore.Qt.red)
        series.setIncreasingColor(QtCore.Qt.green)
        tm = []  # stores str type data
        ma5 = QtChart.QLineSeries()


        num = 0
        # in a loop,  series and ma5 append corresponding data
        for date, o, h, l, c, m, asset in XY_corresponding_date_times:
            num = num + 1
            if num == 30:
                series.append(QtChart.QCandlestickSet(low_val, high_val, low_val, low_val))
                ma5.append(QtCore.QPointF(num, m))
            elif num == 30+step_val:
                series.append(QtChart.QCandlestickSet(low_val, high_val, low_val, low_val))
                ma5.append(QtCore.QPointF(num, m))
            series.append(QtChart.QCandlestickSet(o, h, l, c))
            ma5.append(QtCore.QPointF(num, m))
            tm.append(date)
            asset = asset
        self.label_PATH_open_3.setText(str(asset))
        chart = QtChart.QChart()
        chart.addSeries(series)  # candle
        chart.addSeries(ma5)


        chart.setAnimationOptions(QtChart.QChart.SeriesAnimations)
        chart.createDefaultAxes()
        chart.legend().hide()
        chart.axisX(series).setCategories(tm)

        chartview = QtChart.QChartView(chart)

        self.chart_container2.setContentsMargins(0, 0, 0, 0)
        lay = QtWidgets.QHBoxLayout(self.chart_container2)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(chartview)
        lay.deleteLater()

    @pyqtSlot()
    def on_pushButton_sel_tab7_clicked(self):

        XY_corresponding_date_times = tuple(self.XY_corresponding_date_times[6][1])
        print(XY_corresponding_date_times)
        self.label_PATH_open_4.setText(str(self.XY_corresponding_date_times[6][0]))
        high_val = self.XY_corresponding_date_times[6][3]
        low_val = self.XY_corresponding_date_times[6][4]
        step_val = self.XY_corresponding_date_times[6][2]
        series = QtChart.QCandlestickSeries()
        series.setDecreasingColor(QtCore.Qt.red)
        series.setIncreasingColor(QtCore.Qt.green)
        tm = []  # stores str type data
        ma5 = QtChart.QLineSeries()


        num = 0
        # in a loop,  series and ma5 append corresponding data
        for date, o, h, l, c, m, asset in XY_corresponding_date_times:
            num = num + 1
            if num == 30:
                series.append(QtChart.QCandlestickSet(low_val, high_val, low_val, low_val))
                ma5.append(QtCore.QPointF(num, m))
            elif num == 30+step_val:
                series.append(QtChart.QCandlestickSet(low_val, high_val, low_val, low_val))
                ma5.append(QtCore.QPointF(num, m))
            series.append(QtChart.QCandlestickSet(o, h, l, c))
            ma5.append(QtCore.QPointF(num, m))
            tm.append(date)
            asset = asset
        self.label_PATH_open_3.setText(str(asset))
        chart = QtChart.QChart()
        chart.addSeries(series)  # candle
        chart.addSeries(ma5)


        chart.setAnimationOptions(QtChart.QChart.SeriesAnimations)
        chart.createDefaultAxes()
        chart.legend().hide()
        chart.axisX(series).setCategories(tm)

        chartview = QtChart.QChartView(chart)

        self.chart_container2.setContentsMargins(0, 0, 0, 0)
        lay = QtWidgets.QHBoxLayout(self.chart_container2)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(chartview)
        lay.deleteLater()



def main():
    app = QApplication(sys.argv)
    ex = Gui()
    ex.show()

    app.setWindowIcon(QIcon('icon.ico'))
    ex.setWindowIcon(QIcon('icon.ico'))
    sys.exit(app.exec_())



if __name__ == '__main__':
    main()
