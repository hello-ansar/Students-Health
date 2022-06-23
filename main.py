from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  # plotting
import numpy as np  # linear algebra
import os  # accessing directory structure
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import time

# Баған деректерінің таралу графиктері (гистограмма/бағаналық график)
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if 4 < nunique[col] < 10]]

    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 2) / nGraphPerRow
    plt.figure(num=None, figsize=(6 * nGraphPerRow, 10 * nGraphRow),
               dpi=60, facecolor='w', edgecolor='k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(int(nGraphRow), int(nGraphPerRow), i + 1)
        columnDf = df.iloc[:, i]
        if not np.issubdtype(type(columnDf.iloc[0]), np.number):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation=90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad=10.0, w_pad=0.14, h_pad=2.0)
    plt.show()

# Шашырау және тығыздық графиктер
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include=[np.number])
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]]
    columnNames = list(df)
    if len(columnNames) > 10:
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize,
                                                             plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k=1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2),
                          xycoords='axes fraction', ha='center',
                          va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()

# Корреляциялық матрицасын құратын функция
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]]
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of '
              f'non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80,
               facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()





nRowsRead = 1000
df1 = pd.read_csv('C:/Users/Ансар/OneDrive/Рабочий стол/Students drinking alchogol/student-mat.csv', delimiter=',',
                  nrows=nRowsRead)
df1.dataframeName = 'student-mat.csv'
nRow, nCol = df1.shape
print(f'Бізде {nRow} жолдар мен {nCol} бағандар бар')

print(df1.head(5))
# plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 8)
# plotScatterMatrix(df1, 20, 10)

nRowsRead = 1000
df2 = pd.read_csv('C:/Users/Ансар/OneDrive/Рабочий стол/Students drinking alchogol/student-por.csv', delimiter=',',
                  nrows=nRowsRead)
df2.dataframeName = 'student-por.csv'
nRow, nCol = df2.shape
print(f'Бізде {nRow} жолдар мен {nCol} бағандар бар')

print(df2.head(5))
plotPerColumnDistribution(df2, 10, 5)
plotCorrelationMatrix(df2, 8)
plotScatterMatrix(df2, 20, 10)



