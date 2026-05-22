
# #1 общ
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# from sklearn.datasets import make_blobs
# from sklearn.cluster import KMeans
# import pandas as pd

# dataset, classes = make_blobs(n_samples=200, n_features=2,
#                               centers=4, cluster_std=0.5, random_state=0)
# df = pd.DataFrame(dataset, columns=['var1', 'var2'])
# print(df.head(2))

# inertias = []
# k_range = range(1, 12)
# for k in k_range:
#     model = KMeans(n_clusters=k, random_state=0)
#     model.fit(df)
#     inertias.append(model.inertia_)

# plt.figure()
# plt.plot(list(k_range), inertias, 'bo-')
# plt.xlabel('k')
# plt.ylabel('Distortion score')
# plt.title('Elbow Method')
# plt.savefig('elbow_plot.png')
# plt.close() 
# print("Elbow plot saved as elbow_plot.png")

# kmeans = KMeans(n_clusters=4, init='k-means++',
# random_state=0).fit(df)
# print (kmeans.labels_)
# print (kmeans.cluster_centers_)
# print(kmeans.inertia_)
# print(kmeans.n_iter_)

# from collections import Counter
# Counter(kmeans.labels_)
# print(Counter(kmeans.labels_))

# import seaborn as sns

# sns.scatterplot(data=df, x='var1', y='var2', hue=kmeans.labels_)
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="X", c="r", s=80, label="centroids")
# plt.grid(True)
# plt.savefig('scatter_plot.png')
# plt.close() 
# print("Scatter plot saved as scatter_plot.png") 

# #2 инд 3 вариант
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from collections import Counter


# Блок 1. Генерация исходных данных по варианту 3
# Первый параметр: [0; 1]
# Второй параметр: [-2; 2]
# Третий параметр: "да" / "нет"


np.random.seed(42)
n_samples = 200

param1 = np.random.uniform(0, 1, n_samples)
param2 = np.random.uniform(-2, 2, n_samples)
param3_text = np.random.choice(['да', 'нет'], size=n_samples)

# Преобразуем категориальный параметр в числовой для KMeans
param3_num = np.where(param3_text == 'да', 1, 0)

df = pd.DataFrame({
    'param1': param1,
    'param2': param2,
    'param3_text': param3_text,
    'param3_num': param3_num
})

print("Первые 10 строк набора данных:")
print(df.head(10))
print()


# Блок 2. Подготовка признаков для кластеризации
# KMeans работает только с числовыми признаками


X = df[['param1', 'param2', 'param3_num']]


# Блок 3. Поиск оптимального числа кластеров методом локтя

inertias = []
k_range = range(1, 11)

for k in k_range:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    inertias.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(k_range), inertias, 'bo-')
plt.xlabel('Количество кластеров k')
plt.ylabel('Inertia / Distortion score')
plt.title('Метод локтя для варианта 3')
plt.grid(True)
plt.savefig('variant3_elbow_plot.png')
plt.close()

print("График метода локтя сохранен как variant3_elbow_plot.png")
print()


# Блок 4. Обучение итоговой модели KMeans
# Для примера выбрано 3 кластера


kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

df['cluster'] = kmeans.labels_


# Блок 5. Вывод результатов кластеризации в консоль


print("Предсказанные кластеры для каждой точки:")
print(kmeans.labels_)
print()

print("Координаты центроидов:")
print(kmeans.cluster_centers_)
print()

print("Внутрикластерная сумма квадратов (inertia):")
print(kmeans.inertia_)
print()

print("Количество итераций:")
print(kmeans.n_iter_)
print()

cluster_sizes = Counter(kmeans.labels_)
print("Размер каждого кластера:")
print(cluster_sizes)
print()

print("Таблица с результатами кластеризации:")
print(df.head(20))
print()


# Блок 6. Визуализация кластеров
# Для 2D-графика используем param1 и param2
# Цвет - по метке кластера
# Стиль - по признаку "да/нет"


plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x='param1',
    y='param2',
    hue='cluster',
    style='param3_text',
    s=70
)

# Наносим центроиды только по первым двум координатам
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker='X',
    c='red',
    s=200,
    label='Центроиды'
)

plt.title('Кластеризация данных, вариант 3')
plt.xlabel('Первый параметр [0;1]')
plt.ylabel('Второй параметр [-2;2]')
plt.grid(True)
plt.legend()
plt.savefig('variant3_scatter_plot.png')
plt.close()

print("Диаграмма рассеяния сохранена как variant3_scatter_plot.png")
