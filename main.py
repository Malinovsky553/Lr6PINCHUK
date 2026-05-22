
#1 общ
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import pandas as pd

dataset, classes = make_blobs(n_samples=200, n_features=2,
                              centers=4, cluster_std=0.5, random_state=0)
df = pd.DataFrame(dataset, columns=['var1', 'var2'])
print(df.head(2))

inertias = []
k_range = range(1, 12)
for k in k_range:
    model = KMeans(n_clusters=k, random_state=0)
    model.fit(df)
    inertias.append(model.inertia_)

plt.figure()
plt.plot(list(k_range), inertias, 'bo-')
plt.xlabel('k')
plt.ylabel('Distortion score')
plt.title('Elbow Method')
plt.savefig('elbow_plot.png')
plt.close() 
print("Elbow plot saved as elbow_plot.png")

kmeans = KMeans(n_clusters=4, init='k-means++',
random_state=0).fit(df)
print (kmeans.labels_)
print (kmeans.cluster_centers_)
print(kmeans.inertia_)
print(kmeans.n_iter_)

from collections import Counter
Counter(kmeans.labels_)
print(Counter(kmeans.labels_))

import seaborn as sns

sns.scatterplot(data=df, x='var1', y='var2', hue=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="X", c="r", s=80, label="centroids")
plt.grid(True)
plt.savefig('scatter_plot.png')
plt.close() 
print("Scatter plot saved as scatter_plot.png") 

# #2 инд 3 вариант
# Подключение библиотек
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# import numpy as np
# import pandas as pd

# from sklearn.cluster import KMeans

# # Генерируем данные по 3 варианту
# np.random.seed(0)
# n_samples = 200

# # Первый параметр от 0 до 1
# param1 = np.random.uniform(0, 1, n_samples)

# # Второй параметр от -2 до 2
# param2 = np.random.uniform(-2, 2, n_samples)

# # Третий параметр: да или нет
# param3_text = np.random.choice(['да', 'нет'], size=n_samples)

# # Переводим да/нет в числа для KMeans
# param3_num = np.where(param3_text == 'да', 1, 0)

# # Собираем данные в таблицу
# df = pd.DataFrame({
#     'var1': param1,
#     'var2': param2,
#     'var3_text': param3_text,
#     'var3_num': param3_num
# })

# # Смотрим первые строки
# print(df.head(2))

# # Берем признаки для кластеризации
# X = df[['var1', 'var2', 'var3_num']]

# # Ищем оптимальное число кластеров методом локтя
# inertias = []
# k_range = range(1, 12)

# for k in k_range:
#     model = KMeans(n_clusters=k, random_state=0, n_init=10)
#     model.fit(X)
#     inertias.append(model.inertia_)

# # Строим график локтя
# plt.figure()
# plt.plot(list(k_range), inertias, 'bo-')
# plt.xlabel('k')
# plt.ylabel('Distortion score')
# plt.title('Elbow Method')
# plt.savefig('var3elbow_plot.png')
# plt.close()
# print("Elbow plot saved as elbow_plot.png")

# # Обучаем итоговую модель KMeans
# kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0, n_init=10).fit(X)

# # Выводим результаты
# print(kmeans.labels_)
# print(kmeans.cluster_centers_)
# print(kmeans.inertia_)
# print(kmeans.n_iter_)

# # Считаем размер каждого кластера
# from collections import Counter
# print(Counter(kmeans.labels_))

# # Рисуем диаграмму рассеяния
# import seaborn as sns

# plt.figure()
# sns.scatterplot(data=df, x='var1', y='var2', hue=kmeans.labels_)

# # Показываем центроиды
# plt.scatter(
#     kmeans.cluster_centers_[:, 0],
#     kmeans.cluster_centers_[:, 1],
#     marker="X",
#     c="r",
#     s=80,
#     label="centroids"
# )

# plt.grid(True)
# plt.legend(loc='upper right')
# plt.savefig('var3scatter_plot.png')
# plt.close()
# print("Scatter plot saved as scatter_plot.png")
