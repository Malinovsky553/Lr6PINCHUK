
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


# инд 4 вариант 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from collections import Counter
import seaborn as sns

# Генерация данных по варианту 4
np.random.seed(0)
n_samples = 200

param1 = np.random.uniform(-10, 1, n_samples)
param2 = np.random.uniform(1, 2, n_samples)
param3_text = np.random.choice(['отрицательное значение', 'положительное значение'], size=n_samples)

# Переводим текст в числа
mapping = {
    'отрицательное значение': 0,
    'положительное значение': 1
}
param3_num = [mapping[x] for x in param3_text]

df = pd.DataFrame({
    'var1': param1,
    'var2': param2,
    'var3_text': param3_text,
    'var3_num': param3_num
})

print(df.head(2))

X = df[['var1', 'var2', 'var3_num']]

inertias = []
k_range = range(1, 12)

for k in k_range:
    model = KMeans(n_clusters=k, random_state=0, n_init=10)
    model.fit(X)
    inertias.append(model.inertia_)

plt.figure()
plt.plot(list(k_range), inertias, 'bo-')
plt.xlabel('k')
plt.ylabel('Distortion score')
plt.title('Elbow Method')
plt.savefig('var4elbow_plot.png')
plt.close()
print("Elbow plot saved as elbow_plot.png")

kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0, n_init=10).fit(X)

print(kmeans.labels_)
print(kmeans.cluster_centers_)
print(kmeans.inertia_)
print(kmeans.n_iter_)
print(Counter(kmeans.labels_))

plt.figure()
sns.scatterplot(data=df, x='var1', y='var2', hue=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="X", c="r", s=80, label="centroids")
plt.grid(True)
plt.legend(loc='upper right')
plt.savefig('var4scatter_plot.png')
plt.close()
print("Scatter plot saved as scatter_plot.png")
#  инд 3 вариант
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


# #2 вариант
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# import numpy as np
# import pandas as pd

# from sklearn.cluster import KMeans
# from collections import Counter
# import seaborn as sns

# # Генерация данных по варианту 2
# np.random.seed(0)
# n_samples = 200

# param1 = np.random.uniform(0.01, 1, n_samples)
# param2 = np.random.uniform(1, 300, n_samples)
# param3_text = np.random.choice(['Самара', 'Тольятти', 'Москва'], size=n_samples)

# # Переводим города в числа
# mapping = {'Самара': 1, 'Тольятти': 2, 'Москва': 3}
# param3_num = [mapping[x] for x in param3_text]

# df = pd.DataFrame({
#     'var1': param1,
#     'var2': param2,
#     'var3_text': param3_text,
#     'var3_num': param3_num
# })

# print(df.head(2))

# X = df[['var1', 'var2', 'var3_num']]

# inertias = []
# k_range = range(1, 12)

# for k in k_range:
#     model = KMeans(n_clusters=k, random_state=0, n_init=10)
#     model.fit(X)
#     inertias.append(model.inertia_)

# plt.figure()
# plt.plot(list(k_range), inertias, 'bo-')
# plt.xlabel('k')
# plt.ylabel('Distortion score')
# plt.title('Elbow Method')
# plt.savefig('var2elbow_plot.png')
# plt.close()
# print("Elbow plot saved as elbow_plot.png")

# kmeans = KMeans(n_clusters=3, init='k-means++', random_state=0, n_init=10).fit(X)

# print(kmeans.labels_)
# print(kmeans.cluster_centers_)
# print(kmeans.inertia_)
# print(kmeans.n_iter_)
# print(Counter(kmeans.labels_))

# plt.figure()
# sns.scatterplot(data=df, x='var1', y='var2', hue=kmeans.labels_)
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="X", c="r", s=80, label="centroids")
# plt.grid(True)
# plt.legend(loc='upper right')
# plt.savefig('var2scatter_plot.png')
# plt.close()
# print("Scatter plot saved as scatter_plot.png")

# #1 вариант
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# import numpy as np
# import pandas as pd

# from sklearn.cluster import KMeans
# from collections import Counter
# import seaborn as sns

# # Генерация данных по варианту 1
# np.random.seed(0)
# n_samples = 200

# param1 = np.random.uniform(0.1, 5, n_samples)
# param2 = np.random.uniform(0.1, 3, n_samples)
# param3_text = np.random.choice(['10%', '20%', '80%', '90%'], size=n_samples)

# # Переводим проценты в числа
# mapping = {'10%': 10, '20%': 20, '80%': 80, '90%': 90}
# param3_num = [mapping[x] for x in param3_text]

# df = pd.DataFrame({
#     'var1': param1,
#     'var2': param2,
#     'var3_text': param3_text,
#     'var3_num': param3_num
# })

# print(df.head(2))

# X = df[['var1', 'var2', 'var3_num']]

# inertias = []
# k_range = range(1, 12)

# for k in k_range:
#     model = KMeans(n_clusters=k, random_state=0, n_init=10)
#     model.fit(X)
#     inertias.append(model.inertia_)

# plt.figure()
# plt.plot(list(k_range), inertias, 'bo-')
# plt.xlabel('k')
# plt.ylabel('Distortion score')
# plt.title('Elbow Method')
# plt.savefig('var1elbow_plot.png')
# plt.close()
# print("Elbow plot saved as elbow_plot.png")

# kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0, n_init=10).fit(X)

# print(kmeans.labels_)
# print(kmeans.cluster_centers_)
# print(kmeans.inertia_)
# print(kmeans.n_iter_)
# print(Counter(kmeans.labels_))

# plt.figure()
# sns.scatterplot(data=df, x='var1', y='var2', hue=kmeans.labels_)
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="X", c="r", s=80, label="centroids")
# plt.grid(True)
# plt.legend(loc='upper right')
# plt.savefig('var1scatter_plot.png')
# plt.close()
# print("Scatter plot saved as scatter_plot.png")