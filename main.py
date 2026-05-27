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

# #инд_вар1
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# import numpy as np
# from sklearn.cluster import KMeans
# import pandas as pd
# from mpl_toolkits.mplot3d import Axes3D

# # Генерация данных по варианту 1
# dataset1 = np.random.uniform(0.1, 5, 200)
# dataset2 = np.random.uniform(0.1, 3, 200)
# dataset3_text = np.random.choice(['10%', '20%', '80%', '90%'], 200)

# mapping = {'10%': 10, '20%': 20, '80%': 80, '90%': 90}
# dataset3 = [mapping[x] for x in dataset3_text]

# df = pd.DataFrame({
#     'var1': dataset1,
#     'var2': dataset2,
#     'var3': dataset3
# })

# print(df.head(2))

# # Поиск числа кластеров методом локтя
# inertias = []
# k_range = range(1, 12)
# for k in k_range:
#     model = KMeans(n_clusters=k, random_state=0, n_init=10)
#     model.fit(df)
#     inertias.append(model.inertia_)

# plt.figure()
# plt.plot(list(k_range), inertias, 'bo-')
# plt.xlabel('k')
# plt.ylabel('Distortion score')
# plt.title('Elbow Method')
# plt.savefig('elbow_var1.png')
# plt.close()
# print("Elbow plot saved as elbow_var1.png")

# # Обучение итоговой модели
# kmeans = KMeans(n_clusters=4, init='k-means++',
# random_state=0, n_init=10).fit(df)
# print(kmeans.labels_)
# print(kmeans.cluster_centers_)
# print(kmeans.inertia_)
# print(kmeans.n_iter_)

# from collections import Counter
# print(Counter(kmeans.labels_))

# # Построение 3D-графика
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(df['var1'], df['var2'], df['var3'], c=kmeans.labels_)
# ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
#            kmeans.cluster_centers_[:, 2], marker="X", c="r", s=80, label="centroids")
# ax.set_xlabel('var1')
# ax.set_ylabel('var2')
# ax.set_zlabel('var3')
# plt.legend()
# plt.savefig('scatter_var1.png')
# plt.close()
# print("Scatter plot saved as scatter_var1.png")


# #вар2
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# import numpy as np
# from sklearn.cluster import KMeans
# import pandas as pd
# from mpl_toolkits.mplot3d import Axes3D

# # Генерация данных по варианту 2
# dataset1 = np.random.uniform(0.01, 1, 200)
# dataset2 = np.random.uniform(1, 300, 200)
# dataset3_text = np.random.choice(['Самара', 'Тольятти', 'Москва'], 200)

# mapping = {'Самара': 1, 'Тольятти': 2, 'Москва': 3}
# dataset3 = [mapping[x] for x in dataset3_text]

# df = pd.DataFrame({
#     'var1': dataset1,
#     'var2': dataset2,
#     'var3': dataset3
# })

# print(df.head(2))

# # Поиск числа кластеров методом локтя
# inertias = []
# k_range = range(1, 12)
# for k in k_range:
#     model = KMeans(n_clusters=k, random_state=0, n_init=10)
#     model.fit(df)
#     inertias.append(model.inertia_)

# plt.figure()
# plt.plot(list(k_range), inertias, 'bo-')
# plt.xlabel('k')
# plt.ylabel('Distortion score')
# plt.title('Elbow Method')
# plt.savefig('elbow_var2.png')
# plt.close()
# print("Elbow plot saved as elbow_var2.png")

# # Обучение итоговой модели
# kmeans = KMeans(n_clusters=3, init='k-means++',
# random_state=0, n_init=10).fit(df)
# print(kmeans.labels_)
# print(kmeans.cluster_centers_)
# print(kmeans.inertia_)
# print(kmeans.n_iter_)

# from collections import Counter
# print(Counter(kmeans.labels_))

# # Построение 3D-графика
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(df['var1'], df['var2'], df['var3'], c=kmeans.labels_)
# ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
#            kmeans.cluster_centers_[:, 2], marker="X", c="r", s=80, label="centroids")
# ax.set_xlabel('var1')
# ax.set_ylabel('var2')
# ax.set_zlabel('var3')
# plt.legend()
# plt.savefig('scatter_var2.png')
# plt.close()
# print("Scatter plot saved as scatter_var2.png")


# 
# #вар4
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Генерация данных по варианту 4
dataset1 = np.random.uniform(-10, 1, 200)
dataset2 = np.random.uniform(1, 2, 200)
dataset3_text = np.random.choice(['отрицательное значение', 'положительное значение'], 200)

mapping = {'отрицательное значение': 0, 'положительное значение': 1}
dataset3 = [mapping[x] for x in dataset3_text]

df = pd.DataFrame({
    'var1': dataset1,
    'var2': dataset2,
    'var3': dataset3
})

print(df.head(2))

# Поиск числа кластеров методом локтя
inertias = []
k_range = range(1, 12)
for k in k_range:
    model = KMeans(n_clusters=k, random_state=0, n_init=10)
    model.fit(df)
    inertias.append(model.inertia_)

plt.figure()
plt.plot(list(k_range), inertias, 'bo-')
plt.xlabel('k')
plt.ylabel('Distortion score')
plt.title('Elbow Method')
plt.savefig('elbow_var4.png')
plt.close()
print("Elbow plot saved as elbow_var4.png")

# Обучение итоговой модели
kmeans = KMeans(n_clusters=4, init='k-means++',
random_state=0, n_init=10).fit(df)
print(kmeans.labels_)
print(kmeans.cluster_centers_)
print(kmeans.inertia_)
print(kmeans.n_iter_)

from collections import Counter
print(Counter(kmeans.labels_))

# Построение 3D-графика
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['var1'], df['var2'], df['var3'], c=kmeans.labels_)
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
           kmeans.cluster_centers_[:, 2], marker="X", c="r", s=80, label="centroids")
ax.set_xlabel('var1')
ax.set_ylabel('var2')
ax.set_zlabel('var3')
plt.legend()
plt.savefig('scatter_var4.png')
plt.close()
print("Scatter plot saved as scatter_var4.png")