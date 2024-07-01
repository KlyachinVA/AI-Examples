from annoy import AnnoyIndex
import numpy as np

n = 2
N = 1000
num_tree = 7
k = 5
data = np.random.random((N,n))
# Инициализация индекса
t = AnnoyIndex(n, 'angular')
# Варианты метрик -- "angular", "euclidean", "manhattan", "hamming", "dot"

# Добавление данных в индекс

for i in range(N):
    t.add_item(i,data[i])

# Построение индекса с num_tree деревьями
t.build(num_tree)

# Сохранение индекса
t.save('my_index.ann')

# Загрузка индекса
u = AnnoyIndex(n, 'angular')
u.load('my_index.ann')

# Найдите 5 ближайших соседей для элемента 0
print(t.get_nns_by_item(k, 5))

# Найдите 5 ближайших соседей для заданного вектора
indxs = t.get_nns_by_vector(data[k],5)
print(indxs)
print(data[0])
print(data[indxs])

