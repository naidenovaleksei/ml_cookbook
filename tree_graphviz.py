
# берем дерево
import joblib
model = joblib.load("tree.joblib")
tree = model['model']
features = model['features']


# вариант 1 graphviz (лучше)
# sudo apt-get install graphviz
# sudo pip install graphviz
from graphviz import Source
from sklearn.tree import export_graphviz

# DOT формат дерева (строка)
tree_data = export_graphviz(tree, feature_names=features, class_names=["renew", "churn"], label='all', proportion=True, precision=3)
# отображаем дерево, созданное из DOT формата
Source(tree_data)


# вариант 2 matplotlib (хуже)
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# параметры plot_tree почти те же, что и в export_graphviz
plot_tree(tree, feature_names=features, class_names=["renew", "churn"], label='all', ax=ax)
plt.show()
