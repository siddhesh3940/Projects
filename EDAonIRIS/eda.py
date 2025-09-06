# %%
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np

# %%
iris_d = sb.load_dataset("iris")

# %%
iris_d.head()


# %%
iris_d.tail()

# %%
iris_d.shape

# %%
iris_d.info()

# %%
iris_d['sepal_length'].describe()

# %%
iris_d.describe()

# %%
iris_d.isnull().sum()

# %%
plt.scatter(iris_d['sepal_length'],iris_d['sepal_width'], color='red')
plt.title("scatter plot")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.show()

# %%
plt.hist(iris_d['sepal_width'], bins=40)
plt.title("Histogram")
plt.xlabel("Sepal width")
plt.ylabel("Frequency")
plt.show()

# %%
plt.hist(iris_d['sepal_width'], bins=15)
plt.title("Histogram")
plt.xlabel("Sepal width")
plt.ylabel("Frequency")
plt.show()

# %%
plt.hist(iris_d['sepal_width'], bins='auto')
plt.title("Histogram")
plt.xlabel("Sepal width")
plt.ylabel("Frequency")
plt.show()

# %%
plt.hist(iris_d['petal_width'], bins=40)
plt.title("Histogram")
plt.xlabel("Petal width")
plt.ylabel("Frequency")
plt.show()

# %%
plt.hist(iris_d['petal_width'], bins=5)
plt.title("Histogram")
plt.xlabel("Petal width")
plt.ylabel("Frequency")
plt.show()

# %%
sb.boxplot(x="sepal_width", data=iris_d)
plt.title("Box Plot")

# %%
sb.boxplot(x="sepal_length", data=iris_d)
plt.title("Box Plot")

# %%
import scipy.stats as stats

stats.probplot(iris_d['petal_length'], dist="norm", plot=plt)
plt.title("Q-Q Plot of Sepal Width (Normal Distribution)")
plt.grid(True)
plt.show()

# %%
import scipy.stats as stats

stats.probplot(iris_d['sepal_length'], dist="norm", plot=plt)
plt.title("Q-Q Plot of Sepal Length (Normal Distribution)")
plt.grid(True)
plt.show()

# %%
import scipy.stats as stats

stats.probplot(iris_d['petal_length'], dist="uniform", plot=plt)
plt.title("Q-Q Plot of Sepal Width (Uniform Distribution)")
plt.grid(True)
plt.show()


