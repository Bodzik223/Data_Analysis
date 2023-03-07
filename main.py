import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

df_auto = pd.read_csv('clean_df_auto.csv')

print(df_auto.dtypes)  # Вивід типів даних всіх стовпців
print(df_auto.corr())  # вивід таблиці кореляції всіх стовпців між собою


# Пергляд кореляції між різними змінними
print(df_auto[["engine-size", "price"]].corr())
sns.regplot(x="engine-size", y="price", data=df_auto)
plt.ylim(0,)
# plt.savefig('1.png')
plt.show()

print(df_auto[["highway-L/100km", "price"]].corr())
print(df_auto[["stroke", "price"]].corr())

sns.regplot(x="stroke", y="price", data=df_auto)
plt.ylim(0,)
# plt.savefig('2.png')
plt.show()



# Використання boxplot для демонстрації зв'язків між змінними
sns.boxplot(x="body-style", y="price", data=df_auto)
plt.ylim(0,)
# plt.savefig('3.png')
plt.show()

sns.boxplot(x="engine-location", y="price", data=df_auto)
plt.ylim(0,)
# plt.savefig('4.png')
plt.show()

sns.boxplot(x="drive-wheels", y="price", data=df_auto)
plt.ylim(0,)
# plt.savefig('5.png')
plt.show()

print(df_auto.describe()) # Вивід загальної статистики для стовпців які не є типом object

print(df_auto['drive-wheels'].value_counts()) # Підрахунок значень в совпці

drive_wheels_counts = df_auto['drive-wheels'].value_counts().to_frame() # Підрахунок значень в совпці і пертворення цього в data frame
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True) # Зміна назви стовпця в data frame
drive_wheels_counts.index.name = 'drive-wheels' # Зміна назви індексу
print(drive_wheels_counts)

engine_loc_counts = df_auto['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
print(engine_loc_counts.head(10))



# Групування за стовпцями
print(df_auto['drive-wheels'].unique()) # Вивід унікальних значень стовпця

df_group_one = df_auto[['drive-wheels','body-style','price']] # Вибір стовпців
df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean() # Групування за одним стовпцем і обрахування середньої ціни
print(df_group_one)

df_group_body_style = df_auto[['body-style','price']] # Вибір стовпців
df_group_body_style = df_group_body_style.groupby(['body-style'],as_index=False).mean() # Групування за одним стовпцем і обрахування середньої ціни
print(df_group_body_style)

df_gptest = df_auto[['drive-wheels','body-style','price']] # Вибір стовпців
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean() # Групування за декількома стовпцями і обрахування середньої ціни
print(grouped_test1)

grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')  # Створення зведеної таблиці для кращої наглядності
grouped_pivot = grouped_pivot.fillna(0) # Заміна всіх NaN на 0 в зведені таблиці
print(grouped_pivot)


# Теплова карта для grouped_pivot
fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

plt.xticks(rotation=30)

fig.colorbar(im)
# plt.savefig('6.png')
plt.show()


#Розрахування коефіцієнту кореляції Пірсона та P-значення для wheel-base та price
pearson_coef, p_value = stats.pearsonr(df_auto['wheel-base'], df_auto['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#Розрахування коефіцієнту кореляції Пірсона та P-значення для horsepower та price
pearson_coef, p_value = stats.pearsonr(df_auto['horsepower'], df_auto['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

#Розрахування коефіцієнту кореляції Пірсона та P-значення для length та price
pearson_coef, p_value = stats.pearsonr(df_auto['length'], df_auto['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

#Розрахування коефіцієнту кореляції Пірсона та P-значення для width та price
pearson_coef, p_value = stats.pearsonr(df_auto['width'], df_auto['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value )

#Розрахування коефіцієнту кореляції Пірсона та P-значення для curb-weight та price
pearson_coef, p_value = stats.pearsonr(df_auto['curb-weight'], df_auto['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

#Розрахування коефіцієнту кореляції Пірсона та P-значення для engine-size та price
pearson_coef, p_value = stats.pearsonr(df_auto['engine-size'], df_auto['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#Розрахування коефіцієнту кореляції Пірсона та P-значення для bore та price
pearson_coef, p_value = stats.pearsonr(df_auto['bore'], df_auto['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =  ", p_value )

#Розрахування коефіцієнту кореляції Пірсона та P-значення для city-mpg та price
pearson_coef, p_value = stats.pearsonr(df_auto['city-mpg'], df_auto['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

#Розрахування коефіцієнту кореляції Пірсона та P-значення для highway-L/100km та price
pearson_coef, p_value = stats.pearsonr(df_auto['highway-L/100km'], df_auto['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value )


grouped_test2=df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])

# Отримання результат F-тесту та P-значення дисперсійного аналізу груп : fwd, 4wd, rwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'],grouped_test2.get_group('4wd')['price'])
print("ANOVA results: F=", f_val, ", P =", p_val)

# Отримання результат F-тесту та P-значення дисперсійного аналізу груп : fwd, rwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])
print("ANOVA results: F=", f_val, ", P =", p_val)

# Отримання результат F-тесту та P-значення дисперсійного аналізу груп : 4wd, rwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price'])
print("ANOVA results: F=", f_val, ", P =", p_val)

# Отримання результат F-тесту та P-значення дисперсійного аналізу груп : 4wd, fwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('fwd')['price'])
print("ANOVA results: F=", f_val, ", P =", p_val)