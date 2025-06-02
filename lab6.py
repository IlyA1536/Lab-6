import numpy as np
import matplotlib.pyplot as plt


# --- Завдання 1 ---
# 1. Генеруємо дані навколо прямої y = kx + b
true_k = -1.7
true_b = 4.2
x = np.linspace(-10, 10, 100)
noise = np.random.normal(0, 3, size=x.shape)
y = true_k * x + true_b + noise

# 2. Метод найменших квадратів
def least_squares(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    k = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    b = y_mean - k * x_mean
    return k, b

k_least_squares, b_least_squares = least_squares(x, y)

# 3. Порівняння з np.polyfit
k_np, b_np = np.polyfit(x, y, 1)

print("Початкові параметри: k =", true_k, ", b =", true_b)
print("Метод найменших квадратів: k =", k_least_squares, ", b =", b_least_squares)
print("np.polyfit: k =", k_np, ", b =", b_np)

# 4. Побудова графіку
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Дані', color='lightblue')
plt.plot(x, true_k * x + true_b, label='Істинна лінія', linestyle='--', color='black')
plt.plot(x, k_least_squares * x + b_least_squares, label='МНК', color='green')
plt.plot(x, k_np * x + b_np, label='np.polyfit', color='red')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Графік 1 завдання')
plt.grid(True)
plt.show()

# --- Завдання 2 ---
# 1. Градієнтний спуск
def gradient_descent(x, y, learning_rate=0.001, n_iter=1000):
    k = 0.0
    b = 0.0
    n = len(x)
    errors = []

    for i in range(n_iter):
        y_pred = k * x + b
        error = np.mean((y - y_pred) ** 2)
        errors.append(error)

        # Градієнти
        dk = -2 * np.sum(x * (y - y_pred)) / n
        db = -2 * np.sum(y - y_pred) / n

        # Оновлення
        k -= learning_rate * dk
        b -= learning_rate * db

    return k, b, errors

k_gd, b_gd, errors = gradient_descent(x, y, learning_rate=0.001, n_iter=1000)

print("Градієнтний спуск: k =", k_gd, ", b =", b_gd)

# 2. Додамо на графік
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Дані', color='lightblue')
plt.plot(x, true_k * x + true_b, label='Істинна лінія', linestyle='--', color='black')
plt.plot(x, k_least_squares * x + b_least_squares, label='МНК', color='green')
plt.plot(x, k_np * x + b_np, label='np.polyfit', color='red')
plt.plot(x, k_gd * x + b_gd, label='Градієнтний спуск', color='blue')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Графік 2 завдання')
plt.grid(True)
plt.show()

# 3. Графік похибки
plt.figure(figsize=(10, 6))
plt.plot(errors, label='Похибка')
plt.xlabel('Ітерація')
plt.ylabel('Похибка')
plt.title('Графік похибки')
plt.grid(True)
plt.legend()
plt.show()
