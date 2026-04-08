import numpy as np
import matplotlib.pyplot as plt

class EasyLogisticRegression:
    
    def __init__(self, learning_rate = 0.01, num_iterations = 1000):
        """
        learning_rate - коэф обучения, рекомендуется для начала от 0.01 до 0.1
        num_iterations - через какое количество будем переобучаться
        w - вес
        b - смещение
        """
        self.lr = learning_rate
        self.num_iterations = num_iterations
        self.w = None
        self.b = None 
        self.history = []
    
    def sigmoid(self, z):
        """
        Сигмоидная функция. σ(z) = 1 / (1 + e^(-z))
        Служит для конвертации в вероятность от 0 до 1
        """
        z = np.clip(z, -500, 500) # Без этой строки пробовал запустить и выдало ошибку, что слишком большое значение
        return 1 / (1 + np.exp(-z))
    
    def intercept(self, y):
        """
        Инициализурую смещение через лог среднего
        """
        y_mean = np.mean(y)
        # Защита от log(0) и log(inf)
        y_mean = np.clip(y_mean, 1e-8, 1 - 1e-8)
        
        return np.log(y_mean / (1 - y_mean))
    
    def fit(self, X, y, plot=True):
        """
        Обучение модели
        X - массив с признаками (образцы, признаки)
        y - массив с ответами 
        """
        self.history = []
        n_samples, n_features = X.shape
        # Начало с нулевыми весами
        self.w = np.zeros(n_features)
        self.b = self.intercept(y)
        for i in range(self.num_iterations):
            # np.dot - вычисляет скалярное произведение
            # применение формулы w1*x1 + w2*x2 + ... + b
            z = np.dot(X, self.w) + self.b
            
            # sigmoid
            y_pred = self.sigmoid(z)
            
            # смотрим, сильно ли ошиблись
            error = y_pred - y
            
            # обновление веса, градиентный спуск, формулы нашел во Всемирной паутине.
            # self.lr - скорость обучения, (1/n_samples) - усреднение по всем образцам, np.dot(X.T,error) - как сильно признак ошибся
            self.w = self.w - self.lr * (1/n_samples) * np.dot(X.T, error)
            self.b = self.b - self.lr * (1/n_samples) * np.sum(error)
            
            if i % 100 == 0:
                # Просто считаем, сколько раз угадали
                predictions = (y_pred >= 0.5).astype(int)
                accuracy = np.mean(predictions == y)
                self.history.append((i, accuracy))
                print(f"Шаг {i}, Точность: {accuracy:.4f}")
                
        # Рисуем график после обучения
        if plot:
            self.plot_convergence()
        
        return self    
    
    def plot_convergence(self):
        """Рисует график сходимости"""
        if not self.history:
            print("Нет данных для графика. Сначала обучите модель.")
            return
        
        iterations = [h[0] for h in self.history]
        accuracies = [h[1] for h in self.history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, accuracies, 'b-o', linewidth=2, markersize=6)
        plt.xlabel('Итерация', fontsize=12)
        plt.ylabel('Точность (Accuracy)', fontsize=12)
        plt.title('Сходимость логистической регрессии', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Добавляем финальную точку
        plt.plot(iterations[-1], accuracies[-1], 'ro', markersize=10, label=f'Финальная точность: {accuracies[-1]:.4f}')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    def predict(self,X):
        z = np.dot(X, self.w) + self.b
        predicted = self.sigmoid(z)
        return(predicted >= 0.5).astype(int)
    
    def predict_proba(self, X):
        """Возвращает вероятности классов"""
        z = np.dot(X, self.w) + self.b
        proba = self.sigmoid(z)
        return np.column_stack((1 - proba, proba))