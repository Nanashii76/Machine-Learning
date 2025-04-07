import numpy as np
import matplotlib.pyplot as plt

# Passo 1: Gerar Dados Sintéticos
np.random.seed(42)  # Semente para reprodutibilidade
tempo_de_estudo = 10 * np.random.rand(100, 1)  # Tempo de estudo entre 0 e 10 horas
nota_base = 5  # Nota base assumida sem estudo
nota_por_hora = 0.5  # Ganho de nota por hora de estudo

notas = nota_base + nota_por_hora * tempo_de_estudo + np.random.randn(100, 1)  # Notas simuladas

# Passo 2: Cálculo dos Coeficientes com Mínimos Quadrados
X_b = np.c_[np.ones((100, 1)), tempo_de_estudo]  # Adicionando termo de interceptação
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(notas)

# Passo 3: Visualização dos Dados e do Ajuste Linear
plt.scatter(tempo_de_estudo, notas, color='blue', label='Dados simulados')
line_x = np.linspace(0, 10, 100).reshape(100, 1)
line_y = theta_best[0] + theta_best[1] * line_x
plt.plot(line_x, line_y, color='red', label='Ajuste Linear')
plt.xlabel('Tempo de Estudo (horas)')
plt.ylabel('Nota')
plt.title('Influência do Tempo de Estudo na Nota do Aluno')
plt.legend()
plt.show()

# Passo 4: Predição de Notas com o Modelo
X_new = np.array([[0], [10]])  # Aluno que não estudou e aluno que estudou 10 horas
X_new_b = np.c_[np.ones((2, 1)), X_new]
notas_previstas = X_new_b.dot(theta_best)

notas_previstas