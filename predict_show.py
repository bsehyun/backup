import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import random

# 예시용: 가짜 모델 클래스 (사용자는 실제 모델로 대체)
class DummyModel:
    def predict(self, x):
        # 단순 선형 모델: 가짜 예측 (예: y = 2 * x[0] + noise)
        return 2 * x[0] + random.uniform(-2, 2)

# 가상의 입력 x, 출력 y 데이터 준비
num_samples = 200
x = np.random.rand(num_samples, 1) * 50  # 입력값
y = 2 * x[:, 0] + np.random.normal(0, 5, size=num_samples)  # 실제값

# 모델 인스턴스 (사용자는 자신이 정의한 model 사용)
model = DummyModel()

class RealtimePlotApp:
    def __init__(self, root, x_data, y_data, model):
        self.root = root
        self.root.title("회귀 예측 vs Ground Truth 실시간 시연")
        self.root.geometry("800x600")

        # 데이터
        self.x_all = x_data
        self.y_all = y_data
        self.model = model
        self.index = 0

        # 시각화용 리스트
        self.x_vals = []
        self.gt_vals = []
        self.pred_vals = []

        # matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.gt_line, = self.ax.plot([], [], 'g-', label='Ground Truth')
        self.pred_line, = self.ax.plot([], [], 'r--', label='Prediction')
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(min(self.y_all)-10, max(self.y_all)+10)
        self.ax.set_title("회귀 결과 실시간 시연")
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(pady=20)

        self.start_button = ttk.Button(self.root, text="시작", command=self.start_simulation)
        self.start_button.pack()

    def start_simulation(self):
        self.update_plot()

    def update_plot(self):
        if self.index >= len(self.x_all):
            print("모든 데이터를 시연 완료했습니다.")
            return

        # 입력값으로부터 예측
        x_input = self.x_all[self.index].reshape(1, -1)  # 예: (1, feature_dim)
        gt = self.y_all[self.index]
        pred = self.model.predict(x_input)

        # 저장
        self.x_vals.append(self.index)
        self.gt_vals.append(gt)
        self.pred_vals.append(pred)

        # 그래프 업데이트
        self.gt_line.set_data(self.x_vals, self.gt_vals)
        self.pred_line.set_data(self.x_vals, self.pred_vals)

        if self.index > 100:
            self.ax.set_xlim(self.index - 100, self.index + 10)

        self.ax.relim()
        self.ax.autoscale_view(True, True, True)
        self.canvas.draw()

        self.index += 1
        self.root.after(100, self.update_plot)  # 100ms마다 업데이트

# 실행
if __name__ == "__main__":
    root = tk.Tk()
    app = RealtimePlotApp(root, x_data=x, y_data=y, model=model)
    root.mainloop()
