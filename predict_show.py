import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import random

# 예시 모델 (사용자 모델로 교체 가능)
class DummyModel:
    def predict(self, x):
        return 2 * x[0] + random.uniform(-2, 2)

# 데이터 준비
num_samples = 200
x = np.random.rand(num_samples, 1) * 50
y = 2 * x[:, 0] + np.random.normal(0, 5, size=num_samples)
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
        self.paused = True  # 처음에는 일시정지 상태

        # 시각화용 리스트
        self.x_vals = []
        self.gt_vals = []
        self.pred_vals = []

        # 차트 설정
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.gt_line, = self.ax.plot([], [], 'g-', label='Ground Truth')
        self.pred_line, = self.ax.plot([], [], 'r--', label='Prediction')
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(min(self.y_all)-10, max(self.y_all)+10)
        self.ax.set_title("회귀 결과 실시간 시연")
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(pady=20)

        # 버튼 프레임
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        self.start_button = ttk.Button(btn_frame, text="시작", command=self.resume_simulation)
        self.start_button.pack(side="left", padx=10)

        self.pause_button = ttk.Button(btn_frame, text="일시정지", command=self.pause_simulation)
        self.pause_button.pack(side="left", padx=10)

    def resume_simulation(self):
        if not self.paused:
            return
        self.paused = False
        self.update_plot()

    def pause_simulation(self):
        self.paused = True

    def update_plot(self):
        if self.paused:
            return

        if self.index >= len(self.x_all):
            print("모든 데이터를 시연 완료했습니다.")
            return

        x_input = self.x_all[self.index].reshape(1, -1)
        gt = self.y_all[self.index]
        pred = self.model.predict(x_input)

        self.x_vals.append(self.index)
        self.gt_vals.append(gt)
        self.pred_vals.append(pred)

        self.gt_line.set_data(self.x_vals, self.gt_vals)
        self.pred_line.set_data(self.x_vals, self.pred_vals)

        if self.index > 100:
            self.ax.set_xlim(self.index - 100, self.index + 10)

        self.ax.relim()
        self.ax.autoscale_view(True, True, True)
        self.canvas.draw()

        self.index += 1
        self.root.after(100, self.update_plot)  # 100ms 후에 재호출

# 실행
if __name__ == "__main__":
    root = tk.Tk()
    app = RealtimePlotApp(root, x_data=x, y_data=y, model=model)
    root.mainloop()
