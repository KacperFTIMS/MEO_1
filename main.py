import tkinter as tk
from cProfile import label

import customtkinter as ctk
import matplotlib
from openpyxl.utils.units import dxa_to_cm

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from tkinter import messagebox

class LPGui:
    def __init__(self, root):
        self.root = root
        root.configure(background='#2b2b2b')
        root.title("Optymalizacja")
        root.geometry("1150x750")
        ctk.set_appearance_mode("dark")

        self.default_coeffs = [[1.0, -1.0],
                               [2.0,  1.0],
                               [1.0, -1.0]]
        self.default_sense = ['<=','<=','>=']
        self.default_rhs = [1000.0, 2000.0, 0.0]
        self.default_costs = [1.0, 3.0]

        self.extra_constraints = []
        self.constraint_index = 0
        self.build_ui()

    #UI
    def build_ui(self):
        # Top Frame
        top_frame = ctk.CTkFrame(self.root)
        top_frame.pack(fill='x')

        ctk.CTkLabel(top_frame, text="Podstawowe ograniczenia", font=("Segoe UI", 18, "bold")).pack(anchor='center', pady=(10,4))
        self.table = ctk.CTkFrame(top_frame)
        self.table.pack(anchor='center', pady=4)

        headers = ["", "A", "B", "Warunek", "Limit"]
        header_font = ("Segoe UI", 14, "bold")
        entry_font = ("Segoe UI", 13)
        for col, h in enumerate(headers):
            ctk.CTkLabel(self.table, text=h, width=100, font=header_font).grid(row=0, column=col, padx=2)

        self.base_entries = []
        for i, name in enumerate(["S1", "S2", "S3"]):
            ctk.CTkLabel(self.table, text=name, width=10, font=entry_font).grid(row=i+1, column=0)
            eA = ctk.CTkEntry(self.table, width=50, font=entry_font); eA.grid(row=i+1, column=1); eA.insert(0, str(self.default_coeffs[i][0]))
            eB = ctk.CTkEntry(self.table, width=50, font=entry_font); eB.grid(row=i+1, column=2); eB.insert(0, str(self.default_coeffs[i][1]))
            sense = ctk.CTkComboBox(self.table, values=['<=','>=','='], width=75, font=entry_font); sense.grid(row=i+1, column=3); sense.set(self.default_sense[i])
            rhs = ctk.CTkEntry(self.table, width=100, font=entry_font); rhs.grid(row=i+1, column=4); rhs.insert(0, str(self.default_rhs[i]))
            self.base_entries.append((eA, eB, sense, rhs))

        # Funkcja Celu
        cost_frame = ctk.CTkFrame(top_frame)
        cost_frame.pack(anchor='center', pady=6)
        ctk.CTkLabel(cost_frame, text="Funkcja celu: ").pack(side='left')
        ctk.CTkLabel(cost_frame, text="A").pack(side='left', padx=(8,0))
        self.costA = ctk.CTkEntry(cost_frame, width=50); self.costA.pack(side='left'); self.costA.insert(0, str(self.default_costs[0]))
        ctk.CTkLabel(cost_frame, text="B").pack(side='left', padx=(8,0))
        self.costB = ctk.CTkEntry(cost_frame, width=50); self.costB.pack(side='left'); self.costB.insert(0, str(self.default_costs[1]))

        self.mode_var = tk.StringVar(value='max')
        ctk.CTkRadioButton(cost_frame, text='Max', variable=self.mode_var, value='max').pack(side='left', padx=6)
        ctk.CTkRadioButton(cost_frame, text='Min', variable=self.mode_var, value='min').pack(side='left', padx=6)

        # Mid Frame
        mid_frame = ctk.CTkFrame(self.root)
        mid_frame.pack(fill='x')
        ctk.CTkLabel(mid_frame, text="Dodatkowe ograniczenia").pack(anchor='center')

        add_row = ctk.CTkFrame(mid_frame)
        add_row.pack(anchor='center', pady=4)
        ctk.CTkLabel(add_row, text="A:").pack(side='left')
        self.add_a1 = ctk.CTkEntry(add_row, width=50); self.add_a1.pack(side='left', padx=2)
        ctk.CTkLabel(add_row, text="B:").pack(side='left')
        self.add_a2 = ctk.CTkEntry(add_row, width=50); self.add_a2.pack(side='left', padx=2)
        self.add_sense = ctk.CTkComboBox(add_row, values=['<=','>=','='], width=75); self.add_sense.pack(side='left'
                                                                                    , padx=4); self.add_sense.set('<=')
        ctk.CTkLabel(add_row, text="Limit:").pack(side='left')
        self.add_rhs = ctk.CTkEntry(add_row, width=100);
        self.add_rhs.pack(side='left', padx=2)

        button_style = {
            "corner_radius": 16,
            "fg_color": "#5C0828",
            "hover_color": "#730a32",
            "border_color": "#730a32",
            "border_width": 2,
            "height": 40,
            "width": 150,
            "font": ("Segoe UI", 14, "bold")
        }

        # Przyciski nowych ograniczen
        btn_row = ctk.CTkFrame(mid_frame)
        btn_row.pack(anchor='center', pady=(10, 6))

        ctk.CTkButton(btn_row, text="Dodaj", command=self.on_add, **button_style).pack(side='left', padx=8)

        ctk.CTkButton(btn_row, text="Usuń ostatnie dodane", command=self.on_remove, **button_style).pack(side='left', padx=8)

        ctk.CTkLabel(mid_frame, text="Wykres").pack(anchor='center')

        # Przyciski wykresu
        btns = ctk.CTkFrame(mid_frame)
        btns.pack(anchor='center')
        ctk.CTkButton(btns, text="Następne ograniczenie", command=self.show_next_constraint, **button_style).pack(side='left', padx=4)
        ctk.CTkButton(btns, text="Oblicz", command=self.solve_and_plot, **button_style).pack(side='left', padx=4)
        ctk.CTkButton(btns, text="Resetuj wykres", command=self.reset_plot, **button_style).pack(side='left', padx=4)

        # Wykres
        plot_frame = ctk.CTkFrame(self.root)
        plot_frame.pack(fill='both', expand=True)
        self.fig, self.ax = plt.subplots(figsize=(6, 5))

        self.fig.patch.set_facecolor('#2b2b2b')
        self.ax.set_facecolor('#0e1111')
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.xaxis.label.set_color('white')
        self.ax.title.set_color('white')
        self.ax.grid(True, color='#333333')



        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    # Dodawanie ograniczeń
    def on_add(self):
        try:
            a1 = float(self.add_a1.get())
            a2 = float(self.add_a2.get())
            rhs = float(self.add_rhs.get())
            sense_val = self.add_sense.get()
        except ValueError:
            messagebox.showerror("Błąd", "Wprowadź poprawne liczby.")
            return

        row = len(self.base_entries)+len(self.extra_constraints)+1

        entry_font = ("Segoe UI", 13)
        eA = ctk.CTkEntry(self.table, width=50, font=entry_font)
        eA.grid(row=row, column=1)
        eA.insert(0, str(a1))
        eB = ctk.CTkEntry(self.table, width=50, font=entry_font)
        eB.grid(row=row, column=2)
        eB.insert(0, str(a2))
        sense = ctk.CTkComboBox(self.table, values=['<=', '>=', '='], width=75, font=entry_font)
        sense.grid(row=row, column=3)
        sense.set(sense_val)
        rhs_entry = ctk.CTkEntry(self.table, width=100, font=entry_font)
        rhs_entry.grid(row=row, column=4)
        rhs_entry.insert(0, str(rhs))

        self.extra_constraints.append((eA, eB, sense, rhs_entry))

    # Usuwanie Ograniczeń
    def on_remove(self):
        if not self.extra_constraints:
            messagebox.showinfo("Info", "Nie ma żadnych dodatkowych ograniczeń")
            return

        last_constraint = self.extra_constraints.pop()
        for widget in last_constraint:
            if hasattr(widget, 'destroy'):
                widget.destroy()

        last_row = len(self.base_entries)+len(self.extra_constraints)+1
        for child in self.table.grid_slaves(row=last_row):
            child.destroy()


    # Pobierz ograniczenia
    def get_constraints(self):
        cons = []
        for eA, eB, sense, rhs in self.base_entries + self.extra_constraints:
            cons.append((float(eA.get()), float(eB.get()), sense.get(), float(rhs.get())))
        return cons

    # Zarysuj ograniczenia
    def plot_constraint(self, a1, a2, sense, rhs, color='orange', alpha=0.3):
        x = np.linspace(0, 3000, 400)
        y_max = 3000
        y_min = 0

        if abs(a2) > 1e-8:
            y_line = (rhs - a1 * x) / a2
            self.ax.plot(x, y_line, color=color)
            if sense == '<=':
                if a2 > 0:
                    # Obszar pod linią
                    self.ax.fill_between(x, y_min, y_line, color=color, alpha=alpha)
                else:
                    # Obszar nad linią
                    self.ax.fill_between(x, y_line, y_max, color=color, alpha=alpha)
            elif sense == '>=':
                if a2 > 0:
                    # Obszar nad linią
                    self.ax.fill_between(x, y_line, y_max, color=color, alpha=alpha)
                else:
                    # Obszar pod linią
                    self.ax.fill_between(x, y_min, y_line, color=color, alpha=alpha)
            # '=' tylko linia, bez kolorowania
        else:
            # Pionowa linia
            x_val = rhs / a1
            self.ax.axvline(x_val, color=color)
            if sense == '<=':
                if a1 > 0:
                    self.ax.fill_betweenx(np.linspace(y_min, y_max, 400), x_min=0, x_max=x_val, color=color,
                                          alpha=alpha)
                else:
                    self.ax.fill_betweenx(np.linspace(y_min, y_max, 400), x_min=x_val, x_max=3000, color=color,
                                          alpha=alpha)
            elif sense == '>=':
                if a1 > 0:
                    self.ax.fill_betweenx(np.linspace(y_min, y_max, 400), x_min=x_val, x_max=3000, color=color,
                                          alpha=alpha)
                else:
                    self.ax.fill_betweenx(np.linspace(y_min, y_max, 400), x_min=0, x_max=x_val, color=color,
                                          alpha=alpha)

        self.ax.set_xlim(0, 3000)
        self.ax.set_ylim(0, 3000)
        self.ax.grid(True)
        self.ax.set_xlabel("A")
        self.ax.set_ylabel("B")
        self.canvas.draw()


    # Pokaże kolejne ograniczenie
    def show_next_constraint(self):
        cons = self.get_constraints()
        if self.constraint_index >= len(cons):
            messagebox.showinfo("Info", "Wszystkie ograniczenia zostały narysowane.")
            return
        a1, a2, s, rhs = cons[self.constraint_index]
        self.plot_constraint(a1, a2, s, rhs, alpha=1.5/len(cons), color='white')
        self.constraint_index += 1

    # Wynik i rysowanie
    def solve_and_plot(self):
        res_vertices = []
        cons = self.get_constraints()
        c = np.array([float(self.costA.get()), float(self.costB.get())])
        mode = self.mode_var.get()

        self.ax.clear()

        #rysowanie ograniczen
        for (a1, a2, s, rhs) in cons:
            self.plot_constraint(a1, a2, s, rhs, alpha=1.5/len(cons), color='white')

        #obliczanie wierzcholkow
        vertices = self.compute_feasible_vertices(cons)

        if vertices:
            values = [np.dot(c, v) for v in vertices]

            if mode == 'max':
                optimal_value = max(values)
            else:
                optimal_value = min(values)

            for i in range(len(vertices)):
                if np.dot(c, vertices[i]) == optimal_value:
                    res_vertices.append(vertices[i])

            res_vertices = list(set(res_vertices))

            if len(res_vertices) == 1:
                a1, a2 = self.point_belongs_to_constraint(res_vertices)
                if float(self.costA.get()) == a1 and float(self.costB.get()) == a2:
                    t = np.linspace(0, 3000, 1000)
                    dx, dy = a1, -a2

                    x = res_vertices[0][0] + t * dx
                    y = res_vertices[0][1] + t * dy

                    self.ax.plot(x, y, color='red', label='Polprosta Optimum', linewidth=2)

                    messagebox.showinfo("Wynik", f"Optimum polprosta z punktu: A={res_vertices[0][0]}, B={res_vertices[0][1]}. ze współczynnikami {a1} oraz {a2}"
                                                 f", Wartość={optimal_value}")
                else:
                    self.ax.scatter(res_vertices[0][0], res_vertices[0][1], color='red', s=80, label='Optimum')
                    messagebox.showinfo("Wynik", f"Optimum: A={res_vertices[0][0]}, B={res_vertices[0][1]}"
                                                 f", Wartość={optimal_value}")
            elif len(res_vertices) == 2:
                self.ax.plot([res_vertices[0][0], res_vertices[1][0]], [res_vertices[0][1], res_vertices[1][1]]
                             , color='red', label='Odcinek Optimum', linewidth=2)
                messagebox.showinfo("Wynik",
                                    f"Optimum odcinek z punktu: A1={res_vertices[0][0]}, B1={res_vertices[0][1]}"
                                    f" do punktu A2={res_vertices[1][0]}, B2={res_vertices[1][1]}, Wartość={optimal_value}")

            self.ax.legend()
        else:
            messagebox.showinfo("Info", "Brak Rozwiązania")
        self.canvas.draw()

    def point_belongs_to_constraint(self, point):
        cons = self.get_constraints()
        for (a1, a2, s, rhs) in cons:
            res = a1*float(point[0][0]) + a2*float(point[0][1])
            if res == rhs:
                return (a1, a2)

    # Obliczanie możliwych wierzchołków
    def compute_feasible_vertices(self, constraints):
        from itertools import combinations
        vertices = []

        # Kombinacje par ograniczeń
        for (a1, a2, s1, b1), (a3, a4, s2, b2) in combinations(constraints, 2):
            det = a1 * a4 - a2 * a3
            if abs(det) < 1e-8:
                continue
            x = (b1 * a4 - b2 * a2) / det
            y = (a1 * b2 - a3 * b1) / det
            if x >= 0 and y >= 0 and self.is_feasible(x, y, constraints):
                vertices.append((x, y))

        # Przecięcia z osią X
        for a1, a2, s, b in constraints:
            if abs(a1) > 1e-8:
                x = b / a1 if abs(a2) < 1e-8 else (b - a2 * 0) / a1
                y = 0
                if x >= 0 and self.is_feasible(x, y, constraints):
                    vertices.append((x, y))

        # Przecięcia z osią Y
        for a1, a2, s, b in constraints:
            if abs(a2) > 1e-8:
                y = b / a2 if abs(a1) < 1e-8 else (b - a1 * 0) / a2
                x = 0
                if y >= 0 and self.is_feasible(x, y, constraints):
                    vertices.append((x, y))

        # Punkt (0,0)
        if self.is_feasible(0, 0, constraints):
            vertices.append((0, 0))

        # Unikatowe wierzchołki
        unique_vertices = []
        seen = set()
        for vx, vy in vertices:
            key = (round(vx, 6), round(vy, 6))
            if key not in seen:
                seen.add(key)
                unique_vertices.append((vx, vy))

        return unique_vertices

    # Sprawdzanie czy jest w zakresie
    def is_feasible(self, x, y, constraints, tol=1e-6):
        for a, b, s, rhs in constraints:
            val = a * x + b * y
            if s == '<=' and val - rhs > tol:
                return False
            elif s == '>=' and rhs - val > tol:
                return False
            elif s == '=' and abs(val - rhs) > tol:
                return False
        return True

    # Resetowanie wykresu
    def reset_plot(self):
        self.constraint_index = 0
        self.ax.clear()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = LPGui(root)
    root.mainloop()