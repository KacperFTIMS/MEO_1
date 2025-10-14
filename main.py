import tkinter as tk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk, messagebox
from scipy.optimize import linprog

class LPGui:
    def __init__(self, root):
        self.root = root
        root.title("Optymalizacja LP — z zakolorowaniem obszaru dopuszczalnego")
        root.geometry("1150x750")

        self.default_coeffs = [[1.0, -1.0],
                               [2.0,  1.0],
                               [1.0, -1.0]]
        self.default_sense = ['<=','<=','>=']
        self.default_rhs = [1000.0, 2000.0, 0.0]
        self.default_costs = [1.0, 3.0]

        self.extra_constraints = []
        self.constraint_index = 0
        self.build_ui()

    # --- Interfejs graficzny ---
    def build_ui(self):
        top_frame = ttk.Frame(self.root, padding=8)
        top_frame.pack(fill='x')

        ttk.Label(top_frame, text="Podstawowe ograniczenia").pack(anchor='w')
        table = ttk.Frame(top_frame)
        table.pack(anchor='w', pady=4)

        headers = ["", "A", "B", "Relacja", "RHS"]
        for col, h in enumerate(headers):
            ttk.Label(table, text=h, width=12).grid(row=0, column=col, padx=2)

        self.base_entries = []
        for i, name in enumerate(["S1", "S2", "S3"]):
            ttk.Label(table, text=name, width=12).grid(row=i+1, column=0)
            eA = ttk.Entry(table, width=10); eA.grid(row=i+1, column=1); eA.insert(0, str(self.default_coeffs[i][0]))
            eB = ttk.Entry(table, width=10); eB.grid(row=i+1, column=2); eB.insert(0, str(self.default_coeffs[i][1]))
            sense = ttk.Combobox(table, values=['<=','>=','='], width=6); sense.grid(row=i+1, column=3); sense.set(self.default_sense[i])
            rhs = ttk.Entry(table, width=10); rhs.grid(row=i+1, column=4); rhs.insert(0, str(self.default_rhs[i]))
            self.base_entries.append((eA, eB, sense, rhs))

        # Koszty
        cost_frame = ttk.Frame(top_frame)
        cost_frame.pack(anchor='w', pady=6)
        ttk.Label(cost_frame, text="Funkcja celu: ").pack(side='left')
        ttk.Label(cost_frame, text="A").pack(side='left', padx=(8,0))
        self.costA = ttk.Entry(cost_frame, width=8); self.costA.pack(side='left'); self.costA.insert(0, str(self.default_costs[0]))
        ttk.Label(cost_frame, text="B").pack(side='left', padx=(8,0))
        self.costB = ttk.Entry(cost_frame, width=8); self.costB.pack(side='left'); self.costB.insert(0, str(self.default_costs[1]))

        self.mode_var = tk.StringVar(value='max')
        ttk.Radiobutton(cost_frame, text='Max', variable=self.mode_var, value='max').pack(side='left', padx=6)
        ttk.Radiobutton(cost_frame, text='Min', variable=self.mode_var, value='min').pack(side='left', padx=6)

        # Dodatkowe ograniczenia
        mid_frame = ttk.Frame(self.root, padding=8)
        mid_frame.pack(fill='x')
        ttk.Label(mid_frame, text="Dodatkowe ograniczenia").pack(anchor='w')

        add_row = ttk.Frame(mid_frame)
        add_row.pack(anchor='w', pady=4)
        ttk.Label(add_row, text="a1:").pack(side='left')
        self.add_a1 = ttk.Entry(add_row, width=8); self.add_a1.pack(side='left', padx=2)
        ttk.Label(add_row, text="a2:").pack(side='left')
        self.add_a2 = ttk.Entry(add_row, width=8); self.add_a2.pack(side='left', padx=2)
        self.add_sense = ttk.Combobox(add_row, values=['<=','>=','='], width=6); self.add_sense.pack(side='left', padx=4); self.add_sense.set('<=')
        ttk.Label(add_row, text="RHS:").pack(side='left')
        self.add_rhs = ttk.Entry(add_row, width=10); self.add_rhs.pack(side='left', padx=2)
        ttk.Button(add_row, text="Dodaj", command=self.on_add).pack(side='left', padx=4)
        ttk.Button(add_row, text="Usuń zaznaczone", command=self.on_remove).pack(side='left', padx=4)

        self.lb = tk.Listbox(mid_frame, height=6, width=80)
        self.lb.pack(anchor='w', pady=6)
        self.refresh_listbox()

        # Przyciski
        btns = ttk.Frame(self.root, padding=8)
        btns.pack(anchor='w')
        ttk.Button(btns, text="Następne ograniczenie", command=self.show_next_constraint).pack(side='left', padx=4)
        ttk.Button(btns, text="Oblicz i zakoloruj", command=self.solve_and_plot).pack(side='left', padx=4)
        ttk.Button(btns, text="Resetuj wykres", command=self.reset_plot).pack(side='left', padx=4)

        # Wykres
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(fill='both', expand=True)
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    # --- Zarządzanie ograniczeniami ---
    def on_add(self):
        try:
            a1 = float(self.add_a1.get())
            a2 = float(self.add_a2.get())
            rhs = float(self.add_rhs.get())
            sense = self.add_sense.get()
        except ValueError:
            messagebox.showerror("Błąd", "Wprowadź poprawne liczby.")
            return
        self.extra_constraints.append((a1, a2, sense, rhs))
        self.refresh_listbox()

    def on_remove(self):
        sel = self.lb.curselection()
        if not sel: return
        idx = sel[0]
        if idx >= 4:
            extra_idx = idx - 4
            if 0 <= extra_idx < len(self.extra_constraints):
                self.extra_constraints.pop(extra_idx)
        self.refresh_listbox()

    def refresh_listbox(self):
        self.lb.delete(0, 'end')
        for i, name in enumerate(["S1", "S2", "S3"]):
            eA, eB, sense, rhs = self.base_entries[i]
            self.lb.insert('end', f"{name}: {eA.get()}*A + {eB.get()}*B {sense.get()} {rhs.get()}")
        if self.extra_constraints:
            self.lb.insert('end', "---- Dodatkowe ----")
            for (a1, a2, s, b) in self.extra_constraints:
                self.lb.insert('end', f"{a1}*A + {a2}*B {s} {b}")

    # --- Obliczenia i rysowanie ---
    def get_constraints(self):
        cons = []
        for eA, eB, sense, rhs in self.base_entries:
            cons.append((float(eA.get()), float(eB.get()), sense.get(), float(rhs.get())))
        cons.extend(self.extra_constraints)
        return cons

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

    def show_next_constraint(self):
        cons = self.get_constraints()
        if self.constraint_index >= len(cons):
            messagebox.showinfo("Info", "Wszystkie ograniczenia zostały narysowane.")
            return
        a1, a2, s, rhs = cons[self.constraint_index]
        self.plot_constraint(a1, a2, s, rhs)
        self.constraint_index += 1

    def solve_and_plot(self):
        cons = self.get_constraints()
        c = np.array([float(self.costA.get()), float(self.costB.get())])
        if self.mode_var.get() == 'max':
            c = -c

        A_ub, b_ub, A_eq, b_eq = [], [], [], []
        for a1, a2, s, rhs in cons:
            if s == '<=':
                A_ub.append([a1, a2]); b_ub.append(rhs)
            elif s == '>=':
                A_ub.append([-a1, -a2]); b_ub.append(-rhs)
            else:
                A_eq.append([a1, a2]); b_eq.append(rhs)

        res = linprog(c, A_ub=A_ub or None, b_ub=b_ub or None,
                      A_eq=A_eq or None, b_eq=b_eq or None,
                      bounds=(0, None), method='highs')

        self.ax.clear()

        # Rysuj wszystkie ograniczenia
        for (a1, a2, s, rhs) in cons:
            self.plot_constraint(a1, a2, s, rhs, alpha=0.1)

        # Zakoloruj część wspólną
        self.fill_feasible_region(cons)

        if res.success:
            # Wartość optimum
            val_opt = -res.fun if self.mode_var.get() == 'max' else res.fun
            # Zbieramy wszystkie wierzchołki obszaru dopuszczalnego
            vertices = self.compute_feasible_vertices(cons)

            # Sprawdzenie, które wierzchołki dają wartość optimum
            eps = 1e-5
            opt_points = []
            for x, y in vertices:
                z = np.dot(c, [x, y])
                if abs(z - (-val_opt if self.mode_var.get() == 'max' else val_opt)) < eps:
                    opt_points.append((x, y))

            if len(opt_points) == 1:
                x_opt, y_opt = opt_points[0]
                self.ax.scatter(x_opt, y_opt, color='red', s=80, label='Optimum')
                messagebox.showinfo("Wynik", f"Optimum: A={x_opt:.2f}, B={y_opt:.2f}, wartość={val_opt:.2f}")
            else:
                # Narysuj odcinek optimum
                xs, ys = zip(*opt_points)
                self.ax.plot(xs, ys, color='red', linewidth=3, label='Odcinek optimum')
                messagebox.showinfo("Wynik",
                                    f"Optimum występuje na odcinku: A={min(xs):.2f}-{max(xs):.2f}, B={min(ys):.2f}-{max(ys):.2f}, wartość={val_opt:.2f}")

            self.ax.legend()
        else:
            messagebox.showerror("Błąd", "Nie znaleziono rozwiązania.")

        self.canvas.draw()

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
            if x >= -1e-6 and y >= -1e-6 and self.is_feasible(x, y, constraints):
                vertices.append((x, y))

        # Przecięcia z osią X (y = 0)
        for a1, a2, s, b in constraints:
            if abs(a1) > 1e-8:  # żeby uniknąć dzielenia przez 0
                x = b / a1 if abs(a2) < 1e-8 else (b - a2 * 0) / a1
                y = 0
                if x >= -1e-6 and self.is_feasible(x, y, constraints):
                    vertices.append((x, y))

        # Przecięcia z osią Y (x = 0)
        for a1, a2, s, b in constraints:
            if abs(a2) > 1e-8:
                y = b / a2 if abs(a1) < 1e-8 else (b - a1 * 0) / a2
                x = 0
                if y >= -1e-6 and self.is_feasible(x, y, constraints):
                    vertices.append((x, y))

        # Punkt (0,0)
        if self.is_feasible(0, 0, constraints):
            vertices.append((0, 0))

        # Usuwamy duplikaty (zaokrąglenie aby uniknąć różnic numerycznych)
        unique_vertices = []
        seen = set()
        for vx, vy in vertices:
            key = (round(vx, 6), round(vy, 6))
            if key not in seen:
                seen.add(key)
                unique_vertices.append((vx, vy))

        return unique_vertices

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

    def fill_feasible_region(self, constraints):
        x = np.linspace(0, 3000, 400)
        y = np.linspace(0, 3000, 400)
        X, Y = np.meshgrid(x, y)
        feasible = np.ones_like(X, dtype=bool)

        for a1, a2, sense, rhs in constraints:
            lhs = a1 * X + a2 * Y
            if sense == '<=':
                feasible &= (lhs <= rhs)
            elif sense == '>=':
                feasible &= (lhs >= rhs)
            else:
                feasible &= (np.isclose(lhs, rhs, atol=1e-3))

        self.ax.contourf(X, Y, feasible, levels=[0.5, 1], colors=['#aaffaa'], alpha=0.4)

    def reset_plot(self):
        self.constraint_index = 0
        self.ax.clear()
        self.canvas.draw()

# --- Uruchomienie programu ---
if __name__ == "__main__":
    root = tk.Tk()
    app = LPGui(root)
    root.mainloop()

