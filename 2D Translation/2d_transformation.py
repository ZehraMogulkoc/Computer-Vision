import tkinter as tk
import numpy as np

class PointTransformApp:
    def __init__(self, root):
        self.root = root
        self.root.title("2D Translation")
        
        self.points = []
        
        # Left panel for placing points
        self.left_panel = tk.Canvas(root, width=300, height=400, bg='white')
        self.left_panel.grid(row=0, column=0, padx=10, pady=10)
        self.left_panel.bind("<Button-1>", self.place_point)
        
        # Transformation input panel
        self.transform_panel = tk.Frame(root)
        self.transform_panel.grid(row=0, column=1, padx=10, pady=10)
        
        tk.Label(self.transform_panel, text="Transformation Parameters").grid(row=0, columnspan=2)
        tk.Label(self.transform_panel, text="tx").grid(row=1, column=0)
        self.tx_entry = tk.Entry(self.transform_panel)
        self.tx_entry.grid(row=1, column=1)
        tk.Label(self.transform_panel, text="ty").grid(row=2, column=0)
        self.ty_entry = tk.Entry(self.transform_panel)
        self.ty_entry.grid(row=2, column=1)
        tk.Label(self.transform_panel, text="Rotation Angle (degrees)").grid(row=3, column=0)
        self.angle_entry = tk.Entry(self.transform_panel)
        self.angle_entry.grid(row=3, column=1)
        tk.Label(self.transform_panel, text="Scaling Ratio").grid(row=4, column=0)
        self.scaling_entry = tk.Entry(self.transform_panel)
        self.scaling_entry.grid(row=4, column=1)
        transform_button = tk.Button(self.transform_panel, text="Transform", command=self.transform_points)
        transform_button.grid(row=5, columnspan=2)
        
        # Right panel for displaying transformed points
        self.right_panel = tk.Canvas(root, width=300, height=400, bg='white')
        self.right_panel.grid(row=0, column=2, padx=10, pady=10)
    
    def place_point(self, event):
        x, y = event.x, event.y
        self.left_panel.create_oval(x-3, y-3, x+3, y+3, fill='blue')
        self.points.append([x, y])
    
    def transform_points(self):
        tx = float(self.tx_entry.get())
        ty = float(self.ty_entry.get())
        angle = np.radians(float(self.angle_entry.get()))
        scaling = float(self.scaling_entry.get())
        
        transformation_matrix = np.array([
            [scaling * np.cos(angle), -scaling * np.sin(angle), tx],
            [scaling * np.sin(angle), scaling * np.cos(angle), ty],
            [0, 0, 1]
        ])
        
        transformed_points = []
        for point in self.points:
            homogeneous_point = np.array([point[0], point[1], 1])
            transformed_point = np.dot(transformation_matrix, homogeneous_point)
            transformed_points.append((transformed_point[0], transformed_point[1]))
        
        # Clear previous points on the right panel
        self.right_panel.delete("all")
        for point in transformed_points:
            self.right_panel.create_oval(point[0]-3, point[1]-3, point[0]+3, point[1]+3, fill='red')

# Create the main window
root = tk.Tk()
app = PointTransformApp(root)
root.mainloop()