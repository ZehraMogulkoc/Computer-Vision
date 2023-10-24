from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.graphics import Ellipse, Color
from math import cos, sin, radians
import numpy as np

class PointTransformApp(GridLayout):
    def __init__(self, **kwargs):
        super(PointTransformApp, self).__init__(**kwargs)
        self.cols = 3

        self.left_panel = GridLayout(cols=1, size_hint_x=None, width=300)
        self.add_widget(self.left_panel)
        self.left_panel.bind(minimum_height=self.left_panel.setter('height'))
        self.left_panel.bind(on_touch_down=self.place_point)

        self.transform_panel = GridLayout(cols=2, size_hint_x=None, width=250)
        self.add_widget(self.transform_panel)

        self.transform_panel.add_widget(Label(text="tx"))
        self.tx_entry = TextInput(multiline=False, input_filter="float")
        self.transform_panel.add_widget(self.tx_entry)

        self.transform_panel.add_widget(Label(text="ty"))
        self.ty_entry = TextInput(multiline=False, input_filter="float")
        self.transform_panel.add_widget(self.ty_entry)

        self.transform_panel.add_widget(Label(text="Rotation Angle"))
        self.angle_entry = TextInput(multiline=False, input_filter="float")
        self.transform_panel.add_widget(self.angle_entry)

        self.transform_panel.add_widget(Label(text="Scaling Ratio"))
        self.scaling_entry = TextInput(multiline=False, input_filter="float")
        self.transform_panel.add_widget(self.scaling_entry)

        transform_button = Button(text="Transform")
        transform_button.bind(on_press=self.transform_points)
        self.transform_panel.add_widget(transform_button)

        self.right_panel = GridLayout(cols=1, size_hint_x=None, width=400)
        self.add_widget(self.right_panel)
        self.right_panel.bind(minimum_height=self.right_panel.setter('height'))

        self.points = []

    def place_point(self, instance, touch):
        if self.left_panel.collide_point(*touch.pos):
            x, y = touch.pos
            self.points.append([x, y])
            with self.left_panel.canvas:
                Color(0, 1, 0)  # Set color to green
                Ellipse(pos=(x - 3, y - 3), size=(6, 6))


    def transform_points(self, instance):
        tx = float(self.tx_entry.text)
        ty = float(self.ty_entry.text)
        angle = radians(float(self.angle_entry.text))
        scaling = float(self.scaling_entry.text)

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
        self.right_panel.clear_widgets()
    # Draw transformed points as ellipses on the right panel
        with self.right_panel.canvas:
            Color(1, 0, 0)  # Set color to red
            for point in transformed_points:
                Ellipse(pos=(point[0] - 3, point[1] - 3), size=(6, 6))

class PointTransformAppApp(App):
    def build(self):
        return PointTransformApp()

if __name__ == '__main__':
    PointTransformAppApp().run()
