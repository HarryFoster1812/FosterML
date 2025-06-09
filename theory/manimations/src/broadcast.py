from manim import *

class MatrixBroadcasting3D(ThreeDScene):
    def construct(self):
        # Step 0: Start with a modest 3D camera angle (adds subtle depth)

        # Step 1: Create and show original (2,1) matrix
        matrix = Matrix([[1], [2]])
        matrix.move_to(ORIGIN)
        self.play(Write(matrix))
        self.wait()

        # Step 2: Label original shape
        shape_label = Text("(2, 1)", font_size=24)
        shape_label.next_to(matrix, DOWN)
        self.play(FadeIn(shape_label))
        self.wait()

        # Step 3: Smooth camera rotation to top-left-back view
        self.move_camera(phi=120*DEGREES, theta=80*DEGREES, run_time=3)
        self.wait()

        # Step 4: Add a broadcasted copy of the matrix "behind" the original
        matrix_copy = matrix.copy().shift(OUT * 1.5)  # OUT = -Z direction
        self.play(FadeIn(matrix_copy))
        self.wait()
