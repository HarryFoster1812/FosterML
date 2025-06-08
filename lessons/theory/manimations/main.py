from manim import *

class MatrixBroadcasting3D(ThreeDScene):
    def construct(self):
        # Set 3D camera orientation
        
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

        # Step 3: Rotate camera to show 3D depth
        self.move_camera(phi=60 * DEGREES, theta=-60 * DEGREES, run_time=3)
        self.wait()

        # Step 4: Broadcast the matrix by duplicating it horizontally
        broadcasted_matrices = VGroup()
        shifts = [RIGHT * 2, RIGHT * 4]

        for i, shift in enumerate(shifts):
            m_copy = matrix.copy()
            m_copy.shift(shift)
            broadcasted_matrices.add(m_copy)
            self.play(TransformFromCopy(matrix, m_copy), run_time=1)

        # Combine all matrices into a single VGroup to look like one (2,3) matrix
        full_matrix = VGroup(matrix, *broadcasted_matrices)
        full_matrix_center = full_matrix.get_center()
        full_matrix.move_to(ORIGIN)
        self.play(full_matrix.animate.move_to(full_matrix_center))

        self.wait()

        # Step 5: Update label to reflect new shape
        new_shape_label = Text("(2, 3)", font_size=24)
        new_shape_label.next_to(full_matrix, DOWN)
        self.play(Transform(shape_label, new_shape_label))

        # Step 6: Add broadcasting explanation
        explanation = Text("Broadcasting along columns", font_size=30)
        explanation.to_edge(UP)
        self.play(Write(explanation))
        self.wait(3)

        # Optional: Rotate camera again for nice finish
        self.move_camera(phi=75 * DEGREES, theta=90 * DEGREES, run_time=3)
        self.wait()
