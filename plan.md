learn-ai/
│
├── README.md                   # Project overview, usage, learning path
├── LICENSE
│
├── framework/                  # Reusable shared AI code
│   ├── matrix.h / .cpp
│   ├── vector.h / .cpp
│   ├── activations.h / .cpp
│   ├── loss.h / .cpp
│   ├── optimizer.h / .cpp
│   ├── layer.h / .cpp
│   ├── rnn_layer.h / .cpp
│   ├── transformer_layer.h / .cpp
│   └── ...
│
├── lessons/                    # Step-by-step learning modules
│   ├── 01_math_basics/
│   │   ├── vector_lesson.cpp
│   │   ├── matrix_lesson.cpp
│   │   └── README.md  # Explains the step and promotes to framework
│   ├── 02_activations/
│   ├── 03_loss_functions/
│   ├── 04_optimizers/
│   ├── 05_neural_networks/
│   └── ...
│
├── examples/                   # Full AI programs built with the framework
│   ├── mlp_classification.cpp
│   ├── rnn_textgen.cpp
│   ├── transformer_textgen.cpp
│   └── README.md
│
├── data/                       # Datasets like tiny Shakespeare, MNIST
│   ├── tiny_shakespeare.txt
│   └── ...
│
├── scripts/                    # Build/run helpers
│   ├── build.sh
│   └── run_examples.sh
│
└── docs/
    ├── timeline.md
    ├── matrix_math.md
    ├── neural_nets.md
    └── ...
