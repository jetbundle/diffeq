# The Differential Equations Odyssey

> **The most complete journey through the structure of differential equations and their solution spaces**

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Website](https://img.shields.io/badge/Website-jetbundle.github.io-blue)](https://jetbundle.github.io)
[![Discord](https://img.shields.io/badge/Discord-Join-7289da)](https://discord.gg/5mbumWNb5U)

## 🎯 Overview

This repository contains an interactive, comprehensive journey through differential equations—from classical explicit methods to modern categorical approaches. These notebooks represent the most complete pedagogical treatment of differential equations, connecting the philosophy and mathematics of **physics**.

Each notebook is an interactive exploration with Plotly-based widgets that allow you to visualize and manipulate the mathematical structures in real-time. This is not just a textbook—it's an **interactive mathematical experience**.

## 📺 Video Series

This repository is designed to be followed alongside the **Differential Equations** video series:

🎥 **[Watch the Full Playlist](https://youtube.com/playlist?list=PLgrgo-QP8PRbSYhWryLDYwj951w4Wx0V1)**

The series builds upon foundational concepts introduced in:

🎬 **[The Philosophy and Mathematics of Physics, Finance, and Computers](https://youtu.be/gCdjKSx5jqg?si=lrkrxY8lsqllEF6T)**

## 📚 The Journey: Seven Levels of Understanding

### Level 1: Classical Explicit & Quasi-Explicit Arsenal
**Theme:** *The Rise and Fall of the Specific Solution*

**Notebook:** `01_classical_explicit_arsenal.ipynb`

We begin with the optimistic 19th-century pursuit of closed-form solutions. Explore exact methods, special functions, asymptotic analysis, and the limits of explicit formulas. Witness the emergence of chaos and the necessity of renormalization.

**Key Topics:**
- Quadrature and exact differentials
- Bernoulli and Riccati equations
- Deterministic chaos (Lorenz system)
- Special functions (Hypergeometric, Bessel, Legendre, Airy)
- Stokes phenomenon
- WKB approximation
- Poincaré-Lindstedt method
- Borel summation and resummation

---

### Level 2: Functional Analysis, Distributions & Weak Solutions
**Theme:** *The Geometry of Infinite Dimensions*

**Notebook:** `02_functional_analysis_distributions.ipynb`

Enter the rigorous cathedral of functional analysis. Functions become vectors in infinite-dimensional spaces. Differential equations become linear operators. Solving becomes geometry: projections, angles, and orthogonality.

**Key Topics:**
- Distributions and test functions (Schwartz)
- Weak derivatives
- Sobolev spaces ($H^s$)
- Spectral theorem and unbounded operators
- Stone's theorem and semigroups
- Variational methods (Lax-Milgram)
- Galerkin projection and FEM
- Fredholm theory

---

### Level 3: Tensor Fields, Conservation Laws & Geometric Formulation
**Theme:** *Physics is Invariant; Coordinates are Artifacts*

**Notebook:** `03_tensor_fields_conservation_laws.ipynb`

Detach physics from coordinates. Reformulate differential equations on manifolds using the covariant derivative. Understand how parallel transport reveals curvature through holonomy.

**Key Topics:**
- Covariant derivative and Christoffel symbols
- Parallel transport and holonomy
- Systems of conservation laws
- Shock waves and entropy solutions
- Exterior calculus and differential forms
- Hodge star and Laplace-Beltrami operator
- Geometric optics and caustics

---

### Level 4: Symmetry, Lie Theory & Classical Integrability
**Theme:** *Solvability is Symmetry*

**Notebook:** `04_symmetry_lie_theory_integrability.ipynb`

Discover why some equations are exactly solvable: **symmetry**. When a differential equation admits a continuous group of transformations, we can reduce its order or find conservation laws. In rare cases, infinite symmetry leads to completely integrable systems.

**Key Topics:**
- Lie symmetries and prolongation
- Noether's theorem
- Completely integrable systems
- Solitons (KdV equation)
- Lax pairs and inverse scattering transform
- Iso-spectral flow
- Supersymmetric quantum mechanics

---

### Level 5: Stochastic, Rough, Fractional & Nonlocal Dynamics
**Theme:** *Regularity is an Exception; Roughness is the Rule*

**Notebook:** `05_stochastic_rough_fractional_nonlocal.ipynb`

When randomness enters, classical derivatives fail. We need Itô calculus, rough paths, and fractional operators. Discover how roughness is not a bug—it's a feature of reality.

**Key Topics:**
- Brownian motion and Itô calculus
- Feynman-Kac formula
- Rough paths and signatures
- Fractional Brownian motion
- Fractional Laplacian
- Lévy flights
- KPZ equation and regularity structures
- **Karhunen–Loève expansion** (optimal basis for Gaussian processes)

---

### Level 6: Jet Bundles, Exterior Differential Systems & Intrinsic Geometric PDEs
**Theme:** *The PDE is the Manifold*

**Notebook:** `06_jet_bundles_exterior_differential_systems.ipynb`

Invert the perspective: derivatives become independent coordinates. A differential equation is a submanifold in a jet bundle. Solving means finding surfaces that fit inside this submanifold while remaining tangent to a contact structure.

**Key Topics:**
- Jet bundles and contact geometry
- Exterior differential systems
- Cartan's method
- Spencer cohomology
- Ricci flow
- Optimal transport (Monge-Ampère)
- Instantons and gauge theory

---

### Level 7: Microlocal Analysis, D-Modules & Categorical Resolution
**Theme:** *The Phase Space is the Reality*

**Notebook:** `07_microlocal_analysis_dmodules_categorical_resolution.ipynb`

The final synthesis: singularities are geometric objects in phase space. Differential equations become modules over rings of operators. Solutions become sheaves. The Riemann-Hilbert correspondence reveals that analysis is topology in disguise.

**Key Topics:**
- Wave front sets and microlocal analysis
- D-modules and characteristic varieties
- Holonomic systems and Bernstein's inequality
- Resurgence theory and alien calculus
- Riemann-Hilbert correspondence
- Perverse sheaves
- Topological quantum field theory

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Jupyter Notebook or JupyterLab
- Required packages (see `pyproject.toml`)

### Installation

```bash
# Clone the repository
git clone https://github.com/jetbundle/diffeq.git
cd diffeq

# Install dependencies
pip install -e .

# Launch Jupyter
jupyter notebook
# or
jupyter lab
```

### Running the Notebooks

1. Navigate to the `notebooks/` directory
2. Open notebooks in order (01 through 07) for the complete journey
3. Each notebook is self-contained with interactive widgets
4. Run all cells to initialize the interactive visualizations

## 🎨 Interactive Features

Every notebook contains **Plotly-based interactive widgets** that allow you to:

- Manipulate parameters in real-time
- Visualize mathematical structures dynamically
- Explore phase spaces and solution manifolds
- Understand geometric and algebraic relationships through interaction

## 📖 How to Use This Repository

### For Students

1. **Follow sequentially**: Start with Level 1 and progress through Level 7
2. **Watch the videos**: Use the YouTube playlist as your primary guide
3. **Interact**: Don't just read—manipulate the widgets to build intuition
4. **Experiment**: Modify parameters and observe how structures change

### For Researchers

- Each notebook contains state-of-the-art implementations
- Citations point to original papers
- Code is modular and extensible
- Perfect for understanding modern approaches to PDEs

### For Educators

- Use notebooks as interactive lecture materials
- Widgets demonstrate concepts that are difficult to explain statically
- Citations provide a complete reference list
- Structure follows a logical pedagogical progression

## 🔗 Connect & Support

- **🌐 Website:** [jetbundle.github.io](https://jetbundle.github.io)
- **💬 Discord:** [Join our community](https://discord.gg/5mbumWNb5U)
- **🐦 X (Twitter):** [@jetbundle](https://x.com/jetbundle)
- **☕ Support:** [Buy Me a Coffee](https://buymeacoffee.com/jetbundle)
- **📦 Repository:** [github.com/jetbundle/diffeq](https://github.com/jetbundle/diffeq)

## 📚 Complete Bibliography

### Level 1: Classical Methods

* **Borel, É.** (1899). *Mémoire sur les séries divergentes*. Annales Scientifiques de l'École Normale Supérieure.
* **Green, G.** (1828). *An essay on the application of mathematical analysis to the theories of electricity and magnetism*.
* **Lorenz, E. N.** (1963). "Deterministic nonperiodic flow". *Journal of the Atmospheric Sciences*.
* **Poincaré, H.** (1892). *Les méthodes nouvelles de la mécanique céleste*.
* **Riccati, J.** (1724). *Animadversiones in aequationes differentiales secundi gradus*. Acta Eruditorum.
* **Stokes, G. G.** (1857). "On the discontinuity of arbitrary constants which appear in divergent developments". *Transactions of the Cambridge Philosophical Society*.
* **Wentzel, G., Kramers, H. A., & Brillouin, L.** (1926). (Independent papers on the WKB approximation).

### Level 2: Functional Analysis

* **Céa, J.** (1964). *Approximation variationnelle des problèmes aux limites*.
* **Fredholm, I.** (1903). *Sur une classe d'équations fonctionnelles*.
* **Galerkin, B. G.** (1915). *Series developments for some cases of equilibrium of plates and beams*.
* **Lax, P. D., & Milgram, A. N.** (1954). *Parabolic equations*.
* **Schwartz, L.** (1950). *Théorie des distributions*.
* **Sobolev, S. L.** (1938). *On a theorem of functional analysis*.
* **Stone, M. H.** (1932). *Linear transformations in Hilbert space*.
* **von Neumann, J.** (1927). *Mathematische Begründung der Quantenmechanik*.

### Level 3: Geometric Formulation

* **Airy, G. B.** (1838). *On the intensity of light in the neighbourhood of a caustic*.
* **Cartan, É.** (1899). *Sur certaines expressions différentielles et le problème de Pfaff*.
* **Cauchy, A. L.** (1827). *De la pression ou tension dans un corps solide*.
* **Hamilton, W. R.** (1832). *Third supplement to an essay on the theory of systems of rays*.
* **Hodge, W. V. D.** (1941). *The Theory and Applications of Harmonic Integrals*.
* **Kruzhkov, S. N.** (1970). *First order quasilinear equations in several independent variables*.
* **Lax, P. D.** (1957). *Hyperbolic systems of conservation laws II*.
* **Levi-Civita, T.** (1917). *Nozione di parallelismo in una varietà qualunque*.

### Level 4: Symmetry & Integrability

* **Bluman, G. W., & Cole, J. D.** (1974). *Similarity methods for differential equations*.
* **Korteweg, D. J., & de Vries, G.** (1895). *On the change of form of long waves advancing in a rectangular canal*.
* **Lax, P. D.** (1968). *Integrals of nonlinear equations of evolution and solitary waves*.
* **Lie, S.** (1880). *Theorie der Transformationsgruppen*.
* **Noether, E.** (1918). *Invariante Variationsprobleme*.
* **Olver, P. J.** (1986). *Applications of Lie Groups to Differential Equations*.
* **Witten, E.** (1981). *Dynamical breaking of supersymmetry*.
* **Zakharov, V. E., & Shabat, A. B.** (1972). *Exact theory of two-dimensional self-focusing and one-dimensional self-modulation of waves in nonlinear media*.

### Level 5: Stochastic & Rough

* **Boltzmann, L.** (1872). *Weitere Studien über das Wärmegleichgewicht unter Gasmolekülen*.
* **Caffarelli, L., & Silvestre, L.** (2007). *An extension problem related to the fractional Laplacian*.
* **Hairer, M.** (2014). *A theory of regularity structures*.
* **Itô, K.** (1944). *Stochastic integral*.
* **Karhunen, K.** (1947). *Über lineare Methoden in der Wahrscheinlichkeitsrechnung*.
* **Kac, M.** (1949). *On distributions of certain Wiener functionals*.
* **Loève, M.** (1948). *Fonctions aléatoires du second ordre*.
* **Lyons, T.** (1998). *Differential equations driven by rough signals*.
* **Malliavin, P.** (1978). *Stochastic calculus of variation and hypoelliptic operators*.
* **Mandelbrot, B. B., & Van Ness, J. W.** (1968). *Fractional Brownian motions, fractional noises and applications*.
* **Vergara, R. C.** (2025). *Karhunen-Loève expansion of random measures*.

### Level 6: Jet Bundles & Geometric PDEs

* **Atiyah, M. F., Hitchin, N. J., & Singer, I. M.** (1978). *Self-duality in four-dimensional Riemannian geometry*.
* **Cartan, É.** (1901). *Sur l'intégration des systèmes d'équations aux différentielles totales*.
* **Ehresmann, C.** (1952). *Les connexions infinitésimales dans un espace fibré différentiable*.
* **Evans, L. C.** (1982). *Classical solutions of fully nonlinear, convex, second-order elliptic equations*.
* **Hamilton, R. S.** (1982). *Three-manifolds with positive Ricci curvature*.
* **Monge, G.** (1781). *Mémoire sur la théorie des déblais et des remblais*.
* **Perelman, G.** (2002). *The entropy formula for the Ricci flow and its geometric applications*.
* **Spencer, D. C.** (1965). *Deformation of structures on manifolds defined by transitive, continuous pseudogroups*.
* **Yau, S. T.** (1978). *On the Ricci curvature of a compact Kähler manifold and the complex Monge-Ampère equation*.

### Level 7: Microlocal & Categorical

* **Atiyah, M.** (1988). *Topological quantum field theories*.
* **Bernstein, I. N.** (1971). *Modules over a ring of differential operators*.
* **Écalle, J.** (1981). *Les fonctions résurgentes*.
* **Hörmander, L.** (1971). *Fourier integral operators. I*.
* **Kashiwara, M.** (1980). *Faisceaux constructibles et systèmes holonomes d'équations aux dérivées partielles*.
* **Mebkhout, Z.** (1980). *Sur le problème de Riemann-Hilbert*.
* **Sato, M.** (1959). *Theory of hyperfunctions*.

## 🏗️ Repository Structure

```
diffeq/
├── notebooks/           # Interactive Jupyter notebooks (Levels 1-7)
│   ├── 01_classical_explicit_arsenal.ipynb
│   ├── 02_functional_analysis_distributions.ipynb
│   ├── 03_tensor_fields_conservation_laws.ipynb
│   ├── 04_symmetry_lie_theory_integrability.ipynb
│   ├── 05_stochastic_rough_fractional_nonlocal.ipynb
│   ├── 06_jet_bundles_exterior_differential_systems.ipynb
│   └── 07_microlocal_analysis_dmodules_categorical_resolution.ipynb
├── src/                 # Source code for widgets and utilities
│   └── diffeq/
│       ├── widgets/      # Interactive widget implementations
│       ├── visualizations/  # Plotting utilities
│       └── core/        # Core solvers and algorithms
├── pyproject.toml       # Project configuration and dependencies
└── README.md           # This file
```

## 🤝 Contributing

This is an educational resource. Contributions that improve clarity, fix bugs, or add pedagogical value are welcome. Please ensure all code follows the repository's coding standards and includes appropriate documentation.

## 📄 License

This work is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License** (CC BY-NC 4.0).

This means you are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

Under the following terms:
- **Attribution** — You must give appropriate credit
- **NonCommercial** — You may not use the material for commercial purposes
- **No additional restrictions** — You may not apply legal terms or technological measures that restrict others from doing anything the license permits

See the [LICENSE](LICENSE) file for full details.

## 🌟 Philosophy

This repository embodies a fundamental belief: **differential equations are not just tools—they are the language through which we understand the structure of reality**. From the deterministic chaos of weather systems to the stochastic roughness of financial markets, from the geometric beauty of general relativity to the algebraic elegance of quantum field theory, differential equations reveal the deep connections between mathematics, physics, finance, and computation.

The journey from explicit formulas to categorical resolutions is not just a historical progression—it's a map of how human understanding has evolved to grasp increasingly complex structures. Each level builds on the previous, revealing new layers of mathematical beauty and physical insight.

**Welcome to The Differential Equations Odyssey.**

---

*"The differential equation is the most powerful tool humanity has invented for understanding change. This repository is your guide to mastering it."*
