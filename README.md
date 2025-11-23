
# **Master’s Thesis Repository**
## *Bridging Simulation Scales Enables Analysis of Self-Assembly Dynamics of Y-Shaped DNA-Nanomotifs*

**Author:** Xenia Schneider\
**Submission Date:** 25. November 2025\
**Institution:** Karlsruhe Institute of Technology (KIT)

---

## **📘 Abstract**


> *Liquid–liquid phase separation (LLPS) in biological systems arises from multivalent molecular interactions that are challenging to isolate experimentally. DNA nanomotifs offer a platform for constructing synthetic condensates, making them ideal for investigating the physical principles underlying LLPS. The self-assembly behavior of nanomotifs is strongly influenced by structural flexibility, binding kinetics, and environmental conditions, creating an opportunity for the use of computational models that capture these interdependent effects.

A central challenge lies in the high computational cost of nucleotide-level simulations, which limits their applicability to multi-motif systems. 

To address this challenge, a hierarchical modeling framework is introduced in which configurational statistics extracted from oxDNA trajectories are mapped onto a coarse-grained bead–spring representation, enabling efficient simulation of nanomotif network formation.

Each bead approximates one helical turn of the DNA double helix, substantially reducing computational demands while retaining essential structural features. Model fidelity was further enhanced through Bayesian Optimization, which refined nanomotif angle potentials to improve the reproduction of physically grounded behavior. The resulting model successfully captures key characteristics of LLPS, including the formation of interconnected networks across a range of binding kinetics and motif concentrations, as well as the emergence of distinct phase-transition regimes. 

This framework establishes a foundation for mimicking and controlling biomolecular phase separation and opens pathways toward future applications in synthetic biology and programmable materials.*

---

## **📂 Repository Structure**

This repository contains all scripts, workflows, and supplemental materials associated with the thesis.
Each folder corresponds to a major methodological component of the project.

### **1. `Nanomotif_Coarse_Graining/`**

Scripts for converting oxDNA trajectories into a coarse-grained bead–spring representation.

---

### **2. `Nanomotif_Bayes_Opt/`**

Bayesian Optimization framework for predicting bead–spring angle potentials.

---

### **3. `Nanomotif_Phase_Sep/`**

Simulation scripts for network formation and concentration-dependent phase separation.

---

### **4. `environment.yml`**

Conda environment file used to recreate the computational environment for all scripts.

To create the environment:

```bash
conda env create -f environment.yml
conda activate readdy2
```

---

## **📊 Data Availability**

oxDNA trajectories and phase-separation output files are **not included** in this repository due to their large size.

These datasets are **available upon request**.
Please contact:

📧 *x.schneider96@gmail.com*

---



