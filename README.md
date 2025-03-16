# Pull-off force prediction in viscoelastic adhesive Hertzian contact by physics augmented machine learning

## Overview
This repository contains machine learning models that integrate physics-based principles to enhance the prediction accuracy of pull-off force in viscoelastic adhesive Hertzian contact. By leveraging both **data-driven** and **physics-augmented ML** techniques, the approach provides improved generalization. The results are provided in a manuscript written by **Ali Maghami**, **Merten Stender**, and **Antonio Papangelo**.

The Python codes are written by Ali Maghami, a PhD researcher at **Politecnico di Bari, Italy**, and a research scholar at **Technische Universität Berlin, Germany**. The project is a sub-project of the **ERC-Funded project of ERC-2021-STG**, "Towards Future Interfaces With Tuneable Adhesion By Dynamic Excitation" - SURFACE (Project ID: 101039198, CUP: D95F22000430006) at TriboDynamics Lab (https://tribodynamicslab.poliba.it/).

## Features
- **Physics-Augmented Machine Learning**: Incorporates physical laws (Persson & Brenner model) into ML models.
- **Dataset Included**: Provides data for training and validation.
- **Multiple Models**: Compares different ML techniques.
- **Reproducible Code**: Easy-to-follow scripts and notebooks.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/alimaghamii/ML4Adhesion
   cd ML4Adhesion
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Prepare the dataset**  
   - Ensure the dataset is available in the `Data_Files` directory.

2. **Run machine learning models**  
   - To evaluate a data-driven model for pull-off force:
     ```bash
     python ML_Adh_predict.py
     ```
   - To evaluate a physics-augmented model for pull-off force:
     ```bash
     python PML_Adh_predict.py
     ```
   - To evaluate a data-driven model for work to pull off:
     ```bash
     python ML_Adh_predict_W.py
     ```
   - To evaluate a physics-augmented model for work to pull off:
     ```bash
     python PML_Adh_predict_W.py
     ```

## File Structure
```
📂 AI4Adhesion
│── 📄 README.md                 # Project documentation
│── 📄 requirements.txt          # Dependencies
│── 📄 ML_Adh.py                 # Data-driven model training for pull-off force
│── 📄 ML_Adh_predict.py         # Data-driven evaluation, plotting the results for pull-off force
│── 📄 ML_Adh_W.py               # Data-driven model training for work to pull off
│── 📄 ML_Adh_predict_W.py       # Data-driven evaluation, plotting the results for work to pull off
│── 📄 PML_Adh.py                # Physics-aug ML model training for pull-off force
│── 📄 PML_Adh_predict.py        # Physics-aug ML evaluation, plotting the results for pull-off force
│── 📄 PML_Adh_W.py              # Physics-aug ML model training for work to pull off
│── 📄 PML_Adh_predict_W.py      # Physics-aug ML evaluation, plotting the results for work to pull off
│── 📂 Data_Files                # Dataset directory
│── 📂 ML_saved                  # Saved models of data-driven model training for pull-off force
│── 📂 ML_saved_W                # Saved models of data-driven model training for work to pull off
│── 📂 PML_saved                 # Saved models of Physics-aug ML model training for pull-off force
│── 📂 PML_saved_W               # Saved models of Physics-aug ML model training for work to pull off
```

## Citation
If you use this repository, please cite the following papers which are the basis of the codes:

**Pull-off force prediction in viscoelastic adhesive Hertzian contact by physics augmented machine learning**  

A. Maghami, M. Stender, A. Papangelo, *Pull-off force prediction in viscoelastic adhesive Hertzian contact by physics augmented machine learning*, 2025.

**Bulk and fracture process zone contribution to the rate-dependent adhesion amplification in viscoelastic broad-band materials**

A. Maghami, Q. Wang, M. Tricarico, M. Ciavarella, Q. Li, A. Papangelo, *Bulk and fracture process zone contribution to the rate-dependent adhesion amplification in viscoelastic broad-band materials*, Journal of the Mechanics and Physics of Solids, 2024.

**Viscoelastic amplification of the pull-off stress in the detachment of a rigid flat punch from an adhesive soft viscoelastic layer**

A. Maghami, M. Tricarico, M. Ciavarella, A. Papangelo, *Viscoelastic amplification of the pull-off stress in the detachment of a rigid flat punch from an adhesive soft viscoelastic layer*, Engineering Fracture Mechanics, 2024.

## Contributing
We welcome contributions! To contribute:
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-branch`
3. Commit your changes: `git commit -m "Add feature"`
4. Push to the branch: `git push origin feature-branch`
5. Submit a pull request.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

This work was accomplished during a research stay of the first author (Ali Maghami) at the Chair of Cyber-Physical Systems in Mechanical Engineering (https://www.tu.berlin/en/cpsme) of the **Technische Universität Berlin**. A.M. and A.P. were partly supported by the Italian Ministry of University and Research under the Programme **"Department of Excellence" Legge 232/2016 (Grant No. CUP - D93C23000100001)**. A.P. and A.M. were supported by the **European Union (ERC-2021-STG, "Towards Future Interfaces With Tuneable Adhesion By Dynamic Excitation" - SURFACE, Project ID: 101039198, CUP: D95F22000430006)**.

Views and opinions expressed are those of the authors only and do not necessarily reflect those of the **European Union** or the **European Research Council**. Neither the European Union nor the granting authority can be held responsible for them. A.P. was partly supported by the **European Union through the program – Next Generation EU (PRIN-2022-PNRR, "Fighting blindness with two photon polymerization of wet adhesive, biomimetic scaffolds for neurosensory REtina-retinal Pigment epitheliAl Interface Regeneration" - REPAIR, Project ID: P2022TTZZF, CUP: D53D23018570001).**

