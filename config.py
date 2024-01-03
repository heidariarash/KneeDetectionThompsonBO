class Setting:
    n_var       = 5        # Input Dimension
    knees       = 1        # Number of Knees when it is a parameter
    s           = 1        # S parameter in DO2DK. Deos not affect any other problem.
    problem     = "DEB2DK" # The problem to be solved (DEB2DK, DEB3DK, CKP, CBEAM, CYCLONE, DTLZ7 ZDT3)
    EA_util     = "NSGAII" # The EA to be used (NSGAII, SMSEMOA)
    method      = "EHVI"   # The method to be used (HVKTS, HVKTS-EA, EHVI, HV-KNEE)
    repetitions = 20       # Number of optimization repetitions of a single problem with fixed settings