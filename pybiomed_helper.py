# this helper script is from PyBioMed to get protein descriptors. 
# there are incompatibility issues to use the entire library, so I copied and pasted here the src.
# https://pybiomed.readthedocs.io/en/latest/
import math
import string
import re

# import scipy


AALetter = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "E",
    "Q",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]

_Hydrophobicity = {
    "A": 0.62,
    "R": -2.53,
    "N": -0.78,
    "D": -0.90,
    "C": 0.29,
    "Q": -0.85,
    "E": -0.74,
    "G": 0.48,
    "H": -0.40,
    "I": 1.38,
    "L": 1.06,
    "K": -1.50,
    "M": 0.64,
    "F": 1.19,
    "P": 0.12,
    "S": -0.18,
    "T": -0.05,
    "W": 0.81,
    "Y": 0.26,
    "V": 1.08,
}

_hydrophilicity = {
    "A": -0.5,
    "R": 3.0,
    "N": 0.2,
    "D": 3.0,
    "C": -1.0,
    "Q": 0.2,
    "E": 3.0,
    "G": 0.0,
    "H": -0.5,
    "I": -1.8,
    "L": -1.8,
    "K": 3.0,
    "M": -1.3,
    "F": -2.5,
    "P": 0.0,
    "S": 0.3,
    "T": -0.4,
    "W": -3.4,
    "Y": -2.3,
    "V": -1.5,
}

_residuemass = {
    "A": 15.0,
    "R": 101.0,
    "N": 58.0,
    "D": 59.0,
    "C": 47.0,
    "Q": 72.0,
    "E": 73.0,
    "G": 1.000,
    "H": 82.0,
    "I": 57.0,
    "L": 57.0,
    "K": 73.0,
    "M": 75.0,
    "F": 91.0,
    "P": 42.0,
    "S": 31.0,
    "T": 45.0,
    "W": 130.0,
    "Y": 107.0,
    "V": 43.0,
}

_pK1 = {
    "A": 2.35,
    "C": 1.71,
    "D": 1.88,
    "E": 2.19,
    "F": 2.58,
    "G": 2.34,
    "H": 1.78,
    "I": 2.32,
    "K": 2.20,
    "L": 2.36,
    "M": 2.28,
    "N": 2.18,
    "P": 1.99,
    "Q": 2.17,
    "R": 2.18,
    "S": 2.21,
    "T": 2.15,
    "V": 2.29,
    "W": 2.38,
    "Y": 2.20,
}

_pK2 = {
    "A": 9.87,
    "C": 10.78,
    "D": 9.60,
    "E": 9.67,
    "F": 9.24,
    "G": 9.60,
    "H": 8.97,
    "I": 9.76,
    "K": 8.90,
    "L": 9.60,
    "M": 9.21,
    "N": 9.09,
    "P": 10.6,
    "Q": 9.13,
    "R": 9.09,
    "S": 9.15,
    "T": 9.12,
    "V": 9.74,
    "W": 9.39,
    "Y": 9.11,
}

_pI = {
    "A": 6.11,
    "C": 5.02,
    "D": 2.98,
    "E": 3.08,
    "F": 5.91,
    "G": 6.06,
    "H": 7.64,
    "I": 6.04,
    "K": 9.47,
    "L": 6.04,
    "M": 5.74,
    "N": 10.76,
    "P": 6.30,
    "Q": 5.65,
    "R": 10.76,
    "S": 5.68,
    "T": 5.60,
    "V": 6.02,
    "W": 5.88,
    "Y": 5.63,
}


#############################################################################################


def _mean(listvalue):
    """
    ########################################################################################
    The mean value of the list data.
    Usage:
    result=_mean(listvalue)
    ########################################################################################
    """
    return sum(listvalue) / len(listvalue)


##############################################################################################
def _std(listvalue, ddof=1):
    """
    ########################################################################################
    The standard deviation of the list data.
    Usage:
    result=_std(listvalue)
    ########################################################################################
    """
    mean = _mean(listvalue)
    temp = [math.pow(i - mean, 2) for i in listvalue]
    res = math.sqrt(sum(temp) / (len(listvalue) - ddof))
    return res


##############################################################################################
def NormalizeEachAAP(AAP):
    """
    ########################################################################################
    All of the amino acid indices are centralized and
    standardized before the calculation.
    Usage:
    result=NormalizeEachAAP(AAP)
    Input: AAP is a dict form containing the properties of 20 amino acids.
    Output: result is the a dict form containing the normalized properties
    of 20 amino acids.
    ########################################################################################
    """
    if len(AAP.values()) != 20:
        print("You can not input the correct number of properities of Amino acids!")
    else:
        Result = {}
        for i, j in AAP.items():
            Result[i] = (j - _mean(AAP.values())) / _std(AAP.values(), ddof=0)

    return Result


#############################################################################################
#############################################################################################
##################################Type I descriptors#########################################
####################### Pseudo-Amino Acid Composition descriptors############################
#############################################################################################
#############################################################################################
def _GetCorrelationFunction(
    Ri="S", Rj="D", AAP=[_Hydrophobicity, _hydrophilicity, _residuemass]
):
    """
    ########################################################################################
    Computing the correlation between two given amino acids using the above three
    properties.
    Usage:
    result=_GetCorrelationFunction(Ri,Rj)
    Input: Ri and Rj are the amino acids, respectively.
    Output: result is the correlation value between two amino acids.
    ########################################################################################
    """
    Hydrophobicity = NormalizeEachAAP(AAP[0])
    hydrophilicity = NormalizeEachAAP(AAP[1])
    residuemass = NormalizeEachAAP(AAP[2])
    theta1 = math.pow(Hydrophobicity[Ri] - Hydrophobicity[Rj], 2)
    theta2 = math.pow(hydrophilicity[Ri] - hydrophilicity[Rj], 2)
    theta3 = math.pow(residuemass[Ri] - residuemass[Rj], 2)
    theta = round((theta1 + theta2 + theta3) / 3.0, 3)
    return theta


#############################################################################################


def _GetSequenceOrderCorrelationFactor(ProteinSequence, k=1):
    """
    ########################################################################################
    Computing the Sequence order correlation factor with gap equal to k based on
    [_Hydrophobicity,_hydrophilicity,_residuemass].
    Usage:
    result=_GetSequenceOrderCorrelationFactor(protein,k)
    Input: protein is a pure protein sequence.
    k is the gap.
    Output: result is the correlation factor value with the gap equal to k.
    ########################################################################################
    """
    LengthSequence = len(ProteinSequence)
    res = []
    for i in range(LengthSequence - k):
        AA1 = ProteinSequence[i]
        AA2 = ProteinSequence[i + k]
        res.append(_GetCorrelationFunction(AA1, AA2))
    result = round(sum(res) / (LengthSequence - k), 3)
    return result


#############################################################################################


def GetAAComposition(ProteinSequence):
    """
    ########################################################################################
    Calculate the composition of Amino acids
    for a given protein sequence.
    Usage:
    result=CalculateAAComposition(protein)
    Input: protein is a pure protein sequence.
    Output: result is a dict form containing the composition of
    20 amino acids.
    ########################################################################################
    """
    LengthSequence = len(ProteinSequence)
    Result = {}
    for i in AALetter:
        Result[i] = round(float(ProteinSequence.count(i)) / LengthSequence * 100, 3)
    return Result


#############################################################################################
def _GetPseudoAAC1(ProteinSequence, lamda=10, weight=0.05):
    """
    #######################################################################################
    Computing the first 20 of type I pseudo-amino acid compostion descriptors based on
    [_Hydrophobicity,_hydrophilicity,_residuemass].
    ########################################################################################
    """
    rightpart = 0.0
    for i in range(lamda):
        rightpart = rightpart + _GetSequenceOrderCorrelationFactor(
            ProteinSequence, k=i + 1
        )
    AAC = GetAAComposition(ProteinSequence)

    result = {}
    temp = 1 + weight * rightpart
    for index, i in enumerate(AALetter):
        result["PAAC" + str(index + 1)] = round(AAC[i] / temp, 3)

    return result


#############################################################################################
def _GetPseudoAAC2(ProteinSequence, lamda=10, weight=0.05):
    """
    ########################################################################################
    Computing the last lamda of type I pseudo-amino acid compostion descriptors based on
    [_Hydrophobicity,_hydrophilicity,_residuemass].
    ########################################################################################
    """
    rightpart = []
    for i in range(lamda):
        rightpart.append(_GetSequenceOrderCorrelationFactor(ProteinSequence, k=i + 1))

    result = {}
    temp = 1 + weight * sum(rightpart)
    for index in range(20, 20 + lamda):
        result["PAAC" + str(index + 1)] = round(
            weight * rightpart[index - 20] / temp * 100, 3
        )

    return result


#############################################################################################


def _GetPseudoAAC(ProteinSequence, lamda=10, weight=0.05):
    """
    #######################################################################################
    Computing all of type I pseudo-amino acid compostion descriptors based on three given
    properties. Note that the number of PAAC strongly depends on the lamda value. if lamda
    = 20, we can obtain 20+20=40 PAAC descriptors. The size of these values depends on the
    choice of lamda and weight simultaneously.
    AAP=[_Hydrophobicity,_hydrophilicity,_residuemass]
    Usage:
    result=_GetAPseudoAAC(protein,lamda,weight)
    Input: protein is a pure protein sequence.
    lamda factor reflects the rank of correlation and is a non-Negative integer, such as 15.
    Note that (1)lamda should NOT be larger than the length of input protein sequence;
    (2) lamda must be non-Negative integer, such as 0, 1, 2, ...; (3) when lamda =0, the
    output of PseAA server is the 20-D amino acid composition.
    weight factor is designed for the users to put weight on the additional PseAA components
    with respect to the conventional AA components. The user can select any value within the
    region from 0.05 to 0.7 for the weight factor.
    Output: result is a dict form containing calculated 20+lamda PAAC descriptors.
    ########################################################################################
    """
    res = {}
    res.update(_GetPseudoAAC1(ProteinSequence, lamda=lamda, weight=weight))
    res.update(_GetPseudoAAC2(ProteinSequence, lamda=lamda, weight=weight))
    return np.array(list(res.values()))


#############################################################################################
##################################Type II descriptors########################################
###############Amphiphilic Pseudo-Amino Acid Composition descriptors#########################
#############################################################################################
#############################################################################################
def _GetCorrelationFunctionForAPAAC(
    Ri="S", Rj="D", AAP=[_Hydrophobicity, _hydrophilicity]
):
    """
    ########################################################################################
    Computing the correlation between two given amino acids using the above two
    properties for APAAC (type II PseAAC).
    Usage:
    result=_GetCorrelationFunctionForAPAAC(Ri,Rj)
    Input: Ri and Rj are the amino acids, respectively.
    Output: result is the correlation value between two amino acids.
    ########################################################################################
    """
    Hydrophobicity = NormalizeEachAAP(AAP[0])
    hydrophilicity = NormalizeEachAAP(AAP[1])
    theta1 = round(Hydrophobicity[Ri] * Hydrophobicity[Rj], 3)
    theta2 = round(hydrophilicity[Ri] * hydrophilicity[Rj], 3)

    return theta1, theta2


#############################################################################################
def GetSequenceOrderCorrelationFactorForAPAAC(ProteinSequence, k=1):
    """
    ########################################################################################
    Computing the Sequence order correlation factor with gap equal to k based on
    [_Hydrophobicity,_hydrophilicity] for APAAC (type II PseAAC) .
    Usage:
    result=GetSequenceOrderCorrelationFactorForAPAAC(protein,k)
    Input: protein is a pure protein sequence.
    k is the gap.
    Output: result is the correlation factor value with the gap equal to k.
    ########################################################################################
    """
    LengthSequence = len(ProteinSequence)
    resHydrophobicity = []
    reshydrophilicity = []
    for i in range(LengthSequence - k):
        AA1 = ProteinSequence[i]
        AA2 = ProteinSequence[i + k]
        temp = _GetCorrelationFunctionForAPAAC(AA1, AA2)
        resHydrophobicity.append(temp[0])
        reshydrophilicity.append(temp[1])
    result = []
    result.append(round(sum(resHydrophobicity) / (LengthSequence - k), 3))
    result.append(round(sum(reshydrophilicity) / (LengthSequence - k), 3))
    return result


#############################################################################################
def GetAPseudoAAC1(ProteinSequence, lamda=30, weight=0.5):
    """
    ########################################################################################
    Computing the first 20 of type II pseudo-amino acid compostion descriptors based on
    [_Hydrophobicity,_hydrophilicity].
    ########################################################################################
    """
    rightpart = 0.0
    for i in range(lamda):
        rightpart = rightpart + sum(
            GetSequenceOrderCorrelationFactorForAPAAC(ProteinSequence, k=i + 1)
        )
    AAC = GetAAComposition(ProteinSequence)

    result = {}
    temp = 1 + weight * rightpart
    for index, i in enumerate(AALetter):
        result["APAAC" + str(index + 1)] = round(AAC[i] / temp, 3)

    return result


#############################################################################################
def GetAPseudoAAC2(ProteinSequence, lamda=30, weight=0.5):
    """
    #######################################################################################
    Computing the last lamda of type II pseudo-amino acid compostion descriptors based on
    [_Hydrophobicity,_hydrophilicity].
    #######################################################################################
    """
    rightpart = []
    for i in range(lamda):
        temp = GetSequenceOrderCorrelationFactorForAPAAC(ProteinSequence, k=i + 1)
        rightpart.append(temp[0])
        rightpart.append(temp[1])

    result = {}
    temp = 1 + weight * sum(rightpart)
    for index in range(20, 20 + 2 * lamda):
        result["PAAC" + str(index + 1)] = round(
            weight * rightpart[index - 20] / temp * 100, 3
        )

    return result


#############################################################################################
def GetAPseudoAAC(ProteinSequence, lamda=30, weight=0.5):
    """
    #######################################################################################
    Computing all of type II pseudo-amino acid compostion descriptors based on the given
    properties. Note that the number of PAAC strongly depends on the lamda value. if lamda
    = 20, we can obtain 20+20=40 PAAC descriptors. The size of these values depends on the
    choice of lamda and weight simultaneously.
    Usage:
    result=GetAPseudoAAC(protein,lamda,weight)
    Input: protein is a pure protein sequence.
    lamda factor reflects the rank of correlation and is a non-Negative integer, such as 15.
    Note that (1)lamda should NOT be larger than the length of input protein sequence;
    (2) lamda must be non-Negative integer, such as 0, 1, 2, ...; (3) when lamda =0, the
    output of PseAA server is the 20-D amino acid composition.
    weight factor is designed for the users to put weight on the additional PseAA components
    with respect to the conventional AA components. The user can select any value within the
    region from 0.05 to 0.7 for the weight factor.
    Output: result is a dict form containing calculated 20+lamda PAAC descriptors.
    #######################################################################################
    """
    res = {}
    res.update(GetAPseudoAAC1(ProteinSequence, lamda=lamda, weight=weight))
    res.update(GetAPseudoAAC2(ProteinSequence, lamda=lamda, weight=weight))
    return res


#############################################################################################
#############################################################################################
##################################Type I descriptors#########################################
####################### Pseudo-Amino Acid Composition descriptors############################
#############################based on different properties###################################
#############################################################################################
#############################################################################################
def GetCorrelationFunction(Ri="S", Rj="D", AAP=[]):
    """
    ########################################################################################
    Computing the correlation between two given amino acids using the given
    properties.
    Usage:
    result=GetCorrelationFunction(Ri,Rj,AAP)
    Input: Ri and Rj are the amino acids, respectively.
    AAP is a list form containing the properties, each of which is a dict form.
    Output: result is the correlation value between two amino acids.
    ########################################################################################
    """
    NumAAP = len(AAP)
    theta = 0.0
    for i in range(NumAAP):
        temp = NormalizeEachAAP(AAP[i])
        theta = theta + math.pow(temp[Ri] - temp[Rj], 2)
    result = round(theta / NumAAP, 3)
    return result


#############################################################################################
def GetSequenceOrderCorrelationFactor(ProteinSequence, k=1, AAP=[]):
    """
    ########################################################################################
    Computing the Sequence order correlation factor with gap equal to k based on
    the given properities.
    Usage:
    result=GetSequenceOrderCorrelationFactor(protein,k,AAP)
    Input: protein is a pure protein sequence.
    k is the gap.
    AAP is a list form containing the properties, each of which is a dict form.
    Output: result is the correlation factor value with the gap equal to k.
    ########################################################################################
    """
    LengthSequence = len(ProteinSequence)
    res = []
    for i in range(LengthSequence - k):
        AA1 = ProteinSequence[i]
        AA2 = ProteinSequence[i + k]
        res.append(GetCorrelationFunction(AA1, AA2, AAP))
    result = round(sum(res) / (LengthSequence - k), 3)
    return result


#############################################################################################
def GetPseudoAAC1(ProteinSequence, lamda=30, weight=0.05, AAP=[]):
    """
    #######################################################################################
    Computing the first 20 of type I pseudo-amino acid compostion descriptors based on the given
    properties.
    ########################################################################################
    """
    rightpart = 0.0
    for i in range(lamda):
        rightpart = rightpart + GetSequenceOrderCorrelationFactor(
            ProteinSequence, i + 1, AAP
        )
    AAC = GetAAComposition(ProteinSequence)

    result = {}
    temp = 1 + weight * rightpart
    for index, i in enumerate(AALetter):
        result["PAAC" + str(index + 1)] = round(AAC[i] / temp, 3)

    return result


#############################################################################################
def GetPseudoAAC2(ProteinSequence, lamda=30, weight=0.05, AAP=[]):
    """
    #######################################################################################
    Computing the last lamda of type I pseudo-amino acid compostion descriptors based on the given
    properties.
    ########################################################################################
    """
    rightpart = []
    for i in range(lamda):
        rightpart.append(GetSequenceOrderCorrelationFactor(ProteinSequence, i + 1, AAP))

    result = {}
    temp = 1 + weight * sum(rightpart)
    for index in range(20, 20 + lamda):
        result["PAAC" + str(index + 1)] = round(
            weight * rightpart[index - 20] / temp * 100, 3
        )

    return result


#############################################################################################


def GetPseudoAAC(ProteinSequence, lamda=30, weight=0.05, AAP=[]):
    """
    #######################################################################################
    Computing all of type I pseudo-amino acid compostion descriptors based on the given
    properties. Note that the number of PAAC strongly depends on the lamda value. if lamda
    = 20, we can obtain 20+20=40 PAAC descriptors. The size of these values depends on the
    choice of lamda and weight simultaneously. You must specify some properties into AAP.
    Usage:
    result=GetPseudoAAC(protein,lamda,weight)
    Input: protein is a pure protein sequence.
    lamda factor reflects the rank of correlation and is a non-Negative integer, such as 15.
    Note that (1)lamda should NOT be larger than the length of input protein sequence;
    (2) lamda must be non-Negative integer, such as 0, 1, 2, ...; (3) when lamda =0, the
    output of PseAA server is the 20-D amino acid composition.
    weight factor is designed for the users to put weight on the additional PseAA components
    with respect to the conventional AA components. The user can select any value within the
    region from 0.05 to 0.7 for the weight factor.
    AAP is a list form containing the properties, each of which is a dict form.
    Output: result is a dict form containing calculated 20+lamda PAAC descriptors.
    ########################################################################################
    """
    res = {}
    res.update(GetPseudoAAC1(ProteinSequence, lamda, weight, AAP))
    res.update(GetPseudoAAC2(ProteinSequence, lamda, weight, AAP))
    return res


def CalculateAAComposition(ProteinSequence):
    """
    ########################################################################
    Calculate the composition of Amino acids
    for a given protein sequence.
    Usage:
    result=CalculateAAComposition(protein)
    Input: protein is a pure protein sequence.
    Output: result is a dict form containing the composition of
    20 amino acids.
    ########################################################################
    """
    LengthSequence = len(ProteinSequence)
    Result = {}
    for i in AALetter:
        Result[i] = round(float(ProteinSequence.count(i)) / LengthSequence * 100, 3)
    return Result


def CalculateDipeptideComposition(ProteinSequence):
    """
    Calculate the composition of dipeptidefor a given protein sequence.
    Usage:
    result=CalculateDipeptideComposition(protein)
    Input: protein is a pure protein sequence.
    Output: result is a dict form containing the composition of
    400 dipeptides.
    """

    LengthSequence = len(ProteinSequence)
    Result = {}
    for i in AALetter:
        for j in AALetter:
            Dipeptide = i + j
            Result[Dipeptide] = round(
                float(ProteinSequence.count(Dipeptide)) / (LengthSequence - 1) * 100, 2
            )
    return Result


#############################################################################################


def Getkmers():
    """
    ########################################################################
    Get the amino acid list of 3-mers.
    Usage:
    result=Getkmers()
    Output: result is a list form containing 8000 tri-peptides.
    ########################################################################
    """
    kmers = list()
    for i in AALetter:
        for j in AALetter:
            for k in AALetter:
                kmers.append(i + j + k)
    return kmers


def GetSpectrumDict(proteinsequence):
    """
    ########################################################################
    Calcualte the spectrum descriptors of 3-mers for a given protein.
    Usage:
    result=GetSpectrumDict(protein)
    Input: protein is a pure protein sequence.
    Output: result is a dict form containing the composition values of 8000
    3-mers.
    """
    result = {}
    kmers = Getkmers()
    for i in kmers:
        result[i] = len(re.findall(i, proteinsequence))
    return result

import numpy as np

#############################################################################################
def CalculateAADipeptideComposition(ProteinSequence):
    """
    ########################################################################
    Calculate the composition of AADs, dipeptide and 3-mers for a
    given protein sequence.
    Usage:
    result=CalculateAADipeptideComposition(protein)
    Input: protein is a pure protein sequence.
    Output: result is a dict form containing all composition values of
    AADs, dipeptide and 3-mers (8420).
    ########################################################################
    """

    result = {}
    result.update(CalculateAAComposition(ProteinSequence))
    result.update(CalculateDipeptideComposition(ProteinSequence))
    result.update(GetSpectrumDict(ProteinSequence))

    return np.array(list(result.values()))


_repmat = {
    1: ["A", "G", "V"],
    2: ["I", "L", "F", "P"],
    3: ["Y", "M", "T", "S"],
    4: ["H", "N", "Q", "W"],
    5: ["R", "K"],
    6: ["D", "E"],
    7: ["C"],
}

def _Str2Num(proteinsequence):
    """
    translate the amino acid letter into the corresponding class based on the
    given form.
    """
    repmat = {}
    for i in _repmat:
        for j in _repmat[i]:
            repmat[j] = i

    res = proteinsequence
    for i in repmat:
        res = res.replace(i, str(repmat[i]))
    return res


###############################################################################
def CalculateConjointTriad(proteinsequence):
    """
    Calculate the conjoint triad features from protein sequence.
    Useage:
    res = CalculateConjointTriad(protein)
    Input: protein is a pure protein sequence.
    Output is a dict form containing all 343 conjoint triad features.
    """
    res = {}
    proteinnum = _Str2Num(proteinsequence)
    for i in range(1, 8):
        for j in range(1, 8):
            for k in range(1, 8):
                temp = str(i) + str(j) + str(k)
                res[temp] = proteinnum.count(temp)
    return np.array(list(res.values()))


## Distance is the Schneider-Wrede physicochemical distance matrix used by Chou et. al.
_Distance1 = {
    "GW": 0.923,
    "GV": 0.464,
    "GT": 0.272,
    "GS": 0.158,
    "GR": 1.0,
    "GQ": 0.467,
    "GP": 0.323,
    "GY": 0.728,
    "GG": 0.0,
    "GF": 0.727,
    "GE": 0.807,
    "GD": 0.776,
    "GC": 0.312,
    "GA": 0.206,
    "GN": 0.381,
    "GM": 0.557,
    "GL": 0.591,
    "GK": 0.894,
    "GI": 0.592,
    "GH": 0.769,
    "ME": 0.879,
    "MD": 0.932,
    "MG": 0.569,
    "MF": 0.182,
    "MA": 0.383,
    "MC": 0.276,
    "MM": 0.0,
    "ML": 0.062,
    "MN": 0.447,
    "MI": 0.058,
    "MH": 0.648,
    "MK": 0.884,
    "MT": 0.358,
    "MW": 0.391,
    "MV": 0.12,
    "MQ": 0.372,
    "MP": 0.285,
    "MS": 0.417,
    "MR": 1.0,
    "MY": 0.255,
    "FP": 0.42,
    "FQ": 0.459,
    "FR": 1.0,
    "FS": 0.548,
    "FT": 0.499,
    "FV": 0.252,
    "FW": 0.207,
    "FY": 0.179,
    "FA": 0.508,
    "FC": 0.405,
    "FD": 0.977,
    "FE": 0.918,
    "FF": 0.0,
    "FG": 0.69,
    "FH": 0.663,
    "FI": 0.128,
    "FK": 0.903,
    "FL": 0.131,
    "FM": 0.169,
    "FN": 0.541,
    "SY": 0.615,
    "SS": 0.0,
    "SR": 1.0,
    "SQ": 0.358,
    "SP": 0.181,
    "SW": 0.827,
    "SV": 0.342,
    "ST": 0.174,
    "SK": 0.883,
    "SI": 0.478,
    "SH": 0.718,
    "SN": 0.289,
    "SM": 0.44,
    "SL": 0.474,
    "SC": 0.185,
    "SA": 0.1,
    "SG": 0.17,
    "SF": 0.622,
    "SE": 0.812,
    "SD": 0.801,
    "YI": 0.23,
    "YH": 0.678,
    "YK": 0.904,
    "YM": 0.268,
    "YL": 0.219,
    "YN": 0.512,
    "YA": 0.587,
    "YC": 0.478,
    "YE": 0.932,
    "YD": 1.0,
    "YG": 0.782,
    "YF": 0.202,
    "YY": 0.0,
    "YQ": 0.404,
    "YP": 0.444,
    "YS": 0.612,
    "YR": 0.995,
    "YT": 0.557,
    "YW": 0.244,
    "YV": 0.328,
    "LF": 0.139,
    "LG": 0.596,
    "LD": 0.944,
    "LE": 0.892,
    "LC": 0.296,
    "LA": 0.405,
    "LN": 0.452,
    "LL": 0.0,
    "LM": 0.062,
    "LK": 0.893,
    "LH": 0.653,
    "LI": 0.013,
    "LV": 0.133,
    "LW": 0.341,
    "LT": 0.397,
    "LR": 1.0,
    "LS": 0.443,
    "LP": 0.309,
    "LQ": 0.376,
    "LY": 0.205,
    "RT": 0.808,
    "RV": 0.914,
    "RW": 1.0,
    "RP": 0.796,
    "RQ": 0.668,
    "RR": 0.0,
    "RS": 0.86,
    "RY": 0.859,
    "RD": 0.305,
    "RE": 0.225,
    "RF": 0.977,
    "RG": 0.928,
    "RA": 0.919,
    "RC": 0.905,
    "RL": 0.92,
    "RM": 0.908,
    "RN": 0.69,
    "RH": 0.498,
    "RI": 0.929,
    "RK": 0.141,
    "VH": 0.649,
    "VI": 0.135,
    "EM": 0.83,
    "EL": 0.854,
    "EN": 0.599,
    "EI": 0.86,
    "EH": 0.406,
    "EK": 0.143,
    "EE": 0.0,
    "ED": 0.133,
    "EG": 0.779,
    "EF": 0.932,
    "EA": 0.79,
    "EC": 0.788,
    "VM": 0.12,
    "EY": 0.837,
    "VN": 0.38,
    "ET": 0.682,
    "EW": 1.0,
    "EV": 0.824,
    "EQ": 0.598,
    "EP": 0.688,
    "ES": 0.726,
    "ER": 0.234,
    "VP": 0.212,
    "VQ": 0.339,
    "VR": 1.0,
    "VT": 0.305,
    "VW": 0.472,
    "KC": 0.871,
    "KA": 0.889,
    "KG": 0.9,
    "KF": 0.957,
    "KE": 0.149,
    "KD": 0.279,
    "KK": 0.0,
    "KI": 0.899,
    "KH": 0.438,
    "KN": 0.667,
    "KM": 0.871,
    "KL": 0.892,
    "KS": 0.825,
    "KR": 0.154,
    "KQ": 0.639,
    "KP": 0.757,
    "KW": 1.0,
    "KV": 0.882,
    "KT": 0.759,
    "KY": 0.848,
    "DN": 0.56,
    "DL": 0.841,
    "DM": 0.819,
    "DK": 0.249,
    "DH": 0.435,
    "DI": 0.847,
    "DF": 0.924,
    "DG": 0.697,
    "DD": 0.0,
    "DE": 0.124,
    "DC": 0.742,
    "DA": 0.729,
    "DY": 0.836,
    "DV": 0.797,
    "DW": 1.0,
    "DT": 0.649,
    "DR": 0.295,
    "DS": 0.667,
    "DP": 0.657,
    "DQ": 0.584,
    "QQ": 0.0,
    "QP": 0.272,
    "QS": 0.461,
    "QR": 1.0,
    "QT": 0.389,
    "QW": 0.831,
    "QV": 0.464,
    "QY": 0.522,
    "QA": 0.512,
    "QC": 0.462,
    "QE": 0.861,
    "QD": 0.903,
    "QG": 0.648,
    "QF": 0.671,
    "QI": 0.532,
    "QH": 0.765,
    "QK": 0.881,
    "QM": 0.505,
    "QL": 0.518,
    "QN": 0.181,
    "WG": 0.829,
    "WF": 0.196,
    "WE": 0.931,
    "WD": 1.0,
    "WC": 0.56,
    "WA": 0.658,
    "WN": 0.631,
    "WM": 0.344,
    "WL": 0.304,
    "WK": 0.892,
    "WI": 0.305,
    "WH": 0.678,
    "WW": 0.0,
    "WV": 0.418,
    "WT": 0.638,
    "WS": 0.689,
    "WR": 0.968,
    "WQ": 0.538,
    "WP": 0.555,
    "WY": 0.204,
    "PR": 1.0,
    "PS": 0.196,
    "PP": 0.0,
    "PQ": 0.228,
    "PV": 0.244,
    "PW": 0.72,
    "PT": 0.161,
    "PY": 0.481,
    "PC": 0.179,
    "PA": 0.22,
    "PF": 0.515,
    "PG": 0.376,
    "PD": 0.852,
    "PE": 0.831,
    "PK": 0.875,
    "PH": 0.696,
    "PI": 0.363,
    "PN": 0.231,
    "PL": 0.357,
    "PM": 0.326,
    "CK": 0.887,
    "CI": 0.304,
    "CH": 0.66,
    "CN": 0.324,
    "CM": 0.277,
    "CL": 0.301,
    "CC": 0.0,
    "CA": 0.114,
    "CG": 0.32,
    "CF": 0.437,
    "CE": 0.838,
    "CD": 0.847,
    "CY": 0.457,
    "CS": 0.176,
    "CR": 1.0,
    "CQ": 0.341,
    "CP": 0.157,
    "CW": 0.639,
    "CV": 0.167,
    "CT": 0.233,
    "IY": 0.213,
    "VA": 0.275,
    "VC": 0.165,
    "VD": 0.9,
    "VE": 0.867,
    "VF": 0.269,
    "VG": 0.471,
    "IQ": 0.383,
    "IP": 0.311,
    "IS": 0.443,
    "IR": 1.0,
    "VL": 0.134,
    "IT": 0.396,
    "IW": 0.339,
    "IV": 0.133,
    "II": 0.0,
    "IH": 0.652,
    "IK": 0.892,
    "VS": 0.322,
    "IM": 0.057,
    "IL": 0.013,
    "VV": 0.0,
    "IN": 0.457,
    "IA": 0.403,
    "VY": 0.31,
    "IC": 0.296,
    "IE": 0.891,
    "ID": 0.942,
    "IG": 0.592,
    "IF": 0.134,
    "HY": 0.821,
    "HR": 0.697,
    "HS": 0.865,
    "HP": 0.777,
    "HQ": 0.716,
    "HV": 0.831,
    "HW": 0.981,
    "HT": 0.834,
    "HK": 0.566,
    "HH": 0.0,
    "HI": 0.848,
    "HN": 0.754,
    "HL": 0.842,
    "HM": 0.825,
    "HC": 0.836,
    "HA": 0.896,
    "HF": 0.907,
    "HG": 1.0,
    "HD": 0.629,
    "HE": 0.547,
    "NH": 0.78,
    "NI": 0.615,
    "NK": 0.891,
    "NL": 0.603,
    "NM": 0.588,
    "NN": 0.0,
    "NA": 0.424,
    "NC": 0.425,
    "ND": 0.838,
    "NE": 0.835,
    "NF": 0.766,
    "NG": 0.512,
    "NY": 0.641,
    "NP": 0.266,
    "NQ": 0.175,
    "NR": 1.0,
    "NS": 0.361,
    "NT": 0.368,
    "NV": 0.503,
    "NW": 0.945,
    "TY": 0.596,
    "TV": 0.345,
    "TW": 0.816,
    "TT": 0.0,
    "TR": 1.0,
    "TS": 0.185,
    "TP": 0.159,
    "TQ": 0.322,
    "TN": 0.315,
    "TL": 0.453,
    "TM": 0.403,
    "TK": 0.866,
    "TH": 0.737,
    "TI": 0.455,
    "TF": 0.604,
    "TG": 0.312,
    "TD": 0.83,
    "TE": 0.812,
    "TC": 0.261,
    "TA": 0.251,
    "AA": 0.0,
    "AC": 0.112,
    "AE": 0.827,
    "AD": 0.819,
    "AG": 0.208,
    "AF": 0.54,
    "AI": 0.407,
    "AH": 0.696,
    "AK": 0.891,
    "AM": 0.379,
    "AL": 0.406,
    "AN": 0.318,
    "AQ": 0.372,
    "AP": 0.191,
    "AS": 0.094,
    "AR": 1.0,
    "AT": 0.22,
    "AW": 0.739,
    "AV": 0.273,
    "AY": 0.552,
    "VK": 0.889,
}

## Distance is the Grantham chemical distance matrix used by Grantham et. al.
_Distance2 = {
    "GW": 0.923,
    "GV": 0.464,
    "GT": 0.272,
    "GS": 0.158,
    "GR": 1.0,
    "GQ": 0.467,
    "GP": 0.323,
    "GY": 0.728,
    "GG": 0.0,
    "GF": 0.727,
    "GE": 0.807,
    "GD": 0.776,
    "GC": 0.312,
    "GA": 0.206,
    "GN": 0.381,
    "GM": 0.557,
    "GL": 0.591,
    "GK": 0.894,
    "GI": 0.592,
    "GH": 0.769,
    "ME": 0.879,
    "MD": 0.932,
    "MG": 0.569,
    "MF": 0.182,
    "MA": 0.383,
    "MC": 0.276,
    "MM": 0.0,
    "ML": 0.062,
    "MN": 0.447,
    "MI": 0.058,
    "MH": 0.648,
    "MK": 0.884,
    "MT": 0.358,
    "MW": 0.391,
    "MV": 0.12,
    "MQ": 0.372,
    "MP": 0.285,
    "MS": 0.417,
    "MR": 1.0,
    "MY": 0.255,
    "FP": 0.42,
    "FQ": 0.459,
    "FR": 1.0,
    "FS": 0.548,
    "FT": 0.499,
    "FV": 0.252,
    "FW": 0.207,
    "FY": 0.179,
    "FA": 0.508,
    "FC": 0.405,
    "FD": 0.977,
    "FE": 0.918,
    "FF": 0.0,
    "FG": 0.69,
    "FH": 0.663,
    "FI": 0.128,
    "FK": 0.903,
    "FL": 0.131,
    "FM": 0.169,
    "FN": 0.541,
    "SY": 0.615,
    "SS": 0.0,
    "SR": 1.0,
    "SQ": 0.358,
    "SP": 0.181,
    "SW": 0.827,
    "SV": 0.342,
    "ST": 0.174,
    "SK": 0.883,
    "SI": 0.478,
    "SH": 0.718,
    "SN": 0.289,
    "SM": 0.44,
    "SL": 0.474,
    "SC": 0.185,
    "SA": 0.1,
    "SG": 0.17,
    "SF": 0.622,
    "SE": 0.812,
    "SD": 0.801,
    "YI": 0.23,
    "YH": 0.678,
    "YK": 0.904,
    "YM": 0.268,
    "YL": 0.219,
    "YN": 0.512,
    "YA": 0.587,
    "YC": 0.478,
    "YE": 0.932,
    "YD": 1.0,
    "YG": 0.782,
    "YF": 0.202,
    "YY": 0.0,
    "YQ": 0.404,
    "YP": 0.444,
    "YS": 0.612,
    "YR": 0.995,
    "YT": 0.557,
    "YW": 0.244,
    "YV": 0.328,
    "LF": 0.139,
    "LG": 0.596,
    "LD": 0.944,
    "LE": 0.892,
    "LC": 0.296,
    "LA": 0.405,
    "LN": 0.452,
    "LL": 0.0,
    "LM": 0.062,
    "LK": 0.893,
    "LH": 0.653,
    "LI": 0.013,
    "LV": 0.133,
    "LW": 0.341,
    "LT": 0.397,
    "LR": 1.0,
    "LS": 0.443,
    "LP": 0.309,
    "LQ": 0.376,
    "LY": 0.205,
    "RT": 0.808,
    "RV": 0.914,
    "RW": 1.0,
    "RP": 0.796,
    "RQ": 0.668,
    "RR": 0.0,
    "RS": 0.86,
    "RY": 0.859,
    "RD": 0.305,
    "RE": 0.225,
    "RF": 0.977,
    "RG": 0.928,
    "RA": 0.919,
    "RC": 0.905,
    "RL": 0.92,
    "RM": 0.908,
    "RN": 0.69,
    "RH": 0.498,
    "RI": 0.929,
    "RK": 0.141,
    "VH": 0.649,
    "VI": 0.135,
    "EM": 0.83,
    "EL": 0.854,
    "EN": 0.599,
    "EI": 0.86,
    "EH": 0.406,
    "EK": 0.143,
    "EE": 0.0,
    "ED": 0.133,
    "EG": 0.779,
    "EF": 0.932,
    "EA": 0.79,
    "EC": 0.788,
    "VM": 0.12,
    "EY": 0.837,
    "VN": 0.38,
    "ET": 0.682,
    "EW": 1.0,
    "EV": 0.824,
    "EQ": 0.598,
    "EP": 0.688,
    "ES": 0.726,
    "ER": 0.234,
    "VP": 0.212,
    "VQ": 0.339,
    "VR": 1.0,
    "VT": 0.305,
    "VW": 0.472,
    "KC": 0.871,
    "KA": 0.889,
    "KG": 0.9,
    "KF": 0.957,
    "KE": 0.149,
    "KD": 0.279,
    "KK": 0.0,
    "KI": 0.899,
    "KH": 0.438,
    "KN": 0.667,
    "KM": 0.871,
    "KL": 0.892,
    "KS": 0.825,
    "KR": 0.154,
    "KQ": 0.639,
    "KP": 0.757,
    "KW": 1.0,
    "KV": 0.882,
    "KT": 0.759,
    "KY": 0.848,
    "DN": 0.56,
    "DL": 0.841,
    "DM": 0.819,
    "DK": 0.249,
    "DH": 0.435,
    "DI": 0.847,
    "DF": 0.924,
    "DG": 0.697,
    "DD": 0.0,
    "DE": 0.124,
    "DC": 0.742,
    "DA": 0.729,
    "DY": 0.836,
    "DV": 0.797,
    "DW": 1.0,
    "DT": 0.649,
    "DR": 0.295,
    "DS": 0.667,
    "DP": 0.657,
    "DQ": 0.584,
    "QQ": 0.0,
    "QP": 0.272,
    "QS": 0.461,
    "QR": 1.0,
    "QT": 0.389,
    "QW": 0.831,
    "QV": 0.464,
    "QY": 0.522,
    "QA": 0.512,
    "QC": 0.462,
    "QE": 0.861,
    "QD": 0.903,
    "QG": 0.648,
    "QF": 0.671,
    "QI": 0.532,
    "QH": 0.765,
    "QK": 0.881,
    "QM": 0.505,
    "QL": 0.518,
    "QN": 0.181,
    "WG": 0.829,
    "WF": 0.196,
    "WE": 0.931,
    "WD": 1.0,
    "WC": 0.56,
    "WA": 0.658,
    "WN": 0.631,
    "WM": 0.344,
    "WL": 0.304,
    "WK": 0.892,
    "WI": 0.305,
    "WH": 0.678,
    "WW": 0.0,
    "WV": 0.418,
    "WT": 0.638,
    "WS": 0.689,
    "WR": 0.968,
    "WQ": 0.538,
    "WP": 0.555,
    "WY": 0.204,
    "PR": 1.0,
    "PS": 0.196,
    "PP": 0.0,
    "PQ": 0.228,
    "PV": 0.244,
    "PW": 0.72,
    "PT": 0.161,
    "PY": 0.481,
    "PC": 0.179,
    "PA": 0.22,
    "PF": 0.515,
    "PG": 0.376,
    "PD": 0.852,
    "PE": 0.831,
    "PK": 0.875,
    "PH": 0.696,
    "PI": 0.363,
    "PN": 0.231,
    "PL": 0.357,
    "PM": 0.326,
    "CK": 0.887,
    "CI": 0.304,
    "CH": 0.66,
    "CN": 0.324,
    "CM": 0.277,
    "CL": 0.301,
    "CC": 0.0,
    "CA": 0.114,
    "CG": 0.32,
    "CF": 0.437,
    "CE": 0.838,
    "CD": 0.847,
    "CY": 0.457,
    "CS": 0.176,
    "CR": 1.0,
    "CQ": 0.341,
    "CP": 0.157,
    "CW": 0.639,
    "CV": 0.167,
    "CT": 0.233,
    "IY": 0.213,
    "VA": 0.275,
    "VC": 0.165,
    "VD": 0.9,
    "VE": 0.867,
    "VF": 0.269,
    "VG": 0.471,
    "IQ": 0.383,
    "IP": 0.311,
    "IS": 0.443,
    "IR": 1.0,
    "VL": 0.134,
    "IT": 0.396,
    "IW": 0.339,
    "IV": 0.133,
    "II": 0.0,
    "IH": 0.652,
    "IK": 0.892,
    "VS": 0.322,
    "IM": 0.057,
    "IL": 0.013,
    "VV": 0.0,
    "IN": 0.457,
    "IA": 0.403,
    "VY": 0.31,
    "IC": 0.296,
    "IE": 0.891,
    "ID": 0.942,
    "IG": 0.592,
    "IF": 0.134,
    "HY": 0.821,
    "HR": 0.697,
    "HS": 0.865,
    "HP": 0.777,
    "HQ": 0.716,
    "HV": 0.831,
    "HW": 0.981,
    "HT": 0.834,
    "HK": 0.566,
    "HH": 0.0,
    "HI": 0.848,
    "HN": 0.754,
    "HL": 0.842,
    "HM": 0.825,
    "HC": 0.836,
    "HA": 0.896,
    "HF": 0.907,
    "HG": 1.0,
    "HD": 0.629,
    "HE": 0.547,
    "NH": 0.78,
    "NI": 0.615,
    "NK": 0.891,
    "NL": 0.603,
    "NM": 0.588,
    "NN": 0.0,
    "NA": 0.424,
    "NC": 0.425,
    "ND": 0.838,
    "NE": 0.835,
    "NF": 0.766,
    "NG": 0.512,
    "NY": 0.641,
    "NP": 0.266,
    "NQ": 0.175,
    "NR": 1.0,
    "NS": 0.361,
    "NT": 0.368,
    "NV": 0.503,
    "NW": 0.945,
    "TY": 0.596,
    "TV": 0.345,
    "TW": 0.816,
    "TT": 0.0,
    "TR": 1.0,
    "TS": 0.185,
    "TP": 0.159,
    "TQ": 0.322,
    "TN": 0.315,
    "TL": 0.453,
    "TM": 0.403,
    "TK": 0.866,
    "TH": 0.737,
    "TI": 0.455,
    "TF": 0.604,
    "TG": 0.312,
    "TD": 0.83,
    "TE": 0.812,
    "TC": 0.261,
    "TA": 0.251,
    "AA": 0.0,
    "AC": 0.112,
    "AE": 0.827,
    "AD": 0.819,
    "AG": 0.208,
    "AF": 0.54,
    "AI": 0.407,
    "AH": 0.696,
    "AK": 0.891,
    "AM": 0.379,
    "AL": 0.406,
    "AN": 0.318,
    "AQ": 0.372,
    "AP": 0.191,
    "AS": 0.094,
    "AR": 1.0,
    "AT": 0.22,
    "AW": 0.739,
    "AV": 0.273,
    "AY": 0.552,
    "VK": 0.889,
}


#############################################################################################
#############################################################################################
def GetSequenceOrderCouplingNumber(ProteinSequence, d=1, distancematrix=_Distance1):
    """
    ###############################################################################
    Computing the dth-rank sequence order coupling number for a protein.
    Usage:
    result = GetSequenceOrderCouplingNumber(protein,d)
    Input: protein is a pure protein sequence.
    d is the gap between two amino acids.
    Output: result is numeric value.
    ###############################################################################
    """
    NumProtein = len(ProteinSequence)
    tau = 0.0
    for i in range(NumProtein - d):
        temp1 = ProteinSequence[i]
        temp2 = ProteinSequence[i + d]
        tau = tau + math.pow(distancematrix[temp1 + temp2], 2)
    return round(tau, 3)


#############################################################################################
def GetSequenceOrderCouplingNumberp(ProteinSequence, maxlag=30, distancematrix={}):
    """
    ###############################################################################
    Computing the sequence order coupling numbers from 1 to maxlag
    for a given protein sequence based on the user-defined property.
    Usage:
    result = GetSequenceOrderCouplingNumberp(protein, maxlag,distancematrix)
    Input: protein is a pure protein sequence
    maxlag is the maximum lag and the length of the protein should be larger
    than maxlag. default is 30.
    distancematrix is the a dict form containing 400 distance values
    Output: result is a dict form containing all sequence order coupling numbers based
    on the given property
    ###############################################################################
    """
    NumProtein = len(ProteinSequence)
    Tau = {}
    for i in range(maxlag):
        Tau["tau" + str(i + 1)] = GetSequenceOrderCouplingNumber(
            ProteinSequence, i + 1, distancematrix
        )
    return Tau


#############################################################################################
def GetSequenceOrderCouplingNumberSW(
    ProteinSequence, maxlag=30, distancematrix=_Distance1
):
    """
    ###############################################################################
    Computing the sequence order coupling numbers from 1 to maxlag
    for a given protein sequence based on the Schneider-Wrede physicochemical
    distance matrix
    Usage:
    result = GetSequenceOrderCouplingNumberSW(protein, maxlag,distancematrix)
    Input: protein is a pure protein sequence
    maxlag is the maximum lag and the length of the protein should be larger
    than maxlag. default is 30.
    distancematrix is a dict form containing Schneider-Wrede physicochemical
    distance matrix. omitted!
    Output: result is a dict form containing all sequence order coupling numbers based
    on the Schneider-Wrede physicochemical distance matrix
    ###############################################################################
    """
    NumProtein = len(ProteinSequence)
    Tau = {}
    for i in range(maxlag):
        Tau["tausw" + str(i + 1)] = GetSequenceOrderCouplingNumber(
            ProteinSequence, i + 1, distancematrix
        )
    return Tau


#############################################################################################
def GetSequenceOrderCouplingNumberGrant(
    ProteinSequence, maxlag=30, distancematrix=_Distance2
):
    """
    ###############################################################################
    Computing the sequence order coupling numbers from 1 to maxlag
    for a given protein sequence based on the Grantham chemical distance
    matrix.
    Usage:
    result = GetSequenceOrderCouplingNumberGrant(protein, maxlag,distancematrix)
    Input: protein is a pure protein sequence
    maxlag is the maximum lag and the length of the protein should be larger
    than maxlag. default is 30.
    distancematrix is a dict form containing Grantham chemical distance
    matrix. omitted!
    Output: result is a dict form containing all sequence order coupling numbers
    based on the Grantham chemical distance matrix
    ###############################################################################
    """
    NumProtein = len(ProteinSequence)
    Tau = {}
    for i in range(maxlag):
        Tau["taugrant" + str(i + 1)] = GetSequenceOrderCouplingNumber(
            ProteinSequence, i + 1, distancematrix
        )
    return Tau


#############################################################################################
def GetSequenceOrderCouplingNumberTotal(ProteinSequence, maxlag=30):
    """
    ###############################################################################
    Computing the sequence order coupling numbers from 1 to maxlag
    for a given protein sequence.
    Usage:
    result = GetSequenceOrderCouplingNumberTotal(protein, maxlag)
    Input: protein is a pure protein sequence
    maxlag is the maximum lag and the length of the protein should be larger
    than maxlag. default is 30.
    Output: result is a dict form containing all sequence order coupling numbers
    ###############################################################################
    """
    Tau = {}
    Tau.update(GetSequenceOrderCouplingNumberSW(ProteinSequence, maxlag=maxlag))
    Tau.update(GetSequenceOrderCouplingNumberGrant(ProteinSequence, maxlag=maxlag))
    return Tau


#############################################################################################
def GetAAComposition(ProteinSequence):
    """
    ###############################################################################
    Calculate the composition of Amino acids
    for a given protein sequence.
    Usage:
    result=CalculateAAComposition(protein)
    Input: protein is a pure protein sequence.
    Output: result is a dict form containing the composition of
    20 amino acids.
    ###############################################################################
    """
    LengthSequence = len(ProteinSequence)
    Result = {}
    for i in AALetter:
        Result[i] = round(float(ProteinSequence.count(i)) / LengthSequence, 3)
    return Result


#############################################################################################
def GetQuasiSequenceOrder1(ProteinSequence, maxlag=30, weight=0.1, distancematrix={}):
    """
    ###############################################################################
    Computing the first 20 quasi-sequence-order descriptors for
    a given protein sequence.
    Usage:
    result = GetQuasiSequenceOrder1(protein,maxlag,weigt)
    see method GetQuasiSequenceOrder for the choice of parameters.
    ###############################################################################
    """
    rightpart = 0.0
    for i in range(maxlag):
        rightpart = rightpart + GetSequenceOrderCouplingNumber(
            ProteinSequence, i + 1, distancematrix
        )
    AAC = GetAAComposition(ProteinSequence)
    result = {}
    temp = 1 + weight * rightpart
    for index, i in enumerate(AALetter):
        result["QSO" + str(index + 1)] = round(AAC[i] / temp, 6)

    return result


#############################################################################################
def GetQuasiSequenceOrder2(ProteinSequence, maxlag=30, weight=0.1, distancematrix={}):
    """
    ###############################################################################
    Computing the last maxlag quasi-sequence-order descriptors for
    a given protein sequence.
    Usage:
    result = GetQuasiSequenceOrder2(protein,maxlag,weigt)
    see method GetQuasiSequenceOrder for the choice of parameters.
    ###############################################################################
    """
    rightpart = []
    for i in range(maxlag):
        rightpart.append(
            GetSequenceOrderCouplingNumber(ProteinSequence, i + 1, distancematrix)
        )
    AAC = GetAAComposition(ProteinSequence)
    result = {}
    temp = 1 + weight * sum(rightpart)
    for index in range(20, 20 + maxlag):
        result["QSO" + str(index + 1)] = round(weight * rightpart[index - 20] / temp, 6)

    return result


#############################################################################################
def GetQuasiSequenceOrder1SW(
    ProteinSequence, maxlag=30, weight=0.1, distancematrix=_Distance1
):
    """
    ###############################################################################
    Computing the first 20 quasi-sequence-order descriptors for
    a given protein sequence.
    Usage:
    result = GetQuasiSequenceOrder1SW(protein,maxlag,weigt)
    see method GetQuasiSequenceOrder for the choice of parameters.
    ###############################################################################
    """
    rightpart = 0.0
    for i in range(maxlag):
        rightpart = rightpart + GetSequenceOrderCouplingNumber(
            ProteinSequence, i + 1, distancematrix
        )
    AAC = GetAAComposition(ProteinSequence)
    result = {}
    temp = 1 + weight * rightpart
    for index, i in enumerate(AALetter):
        result["QSOSW" + str(index + 1)] = round(AAC[i] / temp, 6)

    return result


#############################################################################################
def GetQuasiSequenceOrder2SW(
    ProteinSequence, maxlag=30, weight=0.1, distancematrix=_Distance1
):
    """
    ###############################################################################
    Computing the last maxlag quasi-sequence-order descriptors for
    a given protein sequence.
    Usage:
    result = GetQuasiSequenceOrder2SW(protein,maxlag,weigt)
    see method GetQuasiSequenceOrder for the choice of parameters.
    ###############################################################################
    """
    rightpart = []
    for i in range(maxlag):
        rightpart.append(
            GetSequenceOrderCouplingNumber(ProteinSequence, i + 1, distancematrix)
        )
    AAC = GetAAComposition(ProteinSequence)
    result = {}
    temp = 1 + weight * sum(rightpart)
    for index in range(20, 20 + maxlag):
        result["QSOSW" + str(index + 1)] = round(
            weight * rightpart[index - 20] / temp, 6
        )

    return result


#############################################################################################
def GetQuasiSequenceOrder1Grant(
    ProteinSequence, maxlag=30, weight=0.1, distancematrix=_Distance2
):
    """
    ###############################################################################
    Computing the first 20 quasi-sequence-order descriptors for
    a given protein sequence.
    Usage:
    result = GetQuasiSequenceOrder1Grant(protein,maxlag,weigt)
    see method GetQuasiSequenceOrder for the choice of parameters.
    ###############################################################################
    """
    rightpart = 0.0
    for i in range(maxlag):
        rightpart = rightpart + GetSequenceOrderCouplingNumber(
            ProteinSequence, i + 1, distancematrix
        )
    AAC = GetAAComposition(ProteinSequence)
    result = {}
    temp = 1 + weight * rightpart
    for index, i in enumerate(AALetter):
        result["QSOgrant" + str(index + 1)] = round(AAC[i] / temp, 6)

    return result


#############################################################################################
def GetQuasiSequenceOrder2Grant(
    ProteinSequence, maxlag=30, weight=0.1, distancematrix=_Distance2
):
    """
    ###############################################################################
    Computing the last maxlag quasi-sequence-order descriptors for
    a given protein sequence.
    Usage:
    result = GetQuasiSequenceOrder2Grant(protein,maxlag,weigt)
    see method GetQuasiSequenceOrder for the choice of parameters.
    ###############################################################################
    """
    rightpart = []
    for i in range(maxlag):
        rightpart.append(
            GetSequenceOrderCouplingNumber(ProteinSequence, i + 1, distancematrix)
        )
    AAC = GetAAComposition(ProteinSequence)
    result = {}
    temp = 1 + weight * sum(rightpart)
    for index in range(20, 20 + maxlag):
        result["QSOgrant" + str(index + 1)] = round(
            weight * rightpart[index - 20] / temp, 6
        )

    return result


#############################################################################################
def GetQuasiSequenceOrder(ProteinSequence, maxlag=30, weight=0.1):
    """
    ###############################################################################
    Computing quasi-sequence-order descriptors for a given protein.
    [1]:Kuo-Chen Chou. Prediction of Protein Subcellar Locations by
    Incorporating Quasi-Sequence-Order Effect. Biochemical and Biophysical
    Research Communications 2000, 278, 477-483.
    Usage:
    result = GetQuasiSequenceOrder(protein,maxlag,weight)
    Input: protein is a pure protein sequence
    maxlag is the maximum lag and the length of the protein should be larger
    than maxlag. default is 30.
    weight is a weight factor.  please see reference 1 for its choice. default is 0.1.
    Output: result is a dict form containing all quasi-sequence-order descriptors
    ###############################################################################
    """
    result = dict()
    result.update(GetQuasiSequenceOrder1SW(ProteinSequence, maxlag, weight, _Distance1))
    result.update(GetQuasiSequenceOrder2SW(ProteinSequence, maxlag, weight, _Distance1))
    result.update(
        GetQuasiSequenceOrder1Grant(ProteinSequence, maxlag, weight, _Distance2)
    )
    result.update(
        GetQuasiSequenceOrder2Grant(ProteinSequence, maxlag, weight, _Distance2)
    )
    return np.array(list(result.values()))



from rdkit import Chem
from rdkit import DataStructs
# these are SMARTS patterns corresponding to the PubChem fingerprints
# https://astro.temple.edu/~tua87106/list_fingerprints.pdf
# ftp://ftp.ncbi.nlm.nih.gov/pubchem/specifications/pubchem_fingerprints.txt

smartsPatts = {
1:('[H]', 3),# 1-115
2:('[H]', 7),
3:('[H]', 15),
4:('[H]', 31),
5:('[Li]', 0),
6:('[Li]', 1),
7:('[B]', 0),
8:('[B]', 1),
9:('[B]', 3),
10:('[C]', 1),
11:('[C]', 3),
12:('[C]', 7),
13:('[C]', 15),
14:('[C]', 31),
15:('[N]', 0),
16:('[N]', 1),
17:('[N]', 3),
18:('[N]', 7),
19:('[O]', 0),
20:('[O]', 1),
21:('[O]', 3),
22:('[O]', 7),
23:('[O]', 15),
24:('[F]', 0),
25:('[F]', 1),
26:('[F]', 3),
27:('[Na]', 0),
28:('[Na]', 1),
29:('[Si]', 0),
30:('[Si]', 1),
31:('[P]', 0),
32:('[P]', 1),
33:('[P]', 3),
34:('[S]', 0),
35:('[S]', 1),
36:('[S]', 3),
37:('[S]', 7),
38:('[Cl]', 0),
39:('[Cl]', 1),
40:('[Cl]', 3),
41:('[Cl]', 7),
42:('[K]', 0),
43:('[K]', 1),
44:('[Br]', 0),
45:('[Br]', 1),
46:('[Br]', 3),
47:('[I]', 0),
48:('[I]', 1),
49:('[I]', 3),
50:('[Be]', 0),
51:('[Mg]', 0),
52:('[Al]', 0),
53:('[Ca]', 0),
54:('[Sc]', 0),
55:('[Ti]', 0),
56:('[V]', 0),
57:('[Cr]', 0),
58:('[Mn]', 0),
59:('[Fe]', 0),
60:('[CO]', 0),
61:('[Ni]', 0),
62:('[Cu]', 0),
63:('[Zn]', 0),
64:('[Ga]', 0),
65:('[Ge]', 0),
66:('[As]', 0),
67:('[Se]', 0),
68:('[Kr]', 0),
69:('[Rb]', 0),
70:('[Sr]', 0),
71:('[Y]', 0),
72:('[Zr]', 0),
73:('[Nb]', 0),
74:('[Mo]', 0),
75:('[Ru]', 0),
76:('[Rh]', 0),
77:('[Pd]', 0),
78:('[Ag]', 0),
79:('[Cd]', 0),
80:('[In]', 0),
81:('[Sn]', 0),
82:('[Sb]', 0),
83:('[Te]', 0),
84:('[Xe]', 0),
85:('[Cs]', 0),
86:('[Ba]', 0),
87:('[Lu]', 0),
88:('[Hf]', 0),
89:('[Ta]', 0),
90:('[W]', 0),
91:('[Re]', 0),
92:('[Os]', 0),
93:('[Ir]', 0),
94:('[Pt]', 0),
95:('[Au]', 0),
96:('[Hg]', 0),
97:('[Tl]', 0),
98:('[Pb]', 0),
99:('[Bi]', 0),
100:('[La]', 0),
101:('[Ce]', 0),
102:('[Pr]', 0),
103:('[Nd]', 0),
104:('[Pm]', 0),
105:('[Sm]', 0),
106:('[Eu]', 0),
107:('[Gd]', 0),
108:('[Tb]', 0),
109:('[Dy]', 0),
110:('[Ho]', 0),
111:('[Er]', 0),
112:('[Tm]', 0),
113:('[Yb]', 0),
114:('[Tc]', 0),
115:('[U]', 0),
116:('[Li&!H0]', 0),#264-881
117:('[Li]~[Li]', 0),
118:('[Li]~[#5]', 0),
119:('[Li]~[#6]', 0),
120:('[Li]~[#8]', 0),
121:('[Li]~[F]', 0),
122:('[Li]~[#15]', 0),
123:('[Li]~[#16]', 0),
124:('[Li]~[Cl]', 0),
125:('[#5&!H0]', 0),
126:('[#5]~[#5]', 0),
127:('[#5]~[#6]', 0),
128:('[#5]~[#7]', 0),
129:('[#5]~[#8]', 0),
130:('[#5]~[F]', 0),
131:('[#5]~[#14]', 0),
132:('[#5]~[#15]', 0),
133:('[#5]~[#16]', 0),
134:('[#5]~[Cl]', 0),
135:('[#5]~[Br]', 0),
136:('[#6&!H0]', 0),
137:('[#6]~[#6]', 0),
138:('[#6]~[#7]', 0),
139:('[#6]~[#8]', 0),
140:('[#6]~[F]', 0),
141:('[#6]~[Na]', 0),
142:('[#6]~[Mg]', 0),
143:('[#6]~[Al]', 0),
144:('[#6]~[#14]', 0),
145:('[#6]~[#15]', 0),
146:('[#6]~[#16]', 0),
147:('[#6]~[Cl]', 0),
148:('[#6]~[#33]', 0),
149:('[#6]~[#34]', 0),
150:('[#6]~[Br]', 0),
151:('[#6]~[I]', 0),
152:('[#7&!H0]', 0),
153:('[#7]~[#7]', 0),
154:('[#7]~[#8]', 0),
155:('[#7]~[F]', 0),
156:('[#7]~[#14]', 0),
157:('[#7]~[#15]', 0),
158:('[#7]~[#16]', 0),
159:('[#7]~[Cl]', 0),
160:('[#7]~[Br]', 0),
161:('[#8&!H0]', 0),
162:('[#8]~[#8]', 0),
163:('[#8]~[Mg]', 0),
164:('[#8]~[Na]', 0),
165:('[#8]~[Al]', 0),
166:('[#8]~[#14]', 0),
167:('[#8]~[#15]', 0),
168:('[#8]~[K]', 0),
169:('[F]~[#15]', 0),
170:('[F]~[#16]', 0),
171:('[Al&!H0]', 0),
172:('[Al]~[Cl]', 0),
173:('[#14&!H0]', 0),
174:('[#14]~[#14]', 0),
175:('[#14]~[Cl]', 0),
176:('[#15&!H0]', 0),
177:('[#15]~[#15]', 0),
178:('[#33&!H0]', 0),
179:('[#33]~[#33]', 0),
180:('[#6](~Br)(~[#6])', 0),
181:('[#6](~Br)(~[#6])(~[#6])', 0),
182:('[#6&!H0]~[Br]', 0),
183:('[#6](~[Br])(:[c])', 0),
184:('[#6](~[Br])(:[n])', 0),
185:('[#6](~[#6])(~[#6])', 0),
186:('[#6](~[#6])(~[#6])(~[#6])', 0),
187:('[#6](~[#6])(~[#6])(~[#6])(~[#6])', 0),
188:('[#6H1](~[#6])(~[#6])(~[#6])', 0),
189:('[#6](~[#6])(~[#6])(~[#6])(~[#7])', 0),
190:('[#6](~[#6])(~[#6])(~[#6])(~[#8])', 0),
191:('[#6H1](~[#6])(~[#6])(~[#7])', 0),
192:('[#6H1](~[#6])(~[#6])(~[#8])', 0),
193:('[#6](~[#6])(~[#6])(~[#7])', 0),
194:('[#6](~[#6])(~[#6])(~[#8])', 0),
195:('[#6](~[#6])(~[Cl])', 0),
196:('[#6&!H0](~[#6])(~[Cl])', 0),
197:('[#6H,#6H2,#6H3,#6H4]~[#6]', 0),
198:('[#6&!H0](~[#6])(~[#7])', 0),
199:('[#6&!H0](~[#6])(~[#8])', 0),
200:('[#6H1](~[#6])(~[#8])(~[#8])', 0),
201:('[#6&!H0](~[#6])(~[#15])', 0),
202:('[#6&!H0](~[#6])(~[#16])', 0),
203:('[#6](~[#6])(~[I])', 0),
204:('[#6](~[#6])(~[#7])', 0),
205:('[#6](~[#6])(~[#8])', 0),
206:('[#6](~[#6])(~[#16])', 0),
207:('[#6](~[#6])(~[#14])', 0),
208:('[#6](~[#6])(:c)', 0),
209:('[#6](~[#6])(:c)(:c)', 0),
210:('[#6](~[#6])(:c)(:n)', 0),
211:('[#6](~[#6])(:n)', 0),
212:('[#6](~[#6])(:n)(:n)', 0),
213:('[#6](~[Cl])(~[Cl])', 0),
214:('[#6&!H0](~[Cl])', 0),
215:('[#6](~[Cl])(:c)', 0),
216:('[#6](~[F])(~[F])', 0),
217:('[#6](~[F])(:c)', 0),
218:('[#6&!H0](~[#7])', 0),
219:('[#6&!H0](~[#8])', 0),
220:('[#6&!H0](~[#8])(~[#8])', 0),
221:('[#6&!H0](~[#16])', 0),
222:('[#6&!H0](~[#14])', 0),
223:('[#6&!H0]:c', 0),
224:('[#6&!H0](:c)(:c)', 0),
225:('[#6&!H0](:c)(:n)', 0),
226:('[#6&!H0](:n)', 0),
227:('[#6H3]', 0),
228:('[#6](~[#7])(~[#7])', 0),
229:('[#6](~[#7])(:c)', 0),
230:('[#6](~[#7])(:c)(:c)', 0),
231:('[#6](~[#7])(:c)(:n)', 0),
232:('[#6](~[#7])(:n)', 0),
233:('[#6](~[#8])(~[#8])', 0),
234:('[#6](~[#8])(:c)', 0),
235:('[#6](~[#8])(:c)(:c)', 0),
236:('[#6](~[#16])(:c)', 0),
237:('[#6](:c)(:c)', 0),
238:('[#6](:c)(:c)(:c)', 0),
239:('[#6](:c)(:c)(:n)', 0),
240:('[#6](:c)(:n)', 0),
241:('[#6](:c)(:n)(:n)', 0),
242:('[#6](:n)(:n)', 0),
243:('[#7](~[#6])(~[#6])', 0),
244:('[#7](~[#6])(~[#6])(~[#6])', 0),
245:('[#7&!H0](~[#6])(~[#6])', 0),
246:('[#7&!H0](~[#6])', 0),
247:('[#7&!H0](~[#6])(~[#7])', 0),
248:('[#7](~[#6])(~[#8])', 0),
249:('[#7](~[#6])(:c)', 0),
250:('[#7](~[#6])(:c)(:c)', 0),
251:('[#7&!H0](~[#7])', 0),
252:('[#7&!H0](:c)', 0),
253:('[#7&!H0](:c)(:c)', 0),
254:('[#7](~[#8])(~[#8])', 0),
255:('[#7](~[#8])(:o)', 0),
256:('[#7](:c)(:c)', 0),
257:('[#7](:c)(:c)(:c)', 0),
258:('[#8](~[#6])(~[#6])', 0),
259:('[#8&!H0](~[#6])', 0),
260:('[#8](~[#6])(~[#15])', 0),
261:('[#8&!H0](~[#16])', 0),
262:('[#8](:c)(:c)', 0),
263:('[#15](~[#6])(~[#6])', 0),
264:('[#15](~[#8])(~[#8])', 0),
265:('[#16](~[#6])(~[#6])', 0),
266:('[#16&!H0](~[#6])', 0),
267:('[#16](~[#6])(~[#8])', 0),
268:('[#14](~[#6])(~[#6])', 0),
269:('[#6]=,:[#6]', 0),
270:('[#6]#[#6]', 0),
271:('[#6]=,:[#7]', 0),
272:('[#6]#[#7]', 0),
273:('[#6]=,:[#8]', 0),
274:('[#6]=,:[#16]', 0),
275:('[#7]=,:[#7]', 0),
276:('[#7]=,:[#8]', 0),
277:('[#7]=,:[#15]', 0),
278:('[#15]=,:[#8]', 0),
279:('[#15]=,:[#15]', 0),
280:('[#6](#[#6])(-,:[#6])', 0),
281:('[#6&!H0](#[#6])', 0),
282:('[#6](#[#7])(-,:[#6])', 0),
283:('[#6](-,:[#6])(-,:[#6])(=,:[#6])', 0),
284:('[#6](-,:[#6])(-,:[#6])(=,:[#7])', 0),
285:('[#6](-,:[#6])(-,:[#6])(=,:[#8])', 0),
286:('[#6](-,:[#6])([Cl])(=,:[#8])', 0),
287:('[#6&!H0](-,:[#6])(=,:[#6])', 0),
288:('[#6&!H0](-,:[#6])(=,:[#7])', 0),
289:('[#6&!H0](-,:[#6])(=,:[#8])', 0),
290:('[#6](-,:[#6])(-,:[#7])(=,:[#6])', 0),
291:('[#6](-,:[#6])(-,:[#7])(=,:[#7])', 0),
292:('[#6](-,:[#6])(-,:[#7])(=,:[#8])', 0),
293:('[#6](-,:[#6])(-,:[#8])(=,:[#8])', 0),
294:('[#6](-,:[#6])(=,:[#6])', 0),
295:('[#6](-,:[#6])(=,:[#7])', 0),
296:('[#6](-,:[#6])(=,:[#8])', 0),
297:('[#6]([Cl])(=,:[#8])', 0),
298:('[#6&!H0](-,:[#7])(=,:[#6])', 0),
299:('[#6&!H0](=,:[#6])', 0),
300:('[#6&!H0](=,:[#7])', 0),
301:('[#6&!H0](=,:[#8])', 0),
302:('[#6](-,:[#7])(=,:[#6])', 0),
303:('[#6](-,:[#7])(=,:[#7])', 0),
304:('[#6](-,:[#7])(=,:[#8])', 0),
305:('[#6](-,:[#8])(=,:[#8])', 0),
306:('[#7](-,:[#6])(=,:[#6])', 0),
307:('[#7](-,:[#6])(=,:[#8])', 0),
308:('[#7](-,:[#8])(=,:[#8])', 0),
309:('[#15](-,:[#8])(=,:[#8])', 0),
310:('[#16](-,:[#6])(=,:[#8])', 0),
311:('[#16](-,:[#8])(=,:[#8])', 0),
312:('[#16](=,:[#8])(=,:[#8])', 0),
313:('[#6]-,:[#6]-,:[#6]#[#6]', 0),
314:('[#8]-,:[#6]-,:[#6]=,:[#7]', 0),
315:('[#8]-,:[#6]-,:[#6]=,:[#8]', 0),
316:('[#7]:[#6]-,:[#16&!H0]', 0),
317:('[#7]-,:[#6]-,:[#6]=,:[#6]', 0),
318:('[#8]=,:[#16]-,:[#6]-,:[#6]', 0),
319:('[#7]#[#6]-,:[#6]=,:[#6]', 0),
320:('[#6]=,:[#7]-,:[#7]-,:[#6]', 0),
321:('[#8]=,:[#16]-,:[#6]-,:[#7]', 0),
322:('[#16]-,:[#16]-,:[#6]:[#6]', 0),
323:('[#6]:[#6]-,:[#6]=,:[#6]', 0),
324:('[#16]:[#6]:[#6]:[#6]', 0),
325:('[#6]:[#7]:[#6]-,:[#6]', 0),
326:('[#16]-,:[#6]:[#7]:[#6]', 0),
327:('[#16]:[#6]:[#6]:[#7]', 0),
328:('[#16]-,:[#6]=,:[#7]-,:[#6]', 0),
329:('[#6]-,:[#8]-,:[#6]=,:[#6]', 0),
330:('[#7]-,:[#7]-,:[#6]:[#6]', 0),
331:('[#16]-,:[#6]=,:[#7&!H0]', 0),
332:('[#16]-,:[#6]-,:[#16]-,:[#6]', 0),
333:('[#6]:[#16]:[#6]-,:[#6]', 0),
334:('[#8]-,:[#16]-,:[#6]:[#6]', 0),
335:('[#6]:[#7]-,:[#6]:[#6]', 0),
336:('[#7]-,:[#16]-,:[#6]:[#6]', 0),
337:('[#7]-,:[#6]:[#7]:[#6]', 0),
338:('[#7]:[#6]:[#6]:[#7]', 0),
339:('[#7]-,:[#6]:[#7]:[#7]', 0),
340:('[#7]-,:[#6]=,:[#7]-,:[#6]', 0),
341:('[#7]-,:[#6]=,:[#7&!H0]', 0),
342:('[#7]-,:[#6]-,:[#16]-,:[#6]', 0),
343:('[#6]-,:[#6]-,:[#6]=,:[#6]', 0),
344:('[#6]-,:[#7]:[#6&!H0]', 0),
345:('[#7]-,:[#6]:[#8]:[#6]', 0),
346:('[#8]=,:[#6]-,:[#6]:[#6]', 0),
347:('[#8]=,:[#6]-,:[#6]:[#7]', 0),
348:('[#6]-,:[#7]-,:[#6]:[#6]', 0),
349:('[#7]:[#7]-,:[#6&!H0]', 0),
350:('[#8]-,:[#6]:[#6]:[#7]', 0),
351:('[#8]-,:[#6]=,:[#6]-,:[#6]', 0),
352:('[#7]-,:[#6]:[#6]:[#7]', 0),
353:('[#6]-,:[#16]-,:[#6]:[#6]', 0),
354:('[Cl]-,:[#6]:[#6]-,:[#6]', 0),
355:('[#7]-,:[#6]=,:[#6&!H0]', 0),
356:('[Cl]-,:[#6]:[#6&!H0]', 0),
357:('[#7]:[#6]:[#7]-,:[#6]', 0),
358:('[Cl]-,:[#6]:[#6]-,:[#8]', 0),
359:('[#6]-,:[#6]:[#7]:[#6]', 0),
360:('[#6]-,:[#6]-,:[#16]-,:[#6]', 0),
361:('[#16]=,:[#6]-,:[#7]-,:[#6]', 0),
362:('[Br]-,:[#6]:[#6]-,:[#6]', 0),
363:('[#7&!H0]-,:[#7&!H0]', 0),
364:('[#16]=,:[#6]-,:[#7&!H0]', 0),
365:('[#6]-,:[#33]-[#8&!H0]', 0),
366:('[#16]:[#6]:[#6&!H0]', 0),
367:('[#8]-,:[#7]-,:[#6]-,:[#6]', 0),
368:('[#7]-,:[#7]-,:[#6]-,:[#6]', 0),
369:('[#6H,#6H2,#6H3]=,:[#6H,#6H2,#6H3]', 0),
370:('[#7]-,:[#7]-,:[#6]-,:[#7]', 0),
371:('[#8]=,:[#6]-,:[#7]-,:[#7]', 0),
372:('[#7]=,:[#6]-,:[#7]-,:[#6]', 0),
373:('[#6]=,:[#6]-,:[#6]:[#6]', 0),
374:('[#6]:[#7]-,:[#6&!H0]', 0),
375:('[#6]-,:[#7]-,:[#7&!H0]', 0),
376:('[#7]:[#6]:[#6]-,:[#6]', 0),
377:('[#6]-,:[#6]=,:[#6]-,:[#6]', 0),
378:('[#33]-,:[#6]:[#6&!H0]', 0),
379:('[Cl]-,:[#6]:[#6]-,:[Cl]', 0),
380:('[#6]:[#6]:[#7&!H0]', 0),
381:('[#7&!H0]-,:[#6&!H0]', 0),
382:('[Cl]-,:[#6]-,:[#6]-,:[Cl]', 0),
383:('[#7]:[#6]-,:[#6]:[#6]', 0),
384:('[#16]-,:[#6]:[#6]-,:[#6]', 0),
385:('[#16]-,:[#6]:[#6&!H0]', 0),
386:('[#16]-,:[#6]:[#6]-,:[#7]', 0),
387:('[#16]-,:[#6]:[#6]-,:[#8]', 0),
388:('[#8]=,:[#6]-,:[#6]-,:[#6]', 0),
389:('[#8]=,:[#6]-,:[#6]-,:[#7]', 0),
390:('[#8]=,:[#6]-,:[#6]-,:[#8]', 0),
391:('[#7]=,:[#6]-,:[#6]-,:[#6]', 0),
392:('[#7]=,:[#6]-,:[#6&!H0]', 0),
393:('[#6]-,:[#7]-,:[#6&!H0]', 0),
394:('[#8]-,:[#6]:[#6]-,:[#6]', 0),
395:('[#8]-,:[#6]:[#6&!H0]', 0),
396:('[#8]-,:[#6]:[#6]-,:[#7]', 0),
397:('[#8]-,:[#6]:[#6]-,:[#8]', 0),
398:('[#7]-,:[#6]:[#6]-,:[#6]', 0),
399:('[#7]-,:[#6]:[#6&!H0]', 0),
400:('[#7]-,:[#6]:[#6]-,:[#7]', 0),
401:('[#8]-,:[#6]-,:[#6]:[#6]', 0),
402:('[#7]-,:[#6]-,:[#6]:[#6]', 0),
403:('[Cl]-,:[#6]-,:[#6]-,:[#6]', 0),
404:('[Cl]-,:[#6]-,:[#6]-,:[#8]', 0),
405:('[#6]:[#6]-,:[#6]:[#6]', 0),
406:('[#8]=,:[#6]-,:[#6]=,:[#6]', 0),
407:('[Br]-,:[#6]-,:[#6]-,:[#6]', 0),
408:('[#7]=,:[#6]-,:[#6]=,:[#6]', 0),
409:('[#6]=,:[#6]-,:[#6]-,:[#6]', 0),
410:('[#7]:[#6]-,:[#8&!H0]', 0),
411:('[#8]=,:[#7]-,:c:c', 0),
412:('[#8]-,:[#6]-,:[#7&!H0]', 0),
413:('[#7]-,:[#6]-,:[#7]-,:[#6]', 0),
414:('[Cl]-,:[#6]-,:[#6]=,:[#8]', 0),
415:('[Br]-,:[#6]-,:[#6]=,:[#8]', 0),
416:('[#8]-,:[#6]-,:[#8]-,:[#6]', 0),
417:('[#6]=,:[#6]-,:[#6]=,:[#6]', 0),
418:('[#6]:[#6]-,:[#8]-,:[#6]', 0),
419:('[#8]-,:[#6]-,:[#6]-,:[#7]', 0),
420:('[#8]-,:[#6]-,:[#6]-,:[#8]', 0),
421:('N#[#6]-,:[#6]-,:[#6]', 0),
422:('[#7]-,:[#6]-,:[#6]-,:[#7]', 0),
423:('[#6]:[#6]-,:[#6]-,:[#6]', 0),
424:('[#6&!H0]-,:[#8&!H0]', 0),
425:('n:c:n:c', 0),
426:('[#8]-,:[#6]-,:[#6]=,:[#6]', 0),
427:('[#8]-,:[#6]-,:[#6]:[#6]-,:[#6]', 0),
428:('[#8]-,:[#6]-,:[#6]:[#6]-,:[#8]', 0),
429:('[#7]=,:[#6]-,:[#6]:[#6&!H0]', 0),
430:('c:c-,:[#7]-,:c:c', 0),
431:('[#6]-,:[#6]:[#6]-,:c:c', 0),
432:('[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
433:('[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#7]', 0),
434:('[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#8]', 0),
435:('[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
436:('[Cl]-,:[#6]:[#6]-,:[#8]-,:[#6]', 0),
437:('c:c-,:[#6]=,:[#6]-,:[#6]', 0),
438:('[#6]-,:[#6]:[#6]-,:[#7]-,:[#6]', 0),
439:('[#6]-,:[#16]-,:[#6]-,:[#6]-,:[#6]', 0),
440:('[#7]-,:[#6]:[#6]-,:[#8&!H0]', 0),
441:('[#8]=,:[#6]-,:[#6]-,:[#6]=,:[#8]', 0),
442:('[#6]-,:[#6]:[#6]-,:[#8]-,:[#6]', 0),
443:('[#6]-,:[#6]:[#6]-,:[#8&!H0]', 0),
444:('[Cl]-,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
445:('[#7]-,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
446:('[#7]-,:[#6]-,:[#6]-,:[#6]-,:[#7]', 0),
447:('[#6]-,:[#8]-,:[#6]-,:[#6]=,:[#6]', 0),
448:('c:c-,:[#6]-,:[#6]-,:[#6]', 0),
449:('[#7]=,:[#6]-,:[#7]-,:[#6]-,:[#6]', 0),
450:('[#8]=,:[#6]-,:[#6]-,:c:c', 0),
451:('[Cl]-,:[#6]:[#6]:[#6]-,:[#6]', 0),
452:('[#6H,#6H2,#6H3]-,:[#6]=,:[#6H,#6H2,#6H3]', 0),
453:('[#7]-,:[#6]:[#6]:[#6]-,:[#6]', 0),
454:('[#7]-,:[#6]:[#6]:[#6]-,:[#7]', 0),
455:('[#8]=,:[#6]-,:[#6]-,:[#7]-,:[#6]', 0),
456:('[#6]-,:c:c:[#6]-,:[#6]', 0),
457:('[#6]-,:[#8]-,:[#6]-,:[#6]:c', 0),
458:('[#8]=,:[#6]-,:[#6]-,:[#8]-,:[#6]', 0),
459:('[#8]-,:[#6]:[#6]-,:[#6]-,:[#6]', 0),
460:('[#7]-,:[#6]-,:[#6]-,:[#6]:c', 0),
461:('[#6]-,:[#6]-,:[#6]-,:[#6]:c', 0),
462:('[Cl]-,:[#6]-,:[#6]-,:[#7]-,:[#6]', 0),
463:('[#6]-,:[#8]-,:[#6]-,:[#8]-,:[#6]', 0),
464:('[#7]-,:[#6]-,:[#6]-,:[#7]-,:[#6]', 0),
465:('[#7]-,:[#6]-,:[#8]-,:[#6]-,:[#6]', 0),
466:('[#6]-,:[#7]-,:[#6]-,:[#6]-,:[#6]', 0),
467:('[#6]-,:[#6]-,:[#8]-,:[#6]-,:[#6]', 0),
468:('[#7]-,:[#6]-,:[#6]-,:[#8]-,:[#6]', 0),
469:('c:c:n:n:c', 0),
470:('[#6]-,:[#6]-,:[#6]-,:[#8&!H0]', 0),
471:('c:[#6]-,:[#6]-,:[#6]:c', 0),
472:('[#8]-,:[#6]-,:[#6]=,:[#6]-,:[#6]', 0),
473:('c:c-,:[#8]-,:[#6]-,:[#6]', 0),
474:('[#7]-,:[#6]:c:c:n', 0),
475:('[#8]=,:[#6]-,:[#8]-,:[#6]:c', 0),
476:('[#8]=,:[#6]-,:[#6]:[#6]-,:[#6]', 0),
477:('[#8]=,:[#6]-,:[#6]:[#6]-,:[#7]', 0),
478:('[#8]=,:[#6]-,:[#6]:[#6]-,:[#8]', 0),
479:('[#6]-,:[#8]-,:[#6]:[#6]-,:[#6]', 0),
480:('[#8]=,:[#33]-,:[#6]:c:c', 0),
481:('[#6]-,:[#7]-,:[#6]-,:[#6]:c', 0),
482:('[#16]-,:[#6]:c:c-,:[#7]', 0),
483:('[#8]-,:[#6]:[#6]-,:[#8]-,:[#6]', 0),
484:('[#8]-,:[#6]:[#6]-,:[#8&!H0]', 0),
485:('[#6]-,:[#6]-,:[#8]-,:[#6]:c', 0),
486:('[#7]-,:[#6]-,:[#6]:[#6]-,:[#6]', 0),
487:('[#6]-,:[#6]-,:[#6]:[#6]-,:[#6]', 0),
488:('[#7]-,:[#7]-,:[#6]-,:[#7&!H0]', 0),
489:('[#6]-,:[#7]-,:[#6]-,:[#7]-,:[#6]', 0),
490:('[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
491:('[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#7]', 0),
492:('[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#8]', 0),
493:('[#6]=,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
494:('[#8]-,:[#6]-,:[#6]-,:[#6]=,:[#6]', 0),
495:('[#8]-,:[#6]-,:[#6]-,:[#6]=,:[#8]', 0),
496:('[#6&!H0]-,:[#6]-,:[#7&!H0]', 0),
497:('[#6]-,:[#6]=,:[#7]-,:[#7]-,:[#6]', 0),
498:('[#8]=,:[#6]-,:[#7]-,:[#6]-,:[#6]', 0),
499:('[#8]=,:[#6]-,:[#7]-,:[#6&!H0]', 0),
500:('[#8]=,:[#6]-,:[#7]-,:[#6]-,:[#7]', 0),
501:('[#8]=,:[#7]-,:[#6]:[#6]-,:[#7]', 0),
502:('[#8]=,:[#7]-,:c:c-,:[#8]', 0),
503:('[#8]=,:[#6]-,:[#7]-,:[#6]=,:[#8]', 0),
504:('[#8]-,:[#6]:[#6]:[#6]-,:[#6]', 0),
505:('[#8]-,:[#6]:[#6]:[#6]-,:[#7]', 0),
506:('[#8]-,:[#6]:[#6]:[#6]-,:[#8]', 0),
507:('[#7]-,:[#6]-,:[#7]-,:[#6]-,:[#6]', 0),
508:('[#8]-,:[#6]-,:[#6]-,:[#6]:c', 0),
509:('[#6]-,:[#6]-,:[#7]-,:[#6]-,:[#6]', 0),
510:('[#6]-,:[#7]-,:[#6]:[#6]-,:[#6]', 0),
511:('[#6]-,:[#6]-,:[#16]-,:[#6]-,:[#6]', 0),
512:('[#8]-,:[#6]-,:[#6]-,:[#7]-,:[#6]', 0),
513:('[#6]-,:[#6]=,:[#6]-,:[#6]-,:[#6]', 0),
514:('[#8]-,:[#6]-,:[#8]-,:[#6]-,:[#6]', 0),
515:('[#8]-,:[#6]-,:[#6]-,:[#8]-,:[#6]', 0),
516:('[#8]-,:[#6]-,:[#6]-,:[#8&!H0]', 0),
517:('[#6]-,:[#6]=,:[#6]-,:[#6]=,:[#6]', 0),
518:('[#7]-,:[#6]:[#6]-,:[#6]-,:[#6]', 0),
519:('[#6]=,:[#6]-,:[#6]-,:[#8]-,:[#6]', 0),
520:('[#6]=,:[#6]-,:[#6]-,:[#8&!H0]', 0),
521:('[#6]-,:[#6]:[#6]-,:[#6]-,:[#6]', 0),
522:('[Cl]-,:[#6]:[#6]-,:[#6]=,:[#8]', 0),
523:('[Br]-,:[#6]:c:c-,:[#6]', 0),
524:('[#8]=,:[#6]-,:[#6]=,:[#6]-,:[#6]', 0),
525:('[#8]=,:[#6]-,:[#6]=,:[#6&!H0]', 0),
526:('[#8]=,:[#6]-,:[#6]=,:[#6]-,:[#7]', 0),
527:('[#7]-,:[#6]-,:[#7]-,:[#6]:c', 0),
528:('[Br]-,:[#6]-,:[#6]-,:[#6]:c', 0),
529:('[#7]#[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
530:('[#6]-,:[#6]=,:[#6]-,:[#6]:c', 0),
531:('[#6]-,:[#6]-,:[#6]=,:[#6]-,:[#6]', 0),
532:('[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
533:('[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
534:('[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#8]', 0),
535:('[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#7]', 0),
536:('[#7]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
537:('[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
538:('[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#7]', 0),
539:('[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#8]', 0),
540:('[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]=,:[#8]', 0),
541:('[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
542:('[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
543:('[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#8]', 0),
544:('[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#7]', 0),
545:('[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
546:('[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#8]', 0),
547:('[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]=,:[#8]', 0),
548:('[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#7]', 0),
549:('[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
550:('[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6](-,:[#6])-,:[#6]', 0),
551:('[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
552:('[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6](-,:[#6])-,:[#6]', 0),
553:('[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#8]-,:[#6]', 0),
554:('[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6](-,:[#8])-,:[#6]', 0),
555:('[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#7]-,:[#6]', 0),
556:('[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6](-,:[#7])-,:[#6]', 0),
557:('[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
558:('[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6](-,:[#8])-,:[#6]', 0),
559:('[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6](=,:[#8])-,:[#6]', 0),
560:('[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6](-,:[#7])-,:[#6]', 0),
561:('[#6]-,:[#6](-,:[#6])-,:[#6]-,:[#6]', 0),
562:('[#6]-,:[#6](-,:[#6])-,:[#6]-,:[#6]-,:[#6]', 0),
563:('[#6]-,:[#6]-,:[#6](-,:[#6])-,:[#6]-,:[#6]', 0),
564:('[#6]-,:[#6](-,:[#6])(-,:[#6])-,:[#6]-,:[#6]', 0),
565:('[#6]-,:[#6](-,:[#6])-,:[#6](-,:[#6])-,:[#6]', 0),
566:('[#6]c1ccc([#6])cc1', 0),
567:('[#6]c1ccc([#8])cc1', 0),
568:('[#6]c1ccc([#16])cc1', 0),
569:('[#6]c1ccc([#7])cc1', 0),
570:('[#6]c1ccc(Cl)cc1', 0),
571:('[#6]c1ccc(Br)cc1', 0),
572:('[#8]c1ccc([#8])cc1', 0),
573:('[#8]c1ccc([#16])cc1', 0),
574:('[#8]c1ccc([#7])cc1', 0),
575:('[#8]c1ccc(Cl)cc1', 0),
576:('[#8]c1ccc(Br)cc1', 0),
577:('[#16]c1ccc([#16])cc1', 0),
578:('[#16]c1ccc([#7])cc1', 0),
579:('[#16]c1ccc(Cl)cc1', 0),
580:('[#16]c1ccc(Br)cc1', 0),
581:('[#7]c1ccc([#7])cc1', 0),
582:('[#7]c1ccc(Cl)cc1', 0),
583:('[#7]c1ccc(Br)cc1', 0),
584:('Clc1ccc(Cl)cc1', 0),
585:('Clc1ccc(Br)cc1', 0),
586:('Brc1ccc(Br)cc1', 0),
587:('[#6]c1cc([#6])ccc1', 0),
588:('[#6]c1cc([#8])ccc1', 0),
589:('[#6]c1cc([#16])ccc1', 0),
590:('[#6]c1cc([#7])ccc1', 0),
591:('[#6]c1cc(Cl)ccc1', 0),
592:('[#6]c1cc(Br)ccc1', 0),
593:('[#8]c1cc([#8])ccc1', 0),
594:('[#8]c1cc([#16])ccc1', 0),
595:('[#8]c1cc([#7])ccc1', 0),
596:('[#8]c1cc(Cl)ccc1', 0),
597:('[#8]c1cc(Br)ccc1', 0),
598:('[#16]c1cc([#16])ccc1', 0),
599:('[#16]c1cc([#7])ccc1', 0),
600:('[#16]c1cc(Cl)ccc1', 0),
601:('[#16]c1cc(Br)ccc1', 0),
602:('[#7]c1cc([#7])ccc1', 0),
603:('[#7]c1cc(Cl)ccc1', 0),
604:('[#7]c1cc(Br)ccc1', 0),
605:('Clc1cc(Cl)ccc1', 0),
606:('Clc1cc(Br)ccc1', 0),
607:('Brc1cc(Br)ccc1', 0),
608:('[#6]c1c([#6])cccc1', 0),
609:('[#6]c1c([#8])cccc1', 0),
610:('[#6]c1c([#16])cccc1', 0),
611:('[#6]c1c([#7])cccc1', 0),
612:('[#6]c1c(Cl)cccc1', 0),
613:('[#6]c1c(Br)cccc1', 0),
614:('[#8]c1c([#8])cccc1', 0),
615:('[#8]c1c([#16])cccc1', 0),
616:('[#8]c1c([#7])cccc1', 0),
617:('[#8]c1c(Cl)cccc1', 0),
618:('[#8]c1c(Br)cccc1', 0),
619:('[#16]c1c([#16])cccc1', 0),
620:('[#16]c1c([#7])cccc1', 0),
621:('[#16]c1c(Cl)cccc1', 0),
622:('[#16]c1c(Br)cccc1', 0),
623:('[#7]c1c([#7])cccc1', 0),
624:('[#7]c1c(Cl)cccc1', 0),
625:('[#7]c1c(Br)cccc1', 0),
626:('Clc1c(Cl)cccc1', 0),
627:('Clc1c(Br)cccc1', 0),
628:('Brc1c(Br)cccc1', 0),
629:('[#6][#6]1[#6][#6][#6]([#6])[#6][#6]1', 0),
630:('[#6][#6]1[#6][#6][#6]([#8])[#6][#6]1', 0),
631:('[#6][#6]1[#6][#6][#6]([#16])[#6][#6]1', 0),
632:('[#6][#6]1[#6][#6][#6]([#7])[#6][#6]1', 0),
633:('[#6][#6]1[#6][#6][#6](Cl)[#6][#6]1', 0),
634:('[#6][#6]1[#6][#6][#6](Br)[#6][#6]1', 0),
635:('[#8][#6]1[#6][#6][#6]([#8])[#6][#6]1', 0),
636:('[#8][#6]1[#6][#6][#6]([#16])[#6][#6]1', 0),
637:('[#8][#6]1[#6][#6][#6]([#7])[#6][#6]1', 0),
638:('[#8][#6]1[#6][#6][#6](Cl)[#6][#6]1', 0),
639:('[#8][#6]1[#6][#6][#6](Br)[#6][#6]1', 0),
640:('[#16][#6]1[#6][#6][#6]([#16])[#6][#6]1', 0),
641:('[#16][#6]1[#6][#6][#6]([#7])[#6][#6]1', 0),
642:('[#16][#6]1[#6][#6][#6](Cl)[#6][#6]1', 0),
643:('[#16][#6]1[#6][#6][#6](Br)[#6][#6]1', 0),
644:('[#7][#6]1[#6][#6][#6]([#7])[#6][#6]1', 0),
645:('[#7][#6]1[#6][#6][#6](Cl)[#6][#6]1', 0),
646:('[#7][#6]1[#6][#6][#6](Br)[#6][#6]1', 0),
647:('Cl[#6]1[#6][#6][#6](Cl)[#6][#6]1', 0),
648:('Cl[#6]1[#6][#6][#6](Br)[#6][#6]1', 0),
649:('Br[#6]1[#6][#6][#6](Br)[#6][#6]1', 0),
650:('[#6][#6]1[#6][#6]([#6])[#6][#6][#6]1', 0),
651:('[#6][#6]1[#6][#6]([#8])[#6][#6][#6]1', 0),
652:('[#6][#6]1[#6][#6]([#16])[#6][#6][#6]1', 0),
653:('[#6][#6]1[#6][#6]([#7])[#6][#6][#6]1', 0),
654:('[#6][#6]1[#6][#6](Cl)[#6][#6][#6]1', 0),
655:('[#6][#6]1[#6][#6](Br)[#6][#6][#6]1', 0),
656:('[#8][#6]1[#6][#6]([#8])[#6][#6][#6]1', 0),
657:('[#8][#6]1[#6][#6]([#16])[#6][#6][#6]1', 0),
658:('[#8][#6]1[#6][#6]([#7])[#6][#6][#6]1', 0),
659:('[#8][#6]1[#6][#6](Cl)[#6][#6][#6]1', 0),
660:('[#8][#6]1[#6][#6](Br)[#6][#6][#6]1', 0),
661:('[#16][#6]1[#6][#6]([#16])[#6][#6][#6]1', 0),
662:('[#16][#6]1[#6][#6]([#7])[#6][#6][#6]1', 0),
663:('[#16][#6]1[#6][#6](Cl)[#6][#6][#6]1', 0),
664:('[#16][#6]1[#6][#6](Br)[#6][#6][#6]1', 0),
665:('[#7][#6]1[#6][#6]([#7])[#6][#6][#6]1', 0),
666:('[#7][#6]1[#6][#6](Cl)[#6][#6][#6]1', 0),
667:('[#7][#6]1[#6][#6](Br)[#6][#6][#6]1', 0),
668:('Cl[#6]1[#6][#6](Cl)[#6][#6][#6]1', 0),
669:('Cl[#6]1[#6][#6](Br)[#6][#6][#6]1', 0),
670:('Br[#6]1[#6][#6](Br)[#6][#6][#6]1', 0),
671:('[#6][#6]1[#6]([#6])[#6][#6][#6][#6]1', 0),
672:('[#6][#6]1[#6]([#8])[#6][#6][#6][#6]1', 0),
673:('[#6][#6]1[#6]([#16])[#6][#6][#6][#6]1', 0),
674:('[#6][#6]1[#6]([#7])[#6][#6][#6][#6]1', 0),
675:('[#6][#6]1[#6](Cl)[#6][#6][#6][#6]1', 0),
676:('[#6][#6]1[#6](Br)[#6][#6][#6][#6]1', 0),
677:('[#8][#6]1[#6]([#8])[#6][#6][#6][#6]1', 0),
678:('[#8][#6]1[#6]([#16])[#6][#6][#6][#6]1', 0),
679:('[#8][#6]1[#6]([#7])[#6][#6][#6][#6]1', 0),
680:('[#8][#6]1[#6](Cl)[#6][#6][#6][#6]1', 0),
681:('[#8][#6]1[#6](Br)[#6][#6][#6][#6]1', 0),
682:('[#16][#6]1[#6]([#16])[#6][#6][#6][#6]1', 0),
683:('[#16][#6]1[#6]([#7])[#6][#6][#6][#6]1', 0),
684:('[#16][#6]1[#6](Cl)[#6][#6][#6][#6]1', 0),
685:('[#16][#6]1[#6](Br)[#6][#6][#6][#6]1', 0),
686:('[#7][#6]1[#6]([#7])[#6][#6][#6][#6]1', 0),
687:('[#7][#6]1[#6](Cl)[#6][#6][#6][#6]1', 0),
688:('[#7][#6]1[#6](Br)[#6][#6][#6][#6]1', 0),
689:('Cl[#6]1[#6](Cl)[#6][#6][#6][#6]1', 0),
690:('Cl[#6]1[#6](Br)[#6][#6][#6][#6]1', 0),
691:('Br[#6]1[#6](Br)[#6][#6][#6][#6]1', 0),
692:('[#6][#6]1[#6][#6]([#6])[#6][#6]1', 0),
693:('[#6][#6]1[#6][#6]([#8])[#6][#6]1', 0),
694:('[#6][#6]1[#6][#6]([#16])[#6][#6]1', 0),
695:('[#6][#6]1[#6][#6]([#7])[#6][#6]1', 0),
696:('[#6][#6]1[#6][#6](Cl)[#6][#6]1', 0),
697:('[#6][#6]1[#6][#6](Br)[#6][#6]1', 0),
698:('[#8][#6]1[#6][#6]([#8])[#6][#6]1', 0),
699:('[#8][#6]1[#6][#6]([#16])[#6][#6]1', 0),
700:('[#8][#6]1[#6][#6]([#7])[#6][#6]1', 0),
701:('[#8][#6]1[#6][#6](Cl)[#6][#6]1', 0),
702:('[#8][#6]1[#6][#6](Br)[#6][#6]1', 0),
703:('[#16][#6]1[#6][#6]([#16])[#6][#6]1', 0),
704:('[#16][#6]1[#6][#6]([#7])[#6][#6]1', 0),
705:('[#16][#6]1[#6][#6](Cl)[#6][#6]1', 0),
706:('[#16][#6]1[#6][#6](Br)[#6][#6]1', 0),
707:('[#7][#6]1[#6][#6]([#7])[#6][#6]1', 0),
708:('[#7][#6]1[#6][#6](Cl)[#6][#6]1', 0),
709:('[#7][#6]1[#6][#6](Br)[#6][#6]1', 0),
710:('Cl[#6]1[#6][#6](Cl)[#6][#6]1', 0),
711:('Cl[#6]1[#6][#6](Br)[#6][#6]1', 0),
712:('Br[#6]1[#6][#6](Br)[#6][#6]1', 0),
713:('[#6][#6]1[#6]([#6])[#6][#6][#6]1', 0),
714:('[#6][#6]1[#6]([#8])[#6][#6][#6]1', 0),
715:('[#6][#6]1[#6]([#16])[#6][#6][#6]1', 0),
716:('[#6][#6]1[#6]([#7])[#6][#6][#6]1', 0),
717:('[#6][#6]1[#6](Cl)[#6][#6][#6]1', 0),
718:('[#6][#6]1[#6](Br)[#6][#6][#6]1', 0),
719:('[#8][#6]1[#6]([#8])[#6][#6][#6]1', 0),
720:('[#8][#6]1[#6]([#16])[#6][#6][#6]1', 0),
721:('[#8][#6]1[#6]([#7])[#6][#6][#6]1', 0),
722:('[#8][#6]1[#6](Cl)[#6][#6][#6]1', 0),
723:('[#8][#6]1[#6](Br)[#6][#6][#6]1', 0),
724:('[#16][#6]1[#6]([#16])[#6][#6][#6]1', 0),
725:('[#16][#6]1[#6]([#7])[#6][#6][#6]1', 0),
726:('[#16][#6]1[#6](Cl)[#6][#6][#6]1', 0),
727:('[#16][#6]1[#6](Br)[#6][#6][#6]1', 0),
728:('[#7][#6]1[#6]([#7])[#6][#6][#6]1', 0),
729:('[#7][#6]1[#6](Cl)[#6][#6]1', 0),
730:('[#7][#6]1[#6](Br)[#6][#6][#6]1', 0),
731:('Cl[#6]1[#6](Cl)[#6][#6][#6]1', 0),
732:('Cl[#6]1[#6](Br)[#6][#6][#6]1', 0),
733:('Br[#6]1[#6](Br)[#6][#6][#6]1', 0)}

PubchemKeys = None


def InitKeys(keyList, keyDict):
  assert len(keyList) == len(keyDict.keys()), 'length mismatch'
  for key in keyDict.keys():
    patt, count = keyDict[key]
    if patt != '?':
      sma = Chem.MolFromSmarts(patt)
      if not sma:
        print('SMARTS parser error for key #%d: %s' % (key, patt))
      else:
        keyList[key - 1] = sma, count


def calcPubChemFingerPart1(mol, **kwargs):
  global PubchemKeys
  if PubchemKeys is None:
    PubchemKeys = [(None, 0)] * len(smartsPatts.keys())

    InitKeys(PubchemKeys, smartsPatts)
  ctor = kwargs.get('ctor', DataStructs.SparseBitVect)

  res = ctor(len(PubchemKeys) + 1)
  for i, (patt, count) in enumerate(PubchemKeys):
    if patt is not None:
      if count == 0:
        res[i + 1] = mol.HasSubstructMatch(patt)
      else:
        matches = mol.GetSubstructMatches(patt)
        if len(matches) > count:
          res[i + 1] = 1
  return res


def func_1(mol,bits):
    ringSize=[]
    temp={3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    AllRingsAtom = mol.GetRingInfo().AtomRings()
    for ring in AllRingsAtom:
        ringSize.append(len(ring))
        for k,v in temp.items():
            if len(ring) == k:
                temp[k]+=1
    if temp[3]>=2:
        bits[0]=1;bits[7]=1
    elif temp[3]==1:
        bits[0]=1
    else:
        pass
    if temp[4]>=2:
        bits[14]=1;bits[21]=1
    elif temp[4]==1:
        bits[14]=1
    else:
        pass
    if temp[5]>=5:
        bits[28]=1;bits[35]=1;bits[42]=1;bits[49]=1;bits[56]=1
    elif temp[5]==4:
        bits[28]=1;bits[35]=1;bits[42]=1;bits[49]=1
    elif temp[5]==3:
        bits[28]=1;bits[35]=1;bits[42]=1
    elif temp[5]==2:
        bits[28]=1;bits[35]=1
    elif temp[5]==1:
        bits[28]=1
    else:
        pass
    if temp[6]>=5:
        bits[63]=1;bits[70]=1;bits[77]=1;bits[84]=1;bits[91]=1
    elif temp[6]==4:
        bits[63]=1;bits[70]=1;bits[77]=1;bits[84]=1
    elif temp[6]==3:
        bits[63]=1;bits[70]=1;bits[77]=1
    elif temp[6]==2:
        bits[63]=1;bits[70]=1
    elif temp[6]==1:
        bits[63]=1
    else:
        pass
    if temp[7]>=2:
        bits[98]=1;bits[105]=1
    elif temp[7]==1:
        bits[98]=1
    else:
        pass
    if temp[8]>=2:
        bits[112]=1;bits[119]=1
    elif temp[8]==1:
        bits[112]=1
    else:
        pass
    if temp[9]>=1:
        bits[126]=1;
    else:
        pass
    if temp[10]>=1:
        bits[133]=1;
    else:
        pass

    return ringSize,bits


def func_2(mol,bits):
    """ *Internal Use Only*
    saturated or aromatic carbon-only ring
    """
    AllRingsBond = mol.GetRingInfo().BondRings()
    ringSize=[]
    temp={3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    for ring in AllRingsBond:
        ######### saturated
        nonsingle = False
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name!='SINGLE':
                nonsingle = True
                break
        if nonsingle == False:
            ringSize.append(len(ring))
            for k,v in temp.items():
                if len(ring) == k:
                    temp[k]+=1
        ######## aromatic carbon-only     
        aromatic = True
        AllCarb = True
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name!='AROMATIC':
                aromatic = False
                break
        for bondIdx in ring:
            BeginAtom = mol.GetBondWithIdx(bondIdx).GetBeginAtom()
            EndAtom = mol.GetBondWithIdx(bondIdx).GetEndAtom()
            if BeginAtom.GetAtomicNum() != 6 or EndAtom.GetAtomicNum() != 6:
                AllCarb = False
                break
        if aromatic == True and AllCarb == True:
            ringSize.append(len(ring))
            for k,v in temp.items():
                if len(ring) == k:
                    temp[k]+=1
    if temp[3]>=2:
        bits[1]=1;bits[8]=1
    elif temp[3]==1:
        bits[1]=1
    else:
        pass
    if temp[4]>=2:
        bits[15]=1;bits[22]=1
    elif temp[4]==1:
        bits[15]=1
    else:
        pass
    if temp[5]>=5:
        bits[29]=1;bits[36]=1;bits[43]=1;bits[50]=1;bits[57]=1
    elif temp[5]==4:
        bits[29]=1;bits[36]=1;bits[43]=1;bits[50]=1
    elif temp[5]==3:
        bits[29]=1;bits[36]=1;bits[43]=1
    elif temp[5]==2:
        bits[29]=1;bits[36]=1
    elif temp[5]==1:
        bits[29]=1
    else:
        pass
    if temp[6]>=5:
        bits[64]=1;bits[71]=1;bits[78]=1;bits[85]=1;bits[92]=1
    elif temp[6]==4:
        bits[64]=1;bits[71]=1;bits[78]=1;bits[85]=1
    elif temp[6]==3:
        bits[64]=1;bits[71]=1;bits[78]=1
    elif temp[6]==2:
        bits[64]=1;bits[71]=1
    elif temp[6]==1:
        bits[64]=1
    else:
        pass
    if temp[7]>=2:
        bits[99]=1;bits[106]=1
    elif temp[7]==1:
        bits[99]=1
    else:
        pass
    if temp[8]>=2:
        bits[113]=1;bits[120]=1
    elif temp[8]==1:
        bits[113]=1
    else:
        pass
    if temp[9]>=1:
        bits[127]=1;
    else:
        pass
    if temp[10]>=1:
        bits[134]=1;
    else:
        pass
    return ringSize, bits


def func_3(mol,bits):
    AllRingsBond = mol.GetRingInfo().BondRings()
    ringSize=[]
    temp={3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    for ring in AllRingsBond:
        ######### saturated
        nonsingle = False
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name!='SINGLE':
                nonsingle = True
                break
        if nonsingle == False:
            ringSize.append(len(ring))
            for k,v in temp.items():
                if len(ring) == k:
                    temp[k]+=1
        ######## aromatic nitrogen-containing    
        aromatic = True
        ContainNitro = False
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name!='AROMATIC':
                aromatic = False
                break
        for bondIdx in ring:
            BeginAtom = mol.GetBondWithIdx(bondIdx).GetBeginAtom()
            EndAtom = mol.GetBondWithIdx(bondIdx).GetEndAtom()
            if BeginAtom.GetAtomicNum() == 7 or EndAtom.GetAtomicNum() == 7:
                ContainNitro = True
                break
        if aromatic == True and ContainNitro == True:
            ringSize.append(len(ring))
            for k,v in temp.items():
                if len(ring) == k:
                    temp[k]+=1
    if temp[3]>=2:
        bits[2]=1;bits[9]=1
    elif temp[3]==1:
        bits[2]=1
    else:
        pass
    if temp[4]>=2:
        bits[16]=1;bits[23]=1
    elif temp[4]==1:
        bits[16]=1
    else:
        pass
    if temp[5]>=5:
        bits[30]=1;bits[37]=1;bits[44]=1;bits[51]=1;bits[58]=1
    elif temp[5]==4:
        bits[30]=1;bits[37]=1;bits[44]=1;bits[51]=1
    elif temp[5]==3:
        bits[30]=1;bits[37]=1;bits[44]=1
    elif temp[5]==2:
        bits[30]=1;bits[37]=1
    elif temp[5]==1:
        bits[30]=1
    else:
        pass
    if temp[6]>=5:
        bits[65]=1;bits[72]=1;bits[79]=1;bits[86]=1;bits[93]=1
    elif temp[6]==4:
        bits[65]=1;bits[72]=1;bits[79]=1;bits[86]=1
    elif temp[6]==3:
        bits[65]=1;bits[72]=1;bits[79]=1
    elif temp[6]==2:
        bits[65]=1;bits[72]=1
    elif temp[6]==1:
        bits[65]=1
    else:
        pass
    if temp[7]>=2:
        bits[100]=1;bits[107]=1
    elif temp[7]==1:
        bits[100]=1
    else:
        pass
    if temp[8]>=2:
        bits[114]=1;bits[121]=1
    elif temp[8]==1:
        bits[114]=1
    else:
        pass
    if temp[9]>=1:
        bits[128]=1;
    else:
        pass
    if temp[10]>=1:
        bits[135]=1;
    else:
        pass
    return ringSize, bits


def func_4(mol,bits):
    AllRingsBond = mol.GetRingInfo().BondRings()
    ringSize=[]
    temp={3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    for ring in AllRingsBond:
        ######### saturated
        nonsingle = False
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name!='SINGLE':
                nonsingle = True
                break
        if nonsingle == False:
            ringSize.append(len(ring))
            for k,v in temp.items():
                if len(ring) == k:
                    temp[k]+=1
        ######## aromatic heteroatom-containing
        aromatic = True
        heteroatom = False
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name!='AROMATIC':
                aromatic = False
                break
        for bondIdx in ring:
            BeginAtom = mol.GetBondWithIdx(bondIdx).GetBeginAtom()
            EndAtom = mol.GetBondWithIdx(bondIdx).GetEndAtom()
            if BeginAtom.GetAtomicNum() not in [1,6] or EndAtom.GetAtomicNum() not in [1,6]:
                heteroatom = True
                break
        if aromatic == True and heteroatom == True:
            ringSize.append(len(ring))
            for k,v in temp.items():
                if len(ring) == k:
                    temp[k]+=1
    if temp[3]>=2:
        bits[3]=1;bits[10]=1
    elif temp[3]==1:
        bits[3]=1
    else:
        pass
    if temp[4]>=2:
        bits[17]=1;bits[24]=1
    elif temp[4]==1:
        bits[17]=1
    else:
        pass
    if temp[5]>=5:
        bits[31]=1;bits[38]=1;bits[45]=1;bits[52]=1;bits[59]=1
    elif temp[5]==4:
        bits[31]=1;bits[38]=1;bits[45]=1;bits[52]=1
    elif temp[5]==3:
        bits[31]=1;bits[38]=1;bits[45]=1
    elif temp[5]==2:
        bits[31]=1;bits[38]=1
    elif temp[5]==1:
        bits[31]=1
    else:
        pass
    if temp[6]>=5:
        bits[66]=1;bits[73]=1;bits[80]=1;bits[87]=1;bits[94]=1
    elif temp[6]==4:
        bits[66]=1;bits[73]=1;bits[80]=1;bits[87]=1
    elif temp[6]==3:
        bits[66]=1;bits[73]=1;bits[80]=1
    elif temp[6]==2:
        bits[66]=1;bits[73]=1
    elif temp[6]==1:
        bits[66]=1
    else:
        pass
    if temp[7]>=2:
        bits[101]=1;bits[108]=1
    elif temp[7]==1:
        bits[101]=1
    else:
        pass
    if temp[8]>=2:
        bits[115]=1;bits[122]=1
    elif temp[8]==1:
        bits[115]=1
    else:
        pass
    if temp[9]>=1:
        bits[129]=1;
    else:
        pass
    if temp[10]>=1:
        bits[136]=1;
    else:
        pass
    return ringSize,bits


def func_5(mol,bits):
    ringSize=[]
    AllRingsBond = mol.GetRingInfo().BondRings()
    temp={3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    for ring in AllRingsBond:
        unsaturated = False
        nonaromatic = True
        Allcarb = True
        ######### unsaturated
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name!='SINGLE':
                unsaturated = True
                break
        ######## non-aromatic
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name=='AROMATIC':
                nonaromatic = False
                break
        ######## allcarb
        for bondIdx in ring:
            BeginAtom = mol.GetBondWithIdx(bondIdx).GetBeginAtom()
            EndAtom = mol.GetBondWithIdx(bondIdx).GetEndAtom()
            if BeginAtom.GetAtomicNum() != 6 or EndAtom.GetAtomicNum() != 6:
                Allcarb = False
                break
        if unsaturated == True and nonaromatic == True and Allcarb == True:
            ringSize.append(len(ring))
            for k,v in temp.items():
                if len(ring) == k:
                    temp[k]+=1
    if temp[3]>=2:
        bits[4]=1;bits[11]=1
    elif temp[3]==1:
        bits[4]=1
    else:
        pass
    if temp[4]>=2:
        bits[18]=1;bits[25]=1
    elif temp[4]==1:
        bits[18]=1
    else:
        pass
    if temp[5]>=5:
        bits[32]=1;bits[39]=1;bits[46]=1;bits[53]=1;bits[60]=1
    elif temp[5]==4:
        bits[32]=1;bits[39]=1;bits[46]=1;bits[53]=1
    elif temp[5]==3:
        bits[32]=1;bits[39]=1;bits[46]=1
    elif temp[5]==2:
        bits[32]=1;bits[39]=1
    elif temp[5]==1:
        bits[32]=1
    else:
        pass
    if temp[6]>=5:
        bits[67]=1;bits[74]=1;bits[81]=1;bits[88]=1;bits[95]=1
    elif temp[6]==4:
        bits[67]=1;bits[74]=1;bits[81]=1;bits[88]=1
    elif temp[6]==3:
        bits[67]=1;bits[74]=1;bits[81]=1
    elif temp[6]==2:
        bits[67]=1;bits[74]=1
    elif temp[6]==1:
        bits[67]=1
    else:
        pass
    if temp[7]>=2:
        bits[102]=1;bits[109]=1
    elif temp[7]==1:
        bits[102]=1
    else:
        pass
    if temp[8]>=2:
        bits[116]=1;bits[123]=1
    elif temp[8]==1:
        bits[116]=1
    else:
        pass
    if temp[9]>=1:
        bits[130]=1;
    else:
        pass
    if temp[10]>=1:
        bits[137]=1;
    else:
        pass
    return ringSize,bits


def func_6(mol,bits):
    ringSize=[]
    AllRingsBond = mol.GetRingInfo().BondRings()
    temp={3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    for ring in AllRingsBond:
        unsaturated = False
        nonaromatic = True
        ContainNitro = False
        ######### unsaturated
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name!='SINGLE':
                unsaturated = True
                break
        ######## non-aromatic
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name=='AROMATIC':
                nonaromatic = False
                break
        ######## nitrogen-containing
        for bondIdx in ring:
            BeginAtom = mol.GetBondWithIdx(bondIdx).GetBeginAtom()
            EndAtom = mol.GetBondWithIdx(bondIdx).GetEndAtom()
            if BeginAtom.GetAtomicNum() == 7 or EndAtom.GetAtomicNum() == 7:
                ContainNitro = True
                break
        if unsaturated == True and nonaromatic == True and ContainNitro== True:
            ringSize.append(len(ring))
            for k,v in temp.items():
                if len(ring) == k:
                    temp[k]+=1
    if temp[3]>=2:
        bits[5]=1;bits[12]=1
    elif temp[3]==1:
        bits[5]=1
    else:
        pass
    if temp[4]>=2:
        bits[19]=1;bits[26]=1
    elif temp[4]==1:
        bits[19]=1
    else:
        pass
    if temp[5]>=5:
        bits[33]=1;bits[40]=1;bits[47]=1;bits[54]=1;bits[61]=1
    elif temp[5]==4:
        bits[33]=1;bits[40]=1;bits[47]=1;bits[54]=1
    elif temp[5]==3:
        bits[33]=1;bits[40]=1;bits[47]=1
    elif temp[5]==2:
        bits[33]=1;bits[40]=1
    elif temp[5]==1:
        bits[33]=1
    else:
        pass
    if temp[6]>=5:
        bits[68]=1;bits[75]=1;bits[82]=1;bits[89]=1;bits[96]=1
    elif temp[6]==4:
        bits[68]=1;bits[75]=1;bits[82]=1;bits[89]=1
    elif temp[6]==3:
        bits[68]=1;bits[75]=1;bits[82]=1
    elif temp[6]==2:
        bits[68]=1;bits[75]=1
    elif temp[6]==1:
        bits[68]=1
    else:
        pass
    if temp[7]>=2:
        bits[103]=1;bits[110]=1
    elif temp[7]==1:
        bits[103]=1
    else:
        pass
    if temp[8]>=2:
        bits[117]=1;bits[124]=1
    elif temp[8]==1:
        bits[117]=1
    else:
        pass
    if temp[9]>=1:
        bits[131]=1;
    else:
        pass
    if temp[10]>=1:
        bits[138]=1;
    else:
        pass
    return ringSize,bits


def func_7(mol,bits):

    ringSize=[]
    AllRingsBond = mol.GetRingInfo().BondRings()
    temp={3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    for ring in AllRingsBond:
        unsaturated = False
        nonaromatic = True
        heteroatom = False
        ######### unsaturated
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name!='SINGLE':
                unsaturated = True
                break
        ######## non-aromatic
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name=='AROMATIC':
                nonaromatic = False
                break
        ######## heteroatom-containing
        for bondIdx in ring:
            BeginAtom = mol.GetBondWithIdx(bondIdx).GetBeginAtom()
            EndAtom = mol.GetBondWithIdx(bondIdx).GetEndAtom()
            if BeginAtom.GetAtomicNum() not in [1,6] or EndAtom.GetAtomicNum() not in [1,6]:
                heteroatom = True
                break
        if unsaturated == True and nonaromatic == True and heteroatom == True:
            ringSize.append(len(ring))
            for k,v in temp.items():
                if len(ring) == k:
                    temp[k]+=1
    if temp[3]>=2:
        bits[6]=1;bits[13]=1
    elif temp[3]==1:
        bits[6]=1
    else:
        pass
    if temp[4]>=2:
        bits[20]=1;bits[27]=1
    elif temp[4]==1:
        bits[20]=1
    else:
        pass
    if temp[5]>=5:
        bits[34]=1;bits[41]=1;bits[48]=1;bits[55]=1;bits[62]=1
    elif temp[5]==4:
        bits[34]=1;bits[41]=1;bits[48]=1;bits[55]=1
    elif temp[5]==3:
        bits[34]=1;bits[41]=1;bits[48]=1
    elif temp[5]==2:
        bits[34]=1;bits[41]=1
    elif temp[5]==1:
        bits[34]=1
    else:
        pass
    if temp[6]>=5:
        bits[69]=1;bits[76]=1;bits[83]=1;bits[90]=1;bits[97]=1
    elif temp[6]==4:
        bits[69]=1;bits[76]=1;bits[83]=1;bits[90]=1
    elif temp[6]==3:
        bits[69]=1;bits[76]=1;bits[83]=1
    elif temp[6]==2:
        bits[69]=1;bits[76]=1
    elif temp[6]==1:
        bits[69]=1
    else:
        pass
    if temp[7]>=2:
        bits[104]=1;bits[111]=1
    elif temp[7]==1:
        bits[104]=1
    else:
        pass
    if temp[8]>=2:
        bits[118]=1;bits[125]=1
    elif temp[8]==1:
        bits[118]=1
    else:
        pass
    if temp[9]>=1:
        bits[132]=1;
    else:
        pass
    if temp[10]>=1:
        bits[139]=1;
    else:
        pass
    return ringSize,bits


def func_8(mol, bits):

    AllRingsBond = mol.GetRingInfo().BondRings()
    temp={'aromatic':0,'heteroatom':0}
    for ring in AllRingsBond:
        aromatic = True
        heteroatom = False
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name!='AROMATIC':
                aromatic = False
                break
        if aromatic==True:
            temp['aromatic']+=1
        for bondIdx in ring:
            BeginAtom = mol.GetBondWithIdx(bondIdx).GetBeginAtom()
            EndAtom = mol.GetBondWithIdx(bondIdx).GetEndAtom()
            if BeginAtom.GetAtomicNum() not in [1,6] or EndAtom.GetAtomicNum() not in [1,6]:
                heteroatom = True
                break
        if heteroatom==True:
            temp['heteroatom']+=1
    if temp['aromatic']>=4:
        bits[140]=1;bits[142]=1;bits[144]=1;bits[146]=1
    elif temp['aromatic']==3:
        bits[140]=1;bits[142]=1;bits[144]=1
    elif temp['aromatic']==2:
        bits[140]=1;bits[142]=1
    elif temp['aromatic']==1:
        bits[140]=1
    else:
        pass
    if temp['aromatic']>=4 and temp['heteroatom']>=4:
        bits[141]=1;bits[143]=1;bits[145]=1;bits[147]=1
    elif temp['aromatic']==3 and temp['heteroatom']==3:
        bits[141]=1;bits[143]=1;bits[145]=1
    elif temp['aromatic']==2 and temp['heteroatom']==2:
        bits[141]=1;bits[143]=1
    elif temp['aromatic']==1 and temp['heteroatom']==1:
        bits[141]=1
    else:
        pass
    return bits


def calcPubChemFingerPart2(mol):# 116-263

    bits=[0]*148
    bits=func_1(mol,bits)[1]
    bits=func_2(mol,bits)[1]
    bits=func_3(mol,bits)[1]
    bits=func_4(mol,bits)[1]
    bits=func_5(mol,bits)[1]
    bits=func_6(mol,bits)[1]
    bits=func_7(mol,bits)[1]
    bits=func_8(mol,bits)

    return bits


def calcPubChemFingerAll(s):
    mol = Chem.MolFromSmiles(s)
    AllBits=[0]*881
    res1=list(calcPubChemFingerPart1(mol).ToBitString())
    for index, item in enumerate(res1[1:116]):
        if item == '1':
            AllBits[index] = 1
    for index2, item2 in enumerate(res1[116:734]):
        if item2 == '1':
            AllBits[index2+115+148] = 1
    res2=calcPubChemFingerPart2(mol)
    for index3, item3 in enumerate(res2):
        if item3==1:
            AllBits[index3+115]=1
    return np.array(AllBits)
# ------------------------------------