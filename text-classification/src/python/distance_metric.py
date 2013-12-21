from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist

###############################################################################################
# Distance metric between strings
##############################################################################################

s1 = "JDL H-YF HJCNJDF-YF-LJYE U GTHDJVFQCRJUJ"
s2 = "JDL HFQJYF U GTHDJVFQCRJUJ HJCNJDC-YF-LJYE"

string1 = s1.replace(" ", "").replace("-", "").upper()
string2 = s2.replace(" ", "").replace("-", "").upper()

gap_open = -1
gap_extend = -1

penalty_matrix = {}
alphabet = "abcdefghijklmnopqrstuvwxyz".upper()

for c1 in alphabet:
	for c2 in alphabet:
		if c1 == c2:
			penalty_matrix[(c1, c2)] = 1
		else:
			penalty_matrix[(c1, c2)] = -1

#print(penalty_matrix) 	

alns = pairwise2.align.localds(string1, string2, penalty_matrix, gap_open, gap_extend)
top_aln = alns[0]
aln_s1, aln_s2, score, begin, end = top_aln

print string1+'\n'+string2
print aln_s1+'\n'+aln_s2
print score
