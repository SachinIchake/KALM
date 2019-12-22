import  torch
import torch.nn.functional as F

input = torch.randn((3, 4, 5, 6))
sum = torch.sum(input, dim = 3)
print()
# tags =[]
# with open('D:\\Git\\KALM\data\\CoNLL-2003\\test.txt','r') as fp:
#     lines = fp.readlines()
#
#     for line in lines:
#         if line != '\n':
#             words = line.split()
#             tags.append(words[3])
#
# print(set(tags))

#
# with open('D:\\Git\\KALM\data\\CoNLL-2003\\train.txt','r') as fp:
#     linesTrain = fp.readlines()
# lines = linesTrain
# vOrg = []
# vLoc =[]
# vPerson =[]
# vGeneral =[]
# for line in lines:
#     if line != '\n':
#         words = line.split()
#         tag = words[3]
#         word = words[0]
#         if tag == 'I-ORG':
#             vOrg.append(word)
#         elif tag == 'I-PER':
#             vPerson.append(word)
#         elif tag == 'I-LOC':
#             vLoc.append(word)
#         elif tag == 'O':
#             vGeneral.append(word)
#
# vOrg = [set(vOrg)]
# vLoc =[set(vLoc)]
# vPerson =[set(vPerson)]
# vGeneral =[set(vGeneral)]
#
#
