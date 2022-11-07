import matlab.engine

eng = matlab.engine.start_matlab()
fis = eng.readfis('mamdanitype1.fis')
M = eng.csvread("results_dataset.csv")
print(M[0])
A = []
output = eng.evalfis(fis, M[0])
print(output)