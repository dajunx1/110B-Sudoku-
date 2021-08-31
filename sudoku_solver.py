import numpy as np
import pandas as pd
import scipy.optimize as sco
import time

def solver(input_):
    I = np.identity(9)
    zero = np.zeros((9,9))
    # row constraint
    temp0 = I
    for i in range(1,9):
        temp0 = np.append(temp0,I,axis=1)
    row = temp0
    for i in range(1,9):
        row = np.append(row,np.zeros((9,81)),axis=1)
    for i in range(1,9):
        temp1 = np.zeros((9,81))
        for j in range(1,9):
            if i==j:
                temp1 = np.append(temp1,temp0,axis=1)
            else:
                temp1 = np.append(temp1,np.zeros((9,81)),axis=1)
        row = np.append(row,temp1,axis=0)
    # col constraint
    temp = np.append(I,np.zeros((9,72)),axis=1)
    col = temp
    for i in range(1,9):
        col = np.append(col,temp,axis=1)
    temp = col
    for i in range(1,9):
        temp1 = np.append(temp[:,729-9*i:],temp[:,:729-9*i],axis=1)
        col = np.append(col,temp1,axis=0)
    # box constraint
    box = np.zeros([9,729])
    for i in range(3):
        for j in range(3):
            temp0 = np.zeros((9,9))
            temp0[3*i:3*i+3,3*j:3*j+3] = 1
            temp0 = temp0.flatten()
            bc1 = np.zeros((9,9))
            for k in temp0:
                if k == 1:
                    temp = I
                else:
                    temp = np.zeros((9,9))
                bc1 = np.append(bc1,temp,axis=1)   
            box = np.append(box,bc1[:,9::],axis=0)
    box = box[9:,:]
    # cell constraint
    cell = []
    for i in range(len(input_)):
        if int(input_[i]) == 0:
            temp = []
            for j in range(i):
                temp.extend(np.zeros(9))
            temp.extend([1,1,1,1,1,1,1,1,1])
            for j in range(80-i):
                temp.extend(np.zeros(9))
            cell.append(temp)
    cell = np.array(cell)
    # clue constraint
    clue = []
    for i in range(len(input_)):
        if int(input_[i])!= 0:
            temp = []
            for j in range(i):
                temp.extend(np.zeros(9))
            temp1 = np.zeros(9)
            temp1[int(input_[i])-1] = 1
            temp.extend(temp1)
            for j in range(80 - i):
                temp.extend(np.zeros(9))
            clue.append(temp)
    clue = np.array(clue)
    A = np.append(row,col,axis=0)
    A = np.append(A,box,axis=0)
    A = np.append(A,cell,axis=0)
    A = np.append(A,clue,axis=0)
    # weighted LP
    x = np.zeros(729)
    w = x
    for i in range(1, 10):
        epsilon = 0.5
        for j in range(len(w)):
            w[j]= 1/(abs(x[j])**(i-1)+epsilon)
        c = np.block([w,w])
        A_eq = np.block([A, -A])
        b_eq = np.ones(len(A))
        G = np.block([[-np.eye(A.shape[1]), np.zeros((A.shape[1], A.shape[1]))],\
                         [np.zeros((A.shape[1], A.shape[1])), -np.eye(A.shape[1])]])
        h = np.zeros(A.shape[1]*2)
        sol = (sco.linprog(c,G, h, A_eq, b_eq ,method='interior-point', \
                         bounds = (0,None), options={'cholesky':False,'sym_pos':False,'sparse':True})).x
        sol0 = sol[:int(len(sol)/2)]
        sol1 = sol[int(len(sol)/2):]
        x_new = sol0-sol1
        if np.linalg.norm(x_new - x) < 1e-10:
            break
        else:
            x = x_new
    z = np.reshape(x,(81,9))
    answer = np.reshape(np.array([np.argmax(d)+1 for d in z]), (9,9))
    return answer

# test
'''
data = pd.read_csv("../input/large1.csv") 
import time
corr_cnt = 0
start = time.time()

if len(data) > 1000:
    np.random.seed(42)
    samples = np.random.choice(len(data), 1000)
else:
    samples = range(len(data))
for i in range(len(samples)):
    quiz = data["quizzes"][samples[i]]
    solu = data["solutions"][samples[i]]
    answer = solver(quiz)
    if np.linalg.norm(answer \
                      - np.reshape([int(c) for c in solu], (9,9)), np.inf) > 0:
        pass
    else:
        #print("CORRECT")
        corr_cnt += 1

    if (i+1) % 20 == 0:
        
        end = time.time()
        print("Aver Time: {t:6.2f} secs. Success rate: {corr} / {all} ".format(t=(end-start)/(i+1), corr=corr_cnt, all=i+1) )

end = time.time()
print("Aver Time: {t:6.2f} secs. Success rate: {corr} / {all} ".format(t=(end-start)/(i+1), corr=corr_cnt, all=i+1) )
'''
