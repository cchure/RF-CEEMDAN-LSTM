import numpy as np
from PyEMD import EMD, CEEMDAN,Visualisation
import pandas as pd

data=pd.read_csv(r'小论文相关\final_data.csv')
# # t=data['date']
data1=data['PM2.5']
S=np.array(data1)
print(S)

# class Emd_lstm(nn.Module):
#     def __init__(self):
#         super(Emd_lstm,self).__init__()
#         self.


# print(S)
# S=np.array([311,334,343,226,266,302,237,183
# ,155,181,164,210,182,90,146,193,175,156,154,127,150,186,259,248,166,181,185,213,246,313,244,270,229,234])
# print(S)
# #t = np.arange(0, 3, 0.01)
# #S = np.sin(13*t + 0.2*t**1.4) - np.cos(3*t)

# # Extract imfs and residue
# # In case of EMD
ceemdan = CEEMDAN()(S)
imfs, res = ceemdan[:-1], ceemdan[-1]
print(imfs.shape)
vis = Visualisation()
vis.plot_imfs(imfs=imfs, residue=res)
#vis.plot_instant_freq(t, imfs=imfs)
vis.show()

imfs=pd.DataFrame(imfs)
res=pd.DataFrame(res)
imfs.to_csv(r'小论文相关\CEEMDAN\PM2.5_ceemdan.csv')
res.to_csv(r'小论文相关\CEEMDAN\PM2.5_res.csv')
# # # # In general:
# # # #components = EEMD()(S)
# # # #imfs, res = components[:-1], components[-1]


# from PyEMD import CEEMDAN,Visualisation
# import numpy as np
# import matplotlib.pyplot as plt

# if __name__ == "__main__":
#     s = np.random.random(100)
#     ceemdan = CEEMDAN()
#     cIMFs = ceemdan(s)
#     imfs, res = ceemdan.get_imfs_and_residue()
# # plt.plot(s)
# # plt.show()
# vis=Visualisation()
# vis.plot_imfs(imfs=imfs,residue=res)
# vis.show()
