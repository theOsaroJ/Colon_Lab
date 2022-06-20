##Random Prior
#Data
data= pd.read_csv('train_Trial.csv')
data.head()
fug= data['fugacity'].dropna().to_numpy()
eps=data['eps_eff'].dropna().to_numpy()
load=data['loading'].dropna().to_numpy()

#Randomlizing
warnings.filterwarnings('ignore')
rng= np.random.RandomState(42)
training_indices= rng.choice(np.arange(fug.size), size=7, replace=False)
fug_pr, eps_pr, load_pr= fug[training_indices],eps[training_indices],load[training_indices]

#Putting into a CSV
pri=[]
for i,j,k in zip(fug_pr,eps_pr,load_pr):
    data_eq= [i,j,k]
    pri.append(data_eq)
    
pri= pd.DataFrame(pri, columns=['fugacity','eps_eff','loading'])
pri.to_csv('Prior2.csv')
