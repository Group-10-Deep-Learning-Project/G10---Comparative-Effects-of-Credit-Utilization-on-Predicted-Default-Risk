#from lr import run_Model as run_lr
#from svm import run_Model as run_svm
from RF import run_Model as run_rf
#from xgboost import run_Model as run_xgb
#from MLP import run_Model as run_mlp
import Preprocessing

#Set seeds for testing
seeds = [42,123,456]


### Preprocessing & Model Running
for seed in seeds:
    
    #Preprocessing
    x_v, y_v, x_train, y_train, x_test, y_test = Preprocessing.valuesWithSeed(seed)

    #Run Models based on seed

    print(f"Seed Number:{seed}")
    
    #print(f"Linear Model - Seed {seed}")
    #run_lr(seed,x_v, y_v, x_train, y_train, x_test, y_test)
    #print("\n")

    #print(f"MLP Model - Seed {seed}")
    #run_mlp(seed,x_v, y_v, x_train, y_train, x_test, y_test)
    #print("\n")

    print(f"Random Forest Model - Seed {seed}")
    run_rf(seed,x_v, y_v, x_train, y_train, x_test, y_test)
    print("\n")

    #print(f"Support Vector Machine Model - Seed {seed}")
    #run_svm(seed,x_v, y_v, x_train, y_train, x_test, y_test)
    #print("\n")

    #print(f"XGBoost Model - Seed {seed}")
    #run_xgb(seed,x_v, y_v, x_train, y_train, x_test, y_test)
    #print("\n")

### Counterfactual Script Running


### Print out results