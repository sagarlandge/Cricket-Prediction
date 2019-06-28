from django.shortcuts import render, render_to_response
from django.http import HttpRequest,HttpResponse

# Create your views here.
def ml_index(request):

    # print(y_pred)
    # ratio_data= {"abc":1}
    # return render(request,"index.html",{"a":ratio_data})
    import numpy
    import pandas
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    # Import the dataset
    dataset = pandas.read_csv(r'C:\Users\Sagar\Desktop\sagar_ml\Salary_Data.csv')
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values

    # Split the dataset into the training set and test set
    # We're splitting the data in 1/3, so out of 30 rows, 20 rows will go into the training set,
    # and 10 rows will go into the testing set.
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 1/3, random_state = 0)

    linearRegressor = LinearRegression()
    linearRegressor.fit(xTrain, yTrain)
    y_pred = linearRegressor.predict(xTest)
    return render(request,"index.html",{"a":y_pred})


def login_page(request):
    # template = loader.get_template("templates/login.html")
    # return HttpResponse(template.render)
    return render_to_response('login.html')

# def index(request):

#     return render(request,'index.html')


def web_mashup(request):
    return render(request,'web_mashup.html')   

def displayPlayer(request):
    print("Hi")
    return render(request,'table.html')  


def strategy(request):
    return render(request, 'strategy.html')    



def winningIndia(request):
    import pandas as pd
    import numpy as np
    from sklearn import model_selection
    from scipy.stats import norm
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    from sklearn.svm import SVR
    from sklearn import preprocessing
    from sklearn import ensemble
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
    from sklearn.metrics import f1_score
    from sklearn import model_selection
    from sklearn.model_selection import cross_val_score
    # from read_file import read_data
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score
    import pickle

    df = pd.read_csv(r'C:\Users\Sagar\Desktop\major project\CricketPrediction\CricketPrediction\dataset\final_dataset.csv')
    X = pd.DataFrame()
    X = pd.concat([X, pd.get_dummies(df['team1'], prefix = 'team1')], axis = 1)
    X = pd.concat([X, pd.get_dummies(df['team2'], prefix = 'team2')], axis = 1)
    X = pd.concat([X, pd.get_dummies(df['city'], prefix = 'city')], axis = 1)
    X = pd.concat([X, df['toss_winner'].to_frame() ], axis = 1)
    X = pd.concat([X, df['first_batting'].to_frame() ], axis = 1)
    X = pd.concat([X, pd.get_dummies(df['team1'], prefix = 'team1')], axis = 1)
    X = pd.concat([X, pd.get_dummies(df['team2'], prefix = 'team2')], axis = 1)
    X = pd.concat([X, pd.get_dummies(df['city'], prefix = 'city')], axis = 1)
    X = pd.concat([X, df['toss_winner'].to_frame() ], axis = 1)
    X = pd.concat([X, df['first_batting'].to_frame() ], axis = 1)
    #Team1_players_info
    Team1 = pd.DataFrame()
    Team1 = pd.concat([Team1, df['team1_player1_bat_rank'].to_frame()], axis = 1)
    Team1 = pd.concat([Team1, df['team1_player1_bowl_rank'].to_frame()], axis = 1)
    Team1 = pd.concat([Team1, df['team1_player2_bat_rank'].to_frame()], axis = 1)
    Team1 = pd.concat([Team1, df['team1_player2_bowl_rank'].to_frame()], axis = 1)
    Team1 = pd.concat([Team1, df['team1_player3_bat_rank'].to_frame()], axis = 1)
    Team1 = pd.concat([Team1, df['team1_player3_bowl_rank'].to_frame()], axis = 1)
    Team1 = pd.concat([Team1, df['team1_player4_bat_rank'].to_frame()], axis = 1)
    Team1 = pd.concat([Team1, df['team1_player4_bowl_rank'].to_frame()], axis = 1)
    Team1 = pd.concat([Team1, df['team1_player5_bat_rank'].to_frame()], axis = 1)
    Team1 = pd.concat([Team1, df['team1_player5_bowl_rank'].to_frame()], axis = 1)
    Team1 = pd.concat([Team1, df['team1_player6_bat_rank'].to_frame()], axis = 1)
    Team1 = pd.concat([Team1, df['team1_player6_bowl_rank'].to_frame()], axis = 1)
    Team1 = pd.concat([Team1, df['team1_player7_bat_rank'].to_frame()], axis = 1)
    Team1 = pd.concat([Team1, df['team1_player7_bowl_rank'].to_frame()], axis = 1)
    Team1 = pd.concat([Team1, df['team1_player8_bat_rank'].to_frame()], axis = 1)
    Team1 = pd.concat([Team1, df['team1_player8_bowl_rank'].to_frame()], axis = 1)
    Team1 = pd.concat([Team1, df['team1_player9_bat_rank'].to_frame()], axis = 1)
    Team1 = pd.concat([Team1, df['team1_player9_bowl_rank'].to_frame()], axis = 1)
    Team1 = pd.concat([Team1, df['team1_player10_bat_rank'].to_frame()], axis = 1)
    Team1 = pd.concat([Team1, df['team1_player10_bowl_rank'].to_frame()], axis = 1)
    Team1 = pd.concat([Team1, df['team1_player11_bat_rank'].to_frame()], axis = 1)
    Team1 = pd.concat([Team1, df['team1_player11_bowl_rank'].to_frame()], axis = 1)

    Team2 = pd.DataFrame()
    Team2 = pd.concat([Team2, df['team2_player1_bat_rank'].to_frame()], axis = 1)
    Team2 = pd.concat([Team2, df['team2_player1_bowl_rank'].to_frame()], axis = 1)
    Team2 = pd.concat([Team2, df['team2_player2_bat_rank'].to_frame()], axis = 1)
    Team2 = pd.concat([Team2, df['team2_player2_bowl_rank'].to_frame()], axis = 1)
    Team2 = pd.concat([Team2, df['team2_player3_bat_rank'].to_frame()], axis = 1)
    Team2 = pd.concat([Team2, df['team2_player3_bowl_rank'].to_frame()], axis = 1)
    Team2 = pd.concat([Team2, df['team2_player4_bat_rank'].to_frame()], axis = 1)
    Team2 = pd.concat([Team2, df['team2_player4_bowl_rank'].to_frame()], axis = 1)
    Team2 = pd.concat([Team2, df['team2_player5_bat_rank'].to_frame()], axis = 1)
    Team2 = pd.concat([Team2, df['team2_player5_bowl_rank'].to_frame()], axis = 1)
    Team2 = pd.concat([Team2, df['team2_player6_bat_rank'].to_frame()], axis = 1)
    Team2 = pd.concat([Team2, df['team2_player6_bowl_rank'].to_frame()], axis = 1)
    Team2 = pd.concat([Team2, df['team2_player7_bat_rank'].to_frame()], axis = 1)
    Team2 = pd.concat([Team2, df['team2_player7_bowl_rank'].to_frame()], axis = 1)
    Team2 = pd.concat([Team2, df['team2_player8_bat_rank'].to_frame()], axis = 1)
    Team2 = pd.concat([Team2, df['team2_player8_bowl_rank'].to_frame()], axis = 1)
    Team2 = pd.concat([Team2, df['team2_player9_bat_rank'].to_frame()], axis = 1)
    Team2 = pd.concat([Team2, df['team2_player9_bowl_rank'].to_frame()], axis = 1)
    Team2 = pd.concat([Team2, df['team2_player10_bat_rank'].to_frame()], axis = 1)
    Team2 = pd.concat([Team2, df['team2_player10_bowl_rank'].to_frame()], axis = 1)
    Team2 = pd.concat([Team2, df['team2_player11_bat_rank'].to_frame()], axis = 1)
    Team2 = pd.concat([Team2, df['team2_player11_bowl_rank'].to_frame()], axis = 1)
        
    X = pd.concat([X, Team1], axis = 1)
    X = pd.concat([X, Team2], axis = 1)
    y = df['match_winner']
    np.random.seed(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=0)


    #logistic 
    model4=LogisticRegression()
    # clf = model4.fit(X_train,y_train)
    # print(model4.score(X_test, y_test))
    # with open('LogisticRegression.pickle','wb') as f:
    #     pickle.dump(clf,f)

    # pickle_in = open('LogisticRegression.pickle','rb')
    # clf= pickle.load(pickle_in) 
    # new_test = pd.read_csv(r'C:\Users\Sagar\Desktop\major project\CricketPrediction\CricketPrediction\dataset\input_winning.csv')
    # predict_winner=clf.predict(new_test)
    # if(predict_winner==0):
    #     winner = "England"
    # else:
    #     winner = "India" 


    model4.fit(X_train,y_train) 
    new_test = pd.read_csv(r'C:\Users\Sagar\Desktop\major project\CricketPrediction\CricketPrediction\dataset\winning_prediction_input.csv')
    predict_winner=model4.predict(new_test)
    if(predict_winner==0):
        winner = "Predicted Winner: England"
    else:
        winner = "Predicted Winner: India" 
    return render(request,"runs.html",{"a":winner})





def rohitScore(request):
    import pandas as pd   
    data = pd.read_csv(r'C:\Users\Sagar\Desktop\major project\CricketPrediction\CricketPrediction\dataset\Ese_Rohit.csv')
    data = data.drop(['BF','Mins','4s','6s','Dismissal','Unnamed: 9','Unnamed: 13','Start Date'],axis=1)
    data = data.set_index("Runs")
    data = data.drop("DNB", axis=0)
    data = data.drop("TDNB",axis=0)
    data = data.reset_index()
    data[['Out','NotOut']]=data.Runs.str.split("*",expand=True)
    X=data.iloc[:,1:6].values
    y =data.iloc[:,6].values
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X_1 = LabelEncoder()
    X[:, 3] = labelencoder_X_1.fit_transform(X[:, 3])
    X[:, 4] = labelencoder_X_1.fit_transform(X[:, 4])
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 500, random_state = 0,max_depth=5)
    regressor.fit(X_train, y_train)
    new_test = pd.read_csv(r'C:\Users\Sagar\Desktop\major project\CricketPrediction\CricketPrediction\dataset\ese_rohit_input.csv')
    runs = regressor.predict(new_test)
    runs = int(runs)
    rohit = "Rohit Sharma is expected to score " + str(runs) 
    return render(request,"runs.html",{"a":rohit})


def viratScore(request):
    import pandas as pd
    data = pd.read_csv(r'C:\Users\Sagar\Desktop\major project\CricketPrediction\CricketPrediction\dataset\Ese_virat 1.csv')
    data = data.drop([10,17,32,55,151,160,171,194], axis=0)
    data = data.drop(['BF','Mins','4s','6s','Dismissal','Unnamed: 9','Unnamed: 13','Start Date'],axis=1)
    data = data.set_index("Runs")
    data = data.drop("DNB", axis=0)
    data = data.drop("TDNB",axis=0)
    data = data.reset_index()
    data[['Out','NotOut']]=data.Runs.str.split("*",expand=True)
    X=data.iloc[:,1:6].values
    y =data.iloc[:,6].values
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X_1 = LabelEncoder()
    X[:, 3] = labelencoder_X_1.fit_transform(X[:, 3])
    X[:, 4] = labelencoder_X_1.fit_transform(X[:, 4])
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 500, random_state = 0,max_depth=5)
    regressor.fit(X_train, y_train)
    new_test = pd.read_csv(r'C:\Users\Sagar\Desktop\major project\CricketPrediction\CricketPrediction\dataset\ese_virat_input.csv')
    runs = regressor.predict(new_test)
    runs = int(runs)
    virat = "Virat Kohli is expected to score " + str(runs)
    return render(request,"runs.html",{"a":virat})


def dhoniScore(request):
    import pandas as pd
    data = pd.read_csv(r'C:\Users\Sagar\Desktop\major project\CricketPrediction\CricketPrediction\dataset\Ese_Dhoni.csv')
    data = data.drop(['BF','Mins','4s','6s','Dismissal','Unnamed: 9','Unnamed: 13','Start Date'],axis=1)
    data = data.set_index("Runs")
    data = data.drop("DNB", axis=0)
    data = data.drop("TDNB",axis=0)
    data = data.reset_index()
    data[['Out','NotOut']]=data.Runs.str.split("*",expand=True)
    X=data.iloc[:,1:6].values
    y =data.iloc[:,6].values
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X_1 = LabelEncoder()
    X[:, 3] = labelencoder_X_1.fit_transform(X[:, 3])
    X[:, 4] = labelencoder_X_1.fit_transform(X[:, 4])
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 500, random_state = 0,max_depth=5)
    regressor.fit(X_train, y_train)
    new_test = pd.read_csv(r'C:\Users\Sagar\Desktop\major project\CricketPrediction\CricketPrediction\dataset\Ese_Dhoni_input.csv')
    runs = regressor.predict(new_test)
    runs = int(runs)
    dhoni = "MS Dhoni is expected to score " + str(runs)
    return render(request,"runs.html",{"a":dhoni})
    

def jasonRoy(request):
    import pandas as pd
    data = pd.read_csv(r'C:\Users\Sagar\Desktop\major project\CricketPrediction\CricketPrediction\dataset\ese_jasonroy.csv')
    data = data.drop(['BF','Mins','4s','6s','Dismissal','Unnamed: 9','Unnamed: 13','Start Date'],axis=1)
    data = data.set_index("Runs")
    data = data.drop("DNB", axis=0)
    data = data.drop("TDNB",axis=0)
    data = data.reset_index()
    data[['Out','NotOut']]=data.Runs.str.split("*",expand=True)
    X=data.iloc[:,1:6].values
    y =data.iloc[:,6].values
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X_1 = LabelEncoder()
    X[:, 3] = labelencoder_X_1.fit_transform(X[:, 3])
    X[:, 4] = labelencoder_X_1.fit_transform(X[:, 4])
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 500, random_state = 0,max_depth=5)
    regressor.fit(X_train, y_train)
    new_test = pd.read_csv(r'C:\Users\Sagar\Desktop\major project\CricketPrediction\CricketPrediction\dataset\ese_jasonroy_input.csv')
    runs = regressor.predict(new_test)
    runs = int(runs)
    jason = "Jason Roy is expected to score " + str(runs)
    return render(request,"runs.html",{"a":jason})

def bumrahScore(request):
    import pandas as pd
    data = pd.read_csv(r'C:\Users\Sagar\Desktop\major project\CricketPrediction\CricketPrediction\dataset\Ese_Bumrah.csv')
    data = data.drop(['Overs','Unnamed: 7','Unnamed: 11','Start Date','Econ'],axis=1)
    X=data.iloc[:,[0,3,4,5,6]].values
    y =data.iloc[:,1:3].values
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X_1 = LabelEncoder()
    X[:, 4] = labelencoder_X_1.fit_transform(X[:, 4])
    X[:, 3] = labelencoder_X_1.fit_transform(X[:, 3])
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 500, random_state = 0,max_depth=5)
    regressor.fit(X_train, y_train)
    new_test = pd.read_csv(r'C:\Users\Sagar\Desktop\major project\CricketPrediction\CricketPrediction\dataset\Ese_Bumrah_input.csv')
    runs = regressor.predict(new_test)
    # a = int(runs[0])
    # b = int(runs[1])
    # runs[0] =a
    # runs[1] =b
    runs = runs.astype(int)
    a = runs[0][0]
    b = runs[0][1]
    bumrah = "Jasprit Bumrah will concede " + str(a) + " runs and will take " + str(b) + " wickets."
    return render(request,"runs.html",{"a":bumrah})


def stokesScore(request):
    import pandas as pd
    import numpy as np 
    import pandas as pd

    # plot stylings
    data = pd.read_csv(r'C:\Users\Sagar\Desktop\major project\CricketPrediction\CricketPrediction\dataset\ese_stokes.csv')
    data = data.drop(['Overs','Unnamed: 7','Unnamed: 11','Start Date','Econ'],axis=1)
    data = data.set_index("Runs")
# 28 chnages


    data = data.drop("-", axis=0)
    data = data.reset_index()
    data.set_index("Mdns")
    data.reset_index()

    X=data.iloc[:,[0,3,4,5,6]].values
    y =data.iloc[:,1:3].values
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X_1 = LabelEncoder()
    X[:, 4] = labelencoder_X_1.fit_transform(X[:, 4])
    X[:, 3] = labelencoder_X_1.fit_transform(X[:, 3])
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 500, random_state = 0,max_depth=5)
    regressor.fit(X_train, y_train)
    new_test = pd.read_csv(r'C:\Users\Sagar\Desktop\major project\CricketPrediction\CricketPrediction\dataset\ese_stokes_input.csv')
    runs = regressor.predict(new_test)
    runs = runs.astype(int)
    a = runs[0][0]
    b = runs[0][1]
    stokes =  "Ben Stokes will concede " + str(a) + " runs and will take " + str(b) + " wickets."
    return render(request,"runs.html",{"a":stokes})    


def chahalScore(request):
    import pandas as pd
    data = pd.read_csv(r'C:\Users\Sagar\Desktop\major project\CricketPrediction\CricketPrediction\dataset\ese_chahal.csv')
    data = data.drop(['Overs','Unnamed: 7','Unnamed: 11','Start Date','Econ'],axis=1)
    data = data.set_index("Runs")
    data = data.drop("-", axis=0)
    data = data.reset_index()
    data.set_index("Mdns")
    data.reset_index()

    X=data.iloc[:,[0,3,4,5,6]].values
    y =data.iloc[:,1:3].values
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X_1 = LabelEncoder()
    X[:, 4] = labelencoder_X_1.fit_transform(X[:, 4])
    X[:, 3] = labelencoder_X_1.fit_transform(X[:, 3])
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 500, random_state = 0,max_depth=5)
    regressor.fit(X_train, y_train)
    new_test = pd.read_csv(r'C:\Users\Sagar\Desktop\major project\CricketPrediction\CricketPrediction\dataset\ese_chahal_input.csv')
    runs = regressor.predict(new_test)
    runs = runs.astype(int)
    a = runs[0][0]
    b = runs[0][1]
    chahal =  "Yuzvendra Chahal will concede " + str(a) + " runs and will take " + str(b) + " wickets."
    return render(request,"runs.html",{"a":chahal})     




# def strategies(request):
#     import numpy as np # linear algebra
#     import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#     dataset = pd.read_csv('bowler_strategy.csv')
#     dataset = dataset.drop(['BBI'],axis=1)
#     X= dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11]].values
#     from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#     labelencoder_X_1 = LabelEncoder()
#     from sklearn import preprocessing
#     le = preprocessing.LabelEncoder()
#     le_name_mapping = dict(zip(le.fit_transform(X[:, 0]), X[:, 0]))
#     print(le_name_mapping)
#     labelencoder_X_1 = LabelEncoder()
#     X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])
#     from sklearn.cluster import KMeans
#     kmeansmodel = KMeans(n_clusters= 3, init='k-means++', random_state=0)
#     y_kmeans= kmeansmodel.fit_predict(X)
#     tmkb = kmeansmodel.labels_
#     high = []
#     for i in range(100):
#         if(tmkb[i] == 1):
#             #High Risk Players
#             key = X[i][0]
#             high.append(le_name_mapping[key])
#     print(high)

#     medium = []
#     for i in range(100):
#         if(tmkb[i] == 0):
#             #Medium Risk Players
#             key = X[i][0]
#             medium.append(le_name_mapping[key])
#     print(medium)

#     low = []
#     for i in range(100):
#         if(tmkb[i] == 2):
#             #Low Risk Players
#             key = X[i][0]
#             low.append(le_name_mapping[key])
#     print(low)


#     india = []
#     england = []
#     india.extend(["Virat Kohli","Rohit Sharma","JJ Bumrah (INDIA)", "YS Chahal (INDIA)","MS Dhoni","RA Jadeja (INDIA)","Kedar Jadhav","Dinesh Karthik","Kuldeep Yadav (INDIA)","B Kumar (INDIA)","Mohammed Shami (INDIA)","HH Pandya (INDIA)","Rishabh Pant","KL Rahul","Vijay Shankar","Shikhar Dhawan"])
#     england.extend(["BA Stokes (ENG)","Jason Roy","MM Ali (ENG)","JC Archer (ENG)","Jonny Bairstow","Jos Buttler","TK Curran (ENG)","Liam Dawson","LE Plunkett (ENG)","AU Rashid (ENG)","Joe Root","Eoin Morgan","James Vince","CR Woakes (ENG)","MA Wood (ENG)","Joe Denly"])  
#     #Strategy for team India
#     for i in range(len(england)):
#         if(england[i] in high):
#             print(england[i],end=",")
#     # print("have very good record. Indian batsmen should play them safely!!")
    
#     for i in range(len(england)):
#     if(england[i] in medium):
#         print(england[i],end=",")
#     print("have decent record. Indian batsmen should look to play them normally and try to attack lose deliveries!!")

#     for i in range(len(england)):
#     if(england[i] in low):
#         print(england[i],end=",")
#     print("have poor record. Indian batsmen should try to attack these bowlers and score as many runs as possible!!")

#     #Strategy for team England
#     for i in range(len(india)):
#         if(india[i] in high):
#             print(india[i],end=",")
#         if(india[i] == "JJ Bumrah (INDIA)"):
#             print(india[i],end=",")
#     print("have very good record. England batsmen should play them safely!!")

#     for i in range(len(india)):
#     if(india[i] =="JJ Bumrah (INDIA)"):
#         continue
#     if(india[i] in medium):
#         print(india[i],end=",")
#     print("have decent record. England batsmen should look to play them normally and try to attack lose deliveries!!")
#     count = 0
#     for i in range(len(india)):
#         if(india[i] in low):
#             count = count + 1
#             print(india[i],end=",")
#     if(count == 0):
#         print("No indian bowlers have poor record.")
#     elif(count != 0):
#         print("have poor record. England batsmen should try to attack these bowlers and score as many runs as possible!!")