import pandas as pd
import numpy as np 

dt=pd.read_csv("microdados.csv",usecols=["ano", "trimestre", 'id_uf', "capital", "id_upa", "id_estrato", "v1008", "v2001", "v2005", "v2007", "v2009", "v2010", "v3001","v3002", "v3009a", "v4001", "v4005","v403312"])

colunas=["ano", "trimestre", 'id_uf', "capital", "id_upa", "id_estrato", "id_dom", "n_pessoas_dom", "cond_dom", "sexo", "idade_morador", "raca", 'alfabetizado', "freq_escolar", "Grau_Estudo", "qtd_trabalhos", "Afastado_Trabalho","rendimentos"]

dt.columns=colunas #para substituir o nome das variáveis

dt["qtd_trabalhos"] = np.where(dt["qtd_trabalhos"] != 1,0, dt["qtd_trabalhos"])
dt["menor_idade"] = np.where(dt["idade_morador"] < 18, 1, 0)
dt["menor_idade_trabalha"] = dt["menor_idade"]*dt["qtd_trabalhos"] ##ESsa variável já consta no novo outcome
dt["id_domicilio"] = dt["id_upa"].astype("str")+dt["id_dom"].astype("str") #identificador do domicilio


#o nível do dado é trimestre-ano-domicilio

dt["sexo"]= np.where(dt["sexo"] != 1,0, dt["sexo"])
dt["raca"]= np.where(dt["raca"] != 1,0, dt["raca"])
dt["alfabetizado"]= np.where(dt["alfabetizado"] != 1,0, dt["alfabetizado"])
dt["freq_escolar"]= np.where(dt["freq_escolar"] != 1,0, dt["freq_escolar"])
dt["Afastado_Trabalho"] = np.where(dt["Afastado_Trabalho"] != 1,0, dt["Afastado_Trabalho"])

dt["novo_outcome"] =  dt["menor_idade"]*dt["qtd_trabalhos"] + dt["menor_idade"]*dt["Afastado_Trabalho"]

dm_estudo=pd.get_dummies(dt["Grau_Estudo"], prefix_sep="_", prefix="grau_estudo")
dm_estudo=pd.concat([dt[['ano', 'trimestre', 'id_domicilio']], dm_estudo], axis=1)

dm_estudo=dm_estudo.groupby(['ano', 'trimestre', 'id_domicilio']).mean()

data=dt.groupby(['ano', 'trimestre', 'id_domicilio']).agg({"id_uf":"first",
                                                                "n_pessoas_dom": "first",
                                                                "sexo":"mean",
                                                                "idade_morador":"mean",
                                                                "raca":"mean",
                                                                "alfabetizado":"mean",
                                                                "Grau_Estudo": "mean",
                                                                "freq_escolar":"mean",
                                                                "qtd_trabalhos":"mean",
                                                                "rendimentos":"sum"})

outcome=dt[["menor_idade_trabalha", "novo_outcome",'ano', 'trimestre', 'id_domicilio']].groupby(['ano', 'trimestre', 'id_domicilio']).sum() ## aqui colocamos menor idade no nível da basee (ano, trimestre e dominicilio)

dm_estudo.reset_index(inplace=True)
data.reset_index(inplace=True)
outcome.reset_index(inplace=True)

data = data.merge(outcome,on=['ano', 'trimestre', 'id_domicilio'])
data = data.merge(dm_estudo,on=['ano', 'trimestre', 'id_domicilio'])
data["menor_idade_trabalha"]= np.where(data["menor_idade_trabalha"] != 0,1, data["menor_idade_trabalha"]) ## ao invés de termos 5 tipos, vamos ter só tem trabalhando ou n
data["novo_outcome"]= np.where(data["novo_outcome"] != 0,1, data["novo_outcome"])

##separando dados
#até 2018 como treinamento e 2019 como teste

data_train=data.query("ano <= 2018") #80% do dado é treinamento.
data_teste=data.query("ano > 2018")

### estimando modelos ###

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=0) ##definindo modelo de ml

clf_model=clf.fit(data_train[['n_pessoas_dom', 'sexo',
       'idade_morador', 'raca', 'alfabetizado', 'freq_escolar',
       'qtd_trabalhos', 'rendimentos', 'grau_estudo_2.0', 'grau_estudo_3.0', 'grau_estudo_4.0',
       'grau_estudo_5.0', 'grau_estudo_6.0', 'grau_estudo_7.0',
       'grau_estudo_8.0', 'grau_estudo_9.0', 'grau_estudo_10.0',
       'grau_estudo_11.0', 'grau_estudo_12.0', 'grau_estudo_13.0',
       'grau_estudo_14.0', 'grau_estudo_15.0']], data_train["novo_outcome"]) ##treinando modelo de ml

y_predict=clf_model.predict(data_teste[['n_pessoas_dom', 'sexo',
       'idade_morador', 'raca', 'alfabetizado', 'freq_escolar',
       'qtd_trabalhos', 'rendimentos', 'grau_estudo_2.0', 'grau_estudo_3.0', 'grau_estudo_4.0',
       'grau_estudo_5.0', 'grau_estudo_6.0', 'grau_estudo_7.0',
       'grau_estudo_8.0', 'grau_estudo_9.0', 'grau_estudo_10.0',
       'grau_estudo_11.0', 'grau_estudo_12.0', 'grau_estudo_13.0',
       'grau_estudo_14.0', 'grau_estudo_15.0']]) ##gerando vetor de previsão do dado de teste a partir do modelo treinado clf_model

data_teste['out_predict']=y_predict ##salvando vetor de previsão na var out_predict

data_teste_trabalha=data_teste[data_teste.menor_idade_trabalha==1] ##filtrando o dado de teste para todas as obs que trabalham

data_teste_trabalha['out_predict'].value_counts(normalize=True) ##definindo o percentual de acerto do modelo para os indivíduos que trabalham

clf_model.score(data_teste[['n_pessoas_dom', 'sexo',
       'idade_morador', 'raca', 'alfabetizado', 'freq_escolar',
       'qtd_trabalhos', 'rendimentos', 'grau_estudo_2.0', 'grau_estudo_3.0', 'grau_estudo_4.0',
       'grau_estudo_5.0', 'grau_estudo_6.0', 'grau_estudo_7.0',
       'grau_estudo_8.0', 'grau_estudo_9.0', 'grau_estudo_10.0',
       'grau_estudo_11.0', 'grau_estudo_12.0', 'grau_estudo_13.0',
       'grau_estudo_14.0', 'grau_estudo_15.0']], data_teste["novo_outcome"])

pd.DataFrame(clf_model.feature_importances_) ## checa o peso dasvariáveis na previsão do y

##Agora vamos estimar o Random Forests Classifiers

from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(random_state=0, n_estimators=50)

clf_RFmodel=clf.fit(data_train[['n_pessoas_dom', 'sexo',
       'idade_morador', 'raca', 'alfabetizado', 'freq_escolar',
       'qtd_trabalhos', 'rendimentos', 'grau_estudo_2.0', 'grau_estudo_3.0', 'grau_estudo_4.0',
       'grau_estudo_5.0', 'grau_estudo_6.0', 'grau_estudo_7.0',
       'grau_estudo_8.0', 'grau_estudo_9.0', 'grau_estudo_10.0',
       'grau_estudo_11.0', 'grau_estudo_12.0', 'grau_estudo_13.0',
       'grau_estudo_14.0', 'grau_estudo_15.0']], data_train["novo_outcome"]) ##treinando modelo de ml

y_pred=clf_RFmodel.predict(data_teste[['n_pessoas_dom', 'sexo',
       'idade_morador', 'raca', 'alfabetizado', 'freq_escolar',
       'qtd_trabalhos', 'rendimentos', 'grau_estudo_2.0', 'grau_estudo_3.0', 'grau_estudo_4.0',
       'grau_estudo_5.0', 'grau_estudo_6.0', 'grau_estudo_7.0',
       'grau_estudo_8.0', 'grau_estudo_9.0', 'grau_estudo_10.0',
       'grau_estudo_11.0', 'grau_estudo_12.0', 'grau_estudo_13.0',
       'grau_estudo_14.0', 'grau_estudo_15.0']])

data_teste['out_predict']=y_pred ##salvando vetor de previsão na var out_predict

data_teste_trabalha=data_teste[data_teste.novo_outcome==1] ##filtrando o dado de teste para todas as obs que trabalham

data_teste_trabalha['out_predict'].value_counts(normalize=True) ##definindo o percentual de acerto do modelo para os indivíduos que trabalham


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(clf_RFmodel, y_pred))
pd.DataFrame(clf_RFmodel.feature_importances_)

###Regressão logistica

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=200, random_state=0)

lr_model=lr.fit(data_train[['n_pessoas_dom', 'sexo',
       'idade_morador', 'raca', 'alfabetizado', 'freq_escolar',
       'qtd_trabalhos', 'rendimentos', 'grau_estudo_2.0', 'grau_estudo_3.0', 'grau_estudo_4.0',
       'grau_estudo_5.0', 'grau_estudo_6.0', 'grau_estudo_7.0',
       'grau_estudo_8.0', 'grau_estudo_9.0', 'grau_estudo_10.0',
       'grau_estudo_11.0', 'grau_estudo_12.0', 'grau_estudo_13.0',
       'grau_estudo_14.0']], data_train["novo_outcome"]) ##treinando modelo de ml

y_pred=lr_model.predict(data_teste[['n_pessoas_dom', 'sexo',
       'idade_morador', 'raca', 'alfabetizado', 'freq_escolar',
       'qtd_trabalhos', 'rendimentos', 'grau_estudo_2.0', 'grau_estudo_3.0', 'grau_estudo_4.0',
       'grau_estudo_5.0', 'grau_estudo_6.0', 'grau_estudo_7.0',
       'grau_estudo_8.0', 'grau_estudo_9.0', 'grau_estudo_10.0',
       'grau_estudo_11.0', 'grau_estudo_12.0', 'grau_estudo_13.0',
       'grau_estudo_14.0']])

data_teste['out_predict']=y_pred ##salvando vetor de previsão na var out_predict

data_teste_trabalha=data_teste[data_teste.novo_outcome==1] ##filtrando o dado de teste para todas as obs que trabalham

data_teste_trabalha['out_predict'].value_counts(normalize=True) ##definindo o percentual de acerto do modelo para os indivíduos que trabalham

## novo teste

data_nao_tratado=data[data.novo_outcome==0]

data_nao_tratado=data_nao_tratado.sample(frac=0.020137281140739986, random_state=0)

data_nao_tratado=pd.concat([data_nao_tratado, data[data.novo_outcome==1]])

data_train=data_nao_tratado.query("ano <= 2018") #80% do dado é treinamento.
data_teste=data_nao_tratado.query("ano > 2018")

clf=RandomForestClassifier(random_state=0, n_estimators=50)

clf_RFmodel=clf.fit(data_train[['n_pessoas_dom', 'sexo',
       'idade_morador', 'raca', 'alfabetizado', 'freq_escolar',
       'qtd_trabalhos', 'rendimentos', 'grau_estudo_2.0', 'grau_estudo_3.0', 'grau_estudo_4.0',
       'grau_estudo_5.0', 'grau_estudo_6.0', 'grau_estudo_7.0',
       'grau_estudo_8.0', 'grau_estudo_9.0', 'grau_estudo_10.0',
       'grau_estudo_11.0', 'grau_estudo_12.0', 'grau_estudo_13.0',
       'grau_estudo_14.0', 'grau_estudo_15.0']], data_train["novo_outcome"]) ##treinando modelo de ml

y_pred=clf_RFmodel.predict(data_teste[['n_pessoas_dom', 'sexo',
       'idade_morador', 'raca', 'alfabetizado', 'freq_escolar',
       'qtd_trabalhos', 'rendimentos', 'grau_estudo_2.0', 'grau_estudo_3.0', 'grau_estudo_4.0',
       'grau_estudo_5.0', 'grau_estudo_6.0', 'grau_estudo_7.0',
       'grau_estudo_8.0', 'grau_estudo_9.0', 'grau_estudo_10.0',
       'grau_estudo_11.0', 'grau_estudo_12.0', 'grau_estudo_13.0',
       'grau_estudo_14.0', 'grau_estudo_15.0']])

data_teste['out_predict']=y_pred ##salvando vetor de previsão na var out_predict

data_teste_trabalha=data_teste[data_teste.novo_outcome==1] ##filtrando o dado de teste para todas as obs que trabalham

data_teste_trabalha['out_predict'].value_counts(normalize=True) ##definindo o percentual de acerto do modelo para os indivíduos que trabalham
