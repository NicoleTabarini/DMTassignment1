import pandas as pd


data = pd.read_csv("dataset_mood_smartphone.csv")
d = data.iloc[:, 1:]
d[['date','time']] = d.time.str.split(" ",expand=True)
d = d[["id","date","time","variable","value"]]
#print(d)

l = []
for value in d["variable"]:
    if value not in l:
        l.append(value)

mod = d.groupby(["id","date","variable"],as_index=False)["value"].mean()

#mod.to_csv("o.csv")
#print(mod)
new=pd.DataFrame(mod["id"],columns=["id","date",'mood', 'circumplex.arousal', 'circumplex.valence', 'activity', 'screen', 'call', 'sms', 'appCat.builtin', 'appCat.communication', 'appCat.entertainment', 'appCat.finance', 'appCat.game', 'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel', 'appCat.unknown', 'appCat.utilities', 'appCat.weather'])
new["date"] = mod["date"]
new_group = new.groupby(["id","date"],as_index=False).mean()

ind = 0
for row in mod.itertuples():
    var = mod["variable"].iloc[row.Index]
    val = mod["value"].iloc[row.Index]
    new_group.at[ind, var] = val
    if row.Index < len(mod.index)-1:
        if mod["date"].iloc[row.Index+1] != mod["date"].iloc[row.Index]:
                ind += 1

new_group.to_csv("new.csv",na_rep='NA')
print(new_group)