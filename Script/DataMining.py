import pandas as pd
from datetime import datetime

# Rearrange original dataset to have each day as a row and all variable as columns
data = pd.read_csv("dataset_mood_smartphone.csv")
d = data.iloc[:, 1:]
d[['date','time']] = d.time.str.split(" ",expand=True)
d = d[["id","date","time","variable","value"]]



l = []
for value in d["variable"]:
    if value not in l:
        l.append(value)

mod = d.groupby(["id","date","variable"],as_index=False)["value"].mean()



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




# Obtain the same as new_group, but with the sum for some variables instead of mean
data_sum = pd.read_csv("mood_data_sum.csv")
rearranged = pd.DataFrame(new_group["id"],columns=["id","date","mean_mood", 'mean_circumplex.arousal', 'mean_circumplex.valence', 'sum_call', 'sum_sms', 'mean_activity', 'sum_screen', 'sum_appCat.builtin', 'sum_appCat.communication', 'sum_appCat.entertainment', 'sum_appCat.finance', 'sum_appCat.game', 'sum_appCat.office', 'sum_appCat.other', 'sum_appCat.social', 'sum_appCat.travel', 'sum_appCat.unknown', 'sum_appCat.utilities', 'sum_appCat.weather'])
rearranged["date"] = new_group["date"]

v_names=list(data_sum.columns.values)
for n in range(2,len(v_names)):
    rearranged[v_names[n]]=data_sum[v_names[n]]




# Remove empty rows
no_missing = rearranged[["mean_mood", 'mean_circumplex.arousal', 'mean_circumplex.valence', 'mean_activity', 'sum_screen', 'sum_appCat.builtin', 'sum_appCat.communication', 'sum_appCat.entertainment', 'sum_appCat.finance', 'sum_appCat.game', 'sum_appCat.office', 'sum_appCat.other', 'sum_appCat.social', 'sum_appCat.travel', 'sum_appCat.unknown', 'sum_appCat.utilities', 'sum_appCat.weather']].isna().all(axis=1)
for row in range(len(no_missing)):
    if no_missing[row]:
        try:
            cleaned = cleaned.drop(index=row)
        except:
            cleaned = rearranged.drop(index=row)




# Remove not consecutive days
patients = [y for x, y in cleaned.groupby('id', as_index=False)]


for p in patients:
    p.reset_index(inplace=True,drop=True)
    dates = [datetime.strptime(d, "%Y-%m-%d") for d in p["date"]]
    date_ints = pd.DataFrame([d.toordinal() for d in dates])
    for el in range(len(date_ints)-5):
        if int(date_ints.iloc[el+5,]) != int(date_ints.iloc[el,]+5):
            try:
                f_clean = f_clean.drop(index=el)
            except:
                f_clean = p.drop(index=el)

        if int(date_ints.iloc[el+5,]) == int(date_ints.iloc[el,]+5) and el == len(date_ints)-6:
            try:
                f_clean
            except:
                f_clean=p
    try:
        final = final.append(f_clean)
    except:
        final = f_clean
    del f_clean





# Insert mean value per patient whenever the mood value is missing
mean_mood=[]
for p in patients:
    mean_mood.append(p["mean_mood"].mean())

patients = [y for x, y in final.groupby('id', as_index=False)]

final.reset_index(inplace=True,drop=True)
i=0
f_row=0
f_final = final
for p in patients:
    for row in p.itertuples():
        if p.isnull().loc[row.Index,"mean_mood"]:
            f_final.at[f_row,"mean_mood"] = mean_mood[i]
        f_row += 1
    i+=1

f_final.to_csv("FINAL_FINAL.csv",na_rep='NA')





# Windowing: average together N days
STEP = 5
patients = [y for x, y in f_final.groupby('id', as_index=False)]

def computeavg(data):
    data.reset_index(inplace=True, drop=True)
    avg = pd.DataFrame(data["id"],
                       columns=["id","date","mean_mood", 'mean_circumplex.arousal', 'mean_circumplex.valence',
                                'sum_call', 'sum_sms', 'mean_activity', 'sum_screen', 'sum_appCat.builtin',
                                'sum_appCat.communication', 'sum_appCat.entertainment', 'sum_appCat.finance',
                                'sum_appCat.game', 'sum_appCat.office', 'sum_appCat.other', 'sum_appCat.social',
                                'sum_appCat.travel', 'sum_appCat.unknown', 'sum_appCat.utilities',
                                'sum_appCat.weather',"TARGET"])
    for row in range(len(data) - STEP):
        for col in range(2, len(data.columns)):
            avg.iat[row, col] = data.iloc[row:row + STEP, col].mean()
    for row in avg.itertuples():
        if row.Index < len(avg)-5:
            avg.at[row.Index,"TARGET"] = data.loc[row.Index+STEP,"mean_mood"]
    return avg


for i in patients:
    try:
        avg_merged = avg_merged.append(computeavg(i))
    except:
        avg_merged = computeavg(i)

# Remove the last 5 empty rows for each patient
avg5_final = avg_merged.dropna(thresh=2)
avg5_final.to_csv("5days_final_final.csv",na_rep='NA')