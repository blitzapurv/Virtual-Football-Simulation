import streamlit as st
import time
import random
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import csv
import pickle
from utils import get_encoder, get_model



model = get_model()
target_encoder = get_encoder(k='target_label')
team_encoder = get_encoder(k='team_name')

df_info = pd.read_csv("data/data.csv", index_col=0)
feature_cols = ['home_team_id', 'away_team_id', 'time', 'home_goals_cum', 'away_goals_cum']
df_info = df_info[feature_cols]


def event_prediction(test_row, clf):
    preds = clf.predict_proba(test_row.reshape(1, -1))
    preds = np.round(preds, 2)
    preds = preds * 100
    preds = preds.reshape(-1)
    preds = preds.astype(int)
    preds[0] = preds[0] - 1
    preds[2] = preds[2] - 1
    preds[6] = preds[6] - 1
    preds[7] = preds[7] - 1
    preds[1] = preds[1] + 2
    preds[3] = preds[3] + 2
    
    temp_list = []
    for ix, ele in enumerate(preds):
        temp_list.extend([ix] * ele)

    random.shuffle(temp_list)
    choice = random.choice(temp_list)
    
    return choice



def simulate_game(t1, t2, t1_name, t2_name):
    clock = 0
    time_limit = 90
    tot_goals = {t1_name:0, t2_name:0}
    
    while clock <= time_limit:

        if clock < time_limit:
            test_row = np.array([t1, t2, clock,	tot_goals[t1_name], tot_goals[t2_name]])
            event = event_prediction(test_row, model)
            event_name = target_encoder.inverse_transform(np.int64(event).reshape(-1))[0]
            
            if event_name == "Attempt_1_1":
                tot_goals[t1_name] += 1
            elif event_name == "Attempt_2_1":
                tot_goals[t2_name] += 1
            else:
                pass
        
        if clock == time_limit:
            if tot_goals[t1_name] != tot_goals[t2_name]:
                winner = sorted(tot_goals.items(), key=lambda x: x[1])[-1][0]
                break
            else:
                time_limit += 30
                    
        clock += 1
        print(clock, event_name)
    return winner


def football_match():

    home_team = np.random.choice(df_info['home_team_id'].unique()) 
    away_team = np.random.choice(df_info['home_team_id'].unique())
    t1, t2 = home_team, away_team

    t1_name  = team_encoder.inverse_transform(home_team.reshape(-1,1))[0]
    t2_name =  team_encoder.inverse_transform(away_team.reshape(-1,1))[0]
    display_df = pd.DataFrame(columns=[t1_name, t2_name])
    tot_goals = {t1_name:0, t2_name:0}
    win_prob = {t1_name: "NA", t2_name: "NA"}
    
    clock = 0
    time_limit = 90
    place_holder = st.empty()
    place_holder1 = st.empty()
    place_holder2 = st.empty()
    
    while clock <= time_limit:
        
        with place_holder.container():
            st.write("Time on Clock :", clock)
            
        with place_holder1.container():
            col1, col2 = st.columns(2)

            if clock < time_limit:
                test_row = np.array([t1, t2, clock,	tot_goals[t1_name], tot_goals[t2_name]])
                event = event_prediction(test_row, model)
                event_name = target_encoder.inverse_transform(np.int64(event).reshape(-1))[0]
                
                if event_name == "Attempt_1_1":
                    tot_goals[t1_name] += 1
                elif event_name == "Attempt_2_1":
                    tot_goals[t2_name] += 1
                else:
                    pass
                
                winners_list = []
                for _ in range(100):
                    game_winner = simulate_game(t1, t2, t1_name, t2_name)
                    winners_list.append(game_winner)
                
                t1_odds = sum([1 for i in winners_list if i==t1_name])
                t2_odds = 100 - t1_odds
                win_prob[t1_name] = t1_odds
                win_prob[t2_name] = t2_odds
            
            if clock == time_limit:
                if tot_goals[t1_name] != tot_goals[t2_name]:
                    winner = sorted(tot_goals.items(), key=lambda x: x[1])[-1][0]
                    win_prob = {k:100 if k==winner else 0 for k in win_prob.keys()}
                    break
                else:
                    time_limit += 30
            
            with col1:
                display_df.loc["Goals", t1_name] = tot_goals[t1_name]
                display_df.loc["Goals", t2_name] = tot_goals[t2_name]
                st.write(display_df)
            with col2:
                st.write("Comments :", event_name)
                st.write('t1_odds : ', t1_odds)
                st.write('t2_odds : ', t2_odds)
                    
            
            clock += 1
            print(clock, event_name)
            time.sleep(1)
    return tot_goals, winner


st.title("Football Match Simulation")


if st.button("Simulate Match"):
    tot_goals, winner = football_match()
    st.write("The Final score:", tot_goals)
    st.write("Winner:", winner)