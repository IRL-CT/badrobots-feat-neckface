import os
import ast
import numpy as np
import pandas as pd


def getEthinicites(df):
    df["Q959"].replace("", np.nan, inplace=True)
    df.dropna(subset=["Q959"], inplace=True)
    participants_to_ignore = [1000, 1001, 1002]
    
    ethnicities = {}
    nationalities = {}
    ethnicity_column_name = "Q11.1"
    nationality_column_name = "Q727"
    for index, row in df[2:].iterrows():
        try:
            participant_id = ast.literal_eval(row["Q959"])
            if isinstance(participant_id, int) and participant_id not in participants_to_ignore:
                ethnicity = row[ethnicity_column_name]
                nationality = row[nationality_column_name]
                ethnicities[ethnicity] = ethnicities.get(ethnicity, 0) + 1
                nationalities[nationality] = nationalities.get(nationality, 0) + 1
        except (ValueError, SyntaxError) as e:
            print(f"Skipping malformed value in row {index}: {row['Q959']}")

    print("--- All Participants Ethnicity Stats ---")
    print(f"Number of Nationalities = {sum(nationalities.values())}")
    print(nationalities)
    print("---"*10)
    for ethnicity, count in ethnicities.items():
        print(f"{ethnicity}: {count}")

def convertSecondsToMins(duration):
    minutes = duration // 60  # Get the number of minutes
    remaining_seconds = duration % 60  # Get the remaining seconds
    total = f"{minutes}.{remaining_seconds}"
    return float(total)

def getStudyCompletionDurations(df):
    df["Q959"].replace("", np.nan, inplace=True)
    df.dropna(subset=["Q959"], inplace=True)
    participants_to_ignore = [1000, 1001, 1002]
    excluded_participants = {13, 17, 27, 29, 30}
    
    study_duration_column_name = "Duration (in seconds)"
    study_durations = []
    for index, row in df[2:].iterrows():
        try:
            participant_id = ast.literal_eval(row["Q959"])
            if isinstance(participant_id, int) and participant_id not in participants_to_ignore and participant_id not in excluded_participants:
                duration = ast.literal_eval(row[study_duration_column_name])
                mins = convertSecondsToMins(duration=duration)
                study_durations.append(mins)
        except (ValueError, SyntaxError) as e:
            print(f"Skipping malformed value in row {index}: {row['Q959']}")

    max_duration, min_duration = max(study_durations), min(study_durations)
    duration_mean, duration_std = np.mean(study_durations), np.std(study_durations)
    print("--- Final Participants Study Durations ---")
    print(f"Max Duration: {max_duration} \nMin Duration: {min_duration}")
    print(f"Duration Mean: {duration_mean} \nDuration Standard Deviation: {duration_std}")

def getParticipantAges(df):
    participant_ages = {
        "Male": [],
        "Female": [],
        "Non-binary": []
    }

    participants_to_ignore = [1000, 1001, 1002]
    excluded_participants = {13, 17, 27, 29, 30}
    df["Q959"].replace("", np.nan, inplace=True)
    df.dropna(subset=["Q959"], inplace=True)

    for index, row in df[2:].iterrows():
        try:
            participant_id = ast.literal_eval(row["Q959"])
            participant_gender = row["Q11.4"]
            participant_age = ast.literal_eval(row["Q726"])
            if isinstance(participant_id, int) and participant_id not in participants_to_ignore:
                participant_ages[participant_gender].append(participant_age)
        except (ValueError, SyntaxError) as e:
            print(f"Skipping malformed value in row {index}: {row['Q959']}")

    # Calculate stats for each gender
    for gender, ages in participant_ages.items():
        max_age = max(ages)
        min_age = min(ages)
        mean_age = np.mean(ages)
        std_age = np.std(ages)
        number_of_participants = len(ages)
        
        print(f"--- Stats for {gender} ---")
        print(f"Number of Participants: {number_of_participants}")
        print(f"Max Age: {max_age}")
        print(f"Min Age: {min_age}")
        print(f"Mean Age: {mean_age:.2f}")
        print(f"Standard Deviation: {std_age:.2f}")
        print("---" * 10)

    all_ages = sum(participant_ages.values(), [])
    all_participants_max_age, all_participants_min_age = max(all_ages), min(all_ages)
    all_participants_age_mean, all_participants_age_std = np.mean(all_ages), np.std(all_ages)

    print("--- All Participants Stats ---")
    print(f"Max Age: {all_participants_max_age} \nMin Age: {all_participants_min_age}")
    print(f"Mean Age: {all_participants_age_mean} \nStandard Deviation: {all_participants_age_std}")


def main():
    qualtrics_data_path = "../data/Bad Robots Empathy - NECKFACE_September 15, 2024_10.09.csv"
    qualtrics_df = pd.read_csv(qualtrics_data_path)
    # getParticipantAges(qualtics_df)
    # getStudyCompletionDurations(qualtics_df)
    getEthinicites(qualtrics_df)


if __name__ == "__main__":
    main()