import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from session_state import session
from config import SAVE_FILEPATH

def load_data() -> list:
    """
    load the result from previous users
    :return: list of dicts
    """
    df = pd.read_csv(SAVE_FILEPATH)
    result = df.to_dict('records')
    return result

def store_data(result_db) -> None:
    """
    should be called after the result has been generated
    :return: None
    """
    if len(result_db) == 0:
        pass
    df = pd.DataFrame(result_db)
    df.to_csv(SAVE_FILEPATH, index=False)

def vader_sentiment_scores(sentence):
    sentimentAnalyzer= SentimentIntensityAnalyzer()
    sentiment_dict = sentimentAnalyzer.polarity_scores(sentence)
    #print("Overall sentiment dictionary is: ", sentiment_dict)
    #print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative")
    #print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral")
    #print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive")
    #print("Sentence Overall Rated As", end = " ")
    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.05 :
        #print("Positive")
        return 1
 
    elif sentiment_dict['compound'] <= - 0.05 :
        #print("Negative")
        return -1
 
    else:
        #print("Neutral")
        return 0
    
if __name__ == "__main__":
    df = pd.read_pickle('/Users/kyle_lee/Desktop/Infineon_Hackathon/hackathon_challenge/data/github_issues.pkl')
    df = df.loc[df['pr'] == 'issue']
    samples = df.sample(n=4, random_state=42)
    for i in range(samples.shape[0]):
        print(f"{i}. ")
        content = samples['body'].iloc[i]        
        vader_sentiment_scores(content)